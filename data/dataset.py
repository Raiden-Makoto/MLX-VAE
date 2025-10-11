import pandas as pd
import numpy as np
from rdkit import Chem
import torch
import os
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Optional

torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])

class QM9GraphDataset(Dataset):
    """
    Custom dataset class to convert SMILES from QM9 CSV to PyTorch Geometric graphs
    with BBBP labels
    """
    def __init__(self, csv_path: str, root: str = None, transform=None, pre_transform=None, force_process=False):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        # Filter out invalid SMILES
        self.df = self.df.dropna(subset=['smiles'])
        # Set processed graphs directory path (but don't create yet)
        self._processed_dir = os.path.join(root or '.', 'processed')
        self.processed_dir_path = self._processed_dir
        # Don't pass root to parent - prevents auto directory creation
        super().__init__(None, transform, pre_transform)
        # Process data if not already processed or force_process is True
        if force_process or not self._check_processed():
            self.process()

    def _check_processed(self) -> bool:
        """Check if processed files exist and match the expected count."""
        if not os.path.exists(self.processed_dir_path):
            return False
        existing_files = [f for f in os.listdir(self.processed_dir_path) if f.endswith('.pt')]
        # Check if we have the expected number of processed files
        # (allowing for some that might have failed)
        return len(existing_files) >= len(self.df) * 0.95  # At least 95% processed

    @property
    def raw_file_names(self) -> List[str]:
        return []  # No raw files to download
    
    @property
    def raw_dir(self) -> str:
        # Return a path but we won't use it
        return self._processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        # Return empty list to prevent PyG from checking files
        return []
    
    @property
    def processed_dir(self) -> str:
        # Return the path - directory will be created only in process()
        return self._processed_dir

    def download(self):
        # Data is already provided as CSV
        pass

    def process(self):
        """Convert SMILES to graphs and save processed data."""
        # Only create the directory when we actually need to process
        os.makedirs(self.processed_dir_path, exist_ok=True)
        print(f"Processing {len(self.df)} molecules...")
        idx = 0
        failed = 0

        for i, row in self.df.iterrows():
            smiles = row['smiles']
            data = self.smiles_to_graph(
                smiles,
                bbbp_label=row.get('bbbp', 0.0)
            )
            if data is not None:
                if self.pre_transform:
                    data = self.pre_transform(data)
                torch.save(data, os.path.join(self.processed_dir_path, f'data_{idx}.pt'))
                idx += 1
            else:
                failed += 1

        print(f"Successfully processed {idx} molecules, failed: {failed}")
        self._indices = list(range(idx))

    def len(self) -> int:
        return len([f for f in os.listdir(self.processed_dir_path) if f.endswith('.pt')])

    def get(self, idx: int) -> Data:
        path = os.path.join(self.processed_dir_path, f'data_{idx}.pt')
        # Option 1: Use weights_only=False (safe since we created these files)
        return torch.load(path, weights_only=False)
        
        # Option 2: Use safe_globals context manager (more secure but verbose)
        # from torch_geometric.data.data import DataEdgeAttr
        # with torch.serialization.safe_globals([DataEdgeAttr, Data]):
        #     return torch.load(path, weights_only=True)

    @staticmethod
    def get_atom_features(atom) -> List[float]:
        """Extract atom features for graph nodes."""
        features = []
        atomic_nums = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        features += [1 if atom.GetAtomicNum() == num else 0 for num in atomic_nums]
        degrees = [0, 1, 2, 3, 4, 5]
        features += [1 if atom.GetDegree() == d else 0 for d in degrees]
        charges = [-2, -1, 0, 1, 2]
        features += [1 if atom.GetFormalCharge() == c else 0 for c in charges]
        hybridizations = [
            Chem.HybridizationType.S, Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2
        ]
        features += [1 if atom.GetHybridization() == hyb else 0 for hyb in hybridizations]
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(1 if atom.IsInRing() else 0)
        return features

    @staticmethod
    def get_bond_features(bond) -> List[float]:
        """Extract bond features for graph edges."""
        features = []
        bond_types = [
            Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
        ]
        features += [1 if bond.GetBondType() == bt else 0 for bt in bond_types]
        features.append(1 if bond.GetIsConjugated() else 0)
        features.append(1 if bond.IsInRing() else 0)
        return features

    def smiles_to_graph(self, smiles: str, bbbp_label: float) -> Optional[Data]:
        """Convert a SMILES string to a PyG Data object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)

        # Node features
        atom_feats = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_feats:
            return None
        x = torch.tensor(atom_feats, dtype=torch.float)

        # Edge indices and features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feats = self.get_bond_features(bond)
            edge_index += [[i, j], [j, i]]
            edge_attr += [feats, feats]

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)

        # Graph-level label (BBBP permeability score)
        y = torch.tensor([bbbp_label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)