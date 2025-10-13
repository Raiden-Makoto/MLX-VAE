import os
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.data import GraphData
from mlx_graphs.loaders import Dataloader  # Note: capital D, lowercase l
import pandas as pd
from rdkit import Chem

class QM9GraphDataset:
    """
    In-memory dataset converting QM9 CSV SMILES → MLX-graphs GraphData with BBBP labels.
    """
    def __init__(
        self,
        csv_path: str,
        smiles_col: str = "smiles",
        label_col: str = "p_np",
        transform=None,
        force_reload: bool = False
    ):
        self.csv_path    = csv_path
        self.smiles_col  = smiles_col
        self.label_col   = label_col
        self.transform   = transform
        self._graphs: List[GraphData] = []
        # Load and process if necessary
        self._load_or_reload(force_reload)

    def _load_or_reload(self, force: bool):
        # Always reload if force=True, else load once
        if self._graphs and not force:
            return
        df = pd.read_csv(self.csv_path).dropna(subset=[self.smiles_col])
        self._graphs.clear()
        for _, row in df.iterrows():
            g = self._smiles_to_graph(row[self.smiles_col], float(row.get(self.label_col, 0.0)))
            if g is not None:
                if self.transform:
                    g = self.transform(g)
                self._graphs.append(g)

    def _smiles_to_graph(self, smiles: str, label: float) -> Optional[GraphData]:
        """Convert RDKit Mol → MLX-graphs GraphData."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        # Node features
        node_feats = []
        for atom in mol.GetAtoms():
            feats = []
            atomic_nums = [1,6,7,8,9,15,16,17,35,53]
            feats += [1.0 if atom.GetAtomicNum()==n else 0.0 for n in atomic_nums]
            degrees = list(range(6))
            feats += [1.0 if atom.GetDegree()==d else 0.0 for d in degrees]
            charges = [-2,-1,0,1,2]
            feats += [1.0 if atom.GetFormalCharge()==c else 0.0 for c in charges]
            hybs = [
                Chem.HybridizationType.S, Chem.HybridizationType.SP,
                Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
                Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2
            ]
            feats += [1.0 if atom.GetHybridization()==h else 0.0 for h in hybs]
            feats.append(1.0 if atom.GetIsAromatic() else 0.0)
            feats.append(1.0 if atom.IsInRing() else 0.0)
            node_feats.append(feats)
        if not node_feats:
            return None
        x = mx.array(node_feats, dtype=mx.float32)

        # Edge index & features
        edges, edge_feats = [], []
        edge_feat_dim = 6  # 4 bond types + conjugated + in_ring
        
        for bond in mol.GetBonds():
            i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feats = []
            bond_types = [
                Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
            ]
            feats += [1.0 if bond.GetBondType()==bt else 0.0 for bt in bond_types]
            feats.append(1.0 if bond.GetIsConjugated() else 0.0)
            feats.append(1.0 if bond.IsInRing() else 0.0)
            for u,v in ((i,j),(j,i)):
                edges.append([u,v])
                edge_feats.append(feats)
        
        edge_index = mx.array(edges, dtype=mx.int64).T if edges else mx.zeros((2,0),dtype=mx.int64)
        edge_attr  = mx.array(edge_feats, dtype=mx.float32) if edge_feats else mx.zeros((0,edge_feat_dim),dtype=mx.float32)

        # Label
        y = mx.array([label], dtype=mx.float32)

        return GraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx: int) -> GraphData:
        return self._graphs[idx]
