"""
Message-Passing Neural Network for Blood-Brain Barrier Permeability Prediction
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import MessagePassing, global_mean_pool


class EdgeNetwork(MessagePassing):
    """Edge Network for message passing."""
    
    def __init__(self, edge_dim, message_dim):
        super().__init__(aggr='add')
        self.message_dim = message_dim
        # Output a message_dim x message_dim weight matrix per edge
        self.lin = nn.Linear(edge_dim, message_dim * message_dim)

    def forward(self, x, edge_index, edge_attr):
        weight = self.lin(edge_attr)
        weight = weight.view(-1, self.message_dim, self.message_dim)
        return self.propagate(edge_index, x=x, weight=weight)

    def message(self, x_j, weight):
        # x_j shape: [E, message_dim]
        return (weight @ x_j.unsqueeze(-1)).squeeze(-1)


class MPNN(nn.Module):
    """Message-Passing Neural Network for molecular property prediction."""
    def __init__(self, node_dim=12, edge_dim=4, message_dim=32, num_steps=4, hidden_dim=512):
        super().__init__()
        self.message_dim = message_dim
        self.num_steps = num_steps
        self.node_lin = nn.Linear(node_dim, message_dim)  # input projection
        self.edge_net = EdgeNetwork(edge_dim, message_dim)
        self.gru = nn.GRUCell(message_dim, message_dim)
        self.readout = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_lin(x)  # project input features
        for _ in range(self.num_steps):  # number of message passing steps
            m = self.edge_net(h, edge_index, edge_attr)
            h = self.gru(m, h)
        hg = global_mean_pool(h, batch)
        return torch.sigmoid(self.readout(hg)).view(-1)


def smiles_to_graph(smiles):
    """Convert a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string representation of the molecule
    
    Returns:
        Data object with molecular graph features or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}")
            return None
        
        # Add hydrogens (needed for correct atom features)
        mol = Chem.AddHs(mol)
        # Note: No 3D embedding needed - MPNN only uses 2D graph structure
        
        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            degree = atom.GetDegree()
            valence = atom.GetTotalValence()
            numH = atom.GetTotalNumHs()
            feat = [
                float(element == symbol) for symbol in ["C","N","O","F","P","S","Cl","Br","I"]
            ] + [degree, valence, numH]
            atom_features.append(feat)
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Extract edge features
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            attr = [
                float(bt == Chem.rdchem.BondType.SINGLE),
                float(bt == Chem.rdchem.BondType.DOUBLE),
                float(bt == Chem.rdchem.BondType.TRIPLE),
                float(bond.GetIsConjugated())
            ]
            edge_index += [[i,j],[j,i]]
            edge_attr += [attr, attr]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None


def predict_single_molecule(model, smiles, device='mps'):
    """Run inference on a single molecule from SMILES string.
    
    Args:
        model: Trained MPNN model
        smiles: SMILES string representation of the molecule
        device: Device to run inference on ('cpu', 'cuda', 'mps')
    
    Returns:
        Prediction score (float) or None if conversion fails
    """
    # Convert SMILES to graph
    data = smiles_to_graph(smiles)
    if data is None:
        return None
    
    # Add batch information (single molecule = batch index 0)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data = data.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        prediction = model(data)
    
    return prediction.item()


def load_checkpoint(checkpoint_path, device='mps'):
    """Load a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on ('cpu', 'cuda', 'mps')
    
    Returns:
        model: Loaded MPNN model
        checkpoint: Full checkpoint dictionary (includes optimizer state)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MPNN(
        node_dim=checkpoint['node_dim'],
        edge_dim=checkpoint['edge_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model, checkpoint