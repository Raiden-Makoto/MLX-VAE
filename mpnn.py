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