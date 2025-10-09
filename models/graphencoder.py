import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn import BatchNorm


class GraphEncoder(nn.Module):
    """
    Graph Neural Network Encoder for molecular graphs following current literature best practices
    
    References:
    - Uses multi-head GAT for attention-based aggregation (Veličković et al., 2017)
    - Incorporates residual connections and layer normalization (He et al., 2016)
    - Multiple pooling strategies for graph-level representation (Xu et al., 2018)
    - Dropout and batch normalization for regularization
    """
    
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        latent_dim,
        num_layers=4,
        heads=4,
        dropout=0.1,
        use_edge_attr=True
    ):
        super(GraphEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # =====================================================================
        # Embedding Layers
        # =====================================================================
        
        # Initial node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Edge embedding (if using edge attributes)
        if use_edge_attr and edge_dim > 0:
            self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # =====================================================================
        # Graph Attention Layers
        # =====================================================================
        # GAT performs better than GCN for molecular graphs
        
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Each layer uses multi-head attention
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=hidden_dim if use_edge_attr else None
                )
            )
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # =====================================================================
        # Pooling Strategy
        # =====================================================================
        # Options: 'mean', 'max', 'add', 'multi'
        # Multi-pooling combines multiple strategies (as recommended in literature)
        
        self.pool_type = 'multi'
        
        if self.pool_type == 'multi':
            pool_dim = hidden_dim * 3  # Concatenate mean + max + add
        else:
            pool_dim = hidden_dim
        
        # =====================================================================
        # Latent Projection Layers
        # =====================================================================
        # Project graph representation to latent distribution parameters
        
        self.pre_latent = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # =====================================================================
        # Weight Initialization
        # =====================================================================
        
        self._init_weights()
    
    
    def _init_weights(self):
        """Initialize weights following current best practices"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through the graph encoder
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch vector [num_nodes] for batching multiple graphs
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        
        # =====================================================================
        # Initial Embeddings
        # =====================================================================
        
        h = F.relu(self.node_embedding(x))
        
        # Edge embeddings (if using edge attributes)
        edge_emb = None
        if self.use_edge_attr and edge_attr is not None:
            edge_emb = self.edge_embedding(edge_attr)
        
        # =====================================================================
        # Message Passing with Residual Connections
        # =====================================================================
        
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            h_input = h
            
            # Apply GAT layer
            h = gat(h, edge_index, edge_attr=edge_emb)
            
            # Apply batch normalization
            h = bn(h)
            
            # Add residual connection (skip first layer as dimensions might differ)
            if i > 0 and h.size(-1) == h_input.size(-1):
                h = h + h_input
            
            # Apply activation and dropout
            h = F.relu(h)
            h = self.dropout_layer(h)
        
        # =====================================================================
        # Graph-Level Pooling
        # =====================================================================
        
        if self.pool_type == 'multi':
            # Combine multiple pooling strategies (recommended approach)
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_add = global_add_pool(h, batch)
            h_graph = torch.cat([h_mean, h_max, h_add], dim=-1)
        elif self.pool_type == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pool_type == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pool_type == 'add':
            h_graph = global_add_pool(h, batch)
        else:
            # Default fallback
            h_graph = global_mean_pool(h, batch)
        
        # =====================================================================
        # Latent Distribution Parameters
        # =====================================================================
        
        # Pre-latent processing
        h_graph = self.pre_latent(h_graph)
        
        # Generate mean and log variance for latent distribution
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        
        return mu, logvar
