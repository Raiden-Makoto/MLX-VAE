import mlx.core as mx
import mlx.nn as nn
import mlx.nn.init as init

from mlx_graphs.nn import (
    GATConv,
    BatchNormalization,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

class MLXGraphEncoder(nn.Module):
    """
    Graph Neural Network Encoder for molecular graphs following current literature best practices
    """
    def __init__(self,
        node_dim,
        edge_dim,
        hidden_dim,
        latent_dim,
        num_layers=4,
        heads=4,
        dropout=0.1,
        use_edge_attr=True
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        if use_edge_attr and edge_dim > 0:
            self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        self.gat_layers = [] # regular lists work in mlx as substitutes for nn.ModuleList
        self.batch_norms = [] # regular lists work in mlx as substitutes for nn.ModuleList
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    node_features_dim=hidden_dim,
                    out_features_dim=hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_features_dim=hidden_dim if use_edge_attr else None,
                )
            )
            self.batch_norms.append(BatchNormalization(hidden_dim))
        self.dropout_layer = nn.Dropout(dropout)
        self.pool_type = 'multi'
        pool_dim = hidden_dim * 3 if self.pool_type == 'multi' else hidden_dim
        self.pre_latent = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Note: MLX layers are automatically initialized with Xavier/Glorot uniform
    
    def __call__(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through the graph encoder
        """
        h = nn.relu(self.node_embedding(x))
        
        # Edge embeddings (if using edge attributes)
        edge_emb = None
        if self.use_edge_attr and edge_attr is not None:
            edge_emb = self.edge_embedding(edge_attr)
        
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            h_input = h
            # mlx_graphs GATConv API: (edge_index, node_features, edge_features)
            h = gat(edge_index, h, edge_emb)
            h = bn(h)
            if i > 0 and h.shape[-1] == h_input.shape[-1]:
                h = h + h_input
            h = nn.relu(h)
            h = self.dropout_layer(h)
        if self.pool_type == 'multi':
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_add = global_add_pool(h, batch)
            h_graph = mx.concatenate([h_mean, h_max, h_add], axis=-1)
        elif self.pool_type == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pool_type == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pool_type == 'add':
            h_graph = global_add_pool(h, batch)
        else:
            h_graph = global_mean_pool(h, batch)
        h_graph = self.pre_latent(h_graph)
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        return mu, logvar
