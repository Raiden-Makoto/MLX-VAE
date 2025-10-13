import mlx.core as mx
import mlx.nn as nn

class MLXGraphDecoder(nn.Module):
    """
    Graph decoder that reconstructs molecular graphs from latent + property conditioning.
    
    Input:  z [batch_size, latent_dim] + properties [batch_size, num_properties]
    Output: Node logits, edge logits, and graph size predictions
    """
    def __init__(
        self,
        latent_dim,
        num_properties,
        node_dim,
        edge_dim,
        hidden_dim,
        max_nodes=20,
        dropout=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_properties = num_properties
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + num_properties, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_nodes)
        )
        self.node_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * node_dim)
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim + 1)
        )
        self.pos_embedding = nn.Embedding(max_nodes, hidden_dim)
        
        # Precompute edge indices once (all pairs i < j)
        edge_indices = []
        for i in range(max_nodes):
            for j in range(i + 1, max_nodes):
                edge_indices.append([i, j])
        self.edge_indices = mx.array(edge_indices, dtype=mx.int32)
        self.num_edges = len(edge_indices)

    def __call__(self, z, target_properties):
        """
        Forward pass to generate molecular graph from latent code
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            target_properties: Target property values [batch_size, num_properties]
        
        Returns:
            dict with:
                - node_logits: [batch_size, max_nodes, node_dim]
                - edge_logits: [batch_size, max_edges, edge_dim + 1]
                - size_probs: [batch_size, max_nodes]
        """
        batch_size = z.shape[0]
        z_cond = mx.concatenate([z, target_properties], axis=-1)
        h_global = self.input_proj(z_cond)
        size_logits = self.size_predictor(h_global)
        size_probs = mx.softmax(size_logits, axis=-1)
        node_logits_flat = self.node_generator(h_global)
        node_logits = mx.reshape(node_logits_flat, [batch_size, self.max_nodes, self.node_dim])
        positions = mx.arange(self.max_nodes)
        pos_embed = self.pos_embedding(positions)
        h_pos = mx.add(mx.expand_dims(h_global, 1), mx.expand_dims(pos_embed, 0))

        # Vectorized edge prediction - compute all edges at once!
        # Use precomputed edge indices (ensure int32 dtype for gather ops)
        edge_i = mx.array(self.edge_indices[:, 0], dtype=mx.int32)  # [num_edges]
        edge_j = mx.array(self.edge_indices[:, 1], dtype=mx.int32)  # [num_edges]
        
        # Gather node features using take operation for all edges at once
        node_i_feat = mx.take(node_logits, edge_i, axis=1)  # [batch_size, num_edges, node_dim]
        node_j_feat = mx.take(node_logits, edge_j, axis=1)  # [batch_size, num_edges, node_dim]
        h_edge = (mx.take(h_pos, edge_i, axis=1) + mx.take(h_pos, edge_j, axis=1)) / 2  # [batch_size, num_edges, hidden_dim]
        
        # Concatenate all edge inputs: [batch_size, num_edges, hidden_dim + 2*node_dim]
        edge_input = mx.concatenate([h_edge, node_i_feat, node_j_feat], axis=-1)
        
        # Reshape to process all edges together: [batch_size * num_edges, hidden_dim + 2*node_dim]
        edge_input_flat = mx.reshape(edge_input, [-1, edge_input.shape[-1]])
        
        # Predict all edges at once
        edge_pred_flat = self.edge_predictor(edge_input_flat)  # [batch_size * num_edges, edge_dim + 1]
        
        # Reshape back: [batch_size, num_edges, edge_dim + 1]
        edge_logits = mx.reshape(edge_pred_flat, [batch_size, self.num_edges, self.edge_dim + 1])
        return {
            'node_logits': node_logits,
            'edge_logits': edge_logits,
            'size_probs': size_probs,
            'edge_indices': self.edge_indices
        }
    
    def sample_graph(self, decoder_output, temperature=1.0):
        """
        Convert decoder logits into discrete graph structure
        
        Args:
            decoder_output: Output dictionary from forward()
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            Dictionary with sampled discrete graph components
        """
        batch_size = decoder_output['node_logits'].shape[0]
        size_probs = decoder_output['size_probs']
        graph_sizes = mx.random.categorical(size_probs, num_samples=1).squeeze(-1)
        node_logits = decoder_output['node_logits'] / temperature
        node_probs = mx.softmax(node_logits, axis=-1)
        edge_logits = decoder_output['edge_logits']
        edge_exist_probs = mx.sigmoid(edge_logits[:, :, -1])
        edge_type_logits = edge_logits[:, :, :-1] / temperature
        edge_type_probs = mx.softmax(edge_type_logits, axis=-1)
        edge_indices = decoder_output['edge_indices']
        return {
            'graph_sizes': graph_sizes,
            'node_probs': node_probs,
            'edge_exist_probs': edge_exist_probs,
            'edge_type_probs': edge_type_probs,
            'edge_indices': edge_indices
        }