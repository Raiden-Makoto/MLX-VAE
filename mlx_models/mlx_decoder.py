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
        max_nodes=50,
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

        edge_logits = []
        edge_indices = []
        for i in range(self.max_nodes):
            for j in range(i + 1, self.max_nodes):
                node_i_feat = node_logits[:, i, :]
                node_j_feat = node_logits[:, j, :]
                h_edge = (h_pos[:, i, :] + h_pos[:, j, :]) / 2
                edge_input = mx.concatenate([h_edge, node_i_feat, node_j_feat], axis=-1)
                edge_pred = self.edge_predictor(edge_input)
                edge_logits.append(edge_pred)
                edge_indices.append([i, j])
        if edge_logits:
            edge_logits = mx.stack(edge_logits, axis=1)
            edge_indices = mx.array(edge_indices, dtype=mx.int32)
        else:
            edge_logits = mx.zeros([batch_size, 0, self.edge_dim + 1])
            edge_indices = mx.array([], dtype=mx.int32).reshape(0, 2)
        return {
            'node_logits': node_logits,
            'edge_logits': edge_logits,
            'size_probs': size_probs,
            'edge_indices': edge_indices
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