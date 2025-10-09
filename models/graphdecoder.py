import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDecoder(nn.Module):
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
        super(GraphDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_properties = num_properties
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # =====================================================================
        # Input Projection Layer
        # =====================================================================
        # Concatenate latent vector z with target properties, then project to hidden space
        
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + num_properties, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # =====================================================================
        # Graph Size Predictor
        # =====================================================================
        # Predicts how many nodes (atoms) the generated molecule should have
        
        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_nodes),
            nn.Softmax(dim=-1)  # Probability distribution over possible sizes
        )
        
        # =====================================================================
        # Node Type Generator
        # =====================================================================
        # Generates atom features for each node position
        
        self.node_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * node_dim)  # Flattened: all nodes
        )
        
        # =====================================================================
        # Edge Predictor
        # =====================================================================
        # For each pair of nodes, predicts edge existence and bond type
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 2 * node_dim, hidden_dim),  # context + 2 node features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim + 1)  # edge features + existence probability
        )
        
        # =====================================================================
        # Positional Embedding
        # =====================================================================
        # Helps with node ordering and spatial relationships
        
        self.pos_embedding = nn.Embedding(max_nodes, hidden_dim)
    
    
    def forward(self, z, target_properties):
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
        batch_size = z.size(0)
        
        # =====================================================================
        # Condition on Properties
        # =====================================================================
        # Combine latent code with target property values
        
        z_cond = torch.cat([z, target_properties], dim=-1)
        h_global = self.input_proj(z_cond)
        
        # =====================================================================
        # Predict Graph Size
        # =====================================================================
        # Determine how many atoms the molecule should have
        
        size_probs = self.size_predictor(h_global)
        
        # =====================================================================
        # Generate Node Features
        # =====================================================================
        # Predict atom features for all possible node positions
        
        node_logits_flat = self.node_generator(h_global)
        node_logits = node_logits_flat.view(batch_size, self.max_nodes, self.node_dim)
        
        # =====================================================================
        # Add Positional Information
        # =====================================================================
        # Incorporate positional embeddings to help with node ordering
        
        positions = torch.arange(self.max_nodes, device=z.device)
        pos_emb = self.pos_embedding(positions)
        
        # Broadcast and add positional info to global context
        h_pos = h_global.unsqueeze(1) + pos_emb.unsqueeze(0)
        
        # =====================================================================
        # Generate Edge Predictions
        # =====================================================================
        # For each pair of nodes, predict bond existence and type
        
        edge_logits = []
        edge_indices = []
        
        # Iterate over all node pairs (upper triangular for undirected graphs)
        for i in range(self.max_nodes):
            for j in range(i + 1, self.max_nodes):
                # Extract node features
                node_i_feat = node_logits[:, i, :]
                node_j_feat = node_logits[:, j, :]
                
                # Compute edge context from positional embeddings
                h_edge = (h_pos[:, i, :] + h_pos[:, j, :]) / 2
                
                # Concatenate: positional context + two node features
                edge_input = torch.cat([h_edge, node_i_feat, node_j_feat], dim=-1)
                
                # Predict edge existence and bond type
                edge_pred = self.edge_predictor(edge_input)
                edge_logits.append(edge_pred)
                edge_indices.append([i, j])
        
        # =====================================================================
        # Stack Edge Predictions
        # =====================================================================
        
        if edge_logits:
            edge_logits = torch.stack(edge_logits, dim=1)
            edge_indices = torch.tensor(edge_indices, device=z.device)
        else:
            edge_logits = torch.empty(batch_size, 0, self.edge_dim + 1, device=z.device)
            edge_indices = torch.empty(0, 2, dtype=torch.long, device=z.device)
        
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
        batch_size = decoder_output['node_logits'].size(0)
        
        # =====================================================================
        # Sample Graph Sizes
        # =====================================================================
        
        size_probs = decoder_output['size_probs']
        graph_sizes = torch.multinomial(size_probs, 1).squeeze(-1)
        
        # =====================================================================
        # Sample Node Types
        # =====================================================================
        # Use temperature-scaled softmax for differentiability
        
        node_logits = decoder_output['node_logits'] / temperature
        node_probs = F.softmax(node_logits, dim=-1)
        
        # =====================================================================
        # Sample Edges
        # =====================================================================
        # Predict both edge existence and bond type
        
        edge_logits = decoder_output['edge_logits']
        
        # Last dimension: edge existence probability
        edge_exist_probs = torch.sigmoid(edge_logits[:, :, -1])
        
        # Remaining dimensions: edge type probabilities
        edge_type_logits = edge_logits[:, :, :-1] / temperature
        edge_type_probs = F.softmax(edge_type_logits, dim=-1)
        
        return {
            'graph_sizes': graph_sizes,
            'node_probs': node_probs,
            'edge_exist_probs': edge_exist_probs,
            'edge_type_probs': edge_type_probs,
            'edge_indices': decoder_output['edge_indices']
        }
