import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from models.graphencoder import GraphEncoder
from models.propertypredictor import PropertyPredictor
from models.graphdecoder import GraphDecoder


class MGCVAE(nn.Module):
    """
    Multi-objective Graph Conditional Variational Autoencoder
    
    Combines:
    - GraphEncoder: molecular graphs → latent space
    - PropertyPredictor: latent space → property predictions
    - GraphDecoder: latent space + target properties → molecular graphs
    """
    
    def __init__(
        self,
        node_dim,
        edge_dim,
        latent_dim=64,
        hidden_dim=128,
        num_properties=2,
        num_layers=4,
        heads=4,
        max_nodes=50,
        beta=1.0,
        gamma=1.0,
        dropout=0.1
    ):
        super(MGCVAE, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.num_properties = num_properties
        self.max_nodes = max_nodes
        
        # =====================================================================
        # Loss Weighting Parameters
        # =====================================================================
        
        self.beta = beta    # KL divergence weight
        self.gamma = gamma  # Property prediction weight
        
        # =====================================================================
        # Model Components
        # =====================================================================
        
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        
        self.property_predictor = PropertyPredictor(
            latent_dim=latent_dim,
            num_properties=num_properties,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            num_properties=num_properties,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            dropout=dropout
        )
        
        # =====================================================================
        # Weight Initialization
        # =====================================================================
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2)
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        
        Returns:
            z: Sampled latent codes [batch_size, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean (no sampling)
            return mu
    
    
    def encode(self, x, edge_index, edge_attr, batch):
        """
        Encode molecular graphs to latent distributions
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch vector [num_nodes]
        
        Returns:
            mu, logvar: Parameters of latent distribution
        """
        return self.encoder(x, edge_index, edge_attr, batch)
    
    
    def decode(self, z, target_properties):
        """
        Decode latent codes + properties to molecular graphs
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            target_properties: Target property values [batch_size, num_properties]
        
        Returns:
            Decoder output dictionary
        """
        return self.decoder(z, target_properties)
    
    
    def forward(self, batch):
        """
        Full forward pass: encode → reparameterize → predict properties → decode
        
        Args:
            batch: PyTorch Geometric batch object
        
        Returns:
            Dictionary containing all model outputs
        """
        # =====================================================================
        # Extract Batch Components
        # =====================================================================
        
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        batch_idx = batch.batch
        true_properties = batch.y
        
        # Ensure true_properties has shape [batch_size, num_properties]
        if true_properties.dim() == 1:
            true_properties = true_properties.unsqueeze(-1)
        
        # =====================================================================
        # Encoding Phase
        # =====================================================================
        # Encode molecular graphs to latent distributions
        
        mu, logvar = self.encode(x, edge_index, edge_attr, batch_idx)
        
        # =====================================================================
        # Reparameterization
        # =====================================================================
        # Sample latent codes using reparameterization trick
        
        z = self.reparameterize(mu, logvar)
        
        # =====================================================================
        # Property Prediction
        # =====================================================================
        # Predict BBB permeability from latent codes
        
        predicted_properties = self.property_predictor(z)
        
        # =====================================================================
        # Decoding Phase
        # =====================================================================
        # Reconstruct molecular graphs conditioned on true properties
        
        decoder_output = self.decode(z, true_properties)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'predicted_properties': predicted_properties,
            'decoder_output': decoder_output,
            'true_properties': true_properties
        }
    
    
    def compute_loss(self, batch, model_output):
        """
        Compute multi-objective VAE loss
        
        Loss = Reconstruction + β*KL_divergence + γ*Property_prediction
        
        Args:
            batch: Input batch
            model_output: Output from forward()
        
        Returns:
            Dictionary of loss components
        """
        mu = model_output['mu']
        logvar = model_output['logvar']
        predicted_props = model_output['predicted_properties']
        true_props = model_output['true_properties']
        decoder_out = model_output['decoder_output']
        
        batch_size = mu.size(0)
        
        # =====================================================================
        # Reconstruction Loss
        # =====================================================================
        # Measures how well the model can recreate input graphs
        
        node_recon_loss = self._compute_node_reconstruction_loss(
            batch, decoder_out, batch_size
        )
        
        edge_recon_loss = self._compute_edge_reconstruction_loss(
            batch, decoder_out, batch_size
        )
        
        total_recon_loss = node_recon_loss + edge_recon_loss
        
        # =====================================================================
        # KL Divergence Loss
        # =====================================================================
        # Regularizes latent distribution: KL(q(z|x) || p(z)) where p(z) = N(0,I)
        
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        
        # =====================================================================
        # Property Prediction Loss
        # =====================================================================
        # Ensures latent space encodes property information
        
        property_loss = F.mse_loss(predicted_props, true_props)
        
        # =====================================================================
        # Combined Loss
        # =====================================================================
        
        total_loss = total_recon_loss + self.beta * kl_loss + self.gamma * property_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': total_recon_loss,
            'node_recon_loss': node_recon_loss,
            'edge_recon_loss': edge_recon_loss,
            'kl_loss': kl_loss,
            'property_loss': property_loss
        }
    
    
    def _compute_node_reconstruction_loss(self, batch, decoder_output, batch_size):
        """
        Compute loss for node reconstruction
        
        Maps variable-size real graphs to fixed-size decoder predictions with masking
        """
        node_logits = decoder_output['node_logits']  # [batch_size, max_nodes, node_dim]
        
        # Create target node features for each graph in batch
        # We'll use a simplified approach: discretize the first feature dimension
        target_nodes = torch.zeros(batch_size, self.max_nodes, dtype=torch.long, device=batch.x.device)
        masks = torch.zeros(batch_size, self.max_nodes, dtype=torch.bool, device=batch.x.device)
        
        # Fill in actual node features from batch
        node_ptr = 0  # Pointer to current position in batch.x
        for i in range(batch_size):
            # Count nodes in this graph
            graph_size = (batch.batch == i).sum().item()
            graph_size = min(graph_size, self.max_nodes)  # Cap at max_nodes
            
            # Extract node features for this graph
            graph_nodes = batch.x[node_ptr:node_ptr + graph_size]
            
            # Convert continuous features to discrete indices (simplified)
            # Use argmax of first 10 dimensions (atom type one-hot)
            node_indices = torch.argmax(graph_nodes[:, :10], dim=1)
            
            # Fill in targets and mask
            target_nodes[i, :graph_size] = node_indices
            masks[i, :graph_size] = True
            
            node_ptr += (batch.batch == i).sum().item()
        
        # Compute masked cross-entropy loss
        node_logits_flat = node_logits.view(-1, self.node_dim)
        target_nodes_flat = target_nodes.view(-1)
        masks_flat = masks.view(-1)
        
        # Only compute loss on actual nodes (not padding)
        if masks_flat.sum() > 0:
            # Use first 10 dimensions of logits (atom types)
            loss = F.cross_entropy(
                node_logits_flat[masks_flat, :10],
                target_nodes_flat[masks_flat],
                reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=batch.x.device)
        
        return loss
    
    
    def _compute_edge_reconstruction_loss(self, batch, decoder_output, batch_size):
        """
        Compute loss for edge reconstruction
        
        Maps real edges to decoder's fixed-size edge predictions
        """
        edge_logits = decoder_output['edge_logits']  # [batch_size, num_possible_edges, edge_dim+1]
        edge_indices = decoder_output['edge_indices']  # [num_possible_edges, 2]
        
        if edge_logits.size(1) == 0:
            return torch.tensor(0.0, device=batch.x.device)
        
        # Create adjacency matrix for each graph in batch
        num_possible_edges = edge_indices.size(0)
        target_edge_exist = torch.zeros(batch_size, num_possible_edges, device=batch.x.device)
        target_edge_type = torch.zeros(batch_size, num_possible_edges, dtype=torch.long, device=batch.x.device)
        
        # Process each graph in batch
        node_offset = 0
        for b in range(batch_size):
            # Get edges for this graph
            graph_size = (batch.batch == b).sum().item()
            graph_size = min(graph_size, self.max_nodes)
            
            # Get real edges for this graph
            graph_edge_mask = (batch.batch[batch.edge_index[0]] == b)
            graph_edges = batch.edge_index[:, graph_edge_mask]
            
            # Offset edges to be relative to this graph
            graph_edges = graph_edges - node_offset
            
            if hasattr(batch, 'edge_attr'):
                graph_edge_attr = batch.edge_attr[graph_edge_mask]
            else:
                graph_edge_attr = None
            
            # Match real edges to decoder's edge predictions
            for edge_idx in range(num_possible_edges):
                i, j = edge_indices[edge_idx]
                
                # Skip if outside this graph's size
                if i >= graph_size or j >= graph_size:
                    continue
                
                # Check if this edge exists in real graph
                edge_exists = False
                if graph_edges.size(1) > 0:
                    # Look for edge (i,j) or (j,i)
                    matches = ((graph_edges[0] == i) & (graph_edges[1] == j)) | \
                             ((graph_edges[0] == j) & (graph_edges[1] == i))
                    
                    if matches.any():
                        edge_exists = True
                        match_idx = torch.where(matches)[0][0]
                        
                        # Get edge type (first 4 dimensions are bond type one-hot)
                        if graph_edge_attr is not None:
                            edge_type = torch.argmax(graph_edge_attr[match_idx, :4])
                            target_edge_type[b, edge_idx] = edge_type
                
                target_edge_exist[b, edge_idx] = 1.0 if edge_exists else 0.0
            
            node_offset += (batch.batch == b).sum().item()
        
        # Compute losses
        # Binary cross-entropy for edge existence
        edge_exist_loss = F.binary_cross_entropy_with_logits(
            edge_logits[:, :, -1],
            target_edge_exist,
            reduction='mean'
        )
        
        # Categorical cross-entropy for edge types (only on existing edges)
        existing_edges_mask = target_edge_exist > 0.5
        if existing_edges_mask.sum() > 0:
            edge_type_logits = edge_logits[:, :, :4].reshape(-1, 4)  # First 4 dims are bond type
            target_types = target_edge_type.reshape(-1)
            mask_flat = existing_edges_mask.reshape(-1)
            
            edge_type_loss = F.cross_entropy(
                edge_type_logits[mask_flat],
                target_types[mask_flat],
                reduction='mean'
            )
        else:
            edge_type_loss = torch.tensor(0.0, device=batch.x.device)
        
        return edge_exist_loss + edge_type_loss
    
    
    def generate(self, target_properties, num_samples=1, temperature=1.0, device='cpu'):
        """
        Generate new molecules with specified target properties
        
        Args:
            target_properties: Desired property values [num_properties]
            num_samples: Number of molecules to generate
            temperature: Sampling temperature (lower = more deterministic)
            device: Device to run on
        
        Returns:
            Generated molecular graphs
        """
        self.eval()
        with torch.no_grad():
            # =====================================================================
            # Sample from Prior
            # =====================================================================
            # Sample latent codes from N(0, I)
            
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # =====================================================================
            # Prepare Target Properties
            # =====================================================================
            
            if isinstance(target_properties, list):
                target_properties = torch.tensor(
                    target_properties, dtype=torch.float, device=device
                )
            
            if target_properties.dim() == 1:
                target_properties = target_properties.unsqueeze(0).repeat(num_samples, 1)
            
            # =====================================================================
            # Decode to Graphs
            # =====================================================================
            
            decoder_output = self.decode(z, target_properties)
            sampled_graphs = self.decoder.sample_graph(decoder_output, temperature)
            
            return sampled_graphs
    
    
    def interpolate(self, properties_start, properties_end, num_steps=10, device='cpu'):
        """
        Generate molecules by interpolating between two property targets
        
        Args:
            properties_start: Starting property values
            properties_end: Ending property values
            num_steps: Number of interpolation steps
            device: Device to run on
        
        Returns:
            sampled_graphs: Generated graphs at each interpolation step
            target_props: Property values at each step
        """
        self.eval()
        with torch.no_grad():
            # =====================================================================
            # Create Interpolation Path
            # =====================================================================
            
            alphas = torch.linspace(0, 1, num_steps, device=device)
            
            start = torch.tensor(properties_start, dtype=torch.float, device=device)
            end = torch.tensor(properties_end, dtype=torch.float, device=device)
            
            interpolated_props = []
            for alpha in alphas:
                props = (1 - alpha) * start + alpha * end
                interpolated_props.append(props)
            
            target_props = torch.stack(interpolated_props)
            
            # =====================================================================
            # Sample and Generate
            # =====================================================================
            
            z = torch.randn(num_steps, self.latent_dim, device=device)
            decoder_output = self.decode(z, target_props)
            sampled_graphs = self.decoder.sample_graph(decoder_output, temperature=0.8)
            
            return sampled_graphs, target_props
