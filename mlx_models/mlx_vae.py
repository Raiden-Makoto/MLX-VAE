import mlx.core as mx
import mlx.nn as nn

from .mlx_encoder import MLXGraphEncoder  # type: ignore
from .mlx_pp import MLXPropertyPredictor  # type: ignore
from .mlx_decoder import MLXGraphDecoder  # type: ignore


class MLXMGCVAE(nn.Module):
    """
    Multi-objective Graph Conditional Variational Autoencoder (MLX version)
    
    Combines:
    - MLXGraphEncoder: molecular graphs → latent space
    - MLXPropertyPredictor: latent space → property predictions
    - MLXGraphDecoder: latent space + target properties → molecular graphs
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
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.num_properties = num_properties
        self.max_nodes = max_nodes
        # Note: self.training is inherited from nn.Module
        
        # =====================================================================
        # Loss Weighting Parameters
        # =====================================================================
        
        self.beta = beta    # KL divergence weight
        self.gamma = gamma  # Property prediction weight
        
        # =====================================================================
        # Model Components
        # =====================================================================
        
        self.encoder = MLXGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        
        self.property_predictor = MLXPropertyPredictor(
            latent_dim=latent_dim,
            num_properties=num_properties,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.decoder = MLXGraphDecoder(
            latent_dim=latent_dim,
            num_properties=num_properties,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            dropout=dropout
        )
    
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
            std = mx.exp(0.5 * logvar)
            eps = mx.random.normal(std.shape)
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
    
    def __call__(self, batch):
        """
        Full forward pass: encode → reparameterize → predict properties → decode
        
        Args:
            batch: MLX GraphData batch object
        
        Returns:
            Dictionary containing all model outputs
        """
        # =====================================================================
        # Extract Batch Components
        # =====================================================================
        
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        batch_idx = batch.batch_indices
        true_properties = batch.y
        
        # Ensure true_properties has shape [batch_size, num_properties]
        if len(true_properties.shape) == 1:
            true_properties = mx.expand_dims(true_properties, -1)
        
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
            model_output: Output from __call__()
        
        Returns:
            Dictionary of loss components
        """
        mu = model_output['mu']
        logvar = model_output['logvar']
        predicted_props = model_output['predicted_properties']
        true_props = model_output['true_properties']
        decoder_out = model_output['decoder_output']
        
        batch_size = mu.shape[0]
        
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
        
        kl_loss = -0.5 * mx.mean(
            mx.sum(1 + logvar - mx.square(mu) - mx.exp(logvar), axis=1)
        )
        
        # =====================================================================
        # Property Prediction Loss
        # =====================================================================
        # Ensures latent space encodes property information
        
        property_loss = mx.mean(mx.square(predicted_props - true_props))
        
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
        target_nodes = mx.zeros((batch_size, self.max_nodes), dtype=mx.int32)
        masks = mx.zeros((batch_size, self.max_nodes), dtype=mx.bool_)
        
        # Fill in actual node features from batch
        node_ptr = 0
        target_nodes_list = []
        masks_list = []
        
        for i in range(batch_size):
            # Count nodes in this graph
            graph_size = int(mx.sum(batch.batch_indices == i).item())
            graph_size = min(graph_size, self.max_nodes)
            
            # Extract node features for this graph
            graph_nodes = batch.x[node_ptr:node_ptr + graph_size]
            
            # Convert continuous features to discrete indices
            # Use argmax of first 10 dimensions (atom type one-hot)
            node_indices = mx.argmax(graph_nodes[:, :10], axis=1)
            
            # Fill in targets and mask for this graph
            graph_targets = mx.zeros(self.max_nodes, dtype=mx.int32)
            graph_mask = mx.zeros(self.max_nodes, dtype=mx.bool_)
            
            graph_targets[:graph_size] = node_indices
            graph_mask[:graph_size] = True
            
            target_nodes_list.append(graph_targets)
            masks_list.append(graph_mask)
            
            node_ptr += int(mx.sum(batch.batch_indices == i).item())
        
        target_nodes = mx.stack(target_nodes_list)
        masks = mx.stack(masks_list)
        
        # Compute masked cross-entropy loss
        node_logits_flat = mx.reshape(node_logits, (-1, self.node_dim))
        target_nodes_flat = mx.reshape(target_nodes, (-1,))
        masks_flat = mx.reshape(masks, (-1,))
        
        # Only compute loss on actual nodes (not padding)
        num_valid = int(mx.sum(masks_flat).item())
        if num_valid > 0:
            # Use first 10 dimensions of logits (atom types)
            # MLX doesn't support boolean indexing - compute loss differently
            # Weight by mask and average only over valid elements
            node_logits_atoms = node_logits_flat[:, :10]  # [total, 10]
            # Compute log_softmax manually: log(softmax(x)) = x - log(sum(exp(x)))
            log_probs = node_logits_atoms - mx.logsumexp(node_logits_atoms, axis=-1, keepdims=True)
            
            # Gather log probs for target classes
            batch_idx = mx.arange(target_nodes_flat.shape[0])
            target_log_probs = log_probs[batch_idx, target_nodes_flat]  # [total]
            
            # Apply mask and compute mean over valid elements
            masked_log_probs = target_log_probs * masks_flat.astype(mx.float32)
            loss = -mx.sum(masked_log_probs) / num_valid
        else:
            loss = mx.array(0.0)
        
        return loss
    
    def _compute_edge_reconstruction_loss(self, batch, decoder_output, batch_size):
        """
        Compute loss for edge reconstruction
        Maps real edges to decoder's fixed-size edge predictions
        """
        edge_logits = decoder_output['edge_logits']  # [batch_size, num_possible_edges, edge_dim+1]
        edge_indices = decoder_output['edge_indices']  # [num_possible_edges, 2]
        
        if edge_logits.shape[1] == 0:
            return mx.array(0.0)
        
        # Create adjacency matrix for each graph in batch
        num_possible_edges = edge_indices.shape[0]
        target_edge_exist_list = []
        target_edge_type_list = []
        
        # Process each graph in batch
        node_offset = 0
        for b in range(batch_size):
            # Get edges for this graph
            graph_size = int(mx.sum(batch.batch_indices == b).item())
            graph_size = min(graph_size, self.max_nodes)
            
            # Get real edges for this graph - avoid boolean indexing
            batch_array = batch.batch_indices[batch.edge_index[0]]
            edge_indices_in_graph = []
            for e in range(batch.edge_index.shape[1]):
                if int(batch_array[e].item()) == b:
                    edge_indices_in_graph.append(e)
            
            if edge_indices_in_graph:
                graph_edges = batch.edge_index[:, edge_indices_in_graph]
                # Offset edges to be relative to this graph
                graph_edges = graph_edges - node_offset
                
                if hasattr(batch, 'edge_attr'):
                    graph_edge_attr = batch.edge_attr[edge_indices_in_graph]
                else:
                    graph_edge_attr = None
            else:
                graph_edges = None
                graph_edge_attr = None
            
            # Match real edges to decoder's edge predictions
            edge_exist_row = []
            edge_type_row = []
            
            for edge_idx in range(num_possible_edges):
                i, j = int(edge_indices[edge_idx, 0].item()), int(edge_indices[edge_idx, 1].item())
                
                # Skip if outside this graph's size
                if i >= graph_size or j >= graph_size:
                    edge_exist_row.append(0.0)
                    edge_type_row.append(0)
                    continue
                
                # Check if this edge exists in real graph
                edge_exists = False
                edge_type = 0
                
                if graph_edges is not None and graph_edges.shape[1] > 0:
                    # Look for edge (i,j) or (j,i)
                    for e_idx in range(graph_edges.shape[1]):
                        src, dst = int(graph_edges[0, e_idx].item()), int(graph_edges[1, e_idx].item())
                        if (src == i and dst == j) or (src == j and dst == i):
                            edge_exists = True
                            
                            # Get edge type (first 4 dimensions are bond type one-hot)
                            if graph_edge_attr is not None:
                                edge_type = int(mx.argmax(graph_edge_attr[e_idx, :4]).item())
                            break
                
                edge_exist_row.append(1.0 if edge_exists else 0.0)
                edge_type_row.append(edge_type)
            
            target_edge_exist_list.append(edge_exist_row)
            target_edge_type_list.append(edge_type_row)
            
            node_offset += int(mx.sum(batch.batch_indices == b).item())
        
        # Convert to arrays
        target_edge_exist = mx.array(target_edge_exist_list, dtype=mx.float32)
        target_edge_type = mx.array(target_edge_type_list, dtype=mx.int32)
        
        # Compute losses
        # Binary cross-entropy for edge existence
        edge_exist_logits = edge_logits[:, :, -1]
        edge_exist_loss = mx.mean(
            mx.maximum(edge_exist_logits, 0) - edge_exist_logits * target_edge_exist + 
            mx.log(1 + mx.exp(-mx.abs(edge_exist_logits)))
        )
        
        # Categorical cross-entropy for edge types (only on existing edges)
        existing_edges_mask = target_edge_exist > 0.5
        num_existing = int(mx.sum(existing_edges_mask).item())
        
        if num_existing > 0:
            edge_type_logits = edge_logits[:, :, :4]
            
            # Flatten
            edge_type_logits_flat = mx.reshape(edge_type_logits, (-1, 4))
            target_types_flat = mx.reshape(target_edge_type, (-1,))
            mask_flat = mx.reshape(existing_edges_mask, (-1,))
            
            # Compute log_softmax manually
            log_probs = edge_type_logits_flat - mx.logsumexp(edge_type_logits_flat, axis=-1, keepdims=True)
            
            # Gather target log probs
            batch_idx = mx.arange(target_types_flat.shape[0])
            target_log_probs = log_probs[batch_idx, target_types_flat]
            
            # Apply mask and average over existing edges
            masked_log_probs = target_log_probs * mask_flat.astype(mx.float32)
            edge_type_loss = -mx.sum(masked_log_probs) / num_existing
        else:
            edge_type_loss = mx.array(0.0)
        
        return edge_exist_loss + edge_type_loss
    
    def generate(self, target_properties, num_samples=1, temperature=1.0):
        """
        Generate new molecules with specified target properties
        
        Args:
            target_properties: Desired property values [num_properties]
            num_samples: Number of molecules to generate
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            Generated molecular graphs
        """
        was_training = self.training
        self.eval()  # Set to evaluation mode
        
        # =====================================================================
        # Sample from Prior
        # =====================================================================
        # Sample latent codes from N(0, I)
        
        z = mx.random.normal((num_samples, self.latent_dim))
        
        # =====================================================================
        # Prepare Target Properties
        # =====================================================================
        
        if isinstance(target_properties, list):
            target_properties = mx.array(target_properties, dtype=mx.float32)
        
        if len(target_properties.shape) == 1:
            target_properties = mx.broadcast_to(
                mx.expand_dims(target_properties, 0),
                (num_samples, target_properties.shape[0])
            )
        
        # =====================================================================
        # Decode to Graphs
        # =====================================================================
        
        decoder_output = self.decode(z, target_properties)
        sampled_graphs = self.decoder.sample_graph(decoder_output, temperature)
        
        if was_training:
            self.train()  # Restore training mode if it was on
        return sampled_graphs
    
    def interpolate(self, properties_start, properties_end, num_steps=10):
        """
        Generate molecules by interpolating between two property targets
        
        Args:
            properties_start: Starting property values
            properties_end: Ending property values
            num_steps: Number of interpolation steps
        
        Returns:
            sampled_graphs: Generated graphs at each interpolation step
            target_props: Property values at each step
        """
        was_training = self.training
        self.eval()  # Set to evaluation mode
        
        # =====================================================================
        # Create Interpolation Path
        # =====================================================================
        
        alphas = mx.linspace(0, 1, num_steps)
        
        start = mx.array(properties_start, dtype=mx.float32)
        end = mx.array(properties_end, dtype=mx.float32)
        
        interpolated_props = []
        for alpha in alphas:
            props = (1 - alpha) * start + alpha * end
            interpolated_props.append(props)
        
        target_props = mx.stack(interpolated_props)
        
        # =====================================================================
        # Sample and Generate
        # =====================================================================
        
        z = mx.random.normal((num_steps, self.latent_dim))
        decoder_output = self.decode(z, target_props)
        sampled_graphs = self.decoder.sample_graph(decoder_output, temperature=0.8)
        
        if was_training:
            self.train()  # Restore training mode if it was on
        return sampled_graphs, target_props