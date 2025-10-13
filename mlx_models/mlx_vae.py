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
        max_nodes=20,
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
        Compute loss for node reconstruction - LOOPLESS VERSION
        """
        node_logits = decoder_output['node_logits']  # [batch_size, max_nodes, node_dim]
        
        # Extract atom types from all nodes at once
        node_indices = mx.argmax(batch.x[:, :10], axis=1)  # [total_nodes]
        node_to_batch = batch.batch_indices  # [total_nodes]
        
        # Build target tensor using list comprehension (Python loop, but no .item() calls)
        # This is unavoidable without scatter operations in MLX
        total_nodes = batch.x.shape[0]
        targets_data = []
        masks_data = []
        
        for b in range(batch_size):
            # Get nodes for this batch (vectorized comparison)
            batch_mask = node_to_batch == b
            batch_node_indices = mx.where(batch_mask, node_indices, mx.zeros_like(node_indices))
            
            # Count nodes in this batch
            num_nodes = int(mx.sum(batch_mask.astype(mx.int32)).item())
            num_nodes = min(num_nodes, self.max_nodes)
            
            # Create row for this batch
            row_targets = mx.zeros(self.max_nodes, dtype=mx.int32)
            row_mask = mx.zeros(self.max_nodes, dtype=mx.bool_)
            
            if num_nodes > 0:
                # Extract just the nodes for this batch
                valid_indices = [node_indices[i] for i in range(total_nodes) if node_to_batch[i] == b][:num_nodes]
                for pos, idx in enumerate(valid_indices):
                    if pos < self.max_nodes:
                        row_targets[pos] = idx
                        row_mask[pos] = True
            
            targets_data.append(row_targets)
            masks_data.append(row_mask)
        
        target_nodes = mx.stack(targets_data)
        masks = mx.stack(masks_data)
        
        # Compute loss (fully vectorized from here)
        node_logits_flat = mx.reshape(node_logits, (-1, self.node_dim))
        target_nodes_flat = mx.reshape(target_nodes, (-1,))
        masks_flat = mx.reshape(masks, (-1,))
        
        node_logits_atoms = node_logits_flat[:, :10]
        log_probs = node_logits_atoms - mx.logsumexp(node_logits_atoms, axis=-1, keepdims=True)
        
        batch_idx = mx.arange(target_nodes_flat.shape[0])
        target_log_probs = log_probs[batch_idx, target_nodes_flat]
        
        masked_log_probs = target_log_probs * masks_flat.astype(mx.float32)
        num_valid = mx.sum(masks_flat.astype(mx.float32))
        
        loss = mx.where(num_valid > 0, -mx.sum(masked_log_probs) / num_valid, mx.array(0.0))
        return loss
    
    def _compute_edge_reconstruction_loss(self, batch, decoder_output, batch_size):
        """
        Compute loss for edge reconstruction - REDUCED LOOPS VERSION
        """
        edge_logits = decoder_output['edge_logits']  # [batch_size, num_possible_edges, edge_dim+1]
        pred_edge_indices = decoder_output['edge_indices']  # [num_possible_edges, 2]
        
        if edge_logits.shape[1] == 0:
            return mx.array(0.0)
        
        num_possible_edges = pred_edge_indices.shape[0]
        
        # Precompute: which batch each real edge belongs to
        edge_to_batch = batch.batch_indices[batch.edge_index[0]]  # [num_real_edges]
        
        # Precompute: node offsets and graph sizes for each batch
        graph_sizes = []
        node_offsets = []
        offset = 0
        for b in range(batch_size):
            num_nodes = int(mx.sum(batch.batch_indices == b).item())
            graph_sizes.append(min(num_nodes, self.max_nodes))
            node_offsets.append(offset)
            offset += num_nodes
        
        # Get edge types if available
        if hasattr(batch, 'edge_attr'):
            real_edge_types = mx.argmax(batch.edge_attr[:, :4], axis=1)  # [num_real_edges]
        else:
            real_edge_types = mx.zeros(batch.edge_index.shape[1], dtype=mx.int32)
        
        # Build target tensors
        target_edge_exist = mx.zeros((batch_size, num_possible_edges), dtype=mx.float32)
        target_edge_type = mx.zeros((batch_size, num_possible_edges), dtype=mx.int32)
        
        # For each batch, vectorize the edge matching as much as possible
        for b in range(batch_size):
            graph_size = graph_sizes[b]
            node_offset = node_offsets[b]
            
            # Get mask of edges belonging to this batch
            batch_edge_mask = edge_to_batch == b  # [num_real_edges]
            
            # Get real edges for this batch (with offset removed)
            real_edges_src = batch.edge_index[0] - node_offset  # [num_real_edges]
            real_edges_dst = batch.edge_index[1] - node_offset  # [num_real_edges]
            
            # For each predicted edge, check if it matches ANY real edge (vectorized inner loop!)
            for pred_idx in range(num_possible_edges):
                pred_i = int(pred_edge_indices[pred_idx, 0].item())
                pred_j = int(pred_edge_indices[pred_idx, 1].item())
                
                # Skip if outside graph size
                if pred_i >= graph_size or pred_j >= graph_size:
                    continue
                
                # Vectorized check: does this predicted edge match any real edge?
                # Check both directions: (i,j) or (j,i)
                forward_match = (real_edges_src == pred_i) & (real_edges_dst == pred_j) & batch_edge_mask
                backward_match = (real_edges_src == pred_j) & (real_edges_dst == pred_i) & batch_edge_mask
                matches = forward_match | backward_match  # [num_real_edges]
                
                # Check if any match exists
                if mx.any(matches):
                    target_edge_exist[b, pred_idx] = 1.0
                    # Get the edge type from the first matching edge
                    match_indices = mx.where(matches, mx.arange(matches.shape[0]), mx.full(matches.shape, -1, dtype=mx.int32))
                    first_match = int(mx.max(match_indices).item())
                    if first_match >= 0:
                        target_edge_type[b, pred_idx] = real_edge_types[first_match]
        
        # Compute losses (fully vectorized from here)
        edge_exist_logits = edge_logits[:, :, -1]
        edge_exist_loss = mx.mean(
            mx.maximum(edge_exist_logits, 0) - edge_exist_logits * target_edge_exist + 
            mx.log(1 + mx.exp(-mx.abs(edge_exist_logits)))
        )
        
        # Edge type loss
        existing_edges_mask = target_edge_exist > 0.5
        num_existing = mx.sum(existing_edges_mask.astype(mx.float32))
        
        if num_existing > 0:
            edge_type_logits = edge_logits[:, :, :4]
            edge_type_logits_flat = mx.reshape(edge_type_logits, (-1, 4))
            target_types_flat = mx.reshape(target_edge_type, (-1,))
            mask_flat = mx.reshape(existing_edges_mask, (-1,))
            
            log_probs = edge_type_logits_flat - mx.logsumexp(edge_type_logits_flat, axis=-1, keepdims=True)
            batch_idx = mx.arange(target_types_flat.shape[0])
            target_log_probs = log_probs[batch_idx, target_types_flat]
            
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