import mlx.core as mx
import mlx.nn as nn
import mlx.nn.init as init

from .mlx_encoder import MLXGraphEncoder  # type: ignore
from .mlx_decoder import MLXGraphDecoder  # type: ignore


class MLXMGCVAE(nn.Module):
    """
    Multi-objective Graph VAE (no property conditioning in the decoder)

    Differences from the conditional version:
    - Keeps the PropertyPredictor and associated loss
    - Decoder IGNORES provided properties; graphs are decoded from z only
    - Public API, class name and signatures remain the same for compatibility
    """
    
    def __init__(
        self,
        node_dim,
        edge_dim,
        latent_dim=64,
        hidden_dim=128,
        num_properties=2,
        num_layers=3,
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
        # Total Loss = Reconstruction + Property + γ⋅|KL - C_t|
        
        self.beta = beta    # Deprecated: kept for backwards compatibility
        self.gamma = gamma  # Capacity-controlled KL divergence weight
        
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
        
        self.decoder = MLXGraphDecoder(
            latent_dim=latent_dim,
            num_properties=num_properties,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            dropout=dropout
        )

        self.log_sigma_sq_node = mx.zeros(1)
            
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2)
        """
        if self.training:
            std = mx.exp(0.5 * logvar)
            eps = mx.random.normal(std.shape)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)
    
    def decode(self, z, target_properties):
        """
        Decode latent codes to molecular graphs, WITHOUT conditioning on properties.
        The target_properties argument is accepted for API compatibility, but is ignored.
        """
        batch_size = z.shape[0]
        # Create a zero tensor for properties to pass through the decoder,
        # ensuring identical shapes while removing conditioning.
        ignored_props = mx.zeros((batch_size, self.num_properties), dtype=mx.float32)
        return self.decoder(z, ignored_props)
    
    def __call__(self, batch):
        """
        Full forward pass: encode → reparameterize → predict properties → decode
        Note: Decoding ignores property conditioning in this variant.
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        batch_idx = batch.batch_indices
        true_properties = batch.y
        
        if len(true_properties.shape) == 1:
            true_properties = mx.expand_dims(true_properties, -1)
        
        mu, logvar = self.encode(x, edge_index, edge_attr, batch_idx)
        z = self.reparameterize(mu, logvar)
        
        # No property prediction in pure VAE variant
        predicted_properties = mx.zeros((z.shape[0], self.num_properties), dtype=mx.float32)

        # Decode WITHOUT conditioning on properties
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
        Compute multi-objective VAE loss with capacity control
        
        Loss = Reconstruction + Property + γ⋅|KL - C_t|
        """
        mu = model_output['mu']
        logvar = model_output['logvar']
        # No property supervision in pure VAE
        decoder_out = model_output['decoder_output']
        
        batch_size = mu.shape[0]
        
        # Reconstruction
        node_recon_loss = self._compute_node_reconstruction_loss(
            batch, decoder_out, batch_size
        )
        edge_recon_loss = self._compute_edge_reconstruction_loss(
            batch, decoder_out, batch_size
        )
        total_recon_loss = node_recon_loss + edge_recon_loss
        
        # KL with capacity control
        kl_per_graph = -0.5 * mx.sum(1 + logvar - mx.square(mu) - mx.exp(logvar), axis=1)
        C_t = getattr(self, 'current_capacity', self.latent_dim * 0.8)
        kl_divergence = mx.mean(kl_per_graph)
        kl_loss = mx.mean(mx.abs(kl_per_graph - C_t))
        
        # No property supervision: set to zero to keep external interfaces stable
        property_loss = mx.array(0.0)
        
        # Total
        total_loss = total_recon_loss + property_loss + self.gamma * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': total_recon_loss,
            'node_recon_loss': node_recon_loss,
            'edge_recon_loss': edge_recon_loss,
            'kl_loss': kl_loss,
            'kl_divergence': kl_divergence,
            'property_loss': property_loss
        }
    
    def _compute_node_reconstruction_loss(self, batch, decoder_output, batch_size):
        # Same as conditional version
        preds = decoder_output['node_logits']
        padded_truths = []
        for b in range(batch_size):
            batch_nodes = []
            for i in range(batch.x.shape[0]):
                if batch.batch_indices[i].item() == b:
                    batch_nodes.append(batch.x[i])
            batch_nodes = mx.stack(batch_nodes)
            num_nodes = min(batch_nodes.shape[0], self.max_nodes)
            if num_nodes < self.max_nodes:
                padding = mx.zeros((self.max_nodes - num_nodes, self.node_dim))
                batch_nodes = mx.concatenate([batch_nodes[:num_nodes], padding], axis=0)
            else:
                batch_nodes = batch_nodes[:self.max_nodes]
            padded_truths.append(batch_nodes)
        truths = mx.stack(padded_truths)
        squared_error = mx.sum(mx.square(preds - truths), axis=-1)
        log_sigma_sq = self.log_sigma_sq_node[0]
        sigma_sq = mx.exp(log_sigma_sq)
        nll = squared_error / (2.0 * sigma_sq) + 0.5 * log_sigma_sq
        if hasattr(batch, "node_mask"):
            mask = batch.node_mask
            nll = nll * mask
            counts = mx.maximum(mx.sum(mask, axis=1, keepdims=True), 1.0)
            per_graph_loss = mx.sum(nll, axis=1, keepdims=True) / counts
            per_graph_loss = per_graph_loss.squeeze(axis=1)
        else:
            per_graph_loss = mx.mean(nll, axis=1)
        return mx.mean(per_graph_loss)
    
    def _compute_edge_reconstruction_loss(self, batch, decoder_output, batch_size):
        # Same as conditional version
        edge_logits = decoder_output['edge_logits']
        pred_edge_indices = decoder_output['edge_indices']
        if edge_logits.shape[1] == 0:
            return mx.array(0.0)
        num_possible_edges = pred_edge_indices.shape[0]
        edge_to_batch = batch.batch_indices[batch.edge_index[0]]
        graph_sizes = []
        node_offsets = []
        offset = 0
        for b in range(batch_size):
            num_nodes = int(mx.sum(batch.batch_indices == b).item())
            graph_sizes.append(min(num_nodes, self.max_nodes))
            node_offsets.append(offset)
            offset += num_nodes
        if hasattr(batch, 'edge_attr'):
            real_edge_types = mx.argmax(batch.edge_attr[:, :4], axis=1)
        else:
            real_edge_types = mx.zeros(batch.edge_index.shape[1], dtype=mx.int32)
        target_edge_exist = mx.zeros((batch_size, num_possible_edges), dtype=mx.float32)
        target_edge_type = mx.zeros((batch_size, num_possible_edges), dtype=mx.int32)
        for b in range(batch_size):
            graph_size = graph_sizes[b]
            node_offset = node_offsets[b]
            batch_edge_mask = edge_to_batch == b
            real_edges_src = batch.edge_index[0] - node_offset
            real_edges_dst = batch.edge_index[1] - node_offset
            for pred_idx in range(num_possible_edges):
                pred_i = int(pred_edge_indices[pred_idx, 0].item())
                pred_j = int(pred_edge_indices[pred_idx, 1].item())
                if pred_i >= graph_size or pred_j >= graph_size:
                    continue
                forward_match = (real_edges_src == pred_i) & (real_edges_dst == pred_j) & batch_edge_mask
                backward_match = (real_edges_src == pred_j) & (real_edges_dst == pred_i) & batch_edge_mask
                matches = forward_match | backward_match
                if mx.any(matches):
                    target_edge_exist[b, pred_idx] = 1.0
                    match_indices = mx.where(matches, mx.arange(matches.shape[0]), mx.full(matches.shape, -1, dtype=mx.int32))
                    first_match = int(mx.max(match_indices).item())
                    if first_match >= 0:
                        target_edge_type[b, pred_idx] = real_edge_types[first_match]
        edge_exist_logits = edge_logits[:, :, -1]
        edge_exist_loss = mx.mean(
            mx.maximum(edge_exist_logits, 0) - edge_exist_logits * target_edge_exist + 
            mx.log(1 + mx.exp(-mx.abs(edge_exist_logits)))
        )
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


