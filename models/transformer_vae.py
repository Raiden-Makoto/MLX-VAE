import mlx.core as mx
import mlx.nn as nn

from models.transformer_encoder import SelfiesTransformerEncoder
from models.transformer_decoder import SelfiesTransformerDecoder

class SelfiesTransformerVAE(nn.Module):
    """Transformer-based VAE for SELFIES generation"""
    def __init__(
        self,
        vocab_size,
        embedding_dim: int=128,
        hidden_dim: int=256,
        latent_dim: int=64,
        num_heads: int=8,
        num_layers: int=6,
        dropout: float=0.1,
    ):
        super().__init__()
        self.encoder = SelfiesTransformerEncoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, 
            num_heads, num_layers, dropout
        )
        self.decoder = SelfiesTransformerDecoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim,
            num_heads, num_layers, dropout
        )
        self.latent_dim = latent_dim
        
        # Property embedding for conditional generation (optional)
        self.num_properties = 2  # logp, tpsa
        self.property_embedding = None  # Will be initialized if needed

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mu.shape)
        return mu + std * eps

    def __call__(self, x, training=True, noise_std=0.05):
        # x: [B, T] - input sequences
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]  # Shift for next token prediction
        
        # Encode to latent space
        mu, logvar = self.encoder(input_seq)
        z = self.reparameterize(mu, logvar)
        
        # Add Gaussian noise during training for decoder robustness
        if training and noise_std > 0:
            noise = mx.random.normal(z.shape) * noise_std
            z = z + noise
        
        # Decode from latent space
        logits = self.decoder(z, input_seq)
        
        return logits, mu, logvar

    def generate_conditional(self, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
        """Generate molecules with target LogP and TPSA values"""
        print(f"🎯 Generating molecules with LogP={target_logp}, TPSA={target_tpsa}")
        
        # Initialize property embedding if not already done
        if self.property_embedding is None:
            self.property_embedding = nn.Linear(self.num_properties, self.latent_dim)
        
        # Create property embeddings
        properties_array = mx.array([[target_logp, target_tpsa]] * num_samples)
        property_latent = self.property_embedding(properties_array)  # [num_samples, latent_dim]
        
        # Sample base latent vectors
        base_z = mx.random.normal((num_samples, self.latent_dim))
        
        # Combine base latent with property guidance
        z = base_z + property_latent
        
        # Decode to generate molecules
        samples = self._decode_conditional(z, temperature, top_k)
        return samples

    def _decode_conditional(self, z, temperature, top_k):
        """Autoregressive decoding with property guidance"""
        batch_size = z.shape[0]
        START = 1
        seq = mx.full((batch_size, 1), START, dtype=mx.int32)
        samples = seq
        max_length = 50
        
        for _ in range(max_length - 1):
            logits = self.decoder(z, seq)[:, -1, :]
            
            # Apply top-k sampling
            if top_k > 0:
                top_k_logits = mx.topk(logits, top_k, axis=-1)
                kth_largest = top_k_logits[:, top_k-1:top_k]
                logits = mx.where(
                    logits < kth_largest,
                    -1e9,
                    logits
                )
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs)
            next_token = mx.expand_dims(next_token, axis=1)
            samples = mx.concatenate([samples, next_token], axis=1)
            
            # Stop if all sequences hit END token
            if mx.all(next_token == 2):
                break
                
        return samples
