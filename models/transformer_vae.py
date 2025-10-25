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
        
        # FiLM layers for conditional generation
        self.num_properties = 2  # logp, tpsa
        self.film_gamma = nn.Linear(self.num_properties, latent_dim)  # Scaling parameters
        self.film_beta = nn.Linear(self.num_properties, latent_dim)   # Shifting parameters
        
        # Property normalization (dataset statistics)
        self.logp_mean = 0.33
        self.logp_std = 0.95
        self.tpsa_mean = 36.53
        self.tpsa_std = 20.82

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mu.shape)
        return mu + std * eps

    def __call__(self, x, properties=None, training=True, noise_std=0.05):
        # x: [B, T] - input sequences
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]  # Shift for next token prediction
        
        # Encode to latent space
        mu, logvar = self.encoder(input_seq)
        z = self.reparameterize(mu, logvar)
        
        # Apply FiLM conditioning if properties provided
        if properties is not None:
            # Normalize properties for better FiLM learning
            normalized_props = mx.zeros_like(properties)
            normalized_props[:, 0] = (properties[:, 0] - self.logp_mean) / self.logp_std
            normalized_props[:, 1] = (properties[:, 1] - self.tpsa_mean) / self.tpsa_std
            
            # Compute FiLM parameters (scaling and shifting)
            gamma = self.film_gamma(normalized_props)  # [batch_size, latent_dim]
            beta = self.film_beta(normalized_props)    # [batch_size, latent_dim]
            
            # Apply feature-wise linear modulation
            z = gamma * z + beta
        
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
        
        # Create property arrays for FiLM conditioning
        properties_array = mx.array([[target_logp, target_tpsa]] * num_samples)
        
        # Normalize properties for better FiLM learning
        normalized_props = mx.zeros_like(properties_array)
        normalized_props[:, 0] = (properties_array[:, 0] - self.logp_mean) / self.logp_std
        normalized_props[:, 1] = (properties_array[:, 1] - self.tpsa_mean) / self.tpsa_std
        
        # Sample base latent vectors
        base_z = mx.random.normal((num_samples, self.latent_dim))
        
        # Apply FiLM conditioning
        gamma = self.film_gamma(normalized_props)  # [num_samples, latent_dim]
        beta = self.film_beta(normalized_props)    # [num_samples, latent_dim]
        z = gamma * base_z + beta
        
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
