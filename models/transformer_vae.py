import mlx.core as mx
import mlx.nn as nn

from models.transformer_encoder import SelfiesTransformerEncoder
from models.transformer_decoder import SelfiesTransformerDecoder

class SelfiesTransformerVAE(nn.Module):
    """Transformer-based VAE for SELFIES generation with property conditioning"""
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
        
        # Property conditioning layers - FILM architecture (best practice)
        self.num_properties = 2  # logp, tpsa
        # Project properties to embedding_dim for FILM conditioning
        self.property_encoder = nn.Sequential(
            nn.Linear(self.num_properties, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)  # Output to embedding_dim for FILM
        )
        
        # Property normalization parameters (will be set from training)
        self.logp_mean = None
        self.logp_std = None
        self.tpsa_mean = None
        self.tpsa_std = None

    def set_property_normalization(self, logp_mean, logp_std, tpsa_mean, tpsa_std):
        """Set property normalization parameters"""
        self.logp_mean = logp_mean
        self.logp_std = logp_std
        self.tpsa_mean = tpsa_mean
        self.tpsa_std = tpsa_std
    
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
        
        # Add Gaussian noise during training for decoder robustness
        if training and noise_std > 0:
            noise = mx.random.normal(z.shape) * noise_std
            z = z + noise
        
        # Get property embedding for FILM conditioning (best practice)
        if properties is not None:
            property_embedding = self.property_encoder(properties)  # [B, embedding_dim]
        else:
            property_embedding = None
        
        # Decode from latent space with FILM conditioning
        logits = self.decoder(z, input_seq, property_embedding)
        
        return logits, mu, logvar

    def generate_conditional(self, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
        """Generate molecules with target LogP and TPSA values"""
        print(f"🎯 Generating molecules with LogP={target_logp}, TPSA={target_tpsa}")
        
        # Normalize properties if normalization params are set
        if self.logp_mean is not None and self.logp_std is not None:
            norm_logp = (target_logp - self.logp_mean) / self.logp_std
            norm_tpsa = (target_tpsa - self.tpsa_mean) / self.tpsa_std
        else:
            norm_logp = target_logp
            norm_tpsa = target_tpsa
        
        # Create property embeddings for FILM conditioning
        properties_array = mx.array([[norm_logp, norm_tpsa]] * num_samples)
        property_embedding = self.property_encoder(properties_array)  # [num_samples, embedding_dim]
        
        # Sample from prior N(0,1) - matches training where encoder learns q(z|x)
        z = mx.random.normal((num_samples, self.latent_dim))
        
        # Decode with FILM conditioning
        samples = self._decode_conditional(z, property_embedding, temperature, top_k)
        return samples

    def _decode_conditional(self, z, property_embedding, temperature, top_k):
        """Autoregressive decoding with property guidance"""
        batch_size = z.shape[0]
        START = 1
        seq = mx.full((batch_size, 1), START, dtype=mx.int32)
        samples = seq
        max_length = 50
        
        for _ in range(max_length - 1):
            logits = self.decoder(z, seq, property_embedding)[:, -1, :]
            
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
