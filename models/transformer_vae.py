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
        latent_dim: int=256,
        num_heads: int=4,
        num_layers: int=4,
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
        
        # Property-conditioned latent distribution parameters
        # Generate mu and sigma from properties for better conditioning
        self.property_mu = nn.Sequential(
            nn.Linear(self.num_properties, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.property_logvar = nn.Sequential(
            nn.Linear(self.num_properties, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Initialize property_logvar to produce reasonable variance (~0.01 to 1)
        # This prevents KL explosion during early training
        # Note: MLX doesn't support direct bias manipulation, will use Xavier init
        
        # Property prediction head (CVAE requirement)
        # Predict properties from latent code z
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_properties)  # Predict logp, tpsa
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

    def __call__(self, x, properties=None, training=True, noise_std=0.0):
        # x: [B, T] - input sequences
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]  # Shift for next token prediction
        
        # Get property embedding for conditioning (FILM conditioning)
        if properties is not None:
            property_embedding = self.property_encoder(properties)  # [B, embedding_dim]
        else:
            property_embedding = None
        
        # Encoder learns q(z | x, c) - posterior distribution
        mu, logvar = self.encoder(input_seq, property_embedding)
        z = self.reparameterize(mu, logvar)  # Use encoder's z
        
        # Property-conditioned prior p(z | c)
        if properties is not None:
            property_mu = self.property_mu(properties)
            property_logvar = self.property_logvar(properties)
        else:
            property_mu = None
            property_logvar = None
        
        # Use encoder's z during training - property networks trained via KL divergence
        # Add Gaussian noise during training for decoder robustness
        if training and noise_std > 0:
            noise = mx.random.normal(z.shape) * noise_std
            z = z + noise
        
        # Decode with FILM conditioning
        logits = self.decoder(z, input_seq, property_embedding)
        
        # Property prediction head
        predicted_properties = self.property_predictor(z)
        
        return logits, mu, logvar, predicted_properties, property_mu, property_logvar

    def generate_conditional(self, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
        """Generate molecules with target LogP and TPSA values"""
        print(f" Generating molecules with LogP={target_logp}, TPSA={target_tpsa}")
        
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
        
        # Sample z from property-conditioned distribution (best practice for CVAE)
        # Generate mu and sigma from properties
        property_mu = self.property_mu(properties_array)  # [num_samples, latent_dim]
        property_logvar = self.property_logvar(properties_array)  # [num_samples, latent_dim]
        
        # Sample z using reparameterization trick with property conditioning
        std = mx.exp(0.5 * property_logvar)
        noise = mx.random.normal((num_samples, self.latent_dim))
        z = property_mu + std * noise
        
        # Decode with FILM conditioning
        samples = self._decode_conditional(z, property_embedding, temperature, top_k)
        return samples

    def _decode_conditional(self, z, property_embedding, temperature, top_k):
        """Autoregressive decoding with property guidance"""
        batch_size = z.shape[0]
        START = 1
        END = 2
        seq = mx.full((batch_size, 1), START, dtype=mx.int32)
        max_length = 50
        
        for step in range(max_length - 1):
            # Get logits for next token
            logits = self.decoder(z, seq, property_embedding)[:, -1, :]
            
            # Check for NaN/Inf for safety
            if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
                logits = mx.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
            
            # Apply top-k sampling
            if top_k > 0:
                top_k = min(top_k, logits.shape[-1])  # Cap at vocab size
                top_k_vals = mx.sort(logits, axis=-1)[:, -top_k:]  # Get top-k values
                kth_largest = top_k_vals[:, :1]  # kth largest value
                logits = mx.where(
                    logits < kth_largest,
                    -1e9,
                    logits
                )
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            # Greedy decoding (argmax) for better accuracy
            # Use argmax instead of sampling to get highest-confidence predictions
            next_token = mx.argmax(logits, axis=-1, keepdims=True)
            next_token = next_token.astype(mx.int32)
            
            # Update seq so decoder sees previous tokens
            seq = mx.concatenate([seq, next_token], axis=1)
            
            # Stop if all sequences hit END token
            if mx.all(next_token == END):
                break
                
        return seq
