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
        
        # Joint property conditioning (takes both LogP and TPSA)
        # ONE joint prior network that takes [logp, tpsa] as input
        self.joint_property_mu = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [logp, tpsa]
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.joint_property_logvar = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [logp, tpsa]
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Property prediction heads
        self.tpsa_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predict normalized TPSA
        )
        self.logp_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predict normalized LogP
        )
        
        # Property normalization parameters
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
        
        # Extract TPSA and LogP properties
        tpsa_properties = None
        logp_properties = None
        
        if properties is not None:
            if properties.shape[1] == 2:
                # Separate properties: [logp, tpsa]
                logp_properties = properties[:, :1]  # [B, 1]
                tpsa_properties = properties[:, 1:2]  # [B, 1]
            else:
                # Single property (assume TPSA)
                tpsa_properties = properties  # [B, 1]
        
        # Encoder
        mu, logvar = self.encoder(input_seq)
        z = self.reparameterize(mu, logvar)
        
        # Joint property-conditioned prior p(z | [logp, tpsa])
        # ONE joint prior network (not averaging separate priors)
        if tpsa_properties is not None and logp_properties is not None:
            # Joint input: concatenate both properties [B, 2]
            joint_input = mx.concatenate([logp_properties, tpsa_properties], axis=-1)  # [B, 2]
            joint_mu = self.joint_property_mu(joint_input)
            joint_logvar = self.joint_property_logvar(joint_input)
        elif tpsa_properties is not None:
            # Single property: use joint with zero-padded logp
            # For backwards compatibility, treat TPSA-only as joint with logp=0
            joint_input = mx.concatenate([mx.zeros_like(tpsa_properties), tpsa_properties], axis=-1)
            joint_mu = self.joint_property_mu(joint_input)
            joint_logvar = self.joint_property_logvar(joint_input)
        else:
            joint_mu = None
            joint_logvar = None
        
        # Add noise during training
        if training and noise_std > 0:
            noise = mx.random.normal(z.shape) * noise_std
            z = z + noise
        
        # Concatenate properties with z for decoder conditioning (CVAE best practice)
        if tpsa_properties is not None and logp_properties is not None:
            properties_combined = mx.concatenate([logp_properties, tpsa_properties], axis=-1)  # [B, 2]
            z_conditioned = mx.concatenate([z, properties_combined], axis=-1)  # [B, latent_dim+2]
        else:
            z_conditioned = z
        
        logits = self.decoder(z_conditioned, input_seq)
        
        # Predict both properties
        predicted_tpsa = self.tpsa_predictor(z)  # [B, 1]
        predicted_logp = self.logp_predictor(z)  # [B, 1]
        predicted_properties = mx.concatenate([predicted_logp, predicted_tpsa], axis=1)  # [B, 2]
        
        return logits, mu, logvar, predicted_properties, joint_mu, joint_logvar, None, None, None, None

    def generate_conditional(self, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
        """Generate molecules with target LogP and TPSA values using JOINT prior"""
        print(f" Generating molecules with LogP={target_logp}, TPSA={target_tpsa}")
        
        # Normalize properties if normalization params are set
        if self.tpsa_mean is not None and self.tpsa_std is not None:
            norm_tpsa = (target_tpsa - self.tpsa_mean) / self.tpsa_std
        else:
            norm_tpsa = target_tpsa
        
        if self.logp_mean is not None and self.logp_std is not None:
            norm_logp = (target_logp - self.logp_mean) / self.logp_std
        else:
            norm_logp = target_logp
        
        # JOINT prior input: concatenate [logp, tpsa]
        joint_input = mx.array([[norm_logp, norm_tpsa]] * num_samples)  # [num_samples, 2]
        joint_mu = self.joint_property_mu(joint_input)
        joint_logvar = self.joint_property_logvar(joint_input)
        
        # Sample z from JOINT prior
        std = mx.exp(0.5 * joint_logvar)
        noise = mx.random.normal((num_samples, self.latent_dim))
        z = joint_mu + std * noise
        
        # Concatenate properties with z for conditioning (CVAE best practice)
        properties_combined = mx.array([[norm_logp, norm_tpsa]] * num_samples)  # [num_samples, 2]
        z_conditioned = mx.concatenate([z, properties_combined], axis=-1)  # [num_samples, latent_dim+2]
        
        # Decode with conditioned z
        samples = self._decode_conditional(z_conditioned, None, temperature, top_k)
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
                
            # Sample from logits distribution instead of greedy
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs, axis=-1)
            next_token = mx.expand_dims(next_token, axis=-1)  # Add keepdims dimension
            next_token = next_token.astype(mx.int32)
            
            # Update seq so decoder sees previous tokens
            seq = mx.concatenate([seq, next_token], axis=1)
            
            # Stop if all sequences hit END token
            if mx.all(next_token == END):
                break
                
        return seq
