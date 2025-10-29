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
        
        # TPSA conditioning (prior network for z sampling)
        self.property_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # TPSA only
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # TPSA-conditioned latent distribution parameters (for prior p(z|c))
        self.property_mu = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.property_logvar = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # LogP conditioning - BOTH prior network AND decoder
        # LogP-conditioned latent distribution parameters (for prior p(z|c))
        self.logp_mu = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.logp_logvar = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # LogP encoder for decoder conditioning
        self.logp_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
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
        
        # TPSA conditioning for prior network
        tpsa_embedding = None
        if tpsa_properties is not None:
            tpsa_embedding = self.property_encoder(tpsa_properties)  # [B, embedding_dim]
        
        # Encoder gets TPSA conditioning
        mu, logvar = self.encoder(input_seq, tpsa_embedding)
        z = self.reparameterize(mu, logvar)
        
        # Property-conditioned priors p(z | properties)
        # For KL computation, need both property_mu and property_logvar
        # If both properties provided, combine them; else use single property
        if tpsa_properties is not None and logp_properties is not None:
            # Dual property conditioning: average the prior networks
            tpsa_mu = self.property_mu(tpsa_properties)
            tpsa_logvar = self.property_logvar(tpsa_properties)
            logp_mu = self.logp_mu(logp_properties)
            logp_logvar = self.logp_logvar(logp_properties)
            property_mu = (tpsa_mu + logp_mu) / 2
            property_logvar = (tpsa_logvar + logp_logvar) / 2
        elif tpsa_properties is not None:
            property_mu = self.property_mu(tpsa_properties)
            property_logvar = self.property_logvar(tpsa_properties)
        elif logp_properties is not None:
            property_mu = self.logp_mu(logp_properties)
            property_logvar = self.logp_logvar(logp_properties)
        else:
            property_mu = None
            property_logvar = None
        
        # Add noise during training
        if training and noise_std > 0:
            noise = mx.random.normal(z.shape) * noise_std
            z = z + noise
        
        # Combine embeddings for decoder
        # TPSA modulates latent structure, LogP modulates decoder features
        decoder_embedding = tpsa_embedding
        if logp_properties is not None:
            logp_embedding = self.logp_encoder(logp_properties)  # [B, embedding_dim]
            if decoder_embedding is None:
                decoder_embedding = logp_embedding
            else:
                # Combine both embeddings
                decoder_embedding = decoder_embedding + logp_embedding
        
        # Decode with combined conditioning
        logits = self.decoder(z, input_seq, decoder_embedding)
        
        # Predict both properties
        predicted_tpsa = self.tpsa_predictor(z)  # [B, 1]
        predicted_logp = self.logp_predictor(z)  # [B, 1]
        predicted_properties = mx.concatenate([predicted_logp, predicted_tpsa], axis=1)  # [B, 2]
        
        # Return both property priors for KL computation
        # Also return individual tpsa and logp priors if needed
        if tpsa_properties is not None and logp_properties is not None:
            tpsa_kl_mu = self.property_mu(tpsa_properties)
            tpsa_kl_logvar = self.property_logvar(tpsa_properties)
            logp_kl_mu = self.logp_mu(logp_properties)
            logp_kl_logvar = self.logp_logvar(logp_properties)
        else:
            tpsa_kl_mu = None
            tpsa_kl_logvar = None
            logp_kl_mu = None
            logp_kl_logvar = None
        
        return logits, mu, logvar, predicted_properties, property_mu, property_logvar, tpsa_kl_mu, tpsa_kl_logvar, logp_kl_mu, logp_kl_logvar

    def generate_conditional(self, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
        """Generate molecules with target LogP and TPSA values
        
        TPSA conditions the prior (z sampling), LogP conditions the decoder (FILM)"""
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
        
        # Create property arrays for prior networks
        tpsa_array = mx.array([[norm_tpsa]] * num_samples)  # [num_samples, 1]
        logp_array = mx.array([[norm_logp]] * num_samples)  # [num_samples, 1]
        
        tpsa_embedding = self.property_encoder(tpsa_array)  # [num_samples, embedding_dim]
        
        # Sample z from DUAL property-conditioned prior
        # Average the TPSA and LogP prior networks
        tpsa_mu = self.property_mu(tpsa_array)  # [num_samples, latent_dim]
        tpsa_logvar = self.property_logvar(tpsa_array)  # [num_samples, latent_dim]
        logp_mu = self.logp_mu(logp_array)  # [num_samples, latent_dim]
        logp_logvar = self.logp_logvar(logp_array)  # [num_samples, latent_dim]
        
        # Combine both priors
        property_mu = (tpsa_mu + logp_mu) / 2
        property_logvar = (tpsa_logvar + logp_logvar) / 2
        
        std = mx.exp(0.5 * property_logvar)
        noise = mx.random.normal((num_samples, self.latent_dim))
        z = property_mu + std * noise
        
        # Create LogP embedding for decoder conditioning
        logp_embedding = self.logp_encoder(logp_array)  # [num_samples, embedding_dim]
        
        # Combine TPSA + LogP embeddings for decoder
        decoder_embedding = tpsa_embedding + logp_embedding
        
        # Decode with combined conditioning
        samples = self._decode_conditional(z, decoder_embedding, temperature, top_k)
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
