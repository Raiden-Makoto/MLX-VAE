import mlx.core as mx
import mlx.nn as nn

from models.transformer_encoder import SelfiesTransformerEncoder
from models.transformer_decoder import SelfiesTransformerDecoder
from models.tpsa_to_latent import TPSAToLatentPredictor

class SelfiesTransformerVAE(nn.Module):
    """Transformer-based VAE for SELFIES generation with FILM-only conditioning"""
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
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.encoder = SelfiesTransformerEncoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, 
            num_heads, num_layers, dropout
        )
        self.decoder = SelfiesTransformerDecoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim,
            num_heads, num_layers, dropout
        )
        self.latent_dim = latent_dim
        
        # Property encoders (used for latent injection)
        self.property_encoder_logp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.property_encoder_tpsa = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Project property embeddings to latent dimension for direct injection
        self.property_to_latent = nn.Linear(2 * embedding_dim, latent_dim)
        
        # Predictor: TPSA -> latent z
        self.tpsa_predictor = TPSAToLatentPredictor(latent_dim=latent_dim, hidden_dim=hidden_dim)
        
        # Property normalization parameters
        self.logp_mean = None
        self.logp_std = None
        self.tpsa_mean = None
        self.tpsa_std = None

        # Auxiliary property head: predict TPSA from [z || FILM]
        self.aux_property_head = nn.Sequential(
            nn.Linear(latent_dim + 2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Property encoder reconstruction heads (supervise encoders directly)
        self.tpsa_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.logp_reconstructor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_property_normalization(self, logp_mean, logp_std, tpsa_mean, tpsa_std):
        self.logp_mean = logp_mean
        self.logp_std = logp_std
        self.tpsa_mean = tpsa_mean
        self.tpsa_std = tpsa_std
    
    def reparameterize(self, mu, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mu.shape)
        return mu + std * eps

    def __call__(self, x, properties=None, training=True, noise_std=0.0):
        # x: [B, T]
        input_seq = x[:, :-1]
        # Encoder
        mu, logvar = self.encoder(input_seq)
        z = self.reparameterize(mu, logvar)
        # Optional noise
        if training and noise_std > 0:
            z = z + mx.random.normal(z.shape) * noise_std
        # Inject properties directly into latent space if provided
        if properties is not None and properties.shape[1] >= 2:
            logp_properties = properties[:, :1]
            tpsa_properties = properties[:, 1:2]
            # Normalize properties if stats are available
            if self.logp_mean is not None and self.logp_std is not None:
                logp_properties = (logp_properties - self.logp_mean) / self.logp_std
            if self.tpsa_mean is not None and self.tpsa_std is not None:
                tpsa_properties = (tpsa_properties - self.tpsa_mean) / self.tpsa_std
            logp_emb = self.property_encoder_logp(logp_properties)
            tpsa_emb = self.property_encoder_tpsa(tpsa_properties)
            # Concatenate and project to latent dimension
            prop_emb = mx.concatenate([logp_emb, tpsa_emb], axis=-1)  # [B, 2*embedding_dim]
            prop_latent = self.property_to_latent(prop_emb)  # [B, latent_dim]
            # INJECT property directly into latent (key change!)
            z = z + prop_latent
        
        # Decode (no FILM needed anymore!)
        logits = self.decoder(z, input_seq)
        # Auxiliary TPSA prediction (normalized) when properties are provided
        aux_tpsa_pred = None
        if properties is not None and properties.shape[1] >= 2:
            # For aux head, still use property embeddings concatenated with z
            logp_properties = properties[:, :1]
            tpsa_properties = properties[:, 1:2]
            if self.logp_mean is not None and self.logp_std is not None:
                logp_properties = (logp_properties - self.logp_mean) / self.logp_std
            if self.tpsa_mean is not None and self.tpsa_std is not None:
                tpsa_properties = (tpsa_properties - self.tpsa_mean) / self.tpsa_std
            logp_emb = self.property_encoder_logp(logp_properties)
            tpsa_emb = self.property_encoder_tpsa(tpsa_properties)
            prop_emb = mx.concatenate([logp_emb, tpsa_emb], axis=-1)
            z_prop = mx.concatenate([z, prop_emb], axis=-1)
            aux_tpsa_pred = self.aux_property_head(z_prop)
        # Return VAE outputs + optional aux prediction
        return logits, mu, logvar, aux_tpsa_pred

    def generate_conditional(self, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
        """Generate molecules with properties via direct latent injection; z ~ N(0,I) + property."""
        # Normalize properties
        if self.logp_mean is not None and self.logp_std is not None:
            norm_logp = (target_logp - self.logp_mean) / self.logp_std
        else:
            norm_logp = target_logp
        if self.tpsa_mean is not None and self.tpsa_std is not None:
            norm_tpsa = (target_tpsa - self.tpsa_mean) / self.tpsa_std
        else:
            norm_tpsa = target_tpsa
        
        # Get property embedding
        logp_array = mx.array([[norm_logp]] * num_samples)
        tpsa_array = mx.array([[norm_tpsa]] * num_samples)
        logp_emb = self.property_encoder_logp(logp_array)
        tpsa_emb = self.property_encoder_tpsa(tpsa_array)
        
        # Concatenate and project to latent dimension
        prop_emb = mx.concatenate([logp_emb, tpsa_emb], axis=-1)  # [B, 2*embedding_dim]
        prop_latent = self.property_to_latent(prop_emb)  # [B, latent_dim]
        
        # Sample base latent from prior
        z_base = mx.random.normal((num_samples, self.latent_dim))
        
        # INJECT property directly into latent (key change!)
        z_conditioned = z_base + prop_latent
        
        # Decode (no FILM needed!)
        return self._decode_conditional(z_conditioned, temperature, top_k)

    def _decode_conditional(self, z, temperature, top_k):
        batch_size = z.shape[0]
        START = 1
        END = 2
        seq = mx.full((batch_size, 1), START, dtype=mx.int32)
        max_length = 50
        for _ in range(max_length - 1):
            logits = self.decoder(z, seq)[:, -1, :]
            if mx.any(mx.isnan(logits)) or mx.any(mx.isinf(logits)):
                logits = mx.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                top_vals = mx.sort(logits, axis=-1)[:, -k:]
                kth = top_vals[:, :1]
                logits = mx.where(logits < kth, -1e9, logits)
            if temperature != 1.0:
                logits = logits / temperature
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs, axis=-1)
            next_token = mx.expand_dims(next_token, axis=-1).astype(mx.int32)
            seq = mx.concatenate([seq, next_token], axis=1)
            if mx.all(next_token == END):
                break
        return seq

    def generate_conditional_inverse(self, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
        """Use learned TPSA→z mapping + property injection to generate molecules."""
        # Normalize TPSA
        if self.tpsa_mean is not None and self.tpsa_std is not None:
            norm_tpsa = (target_tpsa - self.tpsa_mean) / self.tpsa_std
        else:
            norm_tpsa = target_tpsa
        # Predict z from target TPSA
        tpsa_arr = mx.array([[norm_tpsa]] * num_samples)
        z_base = self.tpsa_predictor(tpsa_arr)
        
        # Inject properties into latent
        if self.logp_mean is not None and self.logp_std is not None:
            norm_logp = (target_logp - self.logp_mean) / self.logp_std
        else:
            norm_logp = target_logp
        logp_arr = mx.array([[norm_logp]] * num_samples)
        logp_emb = self.property_encoder_logp(logp_arr)
        tpsa_emb = self.property_encoder_tpsa(tpsa_arr)
        prop_emb = mx.concatenate([logp_emb, tpsa_emb], axis=-1)
        prop_latent = self.property_to_latent(prop_emb)
        
        # Inject properties into latent
        z_conditioned = z_base + prop_latent
        
        # Decode
        return self._decode_conditional(z_conditioned, temperature, top_k)


    def generate_conditional_with_search(self, target_logp, target_tpsa, num_candidates=1000, top_k=100):
        """
        1. Sample many z ~ N(0,I)
        2. For each z, generate molecule and compute properties
        3. Select top_k z's closest to target properties
        4. Generate final molecules from those z's
        Returns: list of SMILES strings for the best molecules
        """
        # Lazy imports to avoid circular deps
        from utils.sample import tokens_to_selfies
        from utils.validate import batch_validate_selfies

        # Normalize targets
        if self.logp_mean is not None and self.logp_std is not None:
            norm_logp = (target_logp - self.logp_mean) / self.logp_std
        else:
            norm_logp = target_logp
        if self.tpsa_mean is not None and self.tpsa_std is not None:
            norm_tpsa = (target_tpsa - self.tpsa_mean) / self.tpsa_std
        else:
            norm_tpsa = target_tpsa

        # Sample many candidates and inject properties
        z_base = mx.random.normal((num_candidates, self.latent_dim))
        
        # Create property embeddings and inject into latent
        logp_arr = mx.array([[norm_logp]] * num_candidates)
        tpsa_arr = mx.array([[norm_tpsa]] * num_candidates)
        logp_emb = self.property_encoder_logp(logp_arr)
        tpsa_emb = self.property_encoder_tpsa(tpsa_arr)
        prop_emb = mx.concatenate([logp_emb, tpsa_emb], axis=-1)
        prop_latent = self.property_to_latent(prop_emb)
        z_conditioned = z_base + prop_latent

        # Generate molecules and compute their properties
        selfies_list = []
        for i in range(num_candidates):
            z_single = z_conditioned[i:i+1]
            seq = self._decode_conditional(z_single, temperature=1.0, top_k=10)
            # Convert to SELFIES
            seq_selfies = tokens_to_selfies(seq)
            if seq_selfies and isinstance(seq_selfies[0], str) and len(seq_selfies[0]) > 0:
                selfies_list.append(seq_selfies[0])

        if not selfies_list:
            return []

        # Convert to SMILES and compute properties using existing validator
        results = batch_validate_selfies(selfies_list, verbose=False)

        generated_smiles = []
        properties_list = []
        for res in results:
            if res and isinstance(res, dict) and res.get('smiles'):
                generated_smiles.append(res['smiles'])
                properties_list.append([res.get('logp', 0.0), res.get('tpsa', 0.0)])

        if not properties_list:
            return []

        # Select top_k closest to target
        properties_array = mx.array(properties_list)
        target_array = mx.array([[target_logp, target_tpsa]])
        distances = mx.sum((properties_array - target_array) ** 2, axis=1)
        k = min(top_k, len(generated_smiles))
        top_k_indices = mx.argsort(distances)[:k]

        # Return best molecules
        return [generated_smiles[int(i.item())] for i in top_k_indices]
