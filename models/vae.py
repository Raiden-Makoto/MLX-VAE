import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional
from .encoder import MLXEncoder
from .decoder import MLXAutoregressiveDecoder
from .decoder_sampling import MLXAutoregressiveDecoderSampling

class ARCVAE(nn.Module):
    """
    Complete Autoregressive Conditional VAE for molecule generation
    
    Components:
    - MLXEncoder: Encodes molecules to latent space with conditions
    - MLXAutoregressiveDecoder: Reconstructs molecules from latent space
    - Handles training with teacher forcing and KL annealing
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 200,
        num_conditions: int = 6,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        """Initialize complete AR-CVAE"""
        super().__init__()
        
        # Import encoder from previous implementation
        self.encoder = MLXEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_conditions=num_conditions,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = MLXAutoregressiveDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_conditions=num_conditions,
            num_layers=num_layers
        )
        
        # Create sampling decoder for inference
        self.decoder_sampling = MLXAutoregressiveDecoderSampling(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_conditions=num_conditions,
            num_layers=num_layers
        )
        
        self.latent_dim = latent_dim
    
    def __call__(
        self,
        x: mx.array,
        conditions: mx.array,
        target_seq: Optional[mx.array] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Forward pass through complete VAE
        
        Args:
            x: Input molecule sequences [batch_size, seq_length]
            conditions: Property conditions [batch_size, num_conditions]
            target_seq: Target sequences (typically same as x)
            teacher_forcing_ratio: Teacher forcing probability
        
        Returns:
            recon_logits: Reconstructed logits [batch_size, seq_length, vocab_size]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log-variance [batch_size, latent_dim]
            z: Sampled latent vector [batch_size, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x, conditions)
        
        # Sample from latent space
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode with teacher forcing
        recon_logits = self.decoder(
            z,
            conditions,
            target_seq=target_seq,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        return recon_logits, mu, logvar, z
    
    def generate(
        self,
        batch_size: int,
        conditions: mx.array,
        max_length: int = 80,
        temperature: float = 1.0
    ) -> mx.array:
        """
        Generate new molecules from conditions
        
        Args:
            batch_size: Number of molecules to generate
            conditions: Property conditions [batch_size, num_conditions]
            max_length: Maximum sequence length
            temperature: Sampling temperature
        
        Returns:
            molecules: Generated token sequences [batch_size, seq_length]
        """
        # Sample from prior N(0, I)
        z = mx.random.normal(shape=(batch_size, self.latent_dim))
        
        # Generate using temperature sampling decoder
        molecules = self.decoder_sampling.generate_with_temperature(
            z,
            conditions,
            max_length=max_length,
            temperature=temperature
        )
        
        return molecules
