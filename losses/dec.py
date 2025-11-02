import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .recon import reconstruction_loss

def decoder_loss(
    model,
    z: mx.array,
    conditions: mx.array,
    target_seq: mx.array,
    teacher_forcing_ratio: float = 0.9
) -> mx.array:
    """
    Loss for training the decoder (autoregressive generation)
    
    Measures how well the decoder generates token sequences given
    latent codes and conditions.
    
    Args:
        model: MLXAutoregressiveDecoder instance
        z: Latent vectors [batch_size, latent_dim]
        conditions: Property conditions [batch_size, num_conditions]
        target_seq: Target sequences [batch_size, seq_length]
        teacher_forcing_ratio: Probability of using ground truth tokens during training
    
    Returns:
        Decoder loss (scalar)
    """
    # Forward pass through decoder
    logits = model(z, conditions, target_seq=target_seq, teacher_forcing_ratio=teacher_forcing_ratio)
    
    # Reconstruction loss only (not using latent, so no KL)
    recon_loss = reconstruction_loss(logits, target_seq, reduction='mean')
    
    return recon_loss
