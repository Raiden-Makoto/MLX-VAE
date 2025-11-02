import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .kl import kl_divergence

def encoder_loss(
    model,
    x: mx.array,
    conditions: mx.array,
    beta: float = 0.4
) -> tuple:
    """
    Loss for training the encoder only (when decoder is frozen or pre-trained)
    
    Used in:
    - Stage 1 training: Encoder warm-up
    - Property prediction: Encode and predict properties from z
    - Representation learning: Disentangle latent factors
    
    Args:
        model: MLXEncoder instance
        x: Input molecules [batch_size, seq_length]
        conditions: Property conditions [batch_size, num_conditions]
        beta: KL weight for latent regularization
    
    Returns:
        (kl_loss, mu, logvar, z)
    """
    # Forward pass
    mu, logvar = model(x, conditions)
    
    # Sample from latent space
    z = model.reparameterize(mu, logvar)
    
    # KL divergence (main regularization term for encoder)
    kl_loss = kl_divergence(mu, logvar, reduction='mean')
    
    # Optional: Add reconstruction loss if validation set is available
    # This helps prevent posterior collapse
    weighted_kl = beta * kl_loss
    
    return weighted_kl, mu, logvar, z
