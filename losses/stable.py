"""
Stable loss functions with built-in sanity checks and bounds checking
"""
import mlx.core as mx
import numpy as np
from typing import Tuple, Optional


def check_loss_sanity(
    total_loss: mx.array,
    recon_loss: mx.array,
    kl_loss: mx.array,
    mu: mx.array,
    logvar: mx.array,
    max_loss: float = 1000.0,
    max_kl_per_dim: float = 10.0
) -> bool:
    """
    Check if loss values are within reasonable bounds
    
    Args:
        total_loss: Total loss value
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
        mu: Latent means
        logvar: Latent log-variances
        max_loss: Maximum acceptable total loss
        max_kl_per_dim: Maximum acceptable KL per dimension
    
    Returns:
        True if loss is sane, False otherwise
    """
    # Evaluate all at once
    mx.eval(total_loss, recon_loss, kl_loss, mu, logvar)
    
    total_val = float(total_loss)
    recon_val = float(recon_loss)
    kl_val = float(kl_loss)
    
    # Check for NaN or Inf
    if not (np.isfinite(total_val) and np.isfinite(recon_val) and np.isfinite(kl_val)):
        return False
    
    # Check loss magnitudes
    if total_val > max_loss or total_val < -100:
        return False
    
    if recon_val < 0 or recon_val > max_loss * 0.9:
        return False
    
    if kl_val < 0 or kl_val > max_kl_per_dim * mu.shape[1]:
        return False
    
    # Check latent parameter bounds (defensive)
    mu_val = float(mx.max(mx.abs(mu)))
    logvar_val = float(mx.max(logvar))
    logvar_min = float(mx.min(logvar))
    
    if mu_val > 5.0:  # Should be bounded to [-2, 2] by tanh
        return False
    
    if logvar_val > 3.0 or logvar_min < -6.0:  # Should be bounded to [-5, 2] by tanh
        return False
    
    return True


def kl_divergence_stable(
    mu: mx.array,
    logvar: mx.array,
    reduction: str = 'mean',
    free_bits: float = 0.0,
    mu_clip: float = 3.0,
    logvar_clip_min: float = -6.0,
    logvar_clip_max: float = 3.0
) -> mx.array:
    """
    Numerically stable KL divergence with emergency clipping
    
    Args:
        mu: Mean of posterior distribution [batch_size, latent_dim]
        logvar: Log variance of posterior distribution [batch_size, latent_dim]
        reduction: 'mean' or 'sum'
        free_bits: Minimum KL per dimension
        mu_clip: Emergency clip for mu
        logvar_clip_min: Emergency clip minimum for logvar
        logvar_clip_max: Emergency clip maximum for logvar
    
    Returns:
        KL divergence loss (scalar)
    """
    batch_size, latent_dim = mu.shape
    
    # Emergency clipping (should rarely trigger with tanh bounds)
    mu = mx.clip(mu, -mu_clip, mu_clip)
    logvar = mx.clip(logvar, logvar_clip_min, logvar_clip_max)
    
    # Compute variance
    var = mx.exp(logvar)
    
    # KL divergence: -0.5 * (1 + logvar - mu^2 - var)
    kl_per_dim = -0.5 * (1.0 + logvar - mx.square(mu) - var)
    
    # Ensure non-negative
    kl_per_dim = mx.maximum(kl_per_dim, 0.0)
    
    # Apply free bits
    if free_bits > 0.0:
        min_kl_per_dim = free_bits / latent_dim
        kl_per_dim = mx.maximum(kl_per_dim, min_kl_per_dim)
    
    # Sum across dimensions
    kl_per_sample = mx.sum(kl_per_dim, axis=1)
    
    if reduction == 'mean':
        return mx.mean(kl_per_sample)
    elif reduction == 'sum':
        return mx.sum(kl_per_sample)
    else:
        return kl_per_sample

