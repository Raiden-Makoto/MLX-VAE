import mlx.core as mx
import mlx.nn as nn
import numpy as np

def kl_divergence(
    mu: mx.array,
    logvar: mx.array,
    reduction: str = 'mean',
    free_bits: float = 0.0
) -> mx.array:
    """
    KL divergence between learned posterior q(z|x,c) and prior p(z) = N(0, I)
    
    Regularizes the latent space to prevent overfitting and enable sampling
    from the prior during generation.
    
    Args:
        mu: Mean of posterior distribution [batch_size, latent_dim]
        logvar: Log variance of posterior distribution [batch_size, latent_dim]
        reduction: 'mean' or 'sum'
        free_bits: Minimum KL per dimension (free bits constraint to prevent collapse)
    
    Returns:
        KL divergence loss (scalar)
    
    Mathematical formulation:
        KL(q(z|x,c) || p(z)) = -0.5 * sum_j [1 + log(sigma_j^2) - mu_j^2 - sigma_j^2]
        
    where sigma_j^2 = exp(logvar_j)
    
    Simplified derivation:
        - For Gaussian q and standard normal p
        - Term 1: -0.5 * sum(log_var - mu^2 - exp(log_var) + 1)
    """
    batch_size, latent_dim = mu.shape
    
    # Compute variance from log-variance
    var = mx.exp(logvar)
    
    # KL divergence for each dimension
    # KL = -0.5 * (1 + log_var - mu^2 - var)
    kl_per_dim = -0.5 * (1.0 + logvar - mx.square(mu) - var)
    
    # Apply free bits constraint: ensure minimum KL per dimension
    if free_bits > 0.0:
        min_kl_per_dim = free_bits / latent_dim
        kl_per_dim = mx.maximum(kl_per_dim, min_kl_per_dim)
    
    # Sum across latent dimensions
    kl_per_sample = mx.sum(kl_per_dim, axis=1)  # [batch_size]
    
    if reduction == 'mean':
        return mx.mean(kl_per_sample)
    elif reduction == 'sum':
        return mx.sum(kl_per_sample)
    else:
        return kl_per_sample
