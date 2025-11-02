import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .recon import reconstruction_loss
from .kl import kl_divergence

def vae_loss(
    logits: mx.array,
    targets: mx.array,
    mu: mx.array,
    logvar: mx.array,
    beta: float = 0.4,
    reduction: str = 'mean'
) -> tuple:
    """
    Complete VAE loss: Reconstruction + β * KL Divergence
    
    The beta parameter controls the trade-off between reconstruction
    and regularization. Typically annealed during training.
    
    Args:
        logits: Decoder output logits [batch_size, seq_length, vocab_size]
        targets: Target token sequences [batch_size, seq_length]
        mu: Latent mean [batch_size, latent_dim]
        logvar: Latent log-variance [batch_size, latent_dim]
        beta: Weight for KL divergence (0.0 - 1.0)
              - 0.0: Only reconstruction (standard autoencoder)
              - 0.1-0.4: Typical for molecules
              - 1.0: Strong regularization
        reduction: 'mean' or 'sum'
    
    Returns:
        (total_loss, recon_loss, kl_loss, weighted_kl_loss)
    
    Mathematical formulation:
        L_total = L_recon + β * KL(q || p)
        
    Beta annealing schedule (typical):
        Epoch 0-100: β = 0 → 0.4 (linearly)
        Epoch 100+: β = 0.4 (constant)
    """
    # Compute reconstruction loss
    recon_loss = reconstruction_loss(logits, targets, reduction=reduction)
    
    # Compute KL divergence
    kl_loss = kl_divergence(mu, logvar, reduction=reduction)
    
    # Weighted KL for annealing
    weighted_kl_loss = beta * kl_loss
    
    # Total loss
    total_loss = recon_loss + weighted_kl_loss
    
    return total_loss, recon_loss, kl_loss, weighted_kl_loss


def vae_loss_per_sample(
    logits: mx.array,
    targets: mx.array,
    mu: mx.array,
    logvar: mx.array,
    beta: float = 0.4
) -> mx.array:
    """
    Compute per-sample VAE loss (useful for weighted sampling, importance weighting)
    
    Args:
        logits: [batch_size, seq_length, vocab_size]
        targets: [batch_size, seq_length]
        mu: [batch_size, latent_dim]
        logvar: [batch_size, latent_dim]
        beta: KL weight
    
    Returns:
        Per-sample loss [batch_size]
    """
    batch_size, seq_length, vocab_size = logits.shape
    
    # Reconstruction loss per sample
    logits_flat = mx.reshape(logits, (-1, vocab_size))
    targets_flat = mx.reshape(targets, (-1))
    
    # Compute cross-entropy
    logits_max = mx.max(logits_flat, axis=1, keepdims=True)
    logits_stable = logits_flat - logits_max
    exp_logits = mx.exp(logits_stable)
    sum_exp = mx.sum(exp_logits, axis=1, keepdims=True)
    log_softmax = logits_stable - mx.log(sum_exp)
    
    target_log_probs = mx.take_along_axis(
        log_softmax,
        mx.reshape(targets_flat, (-1, 1)),
        axis=1
    )
    target_log_probs = mx.reshape(target_log_probs, (-1,))
    
    # Sum across sequence positions
    recon_loss_per_sample = -mx.reshape(target_log_probs, (batch_size, seq_length))
    recon_loss_per_sample = mx.sum(recon_loss_per_sample, axis=1)  # [batch_size]
    
    # KL loss per sample
    var = mx.exp(logvar)
    kl_per_dim = -0.5 * (1.0 + logvar - mx.square(mu) - var)
    kl_loss_per_sample = mx.sum(kl_per_dim, axis=1)  # [batch_size]
    
    # Total per sample
    total_loss_per_sample = recon_loss_per_sample + beta * kl_loss_per_sample
    
    return total_loss_per_sample
