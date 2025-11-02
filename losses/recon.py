import mlx.core as mx
import mlx.nn as nn
import numpy as np


def reconstruction_loss(
    logits: mx.array,
    targets: mx.array,
    reduction: str = 'mean'
) -> mx.array:
    """
    Cross-entropy reconstruction loss for token prediction
    
    Measures how well the decoder reconstructs the input sequence.
    Each token position is treated independently.
    
    Args:
        logits: Predicted logits [batch_size, seq_length, vocab_size]
        targets: Target token indices [batch_size, seq_length]
        reduction: 'mean' or 'sum'
    
    Returns:
        Reconstruction loss (scalar)
    
    Mathematical formulation:
        L_recon = -1/N * sum_i log P(y_i | logits_i)
                = -1/N * sum_i log softmax(logits_i)[y_i]
    """
    batch_size, seq_length, vocab_size = logits.shape
    
    # Reshape for batch processing
    logits_flat = mx.reshape(logits, (-1, vocab_size))  # [batch*seq, vocab]
    targets_flat = mx.reshape(targets, (-1,))  # [batch*seq]
    
    # Compute log-softmax: log(exp(logits) / sum(exp(logits)))
    logits_max = mx.max(logits_flat, axis=1, keepdims=True)
    logits_stable = logits_flat - logits_max  # Numerical stability
    
    exp_logits = mx.exp(logits_stable)
    sum_exp = mx.sum(exp_logits, axis=1, keepdims=True)
    log_softmax = logits_stable - mx.log(sum_exp)
    
    # Get log probability of target class for each position
    # One-hot encode targets
    one_hot_targets = mx.zeros_like(log_softmax)
    # Note: In production MLX, use proper one-hot or gather operations
    # This is a workaround for demonstration
    target_log_probs = mx.take_along_axis(
        log_softmax,
        mx.reshape(targets_flat, (-1, 1)),
        axis=1
    )  # [batch*seq, 1]
    
    target_log_probs = mx.reshape(target_log_probs, (-1,))  # [batch*seq]
    
    # Cross-entropy: -log(P(target))
    ce_loss = -target_log_probs  # [batch*seq]
    
    if reduction == 'mean':
        return mx.mean(ce_loss)
    elif reduction == 'sum':
        return mx.sum(ce_loss)
    else:
        return ce_loss
