import mlx.core as mx
import os

# Get PAD token from metadata (assumes running from project root)
def get_pad_token():
    try:
        import json
        with open('mlx_data/qm9_cns_selfies.json') as f:
            meta = json.load(f)
        return meta['token_to_idx']['<PAD>']
    except:
        return 0  # Default PAD token

PAD = get_pad_token()

def latent_diversity_loss(z):
    """
    Compute a differentiable diversity penalty on a batch of latent vectors.
    Encourages latent codes to be dissimilar (low average cosine similarity).

    Args:
        z: MLX array of shape (batch_size, latent_dim)

    Returns:
        diversity_loss: scalar (average off-diagonal cosine similarity)
    """
    batch_size = z.shape[0]
    
    # 1. Normalize each latent vector to unit norm
    z_norm = z / mx.sqrt(mx.sum(z * z, axis=1, keepdims=True))  # shape: (B, D)
    
    # 2. Compute cosine similarity matrix: (B, D) @ (D, B) -> (B, B)
    sim_matrix = mx.matmul(z_norm, z_norm.T)
    
    # 3. Create mask to exclude diagonal self-similarities
    mask = mx.ones_like(sim_matrix)
    mask = mx.where(mx.eye(batch_size), mx.zeros_like(mask), mask)
    
    # 4. Sum and average only off-diagonal entries
    # sum of similarities
    sim_sum = mx.sum(sim_matrix * mask)
    # number of off-diagonal pairs = B*(B-1)
    pair_count = batch_size * (batch_size - 1)
    
    # 5. Diversity loss = average pairwise similarity
    diversity_loss = sim_sum / pair_count
    return diversity_loss

def compute_loss(x, logits, mu, logvar, beta: float=1.0, diversity_weight: float=0.01):
    """
    Compute VAE loss: reconstruction + β * KL divergence + diversity penalty
    
    Args:
        x: Input sequences [B, T]
        logits: Model predictions [B, T-1, V]
        mu: Latent mean [B, L]
        logvar: Latent log-variance [B, L]
        beta: KL weight
        diversity_weight: Weight for diversity loss
    
    Returns:
        total_loss, recon_loss, kl_loss, diversity_loss
    """
    target = x[:, 1:]  # Shift for next token prediction
    
    # Reconstruction loss (cross-entropy)
    log_probs = mx.log(mx.softmax(logits, axis=-1))
    recon_loss = -mx.mean(mx.take_along_axis(log_probs, mx.expand_dims(target, axis=-1), axis=-1))
    
    # Apply padding mask
    mask = (target != PAD)
    recon_loss = recon_loss * mx.mean(mask.astype(mx.float32))
    
    # KL divergence loss
    kl_loss = -0.5 * mx.mean(1 + logvar - mx.square(mu) - mx.exp(logvar))
    
    # Clip KL loss to prevent numerical instability (reasonable range)
    kl_loss = mx.clip(kl_loss, -10.0, 10.0)
    
    # Diversity loss (encourage diverse latent representations)
    diversity_loss = latent_diversity_loss(mu)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss + diversity_weight * diversity_loss
    
    return total_loss, recon_loss, kl_loss, diversity_loss