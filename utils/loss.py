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

def compute_loss(x, logits, mu, logvar, beta: float=1.0):
    """
    Compute VAE loss: reconstruction + β * KL divergence
    
    Args:
        x: Input sequences [B, T]
        logits: Model predictions [B, T-1, V]
        mu: Latent mean [B, L]
        logvar: Latent log-variance [B, L]
        beta: KL weight
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    target = x[:, 1:]  # Shift for next token prediction
    
    # Reconstruction loss (cross-entropy)
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_logits - mx.log(mx.sum(mx.exp(logits - max_logits), axis=-1, keepdims=True))
    recon_loss = -mx.mean(mx.take_along_axis(log_probs, mx.expand_dims(target, axis=-1), axis=-1))
    
    # Apply padding mask
    mask = (target != PAD)
    recon_loss = recon_loss * mx.mean(mask.astype(mx.float32))
    
    # KL divergence loss
    kl_loss = -0.5 * mx.sum(1 + logvar - mx.square(mu) - mx.exp(logvar), axis=1)
    kl_loss = mx.mean(kl_loss)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss