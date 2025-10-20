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
    log_probs = mx.log(mx.softmax(logits, axis=-1))
    recon_loss = -mx.mean(mx.take_along_axis(log_probs, mx.expand_dims(target, axis=-1), axis=-1))
    
    # Apply padding mask
    mask = (target != PAD)
    recon_loss = recon_loss * mx.mean(mask.astype(mx.float32))
    
    # KL divergence loss
    kl_loss = -0.5 * mx.mean(1 + logvar - mx.square(mu) - mx.exp(logvar))
    
    # Clip KL loss to prevent numerical instability (reasonable range)
    kl_loss = mx.clip(kl_loss, -10.0, 10.0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss