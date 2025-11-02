import mlx.core as mx
from losses.recon import reconstruction_loss
from losses.kl import kl_divergence
from losses.info import posterior_collapse
from losses.prop import property_prediction_loss

def complete_vae_loss(
    encoder,
    decoder,
    property_predictor,
    x: mx.array,
    conditions: mx.array,
    beta: float = 0.4,
    lambda_prop: float = 0.1,
    lambda_collapse: float = 0.01,
    teacher_forcing_ratio: float = 0.9
) -> dict:
    """
    Complete multi-component loss for full AR-CVAE training
    
    Args:
        encoder: MLXEncoder
        decoder: MLXAutoregressiveDecoder
        property_predictor: Optional property prediction network
        x: Input molecules [batch_size, seq_length]
        conditions: Target properties [batch_size, num_conditions]
        beta: KL weight
        lambda_prop: Property prediction weight
        lambda_collapse: Posterior collapse penalty weight
    
    Returns:
        Dictionary with all loss components
    """
    # Encoding
    mu, logvar = encoder(x, conditions)
    z = encoder.reparameterize(mu, logvar)
    
    # Decoding with teacher forcing
    logits = decoder(z, conditions, target_seq=x, teacher_forcing_ratio=teacher_forcing_ratio)
    
    # Reconstruction loss
    recon_loss = reconstruction_loss(logits, x, reduction='mean')
    
    # KL divergence
    kl_loss = kl_divergence(mu, logvar, reduction='mean')
    
    # Posterior collapse penalty
    collapse_penalty = posterior_collapse(mu, logvar, weight=lambda_collapse)
    
    # Property prediction loss (if predictor available)
    if property_predictor is not None:
        pred_properties = property_predictor(z)
        prop_loss = property_prediction_loss(pred_properties, conditions, reduction='mean')
    else:
        prop_loss = mx.array(0.0)
    
    # Total loss
    total_loss = (
        recon_loss +
        beta * kl_loss +
        collapse_penalty +
        lambda_prop * prop_loss
    )
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'weighted_kl': beta * kl_loss,
        'collapse_penalty': collapse_penalty,
        'prop_loss': prop_loss,
        'weighted_prop_loss': lambda_prop * prop_loss,
        'mu': mu,
        'logvar': logvar,
        'z': z
    }
