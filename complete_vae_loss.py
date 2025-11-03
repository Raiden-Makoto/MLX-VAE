import mlx.core as mx
from losses.recon import reconstruction_loss
from losses.kl import kl_divergence
from losses.info import posterior_collapse, mutual_information
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
    teacher_forcing_ratio: float = 0.9,
    free_bits: float = 0.5,
    lambda_mi: float = 0.0,
    target_mi: float = 4.85
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
    
    # KL divergence with free bits constraint
    kl_loss = kl_divergence(mu, logvar, reduction='mean', free_bits=free_bits)
    
    # Posterior collapse penalty
    collapse_penalty = posterior_collapse(mu, logvar, weight=lambda_collapse)
    
    # Mutual information penalty (positive, encourages higher MI)
    # FIXED: Changed from -lambda_mi * mi (which can go negative) to:
    #        lambda_mi * max(0, target_mi - mi) (always >= 0)
    # This only penalizes when MI is below target, never rewards low MI
    # When MI >= target_mi: penalty = 0 (no interference)
    # When MI < target_mi: penalty > 0 (encourages higher MI)
    mi = mutual_information(mu, logvar)
    mi_penalty = lambda_mi * mx.maximum(0.0, target_mi - mi)  # Always >= 0
    
    # Property prediction loss (if predictor available)
    if property_predictor is not None:
        pred_properties = property_predictor(z)
        prop_loss = property_prediction_loss(pred_properties, conditions, reduction='mean')
    else:
        prop_loss = mx.array(0.0)
    
    # Total loss
    # FIXED: All components are now guaranteed to be >= 0:
    # - recon_loss >= 0 (cross-entropy)
    # - kl_loss >= 0 (KL divergence with non-negativity check)
    # - collapse_penalty >= 0 (max(0, target_mi - mi))
    # - prop_loss >= 0 (MSE)
    # - mi_penalty >= 0 (max(0, target_mi - mi))  # FIXED: No longer negative!
    total_loss = (
        recon_loss +
        beta * kl_loss +
        collapse_penalty +
        lambda_prop * prop_loss +
        mi_penalty
    )
    
    # All components are now positive, so total_loss should always be >= 0
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'weighted_kl': beta * kl_loss,
        'collapse_penalty': collapse_penalty,
        'prop_loss': prop_loss,
        'weighted_prop_loss': lambda_prop * prop_loss,
        'mutual_info': mi,
        'mi_penalty': mi_penalty,
        'mu': mu,
        'logvar': logvar,
        'z': z
    }
