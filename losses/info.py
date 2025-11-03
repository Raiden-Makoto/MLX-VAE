import mlx.core as mx

def mutual_information(mu: mx.array, logvar: mx.array) -> mx.array:
    """
    Compute mutual information between q(z) and p(z)
    
    High MI (~4-5) indicates the model is using latent space effectively.
    Low MI (<1) indicates posterior collapse (model ignores z).
    
    Args:
        mu: Latent means [batch_size, latent_dim]
        logvar: Latent log-variances [batch_size, latent_dim]
    
    Returns:
        Mutual information (scalar)
    
    Formulation:
        MI = E[KL(q(z|x) || p(z))] - KL(E[q(z|x)] || p(z))
        
    The first term is average KL per sample.
    The second term is KL of the aggregated posterior.
    """
    batch_size = mu.shape[0]
    
    # Note: mu should be bounded to [-2, 2] via tanh, logvar to [-5, 2] via tanh by encoder
    # Emergency clipping as defensive safeguard
    mu = mx.clip(mu, -3.0, 3.0)
    logvar = mx.clip(logvar, -6.0, 3.0)
    
    # Per-sample KL
    var = mx.exp(logvar)
    kl_per_sample = -0.5 * mx.sum(1.0 + logvar - mx.square(mu) - var, axis=1)
    mean_kl = mx.mean(kl_per_sample)
    
    # Aggregated posterior KL
    mean_mu = mx.mean(mu, axis=0)
    mean_var = mx.mean(var, axis=0)
    mean_logvar = mx.log(mean_var)
    
    # KL of aggregated
    agg_kl = -0.5 * mx.sum(1.0 + mean_logvar - mx.square(mean_mu) - mean_var)
    
    # Mutual information (batch estimate)
    # Correct: I ≈ E_x[KL(q(z|x)||p(z))] - KL(q(z)||p(z))
    # Do NOT divide the aggregated KL by batch size
    mi = mean_kl - agg_kl
    # Ensure non-negative due to estimation noise
    mi = mx.maximum(mi, 0.0)
    
    return mi


def posterior_collapse(
    mu: mx.array,
    logvar: mx.array,
    target_mi: float = 4.85,
    weight: float = 0.1
) -> mx.array:
    """
    Penalty term to prevent posterior collapse
    
    Add to loss function to maintain KL divergence if it gets too small.
    
    Args:
        mu: Latent means [batch_size, latent_dim]
        logvar: Latent log-variances [batch_size, latent_dim]
        target_mi: Target mutual information value
        weight: Penalty weight
    
    Returns:
        Penalty loss (scalar)
    """
    mi = mutual_information(mu, logvar)
    
    # Penalty if MI is below target
    penalty = weight * mx.maximum(0.0, target_mi - mi)
    
    return penalty
