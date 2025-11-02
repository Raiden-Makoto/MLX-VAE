import mlx.core as mx
import mlx.nn as nn
import numpy as np

def property_prediction_loss(
    z: mx.array,
    predicted_properties: mx.array,
    target_properties: mx.array,
    property_scales: mx.array = None,
    reduction: str = 'mean'
) -> mx.array:
    """
    Loss for predicting molecular properties from latent representation
    
    Used to ensure latent space is disentangled and contains
    property information.
    
    Args:
        z: Latent vectors [batch_size, latent_dim]
        predicted_properties: Model predictions [batch_size, num_properties]
        target_properties: Ground truth properties [batch_size, num_properties]
        property_scales: Optional scaling factors per property
        reduction: 'mean' or 'sum'
    
    Returns:
        Property prediction loss (scalar)
    """
    # Mean squared error for continuous properties
    mse = mx.square(predicted_properties - target_properties)
    
    if property_scales is not None:
        # Weight by property scale (e.g., normalize by std dev)
        mse = mse / (mx.square(property_scales) + 1e-8)
    
    if reduction == 'mean':
        return mx.mean(mse)
    elif reduction == 'sum':
        return mx.sum(mse)
    else:
        return mse
