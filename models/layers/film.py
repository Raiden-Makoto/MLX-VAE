import mlx.core as mx
import mlx.nn as nn

class FILM(nn.Module):
    """Feature-wise Linear Modulation for conditioning"""
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        # Generate scale (gamma) and shift (beta) from conditions
        self.gamma_linear = nn.Linear(condition_dim, feature_dim)
        self.beta_linear = nn.Linear(condition_dim, feature_dim)
    
    def __call__(self, features, conditions):
        """
        Args:
            features: [B, T, D] - features to modulate
            conditions: [B, condition_dim] - conditioning information
        
        Returns:
            modulated_features: [B, T, D] - features modulated by conditions
        """
        # Generate modulation parameters
        gamma = self.gamma_linear(conditions)  # [B, D]
        beta = self.beta_linear(conditions)     # [B, D]
        
        # Broadcast gamma and beta across sequence length
        # features is [B, T, D], so we expand to [B, 1, D] and broadcast
        gamma = mx.expand_dims(gamma, axis=1)  # [B, 1, D]
        beta = mx.expand_dims(beta, axis=1)     # [B, 1, D]
        
        # Apply modulation: gamma * features + beta
        modulated = gamma * features + beta
        
        return modulated

