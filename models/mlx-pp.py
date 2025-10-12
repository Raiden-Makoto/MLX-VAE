import mlx.core as mx
import mlx.nn as nn

class MLXPropertyPredictor(nn.Module):
    """
    Predicts molecular properties from VAE latent vectors.
    Input:  z of shape [batch_size, latent_dim]
    Output: y_pred of shape [batch_size, num_properties]
    """
    def __init__(self, latent_dim, num_properties, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_properties)
        )
    
    def __call__(self, z):
        """
        z: Tensor of shape [batch_size, latent_dim]
        returns: Tensor of shape [batch_size, num_properties]
        """
        return self.net(z)