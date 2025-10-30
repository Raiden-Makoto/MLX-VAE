import mlx.nn as nn


class TPSAToLatentPredictor(nn.Module):
    """Learn mapping: target_TPSA → z that produces molecules with that TPSA"""
    def __init__(self, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),          # Input: normalized TPSA
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # Output: z
        )

    def __call__(self, norm_tpsa):
        return self.network(norm_tpsa)


