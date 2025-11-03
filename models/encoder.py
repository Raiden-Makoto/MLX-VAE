import mlx.core as mx
import mlx.nn as nn
from typing import Tuple

class MLXEncoder(nn.Module):
    """
    Conditional Encoder for AR-CVAE molecule generation using MLX
    
    Architecture:
    - Token Embedding: vocab_size -> embedding_dim
    - Bi-directional processing via sequential LSTM
    - Condition projection: num_conditions -> hidden_dim
    - Latent distribution parameters: combined -> latent_dim
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 200,
        num_conditions: int = 6,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            vocab_size: Size of the tokenized vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension of LSTM cells (512 recommended)
            latent_dim: Dimension of latent space (200 recommended)
            num_conditions: Number of property conditions (1: TPSA)
            num_layers: Number of LSTM layers (3 recommended)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.num_layers = num_layers
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers for sequence encoding
        # MLX only supports single-layer LSTM, so we stack them manually
        for i in range(num_layers):
            setattr(
                self, 
                f'lstm_layer_{i}', 
                nn.LSTM(
                    input_size=embedding_dim if i == 0 else hidden_dim,
                    hidden_size=hidden_dim
                )
            )
        
        # Project conditions to hidden dimension
        self.condition_fc = nn.Linear(num_conditions, hidden_dim)
        
        # Project combined representation to latent parameters
        # Combined dimension: hidden_dim (from LSTM) + hidden_dim (from conditions)
        combined_dim = hidden_dim + hidden_dim
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        # Two-layer projection for logvar for stability
        self.fc_logvar_hidden = nn.Linear(combined_dim, combined_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)
        # Initialize logvar bias to start with modest variance (logvar ≈ -2.5)
        if hasattr(self.fc_logvar, 'bias'):
            # Initialize to target logvar around -2.0 (variance ~ 0.135)
            # Solve tanh(b)*6 - 4 ≈ -2  => tanh(b) ≈ 1/3 => b ≈ 0.3466
            self.fc_logvar.bias = mx.ones_like(self.fc_logvar.bias) * 0.35
        
    def __call__(
        self,
        x: mx.array,
        conditions: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass of the encoder
        
        Args:
            x: Tokenized molecule sequences [batch_size, seq_length]
            conditions: Normalized molecular properties [batch_size, num_conditions]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Embed tokens: [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(x)
        
        # Encode through LSTM layers sequentially
        # MLX LSTM returns: (output, hidden)
        # With input [batch, seq_len, features], output is [batch, seq_len, hidden_dim]
        output = embedded
        for i in range(self.num_layers):
            lstm_layer = getattr(self, f'lstm_layer_{i}')
            output, hidden = lstm_layer(output)
        
        # Extract final hidden state from the last timestep
        # output shape: [batch, seq_len, hidden_dim]
        # We want [batch, hidden_dim] by taking last timestep along dim 1
        final_hidden = output[:, -1, :]  # [batch, hidden_dim]
        
        # Project conditions to hidden dimension
        condition_repr = self.condition_fc(conditions)  # [batch_size, hidden_dim]
        
        # Concatenate sequence representation with condition representation
        combined = mx.concatenate([final_hidden, condition_repr], axis=1)  # [batch_size, 2*hidden_dim]
        
        # Compute latent parameters (raw, unbounded)
        mu_raw = self.fc_mu(combined)  # [batch_size, latent_dim]
        # Two-layer logvar projection for smoother mapping
        logvar_hidden = mx.tanh(self.fc_logvar_hidden(combined))
        logvar_raw = self.fc_logvar(logvar_hidden)  # [batch_size, latent_dim]
        
        # CRITICAL: Bound outputs using tanh for smooth, bounded gradients
        # tanh provides natural bounds and maintains gradients everywhere
        # 
        # μ: Use tanh with scaling to get [-2, 2] range
        #    This gives μ² ∈ [0, 4] per dimension (safe)
        #    For 128 dims: max sum(μ²) = 512 (very reasonable)
        mu = mx.tanh(mu_raw / 2.0) * 2.0  # Range: [-2, 2]
        
        # logvar: Bound tightly to [-2, 0] to keep variance in [0.135, 1.0]
        # This prevents tiny variances that explode KL
        logvar = mx.tanh(logvar_raw / 2.0) * 1.0 - 1.0  # Range: [-2, 0]
        
        return mu, logvar
    
    @staticmethod
    def reparameterize(mu: mx.array, logvar: mx.array) -> mx.array:
        """
        Reparameterization trick for sampling from latent distribution
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        # Compute standard deviation from log variance
        std = mx.exp(0.5 * logvar)
        
        # Sample epsilon from standard normal distribution
        eps = mx.random.normal(mu.shape)
        
        # Reparameterization: z = mu + std * epsilon
        z = mu + eps * std
        
        return z