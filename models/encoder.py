import mlx.core as mx
import mlx.nn as nn

class SelfiesEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim: int=128,
        hidden_dim: int=256,
        latent_dim: int=64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # Bidirectional LSTM using two separate LSTMs
        self.lstm_forward = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm_backward = nn.LSTM(embedding_dim, hidden_dim)
        self.mu = nn.Linear(2 * hidden_dim, latent_dim)
        self.logvar = nn.Linear(2 * hidden_dim, latent_dim)

    def __call__(self, x):
        #x: [B, T]
        x = self.embed(x) # [B, T, E]
        
        # Forward LSTM
        _, h_forward = self.lstm_forward(x) # Returns (output, hidden_state)
        h_forward = h_forward[0] # [B, H] - take the hidden state from the last layer
        
        # Backward LSTM (reverse the sequence)
        x_reversed = x[:, ::-1] # Reverse along sequence dimension using slicing
        _, h_backward = self.lstm_backward(x_reversed) # Returns (output, hidden_state)
        h_backward = h_backward[0] # [B, H] - take the hidden state from the last layer
        
        # Concatenate forward and backward hidden states
        h = mx.concatenate([h_forward, h_backward], axis=-1) # [B, 2*H]
        
        mu = self.mu(h) # [B, L]
        logvar = self.logvar(h) # [B, L]
        return mu, logvar