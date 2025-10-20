import mlx.core as mx
import mlx.nn as nn
from models.custom_lstm import CustomLSTM

class SelfiesDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim: int=128,
        hidden_dim: int=256,
        latent_dim: int=64,
    ):
        super().__init__()
        self.latent_h = nn.Linear(latent_dim, hidden_dim)
        self.latent_c = nn.Linear(latent_dim, hidden_dim)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = CustomLSTM(embedding_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, z, seq):
        #z: [B, L] - latent code
        #seq: [B, T] - input sequence
        init_h = mx.tanh(self.latent_h(z))  # [B, H]
        init_c = mx.tanh(self.latent_c(z))  # [B, H]
        x = self.embed(seq)  # [B, T, E]
        
        # Initialize LSTM with latent-derived hidden states
        initial_state = (init_h, init_c)
        outputs, _ = self.lstm(x, initial_state)
        
        # Apply layer normalization before final projection
        outputs = self.layer_norm(outputs)
        logits = self.fc(outputs)
        return logits