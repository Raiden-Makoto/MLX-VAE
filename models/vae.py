import mlx.core as mx
import mlx.nn as nn

from models.encoder import SelfiesEncoder
from models.decoder import SelfiesDecoder

class SelfiesVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim: int=128,
        hidden_dim: int=256,
        latent_dim: int=64,
    ):
        super().__init__()
        self.E = SelfiesEncoder(vocab_size, embedding_dim, hidden_dim, latent_dim)
        self.D = SelfiesDecoder(vocab_size, embedding_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        #mu: [B, L]
        #logvar: [B, L]
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mu.shape)
        return mu + std * eps

    def __call__(self, x):
        #x: [B, T]
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]  # Shift for next token prediction
        
        # Encode to latent space
        mu, logvar = self.E(input_seq)
        z = self.reparameterize(mu, logvar)
        
        # Decode from latent space
        logits = self.D(z, input_seq)
        
        return logits, mu, logvar, target_seq