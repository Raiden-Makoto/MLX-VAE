import mlx.core as mx
import mlx.nn as nn

from .layers import PositionalEncoding, TransformerEncoderLayer

class SelfiesTransformerEncoder(nn.Module):
    """Transformer encoder for SELFIES sequences"""
    def __init__(
        self,
        vocab_size,
        embedding_dim: int=128,
        hidden_dim: int=256,
        latent_dim: int=256,
        num_heads: int=4,
        num_layers: int=4,
        dropout: float=0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ]
        
        # Output projection to latent space
        self.mu_projection = nn.Linear(embedding_dim, latent_dim)
        self.logvar_projection = nn.Linear(embedding_dim, latent_dim)
        
        # Masked pooling for sequence-level representation
        self.dropout = nn.Dropout(dropout)
    
    def masked_pool(self, x, mask):
        # x: [B, T, D], mask: [B, T]
        mask_expanded = mx.expand_dims(mask, axis=-1)  # [B, T, 1]
        masked_x = x * mask_expanded  # zero out padded positions
        denom = mx.maximum(mx.sum(mask_expanded, axis=1), 1e-6)
        return mx.sum(masked_x, axis=1) / denom
        
    def __call__(self, x, property_embedding=None):
        # x: [B, T] - input token sequences
        # property_embedding: [B, embedding_dim] - optional property conditioning
        batch_size, seq_len = x.shape
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        # Assuming 0 is padding token
        mask = (x != 0).astype(mx.float32)
        
        # Token embedding + positional encoding
        embedded = self.token_embedding(x)  # [B, T, embedding_dim]
        embedded = self.positional_encoding(embedded)
        
        # Add property embedding as a global bias if provided
        if property_embedding is not None:
            prop = mx.expand_dims(property_embedding, axis=1)  # [B, 1, D]
            embedded = embedded + prop
        
        embedded = self.dropout(embedded)
        
        # Pass through transformer encoder layers
        encoder_output = embedded
        for i, layer in enumerate(self.encoder_layers):
            encoder_output = layer(encoder_output, mask)
        
        # Pool sequence to get sequence-level representation
        pooled = self.masked_pool(encoder_output, mask)
        
        # Project to latent space
        mu = self.mu_projection(pooled)
        logvar = self.logvar_projection(pooled)
        
        return mu, logvar