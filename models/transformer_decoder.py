import mlx.core as mx
import mlx.nn as nn

from .layers import PositionalEncoding, TransformerEncoderLayer, FILM

class SelfiesTransformerDecoder(nn.Module):
    """Transformer decoder using self-attention (not cross-attention)"""
    
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
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Project latent to embedding dimension (z alone, not z+properties)
        self.latent_projection = nn.Linear(latent_dim, embedding_dim)
        
        # Self-attention layers (not cross-attention!)
        self.decoder_layers = [
            TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ]
        
        # FILM layers for property conditioning (as per research doc)
        # Combined embeddings: logp_emb + tpsa_emb = 2*embedding_dim
        self.film_layers = [
            FILM(embedding_dim, 2 * embedding_dim)
            for _ in range(num_layers)
        ]
        
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, z, input_seq, property_embedding=None):
        batch_size, seq_len = input_seq.shape
        
        # Create padding mask (1 for valid, 0 for padding)
        padding_mask = (input_seq != 0).astype(mx.float32)
        
        # Create causal mask (1 for valid positions, 0 for future positions)
        # Lower triangular matrix: attend to current and past positions only
        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.float32))
        
        # Combine padding and causal masking
        # Expand padding_mask to [B, 1, T] and causal_mask to [1, T, T]
        # Final mask shape: [B, T, T] where 1 = valid position
        combined_mask = mx.expand_dims(padding_mask, axis=2) * causal_mask
        
        # Embed tokens + add position encoding
        embedded = self.token_embedding(input_seq)  # [B, T, embedding_dim]
        embedded = self.positional_encoding(embedded)
        
        # Add latent as global context (broadcast to all positions)
        latent_embedding = self.latent_projection(z)  # [B, embedding_dim]
        latent_expanded = mx.expand_dims(latent_embedding, axis=1)  # [B, 1, embedding_dim]
        embedded = embedded + latent_expanded  # [B, T, embedding_dim]
        
        # Property conditioning is applied via FILM layers only
        # Skip broadcasting to prevent shape mismatches
        
        embedded = self.dropout(embedded)
        
        # Self-attention layers with FILM conditioning
        decoder_output = embedded
        for i, layer in enumerate(self.decoder_layers):
            # Self-attention with causal+padding mask
            decoder_output = layer(decoder_output, combined_mask)
            
            # Apply FILM conditioning if properties provided
            if property_embedding is not None:
                decoder_output = self.film_layers[i](decoder_output, property_embedding)
        
        logits = self.output_projection(decoder_output)  # [B, T, vocab_size]
        return logits