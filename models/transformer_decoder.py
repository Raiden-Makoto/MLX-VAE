import mlx.core as mx
import mlx.nn as nn

from .layers import PositionalEncoding, TransformerDecoderLayer, FILM

class SelfiesTransformerDecoder(nn.Module):
    """Transformer decoder for SELFIES generation"""
    def __init__(
        self,
        vocab_size,
        embedding_dim: int=128,
        hidden_dim: int=256,
        latent_dim: int=64,
        num_heads: int=8,
        num_layers: int=6,
        dropout: float=0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Latent to hidden projection
        self.latent_projection = nn.Linear(latent_dim, embedding_dim)
        
        # FILM layers for property conditioning (best practice)
        # Condition on properties by modulating feature statistics
        self.film_layers = [
            FILM(embedding_dim, embedding_dim)  # condition_dim = embedding_dim
            for _ in range(num_layers)
        ]
        
        # Transformer decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ]
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, z, input_seq, property_embedding=None):
        # z: [B, latent_dim] - latent codes
        # input_seq: [B, T] - input sequences (for teacher forcing)
        # property_embedding: [B, embedding_dim] - optional property conditioning
        batch_size, seq_len = input_seq.shape
        
        # Project latent code to embedding dimension
        latent_embedding = self.latent_projection(z)  # [B, embedding_dim]
        latent_embedding = mx.expand_dims(latent_embedding, axis=1)  # [B, 1, embedding_dim]
        
        # Token embedding + positional encoding
        embedded = self.token_embedding(input_seq)  # [B, T, embedding_dim]
        embedded = self.positional_encoding(embedded)
        
        # Add latent information to embeddings
        embedded = embedded + mx.tile(latent_embedding, (1, seq_len, 1))
        embedded = self.dropout(embedded)
        
        # Use latent embedding as "encoder output" for cross-attention
        encoder_output = mx.tile(latent_embedding, (1, seq_len, 1))  # [B, T, embedding_dim]
        
        # Pass through transformer decoder layers with FILM conditioning
        decoder_output = embedded
        for i, layer in enumerate(self.decoder_layers):
            decoder_output = layer(decoder_output, encoder_output)
            
            # Apply FILM conditioning if properties provided
            if property_embedding is not None:
                # Condition the decoder output with FILM (Feature-wise Linear Modulation)
                decoder_output = self.film_layers[i](decoder_output, property_embedding)  # [B, T, embedding_dim]
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits