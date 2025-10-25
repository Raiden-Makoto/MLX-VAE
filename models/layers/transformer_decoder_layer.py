import mlx.core as mx
import mlx.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive generation"""
        mask = mx.tril(mx.ones((seq_len, seq_len)))
        return mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0)  # [1, 1, T, T]
        
    def __call__(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention with causal mask
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(x.shape[1])
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
