import mlx.core as mx
import mlx.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len, dtype=mx.float32).reshape(-1, 1)
        
        div_term = mx.exp(mx.arange(0, d_model, 2, dtype=mx.float32) * 
                         (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        
        self.pe = pe.reshape(1, max_len, d_model)
    
    def __call__(self, x):
        # x: [B, T, d_model]
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
