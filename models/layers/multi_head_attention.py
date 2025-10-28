import mlx.core as mx
import mlx.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations and split into heads
        Q = self.w_q(query).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = self.w_k(key).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = self.w_v(value).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.d_k, dtype=mx.float32))
        
        if mask is not None:
            # Handle different mask shapes
            if len(mask.shape) == 2:  # [B, T] - padding mask
                mask = mx.expand_dims(mx.expand_dims(mask, axis=1), axis=1)  # [B, 1, 1, T]
            elif len(mask.shape) == 3:  # [B, T, T] - causal mask or combined mask
                mask = mx.expand_dims(mask, axis=1)  # [B, 1, T, T]
            # mask shape is now [B, 1, T, T] or [B, 1, 1, T]
            scores = mx.where(mask == 0, -1e9, scores)
        
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = mx.matmul(attn_weights, V)
        
        # Concatenate heads and project
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output
