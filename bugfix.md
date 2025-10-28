<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## 🎯 FOUND THE ROOT CAUSE: Decoder Architecture Is Fundamentally Broken

Your 0% accuracy is because **the decoder's cross-attention is useless**.

Look at this line in your decoder:

```python
encoder_output = mx.tile(latent_embedding, (1, seq_len, 1))
```

This creates:

```
encoder_output = [latent_vec, latent_vec, latent_vec, ..., latent_vec]
                  (same latent repeated 50 times!)
```

**The problem:** Cross-attention always attends to the SAME information at every position. There's no positional structure!

***

## Why This Breaks Inference

### Training (with teacher forcing):

- Decoder sees full sequence: `[START, token2, token3, ..., tokenN]`
- Cross-attention repeats same latent (inefficient but works)
- Reconstruction loss drives learning
- ✓ Mostly works


### Inference (autoregressive):

- Iteration 1: `seq=[START]` → predict token2 from repeated latent (random)
- Iteration 2: `seq=[START, token2]` → predict token3 from repeated latent (random)
- Iteration 3: `seq=[START, token2, token3]` → predict token4 from repeated latent (random)
- **Result: 0% accuracy (all random)**

The decoder has no way to know which tokens are new vs old!

***

## The Fix: Replace the Decoder

Your current decoder uses flawed cross-attention. Replace it with one that uses **self-attention only**:

```python
import mlx.core as mx
import mlx.nn as nn
from .layers import PositionalEncoding, TransformerEncoderLayer, FILM

class SelfiesTransformerDecoder(nn.Module):
    """Transformer decoder using self-attention (not cross-attention)"""
    
    def __init__(self, vocab_size, embedding_dim: int=128, hidden_dim: int=256,
                 latent_dim: int=64, num_heads: int=4, num_layers: int=4, dropout: float=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Project latent to embedding dimension
        self.latent_projection = nn.Linear(latent_dim, embedding_dim)
        
        # Self-attention layers (not cross-attention!)
        self.decoder_layers = [
            TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ]
        
        # FILM layers for property conditioning
        self.film_layers = [
            FILM(embedding_dim, embedding_dim)
            for _ in range(num_layers)
        ]
        
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, z, input_seq, property_embedding=None):
        batch_size, seq_len = input_seq.shape
        
        # Create mask (1 for valid, 0 for padding)
        mask = (input_seq != 0).astype(mx.float32)
        
        # Embed tokens + add position encoding
        embedded = self.token_embedding(input_seq)  # [B, T, embedding_dim]
        embedded = self.positional_encoding(embedded)
        
        # ✅ Add latent as global context (broadcast to all positions)
        latent_embedding = self.latent_projection(z)  # [B, embedding_dim]
        latent_expanded = mx.expand_dims(latent_embedding, axis=1)  # [B, 1, embedding_dim]
        embedded = embedded + latent_expanded  # [B, T, embedding_dim]
        
        # Add property conditioning if provided
        if property_embedding is not None:
            property_expanded = mx.expand_dims(property_embedding, axis=1)
            embedded = embedded + property_expanded
        
        embedded = self.dropout(embedded)
        
        # Self-attention layers with FILM conditioning
        decoder_output = embedded
        for i, layer in enumerate(self.decoder_layers):
            # Self-attention with mask
            decoder_output = layer(decoder_output, mask)
            
            # Apply FILM conditioning
            if property_embedding is not None:
                decoder_output = self.film_layers[i](decoder_output, property_embedding)
        
        logits = self.output_projection(decoder_output)  # [B, T, vocab_size]
        return logits
```


***

## Key Changes

1. **Removed cross-attention layer**: No more repeated latent vectors
2. **Use self-attention only**: Learns interdependencies between tokens
3. **Add latent as context**: Broadcast to all positions once
4. **Keep FILM conditioning**: Still modulate with properties
5. **Add proper masking**: Prevents attending to padding

***

## Why This Works

- ✅ Self-attention learns proper sequence structure
- ✅ Latent provides global context
- ✅ FILM modulates based on properties
- ✅ Autoregressive inference works (self-attention on growing seq)
- ✅ No confusing cross-attention to repeated latents

***

## Next Steps

1. **Replace `transformer_decoder.py`** with the code above
2. **Retrain from scratch** (architecture changed, old weights won't work)
3. **Test inference** - should be WAY better

This is a fundamental architectural fix. It explains why nothing else worked.
<span style="display:none">[^1][^2]</span>

<div align="center">⁂</div>

[^1]: transformer_decoder.py

[^2]: transformer_encoder.py

