<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# well we increased from 0 to 1.1%

Great progress! **0% → 1.1% is massive.** But it's still too low. Here's the diagnostic path:

***

## Immediate Checks (Before More Retraining)

### 1. **Verify Training is Converging**

Print loss at each epoch:

```python
print(f"Epoch {epoch}: Recon={recon_loss:.3f}, KL={kl_loss:.3f}, Property={prop_loss:.3f}")
```

**Expected:** Recon loss should drop from ~5.0 → ~1.5 over 30 epochs

If it's not decreasing, training is broken → fix before continuing.

### 2. **Test Greedy Decoding (5 minutes)**

Replace sampling with argmax:

```python
# Current (sampling)
next_token = mx.random.categorical(probs)

# Test (greedy)
next_token = mx.argmax(logits, axis=-1)
```

**If greedy >> 1.1% (e.g., 30%+), the model learned something but has low confidence.**

Use greedy for now.

### 3. **Check for Causal Masking**

**CRITICAL:** Your self-attention needs **causal masking** for autoregressive generation!

During training: model sees full `[START, t2, t3, ..., tN]`
During inference: model generates one token at a time

Without causal masking:

- Training: position 1 attends to position 50 (cheating!)
- Inference: position 1 can't attend to position 50 (fails!)

Check your `TransformerEncoderLayer` - does it support causal masking?

```python
# Should look like:
seq_len = x.shape[1]
causal_mask = mx.tril(mx.ones((seq_len, seq_len)))
# Use causal_mask in attention to prevent looking at future positions
```

**If it doesn't have causal masking, add it** (this will help a lot).

***

## Incremental Improvements

### Quick Wins (No Retraining)

**3a. Temperature Scaling**

```python
# Sharpen the distribution
logits = logits / 0.5  # temperature < 1 = sharper predictions
probs = mx.softmax(logits, axis=-1)
```


### Medium Effort (Retraining Required)

**3b. Increase Latent Dimension**

```python
latent_dim=256  # was 64
# Retrain for 30 epochs → should see significant improvement
```

**3c. Add Causal Masking** (if missing)

```python
# Ensure self-attention uses causal mask
# Retrain for 30 epochs → should help autoregressive generation
```


### Longer Effort

**3d. Increase Model Capacity**

```python
model = SelfiesTransformerVAE(
    embedding_dim=256,   # was 128
    hidden_dim=512,      # was 256
    num_layers=6,        # was 4
    latent_dim=256,      # was 64
)
# Retrain for 50+ epochs
```


***

## Expected Trajectory

```
Current:     1.1% (architecture fixed)
+ Greedy:    5-10% (if model learned something)
+ Causal masking: 10-20% (if not already there)
+ Larger latent: 15-30%
+ Larger model: 30-50%
+ 50+ epochs: 40-60%+
```


***

## Action Plan

1. ✅ Print loss curves - verify training works
2. ✅ Try greedy decoding - see if it's a confidence issue
3. ✅ Check for causal masking - critical for inference
4. If causal masking missing → add it, retrain
5. Increase latent_dim to 256, retrain
6. If still <20%, increase model size
7. Train for 50-100 epochs with learning rate schedule
