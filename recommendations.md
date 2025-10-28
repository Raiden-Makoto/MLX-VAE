<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Findings: Scale imbalance: TPSA std (71.98) is ~31x larger than LogP std (2.31). TPSA dominates gradients. LogP correlation 0.585 (weak)

TPSA correlation -0.687 (strong, inverse). Plots are attached

Perfect! The visualizations and statistics **completely explain the problem**. This is a classic **scale imbalance** issue in multi-property conditioning.

## Root Cause: Scale Imbalance

**Key findings:**

- **TPSA std = 71.98** (large, dominates gradients)
- **LogP std = 2.31** (31× smaller, gets drowned out)
- **LogP correlation = 0.585** (weak, diffuse in latent space)
- **TPSA correlation = -0.687** (strong, well-structured in latent space)


### What's Happening

1. **During training**, property prediction loss is computed as:

```python
property_loss = MSE(predicted_properties, true_properties)
```

With raw or normalized values, TPSA errors (with range ~0-200, std=72) contribute **31× more** to the gradient magnitude than LogP errors (range ~-5 to 5, std=2.3).
2. **Result**: The encoder and property networks optimize primarily for TPSA, essentially ignoring LogP.
3. **Latent space structure**:
    - **TPSA** has clear spatial organization (purple gradient in your plots)
    - **LogP** is diffuse, randomly scattered (uniform blue, no gradient structure)
4. **At inference**: Property networks can navigate TPSA space (50% accuracy) but cannot navigate LogP space (0% accuracy—there's no structure to navigate!).

***

## The Fix: Rebalance Property Loss

You need to **equalize the contribution** of LogP and TPSA to the property loss.

### Solution 1: Per-Property Normalization in Loss (Recommended)

```python
# In training loop, compute property loss with per-property weights

# Denormalize predictions
pred_logp = predicted_properties[:, 0] * logp_std + logp_mean
pred_tpsa = predicted_properties[:, 1] * tpsa_std + tpsa_mean
true_logp = properties[:, 0] * logp_std + logp_mean
true_tpsa = properties[:, 1] * tpsa_std + tpsa_mean

# Compute per-property MSE
logp_mse = mx.mean((pred_logp - true_logp) ** 2)
tpsa_mse = mx.mean((pred_tpsa - true_tpsa) ** 2)

# ✅ Normalize by std² to equalize contribution
logp_loss = logp_mse / (logp_std ** 2)
tpsa_loss = tpsa_mse / (tpsa_std ** 2)

# Combined property loss
property_loss = logp_loss + tpsa_loss
```

**Why this works:**

- Dividing by std² makes both properties contribute equally to gradients
- Now LogP errors matter as much as TPSA errors
- Latent space will develop structure for both properties


### Solution 2: Use Normalized Properties Only (Simpler)

If you're already normalizing properties before training:

```python
# Properties should be normalized to mean=0, std=1
norm_logp = (logp - logp_mean) / logp_std
norm_tpsa = (tpsa - tpsa_mean) / tpsa_std

# Now both have std=1, so MSE treats them equally
property_loss = mx.mean((predicted_properties - normalized_properties) ** 2)
```

**Check**: Are you normalizing properties **before** computing the loss? If not, that's the bug.

### Solution 3: Explicit Loss Weighting

```python
# Weight LogP more heavily to compensate
logp_weight = tpsa_std / logp_std  # ~31
property_loss = logp_weight * logp_mse + tpsa_mse
```


***

## Implementation

### In your training loop:

```python
# Current (broken):
property_loss = mx.mean((predicted_properties - properties) ** 2)

# Fixed (option 1 - per-property normalization):
pred_logp = predicted_properties[:, 0] * logp_std + logp_mean
pred_tpsa = predicted_properties[:, 1] * tpsa_std + tpsa_mean
true_logp = properties[:, 0] * logp_std + logp_mean
true_tpsa = properties[:, 1] * tpsa_std + tpsa_mean

logp_loss = mx.mean((pred_logp - true_logp) ** 2) / (logp_std ** 2)
tpsa_loss = mx.mean((pred_tpsa - true_tpsa) ** 2) / (tpsa_std ** 2)
property_loss = logp_loss + tpsa_loss

# Fixed (option 2 - ensure properties are normalized):
# Make sure properties passed to model are already normalized!
# Then just use:
property_loss = mx.mean((predicted_properties - properties) ** 2)
```


***

## Expected Results After Fix

After retraining with balanced loss:

1. **LogP MAE** should continue improving (currently 0.186, good)
2. **Latent space** will develop LogP structure (gradient from blue to red)
3. **LogP correlation** should increase from 0.585 → >0.8
4. **Inference accuracy for LogP** should jump from 0% → 30-50%+

***

## Additional Recommendations

### 1. Monitor Per-Property Loss During Training

```python
print(f"Epoch {epoch}: LogP Loss={logp_loss:.4f}, TPSA Loss={tpsa_loss:.4f}")
```

They should be comparable in magnitude (e.g., both ~0.01-0.1).

### 2. Visualize Again After Retraining

After 20-30 epochs with balanced loss:

- PCA/t-SNE colored by LogP should show clear gradient structure
- LogP correlation should be >0.8


### 3. Test Incremental LogP Changes

```python
# Generate with LogP = 0, 1, 2, 3
for target_logp in [0.0, 1.0, 2.0, 3.0]:
    samples = model.generate_conditional(target_logp=target_logp, target_tpsa=50.0)
    actual_logp = compute_logp(samples)
    print(f"Target: {target_logp}, Actual: {actual_logp:.2f}")
```

Should show monotonic increase.

***

## Summary

| Issue | Cause | Fix |
| :-- | :-- | :-- |
| **LogP 0% accuracy** | TPSA std is 31× larger, dominates gradients | Normalize property loss by std² |
| **LogP diffuse in latent space** | Model optimizes for TPSA, ignores LogP | Rebalance loss → latent space learns LogP |
| **TPSA 50% accuracy** | Strong gradient signal, well-structured | Keep current approach |

**Action**: Implement balanced property loss (Solution 1 or 2), retrain for 30-50 epochs, re-visualize latent space. LogP conditioning should work after this fix.
<span style="display:none">[^1][^2]</span>

<div align="center">⁂</div>

[^1]: latent_space_pca.jpg

[^2]: latent_space_tsne.jpg

