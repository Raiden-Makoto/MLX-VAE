<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# after training 50 epochs, we are still at 0% accuracy for conditioning on logp. however, our accuracy for conditioning on tpsa increased to 50%. this is with greedy decoding. latent dim was 256, and we are using causal masking.

This is a highly informative result—the conditioning is working for TPSA (50% accuracy), but almost completely fails for LogP (0%).

### Here's what's happening:

- **Property conditioning and prediction are present for both LogP and TPSA.**
- Training is identical for both properties: both are normalized, both networks (`property_mu`, `property_logvar`, and `property_predictor`) learn in parallel, and both logp and tpsa are provided for FILM conditioning and the property networks.[^1]

***

## Likely Root Causes

### 1. **Data Distribution/Correlation**

- **TPSA and target molecules:** If your dataset has many molecules with a wide range of TPSA values, the model can learn to condition and generate diverse molecules for different TPSA targets.
- **LogP:** If your dataset's LogP values are tightly clustered or do not differentiate molecules well (e.g., most molecules have similar LogP), the network may fail to learn meaningful conditioning for logp.


### 2. **Gradient Signal and Loss**

- If the property prediction loss for LogP is much weaker or the variance is much lower than for TPSA, the gradients may be vanishing for logp.
- **Check property prediction loss per property and print MAE for both LogP and TPSA after every epoch.** If LogP prediction MAE is high or flat, it means the latent does not encode logp information.


### 3. **Scale/Normalization Issue**

- Ensure both properties are being normalized with correct mean/std. If logp normalization is off (e.g., large std, small values getting squashed), conditioning may be lost.


### 4. **Network Bias in Property Networks**

- If the initial layers/biases of `property_mu` and `property_predictor` favor TPSA in its output or initialization, learning for logp may stagnate.
- **Check output statistics at initialization and throughout training for logp.**


### 5. **Decoder’s Use of FILM/Conditioning**

- TPSA may be reflected in token sequences more directly (certain tokens correspond to TPSA), while LogP (which is more subtle and depends on overall hydrophobicity) might not correspond to specific tokens.
- **Ablate**: run with FILM conditioning for TPSA only, for logp only, and for both, to see how each conditioning signal affects generation.


### 6. **Latent Bottleneck**

- With latent_dim=256, bottleneck is less likely, but if the decoder doesn't use the property conditioning properly, signals could still be lost.

***

## Next Steps \& Diagnostics

### \#\#\#\#\# Print these metrics after each epoch:

```python
# In training loop
pred_logp = predicted_properties[:, 0] * logp_std + logp_mean
true_logp = properties[:, 0] * logp_std + logp_mean
logp_mae = mx.mean(mx.abs(pred_logp - true_logp))

pred_tpsa = predicted_properties[:, 1] * tpsa_std + tpsa_mean
true_tpsa = properties[:, 1] * tpsa_std + tpsa_mean
tpsa_mae = mx.mean(mx.abs(pred_tpsa - true_tpsa))

print(f"Epoch {epoch}: LogP MAE={logp_mae:.3f}, TPSA MAE={tpsa_mae:.2f}")
```

- **If LogP MAE is flat/high, while TPSA MAE drops significantly, you know where the issue is.**


### \#\#\#\#\# Print property_mu and property_logvar statistics for each property:

```python
print(f"property_mu LogP: {mx.mean(property_mu[:,0]):.3f} ± {mx.std(property_mu[:,0]):.3f}")
print(f"property_mu TPSA: {mx.mean(property_mu[:,1]):.2f} ± {mx.std(property_mu[:,1]):.2f}")
```


### \#\#\#\#\# Check the distribution of logp and tpsa in your training data.

Often, poor conditioning arises when the property is not informative or distributed poorly (e.g., narrow range or highly imbalanced).

***

## Recommendations

- **Run the diagnostics above** and compare signal for logp and tpsa.
- **If logp is flat/imbalanced**, consider stratified sampling or loss weighting to upweight logp at training.
- **Experiment with more direct decoder conditioning:** Add logp as an extra input to decoder, or increase loss weighting for logp in property prediction.
- **Try adding a logp-specific head or auxiliary loss.**
- **If none of this helps, share representative LogP/TPSA statistics from your dataset.**

***

## Summary Table

| Property | Training MAE | Inference Accuracy | Data Distribution |
| :-- | :-- | :-- | :-- |
| LogP | [check] | 0% | [analyze] |
| TPSA | [check] | 50% | [analyze] |

Most likely: **Dataset or loss signal for logp is too weak.** Focus on diagnostics above to confirm.

<div align="center">⁂</div>

[^1]: transformer_vae.py

