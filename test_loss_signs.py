#!/usr/bin/env python3
"""
Diagnostic script to test loss functions for sign errors
Run this to verify all loss components are positive
"""

import mlx.core as mx
import numpy as np
from losses.recon import reconstruction_loss
from losses.kl import kl_divergence
from losses.info import mutual_information, posterior_collapse
from losses.prop import property_prediction_loss

print("=" * 80)
print("Loss Function Sign Check")
print("=" * 80)

# Create dummy data
batch_size = 32
seq_length = 120
vocab_size = 95
latent_dim = 128
num_conditions = 1

print("\n1. Testing Reconstruction Loss...")
logits = mx.random.normal((batch_size, seq_length, vocab_size))
targets = mx.random.randint(0, vocab_size, (batch_size, seq_length))

recon = reconstruction_loss(logits, targets, reduction='mean')
mx.eval(recon)
recon_val = float(recon)

print(f"   Reconstruction loss: {recon_val:.4f}")
if recon_val < 0:
    print(f"   ❌ ERROR: Reconstruction loss is NEGATIVE!")
    print(f"   → Check losses/recon.py line 57: should be '-target_log_probs'")
else:
    print(f"   ✅ Correct: Reconstruction loss is positive")

print("\n2. Testing KL Divergence...")
mu = mx.random.normal((batch_size, latent_dim)) * 0.1  # Small values
logvar = mx.random.normal((batch_size, latent_dim)) * 0.1 - 1.0  # Around -1

kl = kl_divergence(mu, logvar, reduction='mean', free_bits=0.0)
mx.eval(kl)
kl_val = float(kl)

print(f"   KL loss: {kl_val:.4f}")
if kl_val < 0:
    print(f"   ❌ ERROR: KL loss is NEGATIVE!")
    print(f"   → Check losses/kl.py line 48: should be '-0.5 * (...)'")
else:
    print(f"   ✅ Correct: KL loss is positive")

print("\n3. Testing Mutual Information...")
mi = mutual_information(mu, logvar)
mx.eval(mi)
mi_val = float(mi)

print(f"   Mutual Information: {mi_val:.4f}")
print(f"   (MI can be positive or negative, but should be reasonable)")

print("\n4. Testing Posterior Collapse Penalty...")
collapse = posterior_collapse(mu, logvar, target_mi=4.85, weight=0.1)
mx.eval(collapse)
collapse_val = float(collapse)

print(f"   Collapse penalty: {collapse_val:.4f}")
if collapse_val < 0:
    print(f"   ⚠️  WARNING: Collapse penalty is negative (may be intentional)")
else:
    print(f"   ✅ Correct: Collapse penalty is non-negative")

print("\n5. Testing Property Prediction Loss...")
pred_props = mx.random.normal((batch_size, num_conditions))
target_props = mx.random.normal((batch_size, num_conditions))

prop = property_prediction_loss(pred_props, target_props, reduction='mean')
mx.eval(prop)
prop_val = float(prop)

print(f"   Property loss: {prop_val:.4f}")
if prop_val < 0:
    print(f"   ❌ ERROR: Property loss is NEGATIVE!")
else:
    print(f"   ✅ Correct: Property loss is non-negative")

print("\n6. Testing Combined Loss (Complete VAE Loss)...")
from complete_vae_loss import complete_vae_loss
from models.encoder import MLXEncoder
from models.decoder import MLXAutoregressiveDecoder

# Create dummy encoder/decoder (minimal)
# Note: This will fail if models aren't properly initialized
# Just check the formulas manually

print("   Skipping full model test (requires proper initialization)")
print("   → Run training and check first batch output for negative values")

print("\n" + "=" * 80)
print("Summary:")
print("  - Reconstruction loss should be ≥ 0")
print("  - KL divergence should be ≥ 0")
print("  - Property loss should be ≥ 0")
print("  - Total loss should be ≥ 0 (may be negative if MI penalty dominates)")
print("=" * 80)

