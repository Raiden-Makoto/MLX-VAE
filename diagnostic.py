#!/usr/bin/env python3
"""
End-to-end diagnostic to verify encoder/decoder outputs and loss components.
Follows the requested steps exactly.
"""

import numpy as np
import mlx.core as mx

from complete_vae_loss import complete_vae_loss
from models.vae import ARCVAE


def main():
    batch_size = 32
    seq_length = 120
    vocab_size = 95
    latent_dim = 128
    num_conditions = 1

    # Build model (use defaults similar to train.py)
    vae = ARCVAE(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=latent_dim,
        num_conditions=num_conditions,
        num_layers=2,
        dropout=0.2,
    )

    encoder = vae.encoder
    decoder = vae.decoder
    property_predictor = None

    # Dummy batch (random)
    molecules = mx.random.randint(0, vocab_size, (batch_size, seq_length), dtype=mx.uint32)
    conditions = mx.random.normal((batch_size, num_conditions)) * 0.1

    print("=" * 80)
    print("DIAGNOSTIC TEST")
    print("=" * 80)

    # 1) ENCODER OUTPUT
    mu, logvar = encoder(molecules, conditions)
    mx.eval(mu, logvar)
    print("\n1. ENCODER OUTPUT")
    print(f"   μ range: [{float(mx.min(mu)):.3f}, {float(mx.max(mu)):.3f}]")
    print(f"   logvar range: [{float(mx.min(logvar)):.3f}, {float(mx.max(logvar)):.3f}]")

    z = encoder.reparameterize(mu, logvar)
    mx.eval(z)
    print(f"   z range: [{float(mx.min(z)):.3f}, {float(mx.max(z)):.3f}]")

    # 2) DECODER OUTPUT
    logits = decoder(z, conditions, target_seq=molecules, teacher_forcing_ratio=0.9)
    mx.eval(logits)
    print("\n2. DECODER OUTPUT")
    print(f"   logits shape: {tuple(logits.shape)}")
    print(f"   logits range: [{float(mx.min(logits)):.3f}, {float(mx.max(logits)):.3f}]")

    # 3) LOSS COMPONENTS
    loss_dict = complete_vae_loss(
        encoder=encoder,
        decoder=decoder,
        property_predictor=property_predictor,
        x=molecules,
        conditions=conditions,
        beta=0.1,
        lambda_prop=0.1,
        lambda_collapse=0.01,
        teacher_forcing_ratio=0.9,
        free_bits=0.5,
        lambda_mi=0.01,
        target_mi=4.85,
    )
    mx.eval(*[loss_dict[k] for k in ['total_loss','recon_loss','kl_loss','collapse_penalty','prop_loss','mutual_info']])

    print("\n3. LOSS COMPONENTS")
    print(f"   Recon loss: {float(loss_dict['recon_loss']):.6f}")
    print(f"   KL loss: {float(loss_dict['kl_loss']):.6f}")
    print(f"   Collapse penalty: {float(loss_dict['collapse_penalty']):.6f}")
    print(f"   Prop loss: {float(loss_dict['prop_loss']):.6f}")
    print(f"   Total loss: {float(loss_dict['total_loss']):.6f}")
    print(f"   MI: {float(loss_dict['mutual_info']):.6f}")

    # 4) SANITY CHECKS
    print("\n4. SANITY CHECKS")
    for key in ['recon_loss','kl_loss','collapse_penalty','prop_loss','total_loss']:
        val = float(loss_dict[key])
        if not np.isfinite(val):
            print(f"   ❌ {key}: NaN/Inf!")
        elif val < 0:
            print(f"   ❌ {key}: NEGATIVE {val}")
        elif val == 0:
            print(f"   ⚠️  {key}: ZERO (suspicious)")
        else:
            print(f"   ✅ {key}: {val:.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


