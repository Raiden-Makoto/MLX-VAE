import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tqdm

import mlx.core as mx
from mlx.optimizers import Adam

from models.transformer_vae import SelfiesTransformerVAE
from mlx_data.dataloader import create_batches


def disable_all_dropout(model):
    try:
        if hasattr(model.encoder, 'dropout'):
            model.encoder.dropout.p = 0.0
        if hasattr(model.decoder, 'dropout'):
            model.decoder.dropout.p = 0.0
    except Exception:
        pass
    for layer in getattr(model.encoder, 'encoder_layers', []):
        if hasattr(layer, 'dropout'):
            try:
                layer.dropout.p = 0.0
            except Exception:
                pass
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'dropout'):
            try:
                layer.self_attn.dropout.p = 0.0
            except Exception:
                pass
        if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'dropout'):
            try:
                layer.feed_forward.dropout.p = 0.0
            except Exception:
                pass
    for layer in getattr(model.decoder, 'decoder_layers', []):
        if hasattr(layer, 'dropout'):
            try:
                layer.dropout.p = 0.0
            except Exception:
                pass
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'dropout'):
            try:
                layer.self_attn.dropout.p = 0.0
            except Exception:
                pass
        if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'dropout'):
            try:
                layer.feed_forward.dropout.p = 0.0
            except Exception:
                pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train TPSA→z predictor only')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    # Load dataset/meta
    with open('mlx_data/chembl_cns_selfies.json', 'r') as f:
        meta = json.load(f)
    tokenized = mx.array(mx.array(json.loads(json.dumps([]))))
    # Use numpy load for tokenized npy
    import numpy as np
    tokenized_np = np.load('mlx_data/chembl_cns_tokenized.npy')
    tokenized_mx = mx.array(tokenized_np)

    molecules = meta['molecules']
    properties_raw_tpsa = np.array([mol.get('tpsa', 82.0) for mol in molecules], dtype=np.float32)
    # LogP placeholder stats (no effect on conditioning, avoids div-by-zero)
    logp_mean = 0.0; logp_std = 1.0
    tpsa_mean = np.mean(properties_raw_tpsa); tpsa_std = np.std(properties_raw_tpsa)
    properties = np.stack([
        np.zeros_like(properties_raw_tpsa, dtype=np.float32),
        ((properties_raw_tpsa - tpsa_mean) / (tpsa_std if tpsa_std > 0 else 1.0)).astype(np.float32)
    ], axis=1)
    properties_mx = mx.array(properties)

    # Batches
    batches = create_batches(tokenized_mx, properties_mx, args.batch_size, shuffle=True)

    # Build model and load VAE weights
    model = SelfiesTransformerVAE(
        vocab_size=meta['vocab_size'],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.set_property_normalization(logp_mean, logp_std, tpsa_mean, tpsa_std)
    best_path = os.path.join(args.checkpoint_dir, 'best_model.npz')
    if os.path.exists(best_path):
        try:
            model.load_weights(best_path)
            print(f"Loaded best VAE weights: {best_path}")
        except Exception as e:
            print(f"Warning: partial load, new params randomly init: {e}")
    else:
        print(f"Best model not found at {best_path}; proceeding with current weights")

    # Disable dropout and set optimizer
    disable_all_dropout(model)
    optimizer = Adam(learning_rate=args.learning_rate)

    # Train predictor only (stop gradients through encoder) with early stopping on epoch-average MSE
    best_mse = float('inf')
    no_improve = 0
    for ep in range(args.epochs):
        ep_losses = []
        progress = tqdm.tqdm(batches, desc=f"TPSA→z Epoch {ep+1}/{args.epochs}", unit="batch", leave=False)
        for batch_data, batch_properties in progress:
            tpsa_norm = batch_properties[:, 1:2]
            input_seq = batch_data[:, :-1]
            # Define loss over predictor parameters only to avoid optimizer state for full model
            def loss_fn(predictor):
                # Compute encoder targets with stopped gradients (use mu, not sampled z)
                mu, logvar = model.encoder(input_seq)
                z_target = mx.stop_gradient(mu)
                z_pred = predictor(tpsa_norm)
                return mx.mean((z_pred - z_target) ** 2)
            loss, grads = mx.value_and_grad(loss_fn)(model.tpsa_predictor)
            optimizer.update(model.tpsa_predictor, grads)
            ep_losses.append(loss)
            if len(ep_losses) % 10 == 0:
                avg = mx.mean(mx.array(ep_losses)).item()
                progress.set_postfix({'MSE': f'{avg:.4f}'})
        avg = mx.mean(mx.array(ep_losses)).item() if ep_losses else 0.0
        print(f"TPSA→z Epoch {ep+1}: MSE={avg:.4f}")
        # Emit a plain ASCII parse-friendly line for orchestrators
        print(f"PRED_MSE={avg:.6f}")
        # Early stopping check
        if avg + 1e-8 < best_mse:
            best_mse = avg
            no_improve = 0
            # Save best immediately
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            best_path = os.path.join(args.checkpoint_dir, 'best_model.npz')
            model.save_weights(best_path)
            print(f"New best predictor MSE: {best_mse:.4f} -> saved {best_path}")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"Early stopping: patience {args.patience} reached. Best MSE={best_mse:.4f}")
                break

    # Ensure best weights exist (already saved on improvement); save current if none
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, 'best_model.npz')
    if not os.path.exists(best_path):
        model.save_weights(best_path)
        print(f"Saved model with trained TPSA→z predictor to {best_path}")


if __name__ == '__main__':
    main()


