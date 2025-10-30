from models.transformer_vae import SelfiesTransformerVAE
from utils.loss import compute_loss
from mlx_data.dataloader import create_batches

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam

import numpy as np
import json
import argparse
import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Epochs to warm up beta from 0 to 1')
parser.add_argument('--max_beta', type=float, default=1.0, help='Maximum beta value (CVAE best practice: 1.0 for full KL divergence)')
parser.add_argument('--latent_noise_std', type=float, default=0.05, help='Standard deviation of Gaussian noise added to latent vectors during training')
parser.add_argument('--diversity_weight', type=float, default=0.01, help='Weight for latent diversity loss')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (FIXED at 4)')
parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers (FIXED at 4)')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--resume', action='store_true', help='Resume training from best model and last epoch')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')
parser.add_argument('--val_freq', type=int, default=5, help='Validate every N epochs')
 # Predictor training moved to train_predictor.py to keep this file focused

# Load data and metadata
with open('mlx_data/chembl_cns_selfies.json', 'r') as f:
    meta = json.load(f)

tokenized = np.load('mlx_data/chembl_cns_tokenized.npy')
token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
vocab_size = meta['vocab_size']
max_length = meta['max_length']

# Load properties - compute logp and tpsa from dataset
molecules = meta['molecules']
properties_raw = np.array([[mol.get('logp', 3.0), mol.get('tpsa', 82.0)] for mol in molecules], dtype=np.float32)

# Normalize properties (zero mean, unit variance)
logp_mean = np.mean(properties_raw[:, 0])
logp_std = np.std(properties_raw[:, 0])
tpsa_mean = np.mean(properties_raw[:, 1])
tpsa_std = np.std(properties_raw[:, 1])

properties = np.array([
    [(prop[0] - logp_mean) / logp_std, (prop[1] - tpsa_mean) / tpsa_std]
    for prop in properties_raw
], dtype=np.float32)

args = parser.parse_args()

# ENFORCE 4 LAYERS 4 HEADS - NEVER CHANGE THIS
args.num_layers = 4
args.num_heads = 4

# Prepare data
tokenized_mx = mx.array(tokenized)
properties_mx = mx.array(properties)

# Shuffle
n_samples = tokenized_mx.shape[0]
indices = mx.random.permutation(mx.arange(n_samples))
tokenized_mx = tokenized_mx[indices]
properties_mx = properties_mx[indices]

# Batches
batches = create_batches(tokenized_mx, properties_mx, args.batch_size, shuffle=False)
train_size = int(0.9 * len(batches))
train_batches = batches[:train_size]
val_batches = batches[train_size:]

print(f"Total batches: {len(batches)}, Train: {len(train_batches)}, Validation: {len(val_batches)}")

# Initialize model/optimizer
print(f"Initializing Transformer VAE with embedding_dim={args.embedding_dim}, hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}")
print(f"Transformer config: num_heads={args.num_heads}, num_layers={args.num_layers}, dropout={args.dropout}")

model = SelfiesTransformerVAE(
    vocab_size=vocab_size,
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    dropout=args.dropout
)
model.set_property_normalization(logp_mean, logp_std, tpsa_mean, tpsa_std)
optimizer = Adam(learning_rate=args.learning_rate)

# Output dir
os.makedirs(args.output_dir, exist_ok=True)

# Resume
start_epoch = 0
if args.resume:
    print(f"Resuming training...")
    last_epoch_file = os.path.join(args.output_dir, 'last_epoch.txt')
    if os.path.exists(last_epoch_file):
        with open(last_epoch_file, 'r') as f:
            last_completed_epoch = int(f.read().strip())
            start_epoch = last_completed_epoch + 1
        print(f"Resuming from epoch {start_epoch} (last completed: {last_completed_epoch})")
    best_model_path = os.path.join(args.output_dir, 'best_model.npz')
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from: {best_model_path}")
        model.load_weights(best_model_path)
        norm_file = os.path.join(args.output_dir, 'property_norm.json')
        if os.path.exists(norm_file):
            with open(norm_file, 'r') as f:
                norm_stats = json.load(f)
            model.set_property_normalization(
                norm_stats.get('logp_mean', 0.0),
                norm_stats.get('logp_std', 1.0),
                norm_stats['tpsa_mean'],
                norm_stats['tpsa_std']
            )

# Best tracking
best_loss = float('inf')
best_model_path = os.path.join(args.output_dir, 'best_model.npz')

# Training step
def train_step(model, batch_data, batch_properties, optimizer, beta, noise_std=0.05, diversity_weight=0.01):
    stored = {}
    def loss_fn(model):
        logits, mu, logvar = model(batch_data, properties=batch_properties, training=True, noise_std=noise_std)
        total, recon, kl, diversity = compute_loss(batch_data, logits, mu, logvar, beta, diversity_weight)
        stored['recon'] = float(recon.item())
        stored['kl'] = float(kl.item())
        stored['diversity'] = float(diversity.item())
        return total
    total_loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return total_loss, stored['recon'], stored['kl'], stored['diversity']

# Validation
def validate(model, val_batches, beta, noise_std=0.05, diversity_weight=0.01):
    total_val_loss = total_recon = total_kl = total_div = 0.0
    for batch_data, properties in val_batches:
        logits, mu, logvar = model(batch_data, properties=properties, training=False, noise_std=noise_std)
        total, recon, kl, div = compute_loss(batch_data, logits, mu, logvar, beta, diversity_weight)
        total_val_loss += total.item(); total_recon += recon.item(); total_kl += kl.item(); total_div += div.item()
    n = len(val_batches)
    return total_val_loss/n, total_recon/n, total_kl/n, total_div/n

# Beta schedule
def get_beta(epoch, total_epochs, warm, max_beta):
    if epoch < warm:
        p = epoch / warm
        return max_beta * (p ** 2)
    return max_beta

# Train VAE as usual
total_epochs = args.epochs if start_epoch == 0 else start_epoch + args.epochs
print(f"Training model for {total_epochs} epochs")
print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
print(f"Beta warmup: {args.beta_warmup_epochs} epochs (quadratic), Max beta: {args.max_beta}")
print(f"Diversity weight: {args.diversity_weight}")
print("="*67)

for epoch in range(start_epoch, total_epochs):
    beta = get_beta(epoch, args.epochs, args.beta_warmup_epochs, args.max_beta)
    epoch_losses, epoch_recon, epoch_kl = [], [], []
    progress = tqdm.tqdm(train_batches, desc=f"Epoch {epoch+1}/{total_epochs} (β={beta:.3f})", unit="batch", leave=False)
    for batch_idx, (batch_data, batch_properties) in enumerate(progress):
        total, recon, kl, div = train_step(model, batch_data, batch_properties, optimizer, beta, args.latent_noise_std, args.diversity_weight)
        epoch_losses.append(total); epoch_recon.append(recon); epoch_kl.append(kl)
        if batch_idx % 10 == 0:
            avg_loss = mx.mean(mx.array(epoch_losses)).item()
            avg_recon = mx.mean(mx.array(epoch_recon)).item()
            avg_kl = mx.mean(mx.array(epoch_kl)).item()
            progress.set_postfix({ 'Loss': f'{avg_loss:.4f}', 'Recon': f'{avg_recon:.4f}', 'KL': f'{avg_kl:.4f}' })
    final_loss = mx.mean(mx.array(epoch_losses)).item()
    final_recon = mx.mean(mx.array(epoch_recon)).item()
    final_kl = mx.mean(mx.array(epoch_kl)).item()
    print(f"\nEpoch {epoch+1}: Recon={final_recon:.4f} KL={final_kl:.4f} Total={final_loss:.4f}")
    # Save best
    if final_loss < best_loss:
        best_loss = final_loss
        model.save_weights(best_model_path)
        norm_stats = {
            'logp_mean': float(logp_mean), 'logp_std': float(logp_std),
            'tpsa_mean': float(tpsa_mean), 'tpsa_std': float(tpsa_std),
            'embedding_dim': args.embedding_dim, 'hidden_dim': args.hidden_dim,
            'latent_dim': args.latent_dim, 'num_heads': args.num_heads,
            'num_layers': args.num_layers, 'dropout': args.dropout
        }
        with open(f'{args.output_dir}/property_norm.json', 'w') as f:
            json.dump(norm_stats, f)
        print(f"New best model! Loss: {final_loss:.4f} -> Saved to {best_model_path}")
    print("="*67)
    if (epoch + 1) % args.save_every == 0:
        ckpt = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.npz')
        model.save_weights(ckpt)
        print(f"Saved checkpoint: {ckpt}")
    with open(os.path.join(args.output_dir, 'last_epoch.txt'), 'w') as f:
        f.write(str(epoch))

print("Training completed!")

# Predictor training removed from this file; use train_predictor.py
