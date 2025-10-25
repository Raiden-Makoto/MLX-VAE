from models.transformer_vae import SelfiesTransformerVAE
from utils.loss import compute_loss
from utils.sample import sample_from_vae
from mlx_data.dataloader import create_batches, split_train_val

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
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Epochs to warm up beta from 0 to 1')
parser.add_argument('--max_beta', type=float, default=0.1, help='Maximum beta value')
parser.add_argument('--latent_noise_std', type=float, default=0.05, help='Standard deviation of Gaussian noise added to latent vectors during training')
parser.add_argument('--diversity_weight', type=float, default=0.01, help='Weight for latent diversity loss')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--resume', action='store_true', help='Resume training from best model and last epoch')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')
parser.add_argument('--val_freq', type=int, default=5, help='Validate every N epochs')

# Load data and metadata
with open('mlx_data/cns_metadata.json', 'r') as f:
    meta = json.load(f)

tokenized = np.load('mlx_data/cns_tokenized.npy')
token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
vocab_size = meta['vocab_size']
max_length = meta['max_length']

# Load properties for conditional training
logp_values = meta['logp_values']
tpsa_values = meta['tpsa_values']
properties = [[logp, tpsa] for logp, tpsa in zip(logp_values, tpsa_values)]

args = parser.parse_args()

# Prepare data with train/val split
tokenized_mx = mx.array(tokenized)
properties_mx = mx.array(properties)

# Split into train/val (90/10)
(train_tokens, train_properties), (val_tokens, val_properties) = split_train_val(
    tokenized_mx, properties_mx, val_ratio=0.1, shuffle=True
)

# Create batches
train_batches = create_batches(train_tokens, train_properties, args.batch_size, shuffle=True)
val_batches = create_batches(val_tokens, val_properties, args.batch_size, shuffle=False)

print(f"📊 Data split: {len(train_tokens)} train, {len(val_tokens)} val")
print(f"📦 Batches: {len(train_batches)} train, {len(val_batches)} val")

# Initialize model and optimizer
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
optimizer = Adam(learning_rate=args.learning_rate)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Resume from best model if specified
start_epoch = 0
if args.resume:
    print(f"🔄 Resuming training...")
    
    # Load last epoch from text file
    last_epoch_file = os.path.join(args.output_dir, 'last_epoch.txt')
    if os.path.exists(last_epoch_file):
        with open(last_epoch_file, 'r') as f:
            last_completed_epoch = int(f.read().strip())
            start_epoch = last_completed_epoch + 1  # Resume from next epoch
        print(f"📅 Resuming from epoch {start_epoch} (last completed: {last_completed_epoch})")
    else:
        print("No last_epoch.txt found, starting from epoch 1")
    
    # Load best model weights if available
    best_model_path = os.path.join(args.output_dir, 'best_model.npz')
    if os.path.exists(best_model_path):
        print(f"🏆 Loading best model weights from: {best_model_path}")
        model.load_weights(best_model_path)
    else:
        print("❌ No best_model.npz found, starting from scratch...")

print(f"Initialized model and optimizer")
print(f"Checkpoints will be saved to: {args.output_dir}")

# Best model tracking and early stopping
best_loss = float('inf')
best_model_path = os.path.join(args.output_dir, 'best_model.npz')
patience_counter = 0
last_val_loss = float('inf')

# Training function
def train_step(model, batch_data, optimizer, beta, noise_std=0.05, diversity_weight=0.01):
    """Single training step with conditional properties"""
    batch, properties = batch_data  # Unpack batch and properties
    
    def loss_fn(model):
        logits, mu, logvar = model(batch, properties=properties, training=True, noise_std=noise_std)
        return compute_loss(batch, logits, mu, logvar, beta, diversity_weight)
    
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model)
    total_loss, recon_loss, kl_loss, diversity_loss = loss
    
    # Clip gradients to prevent explosion
    def clip_grads(grads):
        clipped = {}
        for key, value in grads.items():
            if isinstance(value, dict):
                clipped[key] = clip_grads(value)
            elif isinstance(value, list):
                clipped[key] = [mx.clip(v, -1.0, 1.0) if hasattr(v, 'shape') else v for v in value]
            else:
                clipped[key] = mx.clip(value, -1.0, 1.0)
        return clipped
    
    grads = clip_grads(grads)
    
    # Update parameters
    optimizer.update(model, grads)
    
    return total_loss.item(), recon_loss.item(), kl_loss.item(), diversity_loss.item()

# Validation function
def validate(model, val_batches, beta, noise_std=0.05, diversity_weight=0.01):
    """Validate model on validation set"""
    total_val_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_diversity_loss = 0.0
    
    for batch_data in val_batches:
        batch, properties = batch_data
        
        # Forward pass without gradients
        logits, mu, logvar = model(batch, properties=properties, training=False, noise_std=noise_std)
        loss = compute_loss(batch, logits, mu, logvar, beta, diversity_weight)
        
        total_loss, recon_loss, kl_loss, diversity_loss = loss
        
        total_val_loss += total_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_diversity_loss += diversity_loss.item()
    
    # Average losses
    num_batches = len(val_batches)
    avg_val_loss = total_val_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_diversity_loss = total_diversity_loss / num_batches
    
    return avg_val_loss, avg_recon_loss, avg_kl_loss, avg_diversity_loss

# Beta annealing function
def get_beta(epoch, total_epochs, warmup_epochs, max_beta):
    """Compute beta value with annealing"""
    if epoch < warmup_epochs:
        # Linear warmup from 0 to max_beta
        return max_beta * (epoch / warmup_epochs)
    else:
        # Keep at max_beta after warmup
        return max_beta

# Train model
if start_epoch > 0:
    # When resuming, train for args.epochs MORE epochs from where we left off
    total_epochs = start_epoch + args.epochs
    print(f"Resuming training for {args.epochs} more epochs (from epoch {start_epoch + 1} to {total_epochs})")
else:
    # When starting fresh, train for args.epochs total epochs
    total_epochs = args.epochs
    print(f"Training model for {total_epochs} epochs")
print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
print(f"Beta warmup: {args.beta_warmup_epochs} epochs, Max beta: {args.max_beta}")
print(f"Latent noise std: {args.latent_noise_std}")
print(f"Diversity weight: {args.diversity_weight}")
print(f"Validation frequency: every {args.val_freq} epochs")
print(f"Early stopping patience: {args.patience} validation checks")
print(f"Total batches per epoch: {len(train_batches)} train, {len(val_batches)} val")
print("="*67)

for epoch in range(start_epoch, total_epochs):
    # Compute beta for this epoch
    current_beta = get_beta(epoch, args.epochs, args.beta_warmup_epochs, args.max_beta)
    
    epoch_losses = []
    epoch_recon_losses = []
    epoch_kl_losses = []
    
    # Training loop with progress bar
    progress_bar = tqdm.tqdm(
        train_batches, 
        desc=f"Epoch {epoch+1}/{total_epochs} (β={current_beta:.3f})",
        unit="batch",
        leave=False
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        # Training step
        total_loss, recon_loss, kl_loss, diversity_loss = train_step(model, batch, optimizer, current_beta, args.latent_noise_std, args.diversity_weight)
        
        # Store losses
        epoch_losses.append(total_loss)
        epoch_recon_losses.append(recon_loss)
        epoch_kl_losses.append(kl_loss)
        
        # Update progress bar
        if batch_idx % 10 == 0:  # Update every 10 batches
            avg_loss = mx.mean(mx.array(epoch_losses)).item()
            avg_recon = mx.mean(mx.array(epoch_recon_losses)).item()
            avg_kl = mx.mean(mx.array(epoch_kl_losses)).item()
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Recon': f'{avg_recon:.4f}',
                'KL': f'{avg_kl:.4f}'
            })
    
    # Calculate final epoch metrics
    final_loss = mx.mean(mx.array(epoch_losses)).item()
    final_recon = mx.mean(mx.array(epoch_recon_losses)).item()
    final_kl = mx.mean(mx.array(epoch_kl_losses)).item()
    
    print(f"📊 Epoch {epoch+1} Results:")
    print(f"  Train Loss: {final_loss:.4f} (Recon: {final_recon:.4f}, KL: {final_kl:.4f})")
    
    # Validation every N epochs (including epoch 0)
    should_validate = epoch == 0 or (epoch + 1) % args.val_freq == 0 or epoch == total_epochs - 1
    
    if should_validate:
        val_loss, val_recon, val_kl, val_diversity = validate(model, val_batches, current_beta, args.latent_noise_std, args.diversity_weight)
        print(f"  Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_weights(best_model_path)
            patience_counter = 0
            print(f"🏆 New best model! Val Loss: {val_loss:.4f} -> Saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Best val loss so far: {best_loss:.4f} (patience: {patience_counter}/{args.patience})")
            
            # Early stopping check
            if patience_counter >= args.patience:
                print(f"🛑 Early stopping! No improvement for {args.patience} validation checks.")
                print(f"Best validation loss: {best_loss:.4f}")
                break
    else:
        print(f"  Skipping validation (next validation at epoch {((epoch + 1) // args.val_freq + 1) * args.val_freq})")
    
    print("="*67)
    
    # Save checkpoint
    if (epoch + 1) % args.save_every == 0:
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.npz')
        model.save_weights(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save last epoch
    last_epoch_file = os.path.join(args.output_dir, 'last_epoch.txt')
    with open(last_epoch_file, 'w') as f:
        f.write(str(epoch))

print("🎉 Training completed!")
