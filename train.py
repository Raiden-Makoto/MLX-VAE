from models.transformer_vae import SelfiesTransformerVAE
from utils.loss import compute_loss
from utils.sample import sample_from_vae
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
parser.add_argument('--property_weight', type=float, default=10.0, help='Weight for property prediction loss (CVAE requires high weight)')
parser.add_argument('--logp_weight', type=float, default=100.0, help='Separate weight for LogP (needs boost due to 25x smaller scale vs TPSA)')
parser.add_argument('--tpsa_weight', type=float, default=1.0, help='Separate weight for TPSA')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (FIXED at 4)')
parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers (FIXED at 4)')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--resume', action='store_true', help='Resume training from best model and last epoch')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')
parser.add_argument('--val_freq', type=int, default=5, help='Validate every N epochs')

# Load data and metadata
with open('mlx_data/chembl_cns_selfies.json', 'r') as f:
    meta = json.load(f)

tokenized = np.load('mlx_data/chembl_cns_tokenized.npy')
token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
vocab_size = meta['vocab_size']
max_length = meta['max_length']

# Load properties and normalize
molecules = meta['molecules']
properties_raw = np.array([[mol['logp'], mol['tpsa']] for mol in molecules], dtype=np.float32)

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

# Use create_batches from dataloader, but pair with properties
# Shuffle data and properties together
if True:  # Always shuffle
    n_samples = tokenized_mx.shape[0]
    indices = mx.random.permutation(mx.arange(n_samples))
    tokenized_mx = tokenized_mx[indices]
    properties_mx = properties_mx[indices]

# Create batches using existing function (handles properties automatically)
batches = create_batches(tokenized_mx, properties_mx, args.batch_size, shuffle=False)  # Already shuffled above

# Split into train and validation sets
train_size = int(0.9 * len(batches))
train_batches = batches[:train_size]
val_batches = batches[train_size:]

print(f"Total batches: {len(batches)}, Train: {len(train_batches)}, Validation: {len(val_batches)}")

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

# Set property normalization
model.set_property_normalization(logp_mean, logp_std, tpsa_mean, tpsa_std)
optimizer = Adam(learning_rate=args.learning_rate)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Resume from best model if specified
start_epoch = 0
if args.resume:
    print(f"Resuming training...")
    
    # Load last epoch from text file
    last_epoch_file = os.path.join(args.output_dir, 'last_epoch.txt')
    if os.path.exists(last_epoch_file):
        with open(last_epoch_file, 'r') as f:
            last_completed_epoch = int(f.read().strip())
            start_epoch = last_completed_epoch + 1  # Resume from next epoch
        print(f"Resuming from epoch {start_epoch} (last completed: {last_completed_epoch})")
    else:
        print("No last_epoch.txt found, starting from epoch 1")
    
    # Load best model weights if available
    best_model_path = os.path.join(args.output_dir, 'best_model.npz')
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from: {best_model_path}")
        model.load_weights(best_model_path)
        # Also load normalization stats if available
        norm_file = os.path.join(args.output_dir, 'property_norm.json')
        if os.path.exists(norm_file):
            import json
            with open(norm_file, 'r') as f:
                norm_stats = json.load(f)
            model.set_property_normalization(
                norm_stats['logp_mean'],
                norm_stats['logp_std'],
                norm_stats['tpsa_mean'],
                norm_stats['tpsa_std']
            )
    else:
        print("No best_model.npz found, starting from scratch...")

# Best model tracking and early stopping
best_loss = float('inf')
best_model_path = os.path.join(args.output_dir, 'best_model.npz')
patience_counter = 0
last_val_loss = float('inf')

# Training function
def train_step(model, batch_data, batch_properties, optimizer, beta, noise_std=0.05, diversity_weight=0.01, property_weight=1.0, logp_weight=100.0, tpsa_weight=1.0):
    """Single training step"""
    # Store losses for access after computation
    stored_losses = {}
    
    def loss_fn(model):
        result = model(batch_data, properties=batch_properties, training=True, noise_std=noise_std)
        logits, mu, logvar, predicted_properties, property_kl_mu, property_kl_logvar = result
        
        # Get reconstruction and diversity losses (skip KL from compute_loss)
        _, recon_loss, _, diversity_loss = compute_loss(batch_data, logits, mu, logvar, beta, diversity_weight)
        
        # CVAE KL divergence: KL(q(z|x,c) || p(z|c))
        # This trains property networks to match encoder's distribution
        if property_kl_mu is not None and property_kl_logvar is not None:
            # KL divergence between encoder posterior and property-conditioned prior
            # KL(q(z|x,c) || p(z|c)) = KL(N(mu, sigma) || N(property_mu, property_sigma))
            # Clip logvar for numerical stability
            logvar_clipped = mx.clip(logvar, -5, 5)
            property_logvar_clipped = mx.clip(property_kl_logvar, -5, 5)
            
            # Compute KL with numerical stability (log-space)
            # KL(q||p) = 0.5 * (log(|Σ_p|/|Σ_q|) + tr(Σ_q * Σ_p^-1) + (μ_q-μ_p)^T * Σ_p^-1 * (μ_q-μ_p) - d)
            log_var_ratio = property_logvar_clipped - logvar_clipped
            mu_diff_sq = (mu - property_kl_mu) ** 2
            
            kl_loss = 0.5 * mx.sum(
                log_var_ratio
                + mx.exp(logvar_clipped - property_logvar_clipped)  # exp(logvar - property_logvar)
                + mu_diff_sq * mx.exp(-property_logvar_clipped)  # (mu_diff^2) * exp(-property_logvar)
                - 1.0,
                axis=1  # Sum over latent dimensions
            )
            
            kl_loss = mx.clip(kl_loss, 0, 1000)  # Clip to prevent explosion
            kl_loss = mx.mean(kl_loss)  # Average over batch
        else:
            # Standard VAE: KL(q(z|x) || N(0,1))
            logvar_clipped = mx.clip(logvar, -10, 10)
            kl_loss = -0.5 * mx.mean(1 + logvar_clipped - mu**2 - mx.exp(logvar_clipped))
        
        # Property prediction loss (ensures z encodes properties)
        # Normalize targets
        normalized_targets = mx.array([
            [(batch_properties[i, 0] - model.logp_mean) / model.logp_std, 
             (batch_properties[i, 1] - model.tpsa_mean) / model.tpsa_std]
            for i in range(batch_properties.shape[0])
        ])
        
        # Separate MSE loss for LogP and TPSA
        # Since properties are already normalized (mean=0, std=1), both should contribute equally
        # No need to divide by std² again (they're already on same scale)
        logp_loss = mx.mean((predicted_properties[:, 0] - normalized_targets[:, 0]) ** 2)
        tpsa_loss = mx.mean((predicted_properties[:, 1] - normalized_targets[:, 1]) ** 2)
        
        # Debug: Check if predicted_properties are in expected range
        if batch_data.shape[0] == args.batch_size:  # Only log first batch
            stored_losses['pred_logp_range'] = f"{float(mx.min(predicted_properties[:, 0]).item()):.3f} to {float(mx.max(predicted_properties[:, 0]).item()):.3f}"
            stored_losses['pred_tpsa_range'] = f"{float(mx.min(predicted_properties[:, 1]).item()):.3f} to {float(mx.max(predicted_properties[:, 1]).item()):.3f}"
        
        # Weighted combination (no std² division needed since already normalized)
        property_loss = logp_weight * logp_loss + tpsa_weight * tpsa_loss
        property_kl = mx.array(0.0)  # Not used anymore
        
        # Compute per-property MAE for diagnostics
        # Denormalize properties for MAE calculation
        pred_logp = predicted_properties[:, 0] * model.logp_std + model.logp_mean
        true_logp = normalized_targets[:, 0] * model.logp_std + model.logp_mean
        logp_mae = mx.mean(mx.abs(pred_logp - true_logp))
        
        pred_tpsa = predicted_properties[:, 1] * model.tpsa_std + model.tpsa_mean
        true_tpsa = normalized_targets[:, 1] * model.tpsa_std + model.tpsa_mean
        tpsa_mae = mx.mean(mx.abs(pred_tpsa - true_tpsa))
        
        # Store losses for later (evaluate immediately)
        stored_losses['recon'] = float(recon_loss.item())
        stored_losses['kl'] = float(kl_loss.item())
        stored_losses['diversity'] = float(diversity_loss.item())
        stored_losses['property'] = float(property_loss.item())
        stored_losses['property_kl'] = float(property_kl.item())
        stored_losses['logp_mae'] = float(logp_mae.item())
        stored_losses['tpsa_mae'] = float(tpsa_mae.item())
        stored_losses['logp_loss'] = float(logp_loss.item())
        stored_losses['tpsa_loss'] = float(tpsa_loss.item())
        
        # Weighted total loss
        total_loss = recon_loss + beta * kl_loss + diversity_weight * diversity_loss + property_weight * property_loss
        
        return total_loss
    
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model)
    
    total_loss = loss
    recon_loss = stored_losses['recon']
    kl_loss = stored_losses['kl']
    diversity_loss = stored_losses['diversity']
    property_loss = stored_losses['property']
    logp_mae = stored_losses['logp_mae']
    tpsa_mae = stored_losses['tpsa_mae']
    logp_loss = stored_losses['logp_loss']
    tpsa_loss = stored_losses['tpsa_loss']
    
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
    
    return total_loss, recon_loss, kl_loss, diversity_loss, property_loss, logp_mae, tpsa_mae, logp_loss, tpsa_loss

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
        logits, mu, logvar, _, _, _ = model(batch, properties=properties, training=False, noise_std=noise_std)
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
print(f"Diversity weight: {args.diversity_weight}")
print(f"Property weight: {args.property_weight}")
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
    
    epoch_logp_maes = []
    epoch_tpsa_maes = []
    epoch_logp_losses = []
    epoch_tpsa_losses = []
    
    for batch_idx, (batch_data, batch_properties) in enumerate(progress_bar):
        # Training step
        total_loss, recon_loss, kl_loss, diversity_loss, property_loss, logp_mae, tpsa_mae, logp_loss, tpsa_loss = train_step(model, batch_data, batch_properties, optimizer, current_beta, args.latent_noise_std, args.diversity_weight, args.property_weight, args.logp_weight, args.tpsa_weight)
        
        # Store losses
        epoch_losses.append(total_loss)
        epoch_recon_losses.append(recon_loss)
        epoch_kl_losses.append(kl_loss)
        epoch_logp_maes.append(logp_mae)
        epoch_tpsa_maes.append(tpsa_mae)
        epoch_logp_losses.append(logp_loss)
        epoch_tpsa_losses.append(tpsa_loss)
        
        # Update progress bar
        if batch_idx % 10 == 0:  # Update every 10 batches
            avg_loss = mx.mean(mx.array(epoch_losses)).item()
            avg_recon = mx.mean(mx.array(epoch_recon_losses)).item()
            avg_kl = mx.mean(mx.array(epoch_kl_losses)).item()
            avg_prop = property_loss  # Already a float from stored_losses
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Recon': f'{avg_recon:.4f}',
                'KL': f'{avg_kl:.4f}',
                'Prop': f'{avg_prop:.4f}'
            })
    
    # Calculate final epoch metrics
    final_loss = mx.mean(mx.array(epoch_losses)).item()
    final_recon = mx.mean(mx.array(epoch_recon_losses)).item()
    final_kl = mx.mean(mx.array(epoch_kl_losses)).item()
    avg_logp_mae = sum(epoch_logp_maes) / len(epoch_logp_maes)
    avg_tpsa_mae = sum(epoch_tpsa_maes) / len(epoch_tpsa_maes)
    
    # Print property prediction MAE and loss diagnostics (per recommendations.md)
    print(f"\nEpoch {epoch+1}: LogP MAE={avg_logp_mae:.3f}, TPSA MAE={avg_tpsa_mae:.2f}")
    
    # Also track per-property loss (they should be comparable since already normalized)
    avg_logp_loss = sum(epoch_logp_losses) / len(epoch_logp_losses)
    avg_tpsa_loss = sum(epoch_tpsa_losses) / len(epoch_tpsa_losses)
    print(f"Per-property losses: LogP={avg_logp_loss:.4f}, TPSA={avg_tpsa_loss:.4f}")
    
    # Save best model if this is the best loss so far
    if final_loss < best_loss:
        best_loss = final_loss
        model.save_weights(best_model_path)
        # Also save normalization stats and architecture params
        import json
        norm_stats = {
            'logp_mean': float(logp_mean),
            'logp_std': float(logp_std),
            'tpsa_mean': float(tpsa_mean),
            'tpsa_std': float(tpsa_std),
            # Save architecture parameters to ensure consistent loading
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'latent_dim': args.latent_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
        with open(f'{args.output_dir}/property_norm.json', 'w') as f:
            json.dump(norm_stats, f)
        print(f"New best model! Loss: {final_loss:.4f} -> Saved to {best_model_path}")
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

print("Training completed!")
