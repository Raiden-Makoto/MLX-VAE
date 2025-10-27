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
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Epochs to warm up beta from 0 to 1')
parser.add_argument('--max_beta', type=float, default=0.1, help='Maximum beta value')
parser.add_argument('--latent_noise_std', type=float, default=0.05, help='Standard deviation of Gaussian noise added to latent vectors during training')
parser.add_argument('--diversity_weight', type=float, default=0.01, help='Weight for latent diversity loss')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--resume', action='store_true', help='Resume training from best model and last epoch')

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

print(f"Property normalization:")
print(f"  LogP: mean={logp_mean:.2f}, std={logp_std:.2f}")
print(f"  TPSA: mean={tpsa_mean:.2f}, std={tpsa_std:.2f}")

args = parser.parse_args()

# Prepare data
tokenized_mx = mx.array(tokenized)
properties_mx = mx.array(properties)

# Create function to create batches with properties
def create_batches_with_properties(data, properties, batch_size, shuffle=True):
    """Create batches from data with corresponding properties"""
    n_samples = data.shape[0]
    
    if shuffle:
        # Create random permutation of indices
        indices = mx.random.permutation(mx.arange(n_samples))
        data = data[indices]
        properties = properties[indices]
    
    n_batches = n_samples // batch_size
    batches = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_data = data[start_idx:end_idx]
        batch_properties = properties[start_idx:end_idx]
        batches.append((batch_data, batch_properties))
    
    return batches

batches = create_batches_with_properties(tokenized_mx, properties_mx, args.batch_size, shuffle=True)

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
        print("❌ No best_model.npz found, starting from scratch...")

print(f"Initialized model and optimizer")
print(f"Checkpoints will be saved to: {args.output_dir}")

# Best model tracking
best_loss = float('inf')
best_model_path = os.path.join(args.output_dir, 'best_model.npz')

# Training function
def train_step(model, batch_data, batch_properties, optimizer, beta, noise_std=0.05, diversity_weight=0.01, property_weight=1.0):
    """Single training step"""
    def loss_fn(model):
        logits, mu, logvar, predicted_properties = model(batch_data, properties=batch_properties, training=True, noise_std=noise_std)
        
        # Get reconstruction, KL, and diversity losses
        recon_loss, kl_loss, diversity_loss = compute_loss(batch_data, logits, mu, logvar, beta, diversity_weight)
        
        # Property prediction loss (CVAE requirement)
        # Normalize targets
        norm_logp = (batch_properties[:, 0] - model.logp_mean) / model.logp_std
        norm_tpsa = (batch_properties[:, 1] - model.tpsa_mean) / model.tpsa_std
        normalized_targets = mx.array([[norm_logp[i], norm_tpsa[i]] for i in range(batch_properties.shape[0])])
        
        # MSE loss between predicted and target properties
        property_loss = mx.mean((predicted_properties - normalized_targets) ** 2)
        
        total_loss = recon_loss + kl_loss + diversity_loss + property_weight * property_loss
        
        return total_loss, recon_loss, kl_loss, diversity_loss, property_loss
    
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model)
    total_loss, recon_loss, kl_loss, diversity_loss, property_loss = loss
    
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
print(f"Total batches per epoch: {len(batches)}")
print("="*67)

for epoch in range(start_epoch, total_epochs):
    # Compute beta for this epoch
    current_beta = get_beta(epoch, args.epochs, args.beta_warmup_epochs, args.max_beta)
    
    epoch_losses = []
    epoch_recon_losses = []
    epoch_kl_losses = []
    
    # Training loop with progress bar
    progress_bar = tqdm.tqdm(
        batches, 
        desc=f"Epoch {epoch+1}/{total_epochs} (β={current_beta:.3f})",
        unit="batch",
        leave=False
    )
    
    for batch_idx, (batch_data, batch_properties) in enumerate(progress_bar):
        # Training step
        total_loss, recon_loss, kl_loss, diversity_loss, property_loss = train_step(model, batch_data, batch_properties, optimizer, current_beta, args.latent_noise_std, args.diversity_weight)
        
        # Store losses
        epoch_losses.append(total_loss)
        epoch_recon_losses.append(recon_loss)
        epoch_kl_losses.append(kl_loss)
        
        # Update progress bar
        if batch_idx % 10 == 0:  # Update every 10 batches
            avg_loss = mx.mean(mx.array(epoch_losses)).item()
            avg_recon = mx.mean(mx.array(epoch_recon_losses)).item()
            avg_kl = mx.mean(mx.array(epoch_kl_losses)).item()
            avg_prop = property_loss.item()
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
    
    # Save best model if this is the best loss so far
    if final_loss < best_loss:
        best_loss = final_loss
        model.save_weights(best_model_path)
        # Also save normalization stats
        import json
        norm_stats = {
            'logp_mean': float(logp_mean),
            'logp_std': float(logp_std),
            'tpsa_mean': float(tpsa_mean),
            'tpsa_std': float(tpsa_std)
        }
        with open(f'{args.output_dir}/property_norm.json', 'w') as f:
            json.dump(norm_stats, f)
        print(f"🏆 New best model! Loss: {final_loss:.4f} -> Saved to {best_model_path}")
    else:
        print(f"Best loss so far: {best_loss:.4f}")
    
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
