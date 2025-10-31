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

# REINFORCE imports
try:
    from utils.reinforce_loss import REINFORCELoss
    REINFORCE_AVAILABLE = True
except ImportError:
    REINFORCE_AVAILABLE = False
    print("Warning: REINFORCE loss not available")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
 # output dir fixed in code to 'checkpoints'
parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Epochs to warm up beta from 0 to 1')
parser.add_argument('--max_beta', type=float, default=1.0, help='Maximum beta value (CVAE best practice: 1.0 for full KL divergence)')
parser.add_argument('--latent_noise_std', type=float, default=0.05, help='Standard deviation of Gaussian noise added to latent vectors during training')
parser.add_argument('--diversity_weight', type=float, default=0.01, help='Weight for latent diversity loss')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--resume', action='store_true', help='Resume training from best model and last epoch')
parser.add_argument('--property_weight', type=float, default=100.0, help='Weight for property reconstruction loss (predict TPSA from z+FILM)')
parser.add_argument('--policy_weight_max', type=float, default=10.0, help='Maximum policy weight (curriculum learning)')
parser.add_argument('--aux_weight', type=float, default=5.0, help='Weight for auxiliary TPSA head MSE (normalized)')
 # RL sampling and curriculum use fixed defaults in code (no CLI flags)
 # Predictor training moved to train_predictor.py to keep this file focused

# Load data and metadata
with open('mlx_data/chembl_cns_selfies.json', 'r') as f:
    meta = json.load(f)

tokenized = np.load('mlx_data/chembl_cns_tokenized.npy')
token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
vocab_size = meta['vocab_size']
max_length = meta['max_length']

# Load properties - TPSA only (dataset stores SELFIES + TPSA)
molecules = meta['molecules']
properties_raw_tpsa = np.array([mol.get('tpsa', 82.0) for mol in molecules], dtype=np.float32)

# Normalize properties (zero mean, unit variance)
# LogP placeholder: zero-mean unit-std to avoid impacting conditioning
logp_mean = 0.0
logp_std = 1.0
tpsa_mean = np.mean(properties_raw_tpsa)
tpsa_std = np.std(properties_raw_tpsa)

# Store raw properties - model will normalize them internally
# First column: LogP (placeholder, zeros), Second column: TPSA (raw)
properties = np.stack([
    np.zeros_like(properties_raw_tpsa, dtype=np.float32),
    properties_raw_tpsa.astype(np.float32)
], axis=1)

args = parser.parse_args()

# Always use REINFORCE
USE_REINFORCE = True

# Model architecture is fixed in code; no CLI for heads/layers

# Setup REINFORCE
reinforce = None
if USE_REINFORCE and REINFORCE_AVAILABLE:
    from selfies import decoder as sf_decoder
    
    # Create vocab_to_selfies mapping: token_id (int) -> SELFIES token string
    vocab_to_selfies = {}
    for token_idx_str, token_str in idx_to_token.items():
        vocab_to_selfies[int(token_idx_str)] = token_str
    
    # Create selfies_to_smiles function
    def selfies_to_smiles(selfies_str):
        try:
            return sf_decoder(selfies_str)
        except:
            return None
    
    # Fixed REINFORCE decoding/reward shaping defaults
    reinforce = REINFORCELoss(
        vocab_to_selfies,
        selfies_to_smiles,
        temperature=1.1,
        top_k=30,
        use_sampling=True,
        tol_band=10.0,
        reward_scale=1.0,
    )
    print("✅ REINFORCE loss initialized")
elif USE_REINFORCE:
    print("❌ REINFORCE requested but not available. Falling back to property loss.")
    USE_REINFORCE = False

# Prepare data with TPSA stratified sampling (always on); fixed constants
BINS = 5
PER_BIN = 4000
# Build bins on raw TPSA (un-normalized)
tpsa_vals = properties_raw_tpsa
bin_edges = np.linspace(tpsa_vals.min(), tpsa_vals.max(), BINS + 1)
# bin ids in [0..BINS-1]
bin_ids = np.digitize(tpsa_vals, bin_edges[1:-1], right=False)
balanced_indices = []
rng = np.random.default_rng(42)
for b in range(BINS):
    bin_idx = np.where(bin_ids == b)[0]
    if bin_idx.size == 0:
        continue
    replace = bin_idx.size < PER_BIN
    take = rng.choice(bin_idx, size=PER_BIN, replace=replace)
    balanced_indices.append(take)
if balanced_indices:
    balanced_indices = np.concatenate(balanced_indices, axis=0)
    tokenized_sel = tokenized[balanced_indices]
    properties_sel = properties[balanced_indices]
    print(f"TPSA stratified sampling: bins={BINS}, per_bin={PER_BIN}, total={len(balanced_indices)}")
else:
    tokenized_sel = tokenized
    properties_sel = properties
    print("TPSA stratified sampling found no bins; using full dataset")

tokenized_mx = mx.array(tokenized_sel)
properties_mx = mx.array(properties_sel)

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
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 256
print(f"Initializing Transformer VAE with embedding_dim={EMBEDDING_DIM}, hidden_dim={HIDDEN_DIM}, latent_dim={LATENT_DIM}")
print(f"Transformer config: dropout={args.dropout}")

model = SelfiesTransformerVAE(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    dropout=args.dropout
)
model.set_property_normalization(logp_mean, logp_std, tpsa_mean, tpsa_std)
optimizer = Adam(learning_rate=args.learning_rate)

# Output dir (fixed)
OUTPUT_DIR = 'checkpoints'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resume
start_epoch = 0
if args.resume:
    print(f"Resuming training...")
    last_epoch_file = os.path.join(OUTPUT_DIR, 'last_epoch.txt')
    if os.path.exists(last_epoch_file):
        with open(last_epoch_file, 'r') as f:
            last_completed_epoch = int(f.read().strip())
            start_epoch = last_completed_epoch + 1
        print(f"Resuming from epoch {start_epoch} (last completed: {last_completed_epoch})")
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.npz')
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from: {best_model_path}")
        model.load_weights(best_model_path)
        norm_file = os.path.join(OUTPUT_DIR, 'property_norm.json')
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
best_model_path = os.path.join(OUTPUT_DIR, 'best_model.npz')

# Training step
def train_step(model, batch_data, batch_properties, optimizer, beta, noise_std=0.05, diversity_weight=0.01, 
               property_weight=0.0, use_reinforce=False, reinforce_loss=None, policy_weight=0.0, aux_weight: float=0.0):
    stored = {}
    PROPERTY_ENCODER_WEIGHT = 10.0
    def loss_fn(model):
        outputs = model(batch_data, properties=batch_properties, training=True, noise_std=noise_std)
        # Backward compat: handle 3- or 4-tuple returns
        if len(outputs) == 4:
            logits, mu, logvar, aux_pred = outputs
        else:
            logits, mu, logvar = outputs
            aux_pred = None
        
        # Standard VAE losses
        total, recon, kl, diversity, prop_loss = compute_loss(
            batch_data, logits, mu, logvar, beta, diversity_weight,
            target_tpsa_raw=None, property_weight=0.0  # Disable property loss if using REINFORCE
        )
        
        stored['recon'] = float(recon.item())
        stored['kl'] = float(kl.item())
        stored['diversity'] = float(diversity.item())
        stored['aux'] = 0.0
        stored['logp_recon'] = 0.0
        stored['tpsa_recon'] = 0.0
        
        # REINFORCE policy gradient loss (if enabled)
        if use_reinforce and reinforce_loss is not None:
            # Extract target TPSA
            target_tpsa_raw = None
            if batch_properties is not None and batch_properties.shape[1] >= 2:
                if isinstance(batch_properties, mx.array):
                    target_tpsa_raw = np.array(batch_properties[:, 1])  # [B] raw TPSA values
                else:
                    target_tpsa_raw = batch_properties[:, 1]  # [B] raw TPSA values
            
            if target_tpsa_raw is not None:
                policy_loss, reward, valid_mask = reinforce_loss(logits, target_tpsa_raw)
                stored['policy'] = float(policy_loss.item())
                stored['reward'] = float(mx.mean(reward).item()) if hasattr(mx.mean(reward), 'item') else float(mx.mean(reward))
                stored['valid_pct'] = float(mx.mean(valid_mask.astype(mx.float32)).item())
                total = total + policy_weight * policy_loss
            else:
                stored['policy'] = 0.0
                stored['reward'] = 0.0
                stored['valid_pct'] = 0.0
            stored['property'] = 0.0  # Not used with REINFORCE
        else:
            # Use standard property loss
            target_tpsa_raw = None
            if batch_properties is not None and batch_properties.shape[1] >= 2:
                if isinstance(batch_properties, mx.array):
                    target_tpsa_raw = np.array(batch_properties[:, 1])
                else:
                    target_tpsa_raw = batch_properties[:, 1]
            _, _, _, _, prop_loss = compute_loss(
                batch_data, logits, mu, logvar, beta, diversity_weight,
                target_tpsa_raw=target_tpsa_raw, property_weight=property_weight
            )
            stored['property'] = float(prop_loss.item())
            stored['policy'] = 0.0
            stored['reward'] = 0.0
            stored['valid_pct'] = 0.0
        
        # Auxiliary normalized TPSA MSE from aux head (hybrid signal)
        if aux_weight > 0.0 and aux_pred is not None and batch_properties is not None and batch_properties.shape[1] >= 2:
            if isinstance(batch_properties, mx.array):
                target_raw = np.array(batch_properties[:, 1:2])
            else:
                target_raw = batch_properties[:, 1:2]
            # Normalize to match model conditioning
            t_mean = model.tpsa_mean if model.tpsa_mean is not None else 0.0
            t_std = model.tpsa_std if model.tpsa_std is not None and model.tpsa_std != 0 else 1.0
            target_norm = (mx.array(target_raw) - t_mean) / t_std
            aux_mse = mx.mean((aux_pred - target_norm) ** 2)
            stored['aux'] = float(aux_mse.item())
            total = total + aux_weight * aux_mse

        # Direct supervision for property encoders (reconstruct normalized properties from embeddings)
        if batch_properties is not None and batch_properties.shape[1] >= 2:
            # Prepare normalized targets
            logp_target = batch_properties[:, 0:1]
            tpsa_target = batch_properties[:, 1:2]
            lp_mean = model.logp_mean if model.logp_mean is not None else 0.0
            lp_std = model.logp_std if model.logp_std is not None and model.logp_std != 0 else 1.0
            tp_mean = model.tpsa_mean if model.tpsa_mean is not None else 0.0
            tp_std = model.tpsa_std if model.tpsa_std is not None and model.tpsa_std != 0 else 1.0
            logp_norm = (logp_target - lp_mean) / lp_std
            tpsa_norm = (tpsa_target - tp_mean) / tp_std

            # Compute embeddings (ensure gradients flow)
            if isinstance(logp_norm, mx.array):
                lp_in = logp_norm
                tp_in = tpsa_norm
            else:
                lp_in = mx.array(logp_norm)
                tp_in = mx.array(tpsa_norm)
            logp_emb = model.property_encoder_logp(lp_in)
            tpsa_emb = model.property_encoder_tpsa(tp_in)

            # Reconstruct normalized properties from embeddings
            logp_pred = model.logp_reconstructor(logp_emb)
            tpsa_pred = model.tpsa_reconstructor(tpsa_emb)
            logp_rec_loss = mx.mean((logp_pred - lp_in) ** 2)
            tpsa_rec_loss = mx.mean((tpsa_pred - tp_in) ** 2)
            enc_rec = logp_rec_loss + tpsa_rec_loss
            stored['logp_recon'] = float(logp_rec_loss.item())
            stored['tpsa_recon'] = float(tpsa_rec_loss.item())
            total = total + PROPERTY_ENCODER_WEIGHT * enc_rec
        
        return total
    total_loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return (total_loss, stored['recon'], stored['kl'], stored['diversity'], 
            stored.get('property', 0.0), stored.get('policy', 0.0), 
            stored.get('reward', 0.0), stored.get('valid_pct', 0.0),
            stored.get('aux', 0.0))

# Validation
def validate(model, val_batches, beta, noise_std=0.05, diversity_weight=0.01, property_weight=0.0):
    total_val_loss = total_recon = total_kl = total_div = total_prop = 0.0
    for batch_data, properties in val_batches:
        outputs = model(batch_data, properties=properties, training=False, noise_std=noise_std)
        if len(outputs) == 4:
            logits, mu, logvar, _ = outputs
        else:
            logits, mu, logvar = outputs
        # Extract raw TPSA target (for actual TPSA computation from decoded molecules)
        target_tpsa_raw = None
        if properties is not None and properties.shape[1] >= 2:
            if isinstance(properties, mx.array):
                target_tpsa_raw = np.array(properties[:, 1])  # [B] raw TPSA values
            else:
                target_tpsa_raw = properties[:, 1]  # [B] raw TPSA values
        total, recon, kl, div, prop = compute_loss(
            batch_data, logits, mu, logvar, beta, diversity_weight,
            target_tpsa_raw=target_tpsa_raw, property_weight=property_weight
        )
        total_val_loss += total.item(); total_recon += recon.item(); total_kl += kl.item(); total_div += div.item()
        total_prop += prop.item()
    n = len(val_batches)
    return total_val_loss/n, total_recon/n, total_kl/n, total_div/n, total_prop/n

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
if USE_REINFORCE:
    print(f"✅ REINFORCE enabled: Policy weight max={args.policy_weight_max} (curriculum learning)")
else:
    print(f"Property weight: {args.property_weight} (forces decoder to respect z+FILM)")
print("="*67)

for epoch in range(start_epoch, total_epochs):
    beta = get_beta(epoch, args.epochs, args.beta_warmup_epochs, args.max_beta)
    
    # Curriculum learning for policy weight (ramp up over epochs)
    if USE_REINFORCE:
        epoch_ratio = epoch / max(total_epochs - 1, 1)
        policy_weight = min(args.policy_weight_max * epoch_ratio, args.policy_weight_max)
    else:
        policy_weight = 0.0
    
    epoch_losses, epoch_recon, epoch_kl, epoch_prop, epoch_policy, epoch_reward, epoch_valid, epoch_aux = [], [], [], [], [], [], [], []
    progress = tqdm.tqdm(train_batches, desc=f"Epoch {epoch+1}/{total_epochs} (β={beta:.3f})", unit="batch", leave=False)
    for batch_idx, (batch_data, batch_properties) in enumerate(progress):
        result = train_step(model, batch_data, batch_properties, optimizer, beta, args.latent_noise_std, 
                           args.diversity_weight, args.property_weight, USE_REINFORCE, reinforce, policy_weight, args.aux_weight)
        total, recon, kl, div, prop, policy, reward, valid_pct, aux = result
        epoch_losses.append(total); epoch_recon.append(recon); epoch_kl.append(kl)
        epoch_prop.append(prop); epoch_policy.append(policy); epoch_reward.append(reward); epoch_valid.append(valid_pct); epoch_aux.append(aux)
        
        if batch_idx % 10 == 0:
            avg_loss = mx.mean(mx.array(epoch_losses)).item()
            avg_recon = mx.mean(mx.array(epoch_recon)).item()
            avg_kl = mx.mean(mx.array(epoch_kl)).item()
            if USE_REINFORCE:
                avg_policy = mx.mean(mx.array(epoch_policy)).item() if epoch_policy else 0.0
                avg_reward = mx.mean(mx.array(epoch_reward)).item() if epoch_reward else 0.0
                avg_valid = mx.mean(mx.array(epoch_valid)).item() if epoch_valid else 0.0
                avg_aux = mx.mean(mx.array(epoch_aux)).item() if epoch_aux else 0.0
                progress.set_postfix({ 
                    'Loss': f'{avg_loss:.4f}', 'Recon': f'{avg_recon:.4f}', 'KL': f'{avg_kl:.4f}',
                    'Policy': f'{avg_policy:.4f}', 'Reward': f'{avg_reward:.2f}', 'Valid': f'{avg_valid*100:.1f}%',
                    'Aux': f'{avg_aux:.4f}'
                })
            else:
                avg_prop = mx.mean(mx.array(epoch_prop)).item() if epoch_prop else 0.0
                progress.set_postfix({ 
                    'Loss': f'{avg_loss:.4f}', 'Recon': f'{avg_recon:.4f}', 'KL': f'{avg_kl:.4f}', 'Prop': f'{avg_prop:.4f}' 
                })
    
    final_loss = mx.mean(mx.array(epoch_losses)).item()
    final_recon = mx.mean(mx.array(epoch_recon)).item()
    final_kl = mx.mean(mx.array(epoch_kl)).item()
    # Run validation on the held-out split
    val_total, val_recon, val_kl, val_div, val_prop = validate(
        model,
        val_batches,
        beta,
        noise_std=args.latent_noise_std,
        diversity_weight=args.diversity_weight,
        property_weight=0.0,
    )
    
    if USE_REINFORCE:
        final_policy = mx.mean(mx.array(epoch_policy)).item() if epoch_policy else 0.0
        final_reward = mx.mean(mx.array(epoch_reward)).item() if epoch_reward else 0.0
        final_valid = mx.mean(mx.array(epoch_valid)).item() if epoch_valid else 0.0
        final_aux = mx.mean(mx.array(epoch_aux)).item() if epoch_aux else 0.0
        print(f"\nEpoch {epoch+1}: Recon={final_recon:.4f} KL={final_kl:.4f} Policy={final_policy:.4f} Reward={final_reward:.2f} Valid={final_valid*100:.1f}% Aux={final_aux:.4f} Total={final_loss:.4f}")
        print(f"Val: Total={val_total:.4f} Recon={val_recon:.4f} KL={val_kl:.4f} Div={val_div:.4f}")
    else:
        final_prop = mx.mean(mx.array(epoch_prop)).item() if epoch_prop else 0.0
        print(f"\nEpoch {epoch+1}: Recon={final_recon:.4f} KL={final_kl:.4f} Prop={final_prop:.4f} Total={final_loss:.4f}")
        print(f"Val: Total={val_total:.4f} Recon={val_recon:.4f} KL={val_kl:.4f} Div={val_div:.4f} Prop={val_prop:.4f}")
    # Save best
    # Track best by validation total loss
    if val_total < best_loss:
        best_loss = val_total
        model.save_weights(best_model_path)
        norm_stats = {
            'logp_mean': float(logp_mean), 'logp_std': float(logp_std),
            'tpsa_mean': float(tpsa_mean), 'tpsa_std': float(tpsa_std),
            'embedding_dim': EMBEDDING_DIM, 'hidden_dim': HIDDEN_DIM,
            'latent_dim': LATENT_DIM, 'dropout': args.dropout
        }
        with open(f'{OUTPUT_DIR}/property_norm.json', 'w') as f:
            json.dump(norm_stats, f)
        print(f"New best model! Val Loss: {val_total:.4f} -> Saved to {best_model_path}")
    print("="*67)
    with open(os.path.join(OUTPUT_DIR, 'last_epoch.txt'), 'w') as f:
        f.write(str(epoch))

print("Training completed!")

# Predictor training removed from this file; use train_predictor.py
