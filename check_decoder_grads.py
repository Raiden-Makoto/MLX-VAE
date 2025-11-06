#!/usr/bin/env python3
import json
import numpy as np
import mlx.core as mx
from mlx_data.dataloader import MoleculeDataset
from models.vae import ARCVAE
from complete_vae_loss import complete_vae_loss
import mlx.optimizers as optim

# Load data
data_path = 'mlx_data/chembl_cns_selfies.json'
with open(data_path, 'r') as f:
    data = json.load(f)

properties = np.array([[mol['tpsa']] for mol in data['molecules']], dtype=np.float32)
sequences = data['tokenized_sequences']

np.random.seed(67)
indices = np.arange(len(sequences))
np.random.shuffle(indices)

n_train = int(0.8 * len(sequences))
train_indices = indices[:n_train]

train_dataset = MoleculeDataset(
    tokenized_molecules=[sequences[i] for i in train_indices],
    properties=properties[train_indices],
    max_length=data['max_length'],
    pad_token=0
)

# Create model
vae = ARCVAE(
    vocab_size=80,
    embedding_dim=128,
    hidden_dim=256,
    latent_dim=128,
    num_conditions=1,
    num_layers=2,
    dropout=0.2
)

decoder = vae.decoder
decoder_optimizer = optim.Adam(learning_rate=5e-5)

# Get one batch
molecules, conditions = next(iter(train_dataset.to_batches(64, shuffle=False)))
mx.eval(molecules, conditions)

# Define loss function
def model_loss_fn(encoder, decoder, x, conditions):
    loss_dict = complete_vae_loss(
        encoder=encoder,
        decoder=decoder,
        property_predictor=None,
        x=x,
        conditions=conditions,
        beta=0.01,
        lambda_prop=0.1,
        lambda_collapse=0.001,
        teacher_forcing_ratio=0.7,
        free_bits=1.0,
        lambda_mi=0.01,
        target_mi=4.85
    )
    return loss_dict['total_loss']

loss_and_grad_fn = mx.value_and_grad(model_loss_fn, argnums=[0, 1])

# Monitor multiple steps
num_steps = 10
print(f"Monitoring {num_steps} training steps...\n")

decoder_params_before = float(mx.mean(mx.abs(decoder.fc_out.weight)))
print(f"Initial decoder weight (mean abs): {decoder_params_before:.6f}\n")

weight_changes = []
gradient_magnitudes = []
losses = []

for step in range(num_steps):
    # Get fresh batch
    molecules, conditions = next(iter(train_dataset.to_batches(64, shuffle=False)))
    mx.eval(molecules, conditions)
    
    # Before training step
    decoder_params_before_step = float(mx.mean(mx.abs(decoder.fc_out.weight)))
    
    # Compute loss and gradients
    loss, grads = loss_and_grad_fn(vae.encoder, decoder, molecules, conditions)
    mx.eval(grads[0], grads[1])
    
    # Get gradient magnitude
    if isinstance(grads[1], dict) and 'fc_out' in grads[1]:
        decoder_grad_mag = float(mx.mean(mx.abs(grads[1]['fc_out']['weight'])))
    else:
        decoder_grad_mag = 0.0
    
    # Update decoder
    decoder_optimizer.update(decoder, grads[1])
    mx.eval(decoder.parameters(), decoder_optimizer.state)
    
    # After training step
    decoder_params_after_step = float(mx.mean(mx.abs(decoder.fc_out.weight)))
    weight_change = abs(decoder_params_after_step - decoder_params_before_step)
    
    loss_val = float(loss)
    
    weight_changes.append(weight_change)
    gradient_magnitudes.append(decoder_grad_mag)
    losses.append(loss_val)
    
    print(f"Step {step+1}:")
    print(f"  Weight change: {weight_change:.6f}")
    print(f"  Gradient magnitude: {decoder_grad_mag:.6f}")
    print(f"  Loss: {loss_val:.4f}")
    print(f"  Weight (mean abs): {decoder_params_after_step:.6f}")
    
    if weight_change < 1e-6:
        print(f"  ❌ NOT UPDATING!")
    else:
        print(f"  ✅ UPDATING")
    print()

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Total steps: {num_steps}")
print(f"Average weight change: {sum(weight_changes)/len(weight_changes):.6f}")
print(f"Min weight change: {min(weight_changes):.6f}")
print(f"Max weight change: {max(weight_changes):.6f}")
print(f"Average gradient magnitude: {sum(gradient_magnitudes)/len(gradient_magnitudes):.6f}")
print(f"Final weight (mean abs): {decoder_params_after_step:.6f}")
print(f"Total cumulative change: {abs(decoder_params_after_step - decoder_params_before):.6f}")

num_updating = sum(1 for wc in weight_changes if wc >= 1e-6)
print(f"\nSteps updating (change >= 1e-6): {num_updating}/{num_steps}")

if num_updating == num_steps:
    print("  ✅ ALL STEPS UPDATING - Gradients flowing correctly!")
elif num_updating > 0:
    print(f"  ⚠️  PARTIAL UPDATING - {num_updating}/{num_steps} steps updated")
else:
    print("  ❌ NO STEPS UPDATING - Gradients NOT flowing!")

