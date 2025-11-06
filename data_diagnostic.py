#!/usr/bin/env python3
"""
Data diagnostic to verify train/val split and identify divergence issues.
Follows the requested steps exactly.
"""

import json
import numpy as np
import mlx.core as mx
from pathlib import Path

from mlx_data.dataloader import MoleculeDataset
from models.vae import ARCVAE
from complete_vae_loss import complete_vae_loss


def compute_loss_on_subset(model, dataset, num_batches=5):
    """
    Compute average loss on a subset of data
    
    Args:
        model: Dictionary with 'encoder' and 'decoder'
        dataset: MoleculeDataset instance
        num_batches: Number of batches to evaluate
    
    Returns:
        Average loss value
    """
    encoder = model['encoder']
    decoder = model['decoder']
    
    total_loss = 0.0
    batches_processed = 0
    batch_size = 64
    beta = 0.01  # Use a small beta for testing
    
    for batch_idx, (molecules, conditions) in enumerate(dataset.to_batches(batch_size, shuffle=False)):
        if batch_idx >= num_batches:
            break
        
        loss_dict = complete_vae_loss(
            encoder=encoder,
            decoder=decoder,
            property_predictor=None,
            x=molecules,
            conditions=conditions,
            beta=beta,
            lambda_prop=0.1,
            lambda_collapse=0.001,
            teacher_forcing_ratio=0.0,
            free_bits=1.0,
            lambda_mi=0.01,
            target_mi=4.85
        )
        
        mx.eval(loss_dict['total_loss'])
        total_loss += float(loss_dict['total_loss'])
        batches_processed += 1
    
    return total_loss / batches_processed if batches_processed > 0 else 0.0


def main():
    # Configuration (matching train.py defaults)
    data_path = 'mlx_data/chembl_cns_selfies.json'
    vocab_size = 80
    embedding_dim = 128
    hidden_dim = 256
    latent_dim = 128
    num_conditions = 1
    num_layers = 2
    dropout = 0.2
    
    # Fixed splits: 80/10/10
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    
    # Set random seed (fixed to 67)
    np.random.seed(67)
    
    # Load dataset
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    properties = np.array([[mol['tpsa']] for mol in data['molecules']], dtype=np.float32)
    sequences = data['tokenized_sequences']
    
    # Shuffle data with seed
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    
    # Split into train/val/test (80/10/10)
    n_total = len(sequences)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create TRAIN dataset first (computes normalization stats)
    train_dataset = MoleculeDataset(
        tokenized_molecules=[sequences[i] for i in train_indices],
        properties=properties[train_indices],
        max_length=data['max_length'],
        pad_token=0
    )
    
    # CRITICAL: Use train set's normalization stats for val and test
    val_dataset = MoleculeDataset(
        tokenized_molecules=[sequences[i] for i in val_indices],
        properties=properties[val_indices],
        max_length=data['max_length'],
        pad_token=0,
        properties_mean=train_dataset.properties_mean,
        properties_std=train_dataset.properties_std
    )
    
    test_dataset = MoleculeDataset(
        tokenized_molecules=[sequences[i] for i in test_indices],
        properties=properties[test_indices],
        max_length=data['max_length'],
        pad_token=0,
        properties_mean=train_dataset.properties_mean,
        properties_std=train_dataset.properties_std
    )
    
    # Create model
    vae = ARCVAE(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_conditions=num_conditions,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model = {
        'encoder': vae.encoder,
        'decoder': vae.decoder
    }
    
    # Check data split
    print("="*80)
    print("DATA DIAGNOSTICS")
    print("="*80)
    
    # 1. Check sizes
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    print(f"Ratio: {len(train_dataset) / len(val_dataset):.2f}x")
    
    # 2. Check a few samples from each
    print(f"\nTrain sample stats:")
    for i in range(3):
        sample = train_dataset[i]
        print(f"  Sample {i}: mol shape {sample['molecule'].shape}, "
              f"conditions shape {sample['properties'].shape}")
    
    print(f"\nVal sample stats:")
    for i in range(3):
        sample = val_dataset[i]
        print(f"  Sample {i}: mol shape {sample['molecule'].shape}, "
              f"conditions shape {sample['properties'].shape}")
    
    # 3. Check if val_dataset is using teacher_forcing=0.0
    print(f"\nDuring validation, check:")
    print(f"  - Teacher forcing ratio should be 0.0")
    print(f"  - Model should be in eval mode (if applicable)")
    print(f"  - No dropout/batch norm should be active")
    
    # 4. Compare loss on train set vs val set WITH SAME MODEL STATE
    print(f"\nLoss on TRAIN set sample:")
    train_loss = compute_loss_on_subset(model, train_dataset, num_batches=5)
    print(f"  Average: {train_loss:.4f}")
    print(f"Loss on VAL set sample:")
    val_loss = compute_loss_on_subset(model, val_dataset, num_batches=5)
    print(f"  Average: {val_loss:.4f}")
    
    print(f"\nIf these are close: problem is data distribution")
    print(f"If val >> train: problem is data or evaluation")


if __name__ == "__main__":
    main()
