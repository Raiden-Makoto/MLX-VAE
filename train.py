#!/usr/bin/env python3
"""
Training script for AR-CVAE molecular generation
"""

import argparse
import json
import numpy as np
from pathlib import Path
import shutil

from mlx_data.dataloader import MoleculeDataset
from models.vae import ARCVAE
from trainer import ARCVAETrainerWithLoss


def main():
    parser = argparse.ArgumentParser(description='Train AR-CVAE for molecular generation')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='mlx_data/chembl_cns_selfies.json',
                        help='Path to dataset JSON file')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=95, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--num_conditions', type=int, default=1, help='Number of conditions')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta_start', type=float, default=0.0, help='Initial beta value')
    parser.add_argument('--beta_end', type=float, default=0.4, help='Final beta value')
    parser.add_argument('--beta_warmup_epochs', type=int, default=100, help='Beta warmup epochs')
    parser.add_argument('--lambda_prop', type=float, default=0.1, help='Property loss weight')
    parser.add_argument('--lambda_collapse', type=float, default=0.01, help='Posterior collapse weight')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Checkpoint frequency (epochs)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training set fraction')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation set fraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (if not specified, clears old checkpoints)')
    
    args = parser.parse_args()
    
    # Validate splits
    if abs(args.train_split + args.val_split + (1.0 - args.train_split - args.val_split) - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test splits must sum to 1.0")
    
    print("=" * 80)
    print("AR-CVAE Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data}")
    print(f"  Model: embedding={args.embedding_dim}, hidden={args.hidden_dim}, latent={args.latent_dim}")
    print(f"  Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"  Beta: start={args.beta_start}, end={args.beta_end}, warmup={args.beta_warmup_epochs}")
    print(f"  Splits: train={args.train_split:.1f}, val={args.val_split:.1f}, test={1-args.train_split-args.val_split:.1f}")
    print("=" * 80)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load dataset
    print("\nLoading dataset...")
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    properties = np.array([[mol['tpsa']] for mol in data['molecules']], dtype=np.float32)
    sequences = data['tokenized_sequences']
    
    # Shuffle data with seed
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    
    # Split into train/val/test
    n_total = len(sequences)
    n_train = int(args.train_split * n_total)
    n_val = int(args.val_split * n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create datasets
    train_dataset = MoleculeDataset(
        tokenized_molecules=[sequences[i] for i in train_indices],
        properties=properties[train_indices],
        max_length=data['max_length'],
        pad_token=0
    )
    
    val_dataset = MoleculeDataset(
        tokenized_molecules=[sequences[i] for i in val_indices],
        properties=properties[val_indices],
        max_length=data['max_length'],
        pad_token=0
    )
    
    test_dataset = MoleculeDataset(
        tokenized_molecules=[sequences[i] for i in test_indices],
        properties=properties[test_indices],
        max_length=data['max_length'],
        pad_token=0
    )
    
    print(f"✓ Loaded {n_total:,} samples")
    print(f"  - Training: {len(train_dataset):,} samples")
    print(f"  - Validation: {len(val_dataset):,} samples")
    print(f"  - Test: {len(test_dataset):,} samples")
    
    # Handle checkpoint clearing/resuming
    checkpoint_dir = Path(args.checkpoint_dir)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        # Resume from checkpoint
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        
        # Load checkpoint
        checkpoint = np.load(str(checkpoint_path), allow_pickle=True)
        start_epoch = int(checkpoint['epoch']) + 1
        best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
        
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best validation loss so far: {best_val_loss:.4f}")
    else:
        # Clear old checkpoints unless resuming
        if checkpoint_dir.exists():
            print(f"\nClearing old checkpoints in {checkpoint_dir}")
            for checkpoint_file in checkpoint_dir.glob("*.npz"):
                checkpoint_file.unlink()
            # Clear training history plot
            history_plot = checkpoint_dir / "training_history.png"
            if history_plot.exists():
                history_plot.unlink()
            print("✓ Cleared old checkpoints")
    
    # Create VAE model
    print("\nCreating VAE model...")
    vae = ARCVAE(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_conditions=args.num_conditions,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    print("✓ VAE model created")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = ARCVAETrainerWithLoss(
        encoder=vae.encoder,
        decoder=vae.decoder,
        property_predictor=None,  # TODO: Add property predictor
        dataset=train_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_warmup_epochs=args.beta_warmup_epochs,
        lambda_prop=args.lambda_prop,
        lambda_collapse=args.lambda_collapse,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir
    )
    print("✓ Trainer created")
    
    # Load checkpoint if resuming
    if args.resume:
        loaded_epoch = trainer.load_checkpoint(args.resume)
        # Override start_epoch and best_val_loss from checkpoint
        if loaded_epoch >= 0:
            start_epoch = loaded_epoch + 1
        print(f"✓ Loaded checkpoint from epoch {loaded_epoch}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train one epoch
        metrics = trainer.train_epoch(epoch=epoch, total_epochs=args.epochs, val_dataset=val_dataset)
        
        # Store metrics
        trainer.history['epoch'].append(epoch)
        trainer.history['train_loss'].append(metrics['train_loss'])
        trainer.history['train_recon'].append(metrics['train_recon'])
        trainer.history['train_kl'].append(metrics['train_kl'])
        trainer.history['train_collapse'].append(metrics['train_collapse'])
        trainer.history['train_prop'].append(metrics['train_prop'])
        trainer.history['val_loss'].append(metrics['val_loss'])
        trainer.history['val_recon'].append(metrics['val_recon'])
        trainer.history['val_kl'].append(metrics['val_kl'])
        trainer.history['val_collapse'].append(metrics['val_collapse'])
        trainer.history['val_prop'].append(metrics['val_prop'])
        trainer.history['beta'].append(metrics['beta'])
        trainer.history['teacher_forcing'].append(metrics['teacher_forcing'])
        trainer.history['learning_rate'].append(args.learning_rate)
        trainer.history['mutual_info'].append(metrics['mutual_info'])
        
        # Save checkpoint
        is_best = metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = metrics['val_loss']
        
        if (epoch + 1) % args.checkpoint_freq == 0 or is_best:
            trainer.save_checkpoint(epoch=epoch, is_best=is_best)
            trainer.save_history(args.checkpoint_dir)
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {metrics['train_loss']:.4f} (Recon: {metrics['train_recon']:.4f}, KL: {metrics['train_kl']:.4f})")
        print(f"  Beta: {metrics['beta']:.4f}, Teacher Forcing: {metrics['teacher_forcing']:.4f}")
        print(f"  Mutual Info: {metrics['mutual_info']:.4f}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    # Save final plot
    print("\nSaving training history plot...")
    trainer.plot_history(save_path=f"{args.checkpoint_dir}/training_history.png")
    
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()

