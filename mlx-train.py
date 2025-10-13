import os
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mlx_data.mlx_dataset import QM9GraphDataset  # type: ignore
from mlx_models.mlx_vae import MLXMGCVAE  # type: ignore
from mlx_graphs.loaders import Dataloader  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import json
import argparse


class MLXMGCVAETrainer:
    """Training class for MLXMGCVAE with comprehensive logging and checkpointing"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        lr=1e-3,
        save_dir='checkpoints/mlx_mgcvae'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        
        # =====================================================================
        # Setup Directories
        # =====================================================================
        
        os.makedirs(save_dir, exist_ok=True)
        
        # =====================================================================
        # Optimizer
        # =====================================================================
        
        self.optimizer = optim.Adam(learning_rate=lr)
        self.initial_lr = lr
        
        # =====================================================================
        # Training History
        # =====================================================================
        
        self.train_metrics = {
            'reconstruction': [],
            'kl': [],
            'property': [],
            'total': []
        }
        
        self.val_metrics = {
            'reconstruction': [],
            'kl': [],
            'property': [],
            'total': []
        }
        
        # =====================================================================
        # Early Stopping
        # =====================================================================
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 30
        self.best_model_weights = None
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model and training state from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint .npz file
        
        Returns:
            epoch: The epoch number to resume from
        """
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        
        # Load model weights
        checkpoint = mx.load(checkpoint_path)
        self.model.update(checkpoint)
        
        # Load metadata
        metadata_path = checkpoint_path.replace('.npz', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Restore training state
            self.train_metrics = {k: list(v) for k, v in metadata.get('train_metrics', {}).items()}
            self.val_metrics = {k: list(v) for k, v in metadata.get('val_metrics', {}).items()}
            self.best_val_loss = metadata.get('best_val_loss', float('inf'))
            
            epoch = metadata.get('epoch', 0)
            
            print(f"  ✓ Loaded checkpoint from epoch {epoch}")
            print(f"  ✓ Best validation loss: {self.best_val_loss:.4f}")
            print(f"  ✓ Training history restored")
            
            return epoch
        else:
            print("  ⚠ Metadata not found, only model weights loaded")
            return 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0,
            'reconstruction': 0,
            'kl': 0,
            'property': 0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        # Define loss and gradient function that returns all losses
        def compute_loss(batch):
            output = self.model(batch)
            losses = self.model.compute_loss(batch, output)
            return losses['total_loss'], losses
        
        loss_and_grad_fn = nn.value_and_grad(self.model, compute_loss)
        
        for batch in pbar:
            # =====================================================================
            # Forward and Backward Pass (single forward pass!)
            # =====================================================================
            
            (loss, loss_dict), grads = loss_and_grad_fn(batch)
            
            # Update model parameters
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters())
            
            # =====================================================================
            # Accumulate Losses
            # =====================================================================
            
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[f'{key}_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'KL': f"{loss_dict['kl_loss'].item():.4f}",
                'Prop': f"{loss_dict['property_loss'].item():.4f}"
            })
        
        # Average losses over batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = {
            'total': 0,
            'reconstruction': 0,
            'kl': 0,
            'property': 0
        }
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Forward pass (no gradients)
            output = self.model(batch)
            loss_dict = self.model.compute_loss(batch, output)
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[f'{key}_loss'].item()
            num_batches += 1
        
        # Average losses over batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        flat_params = tree_flatten(self.model.parameters())
        
        # Save model parameters
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.npz')
        mx.savez(checkpoint_path, **dict(flat_params))
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'best_val_loss': float(self.best_val_loss),
            'train_metrics': {k: [float(v) for v in vals] for k, vals in self.train_metrics.items()},
            'val_metrics': {k: [float(v) for v in vals] for k, vals in self.val_metrics.items()},
            'model_config': {
                'node_dim': self.model.node_dim,
                'edge_dim': self.model.edge_dim,
                'latent_dim': self.model.latent_dim,
                'num_properties': self.model.num_properties,
                'max_nodes': self.model.max_nodes,
                'beta': float(self.model.beta),
                'gamma': float(self.model.gamma),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = checkpoint_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.npz')
            mx.savez(best_path, **dict(flat_params))
            
            best_metadata_path = best_path.replace('.npz', '_metadata.json')
            with open(best_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"New best model saved! Val loss: {self.best_val_loss:.4f}")
    
    def plot_training_curves(self, save_path=None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        
        axes[0, 0].plot(self.train_metrics['total'], label='Train', alpha=0.8)
        axes[0, 0].plot(self.val_metrics['total'], label='Val', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # =====================================================================
        # Reconstruction Loss
        # =====================================================================
        
        axes[0, 1].plot(self.train_metrics['reconstruction'], label='Train', alpha=0.8)
        axes[0, 1].plot(self.val_metrics['reconstruction'], label='Val', alpha=0.8)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # =====================================================================
        # KL Divergence
        # =====================================================================
        
        axes[1, 0].plot(self.train_metrics['kl'], label='Train', alpha=0.8)
        axes[1, 0].plot(self.val_metrics['kl'], label='Val', alpha=0.8)
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # =====================================================================
        # Property Prediction Loss
        # =====================================================================
        
        axes[1, 1].plot(self.train_metrics['property'], label='Train', alpha=0.8)
        axes[1, 1].plot(self.val_metrics['property'], label='Val', alpha=0.8)
        axes[1, 1].set_title('Property Prediction Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def train(self, num_epochs=30, start_epoch=1):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train for
            start_epoch: Epoch to start from (default: 1, useful for resuming training)
        """
        end_epoch = start_epoch + num_epochs - 1
        print(f"Starting MLXMGCVAE training from epoch {start_epoch} to {end_epoch}")
        
        # Count parameters
        total_params = sum(x.size for k, x in tree_flatten(self.model.parameters()))
        print(f"Model parameters: {total_params:,}")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"\nEpoch {epoch}/{end_epoch}")
            
            # =====================================================================
            # Training and Validation
            # =====================================================================
            
            train_losses = self.train_epoch()
            val_losses = self.validate_epoch()
            
            # Record metrics
            for key in train_losses:
                self.train_metrics[key].append(train_losses[key])
                self.val_metrics[key].append(val_losses[key])
            
            # =====================================================================
            # Early Stopping Check
            # =====================================================================
            
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.best_model_weights = {k: v.copy() for k, v in self.model.parameters().items()}
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # =====================================================================
            # Regular Checkpointing
            # =====================================================================
            
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
            
            # =====================================================================
            # Epoch Summary
            # =====================================================================
            
            print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
            print(f"Recon: {train_losses['reconstruction']:.4f} | KL: {train_losses['kl']:.4f} | Prop: {train_losses['property']:.4f}")
            print(f"LR: {self.optimizer.learning_rate.item():.2e} | Patience: {self.patience_counter}/{self.max_patience}")
            
            # =====================================================================
            # Early Stopping
            # =====================================================================
            
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping after epoch {epoch}")
                break
        
        # =====================================================================
        # Training Completion
        # =====================================================================
        
        self.save_checkpoint(epoch)
        self.plot_training_curves(os.path.join(self.save_dir, 'training_curves.png'))
        
        print("Training completed!")
        return self.train_metrics, self.val_metrics


# =============================================================================
# Main Training Script
# =============================================================================

if __name__ == '__main__':
    # =========================================================================
    # Parse Arguments
    # =========================================================================
    
    parser = argparse.ArgumentParser(description='Train MLXMGCVAE model')
    parser.add_argument('--resume', type=str, nargs='?', const='checkpoints/mlx_mgcvae/best_model.npz', default=None,
                        help='Resume from checkpoint. Optionally specify path, defaults to best_model.npz')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MLXMGCVAE Training Script")
    print("="*70)
    
    if args.resume:
        print(f"\n⚡ Resume mode: Will load from {args.resume}")
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    
    print("\n[1] Loading dataset...")
    dataset = QM9GraphDataset(
        csv_path="mlx_data/qm9_mlx_part2.csv",
        smiles_col="smiles",
        label_col="p_np"
    )
    
    # Split dataset
    train_graphs, other_graphs = train_test_split(
        dataset._graphs, 
        test_size=0.2, 
        random_state=67
    )
    val_graphs, test_graphs = train_test_split(
        other_graphs, 
        test_size=0.5, 
        random_state=67
    )
    
    print(f"  Train dataset size: {len(train_graphs)}")
    print(f"  Validation dataset size: {len(val_graphs)}")
    print(f"  Test dataset size: {len(test_graphs)}")
    
    # =========================================================================
    # Create Data Loaders
    # =========================================================================
    
    print("\n[2] Creating data loaders...")
    batch_size = args.batch_size
    
    train_loader = Dataloader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = Dataloader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = Dataloader(test_graphs, batch_size=batch_size, shuffle=False)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: ~{len(train_graphs) // batch_size + (1 if len(train_graphs) % batch_size else 0)}")
    print(f"  Val batches: ~{len(val_graphs) // batch_size + (1 if len(val_graphs) % batch_size else 0)}")
    print(f"  Test batches: ~{len(test_graphs) // batch_size + (1 if len(test_graphs) % batch_size else 0)}")
    
    # =========================================================================
    # Initialize Model
    # =========================================================================
    
    print("\n[3] Initializing model...")
    
    model_config = {
        'node_dim': 24,      # Atom features: 5 atom types (H,C,N,O,F) + 6 degree + 5 charge + 6 hybridization + 1 aromatic + 1 ring
        'edge_dim': 6,       # Bond features
        'latent_dim': 32,
        'hidden_dim': 64,
        'num_properties': 1, # Single property (BBBP)
        'num_layers': 2,
        'heads': 4,
        'max_nodes': 20,
        'beta': 0.01,        # KL divergence weight
        'gamma': 1.0,        # Property prediction weight
        'dropout': 0.1
    }
    
    model = MLXMGCVAE(**model_config)
    
    print("  Model configuration:")
    for key, value in model_config.items():
        print(f"    {key}: {value}")
    
    # Count parameters
    total_params = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"  Total parameters: {total_params:,}")
    
    # =========================================================================
    # Initialize Trainer
    # =========================================================================
    
    print("\n[4] Setting up trainer...")
    
    trainer = MLXMGCVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        save_dir='checkpoints/mlx_mgcvae'
    )
    
    print(f"  Learning rate: {trainer.initial_lr:.2e}")
    print(f"  Early stopping patience: {trainer.max_patience}")
    print(f"  Save directory: {trainer.save_dir}")
    
    # =========================================================================
    # Resume from Checkpoint (if specified)
    # =========================================================================
    
    start_epoch = 1
    if args.resume:
        print("\n[4.5] Loading checkpoint...")
        loaded_epoch = trainer.load_checkpoint(args.resume)
        start_epoch = loaded_epoch + 1
        print(f"  Resuming from epoch {start_epoch}")
    
    # =========================================================================
    # Train Model
    # =========================================================================
    
    print("\n[5] Starting training...")
    print("="*70)
    
    train_metrics, val_metrics = trainer.train(
        num_epochs=args.epochs,
        start_epoch=start_epoch
    )
    
    # =========================================================================
    # Restore Best Model and Evaluate
    # =========================================================================
    
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70)
    
    if trainer.best_model_weights is not None:
        print("\n✓ Restoring best model weights...")
        model.update(trainer.best_model_weights)
        print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    
    # =========================================================================
    # Test Set Evaluation
    # =========================================================================
    
    print("\nEvaluating on test set...")
    model.eval()
    
    test_losses = {
        'total': 0,
        'reconstruction': 0,
        'kl': 0,
        'property': 0
    }
    test_batches = 0
    
    for batch in tqdm(test_loader, desc='Testing'):
        output = model(batch)
        loss_dict = model.compute_loss(batch, output)
        
        for key in test_losses:
            test_losses[key] += loss_dict[f'{key}_loss'].item()
        test_batches += 1
    
    # Average test losses
    for key in test_losses:
        test_losses[key] /= test_batches
    
    print("\nFinal Test Results:")
    print(f"  Total Loss:         {test_losses['total']:.4f}")
    print(f"  Reconstruction:     {test_losses['reconstruction']:.4f}")
    print(f"  KL Divergence:      {test_losses['kl']:.4f}")
    print(f"  Property Loss:      {test_losses['property']:.4f}")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
