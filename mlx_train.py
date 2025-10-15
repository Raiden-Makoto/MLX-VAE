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
from mlx_utils.mlx_metrics import (
    evaluate_property_prediction,
    evaluate_reconstruction_and_kl,
    evaluate_conditioning_latent,
)

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
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'property_loss': []
        }
        
        self.val_metrics = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'property_loss': []
        }
        
        # =====================================================================
        # Early Stopping
        # =====================================================================
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 30
        self.best_model_weights = None
        # Early stopping configuration
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 30
        self.best_model_weights = None
        
        # Capacity control configuration
        self.C_max = self.model.latent_dim * 0.8  # Target capacity (nats per latent dim)
        self.warmup_epochs = 5                     # Epochs to reach full capacity
        self.gamma = 1.0                          # Deprecated: gamma is now in model config
    
    def load_checkpoint(self, checkpoint_path, silent=False):
        """
        Load model and training state from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint .npz file
            silent: If True, suppress print statements
        
        Returns:
            epoch: The epoch number to resume from
        """
        if not silent:
            print(f"\nLoading checkpoint from {checkpoint_path}...")
        
        if not os.path.exists(checkpoint_path):
            if not silent:
                print(f"  ✗ Checkpoint not found: {checkpoint_path}")
                print("  Starting with fresh model weights")
            return 0
        
        # Load checkpoint using MLX's built-in load_weights
        try:
            self.model.load_weights(checkpoint_path)
            if not silent:
                print(f"  ✓ Model weights loaded successfully")
        
        except Exception as e:
            if not silent:
                print(f"  ✗ Failed to load checkpoint: {e}")
                print("  Starting with fresh model weights")
            return 0
        
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
            
            if not silent:
                print(f"  ✓ Loaded checkpoint from epoch {epoch}")
                print(f"  ✓ Best validation loss: {self.best_val_loss:.4f}")
                print(f"  ✓ Training history restored")
            
            return epoch
        else:
            if not silent:
                print("  ⚠ Metadata not found, only model weights loaded")
            return 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0,
            'reconstruction_loss': 0,
            'kl_loss': 0,
            'property_loss': 0
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
                epoch_losses[key] += loss_dict[key].item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'KL_div': f"{loss_dict['kl_divergence'].item():.4f}",  # Actual KL divergence
                'KL_loss': f"{loss_dict['kl_loss'].item():.4f}",  # |KL - Ct|
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
            'total_loss': 0,
            'reconstruction_loss': 0,
            'kl_loss': 0,
            'property_loss': 0
        }
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Forward pass (no gradients)
            output = self.model(batch)
            loss_dict = self.model.compute_loss(batch, output)
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
            
            num_batches += 1
        
        # Average losses over batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint using MLX's built-in save_weights"""
        # Save model parameters using MLX's proper method
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.npz')
        self.model.save_weights(checkpoint_path)
        
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
        
        # Save latest epoch number (for resuming from most recent epoch)
        latest_epoch_path = os.path.join(self.save_dir, 'latest_epoch.txt')
        with open(latest_epoch_path, 'w') as f:
            f.write(str(epoch))
        
        # Save best model (for final evaluation)
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.npz')
            self.model.save_weights(best_path)
            
            best_metadata_path = best_path.replace('.npz', '_metadata.json')
            with open(best_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"New best model saved! Val loss: {self.best_val_loss:.4f}")
    
    def plot_training_curves(self, save_path=None, show_plot=True):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        
        axes[0, 0].plot(self.train_metrics['total_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(self.val_metrics['total_loss'], label='Val', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # =====================================================================
        # Reconstruction Loss
        # =====================================================================
        
        axes[0, 1].plot(self.train_metrics['reconstruction_loss'], label='Train', alpha=0.8)
        axes[0, 1].plot(self.val_metrics['reconstruction_loss'], label='Val', alpha=0.8)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # =====================================================================
        # KL Divergence
        # =====================================================================
        
        axes[1, 0].plot(self.train_metrics['kl_loss'], label='Train', alpha=0.8)
        axes[1, 0].plot(self.val_metrics['kl_loss'], label='Val', alpha=0.8)
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # =====================================================================
        # Property Prediction Loss
        # =====================================================================
        
        axes[1, 1].plot(self.train_metrics['property_loss'], label='Train', alpha=0.8)
        axes[1, 1].plot(self.val_metrics['property_loss'], label='Val', alpha=0.8)
        axes[1, 1].set_title('Property Prediction Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    
    def train(self, num_epochs=30, start_epoch=1, show_plot=True):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train for
            start_epoch: Epoch to start from (default: 1, useful for resuming training)
            show_plot: Whether to display training curves plot (default: True)
        """
        end_epoch = start_epoch + num_epochs - 1
        print(f"Starting MLXMGCVAE training from epoch {start_epoch} to {end_epoch}")
        
        print("\n" + "="*70)
        print("Training MLXMGCVAE")
        print("="*70)
        
        # Count parameters
        total_params = sum(x.size for k, x in tree_flatten(self.model.parameters()))
        print(f"Model parameters: {total_params:,}")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"\nEpoch {epoch}/{end_epoch}")
            
            # =====================================================================
            # Update Capacity Target
            # =====================================================================
            
            # Compute current capacity target based on schedule
            C_t = self.C_max * min(1.0, epoch / self.warmup_epochs)
            
            # Print capacity status
            if epoch == self.warmup_epochs:
                print(f"  Capacity target: {C_t:.4f} nats (warmup complete, now tracking best model)")
            elif epoch > self.warmup_epochs:
                print(f"  Capacity target: {C_t:.4f} nats")
            else:
                print(f"  Capacity target: {C_t:.4f} nats (warmup phase {epoch}/{self.warmup_epochs})")
            
            # Update model's capacity target
            self.model.current_capacity = C_t
            
            # =====================================================================
            # Training and Validation
            # =====================================================================
            
            train_losses = self.train_epoch()
            val_losses = self.validate_epoch()
            
            # Record metrics
            for key in ['total_loss', 'reconstruction_loss', 'kl_loss', 'property_loss']:
                self.train_metrics[key].append(train_losses[key])
                self.val_metrics[key].append(val_losses[key])
            
            # =====================================================================
            # Early Stopping Check (only after warmup)
            # =====================================================================
            
            # Only start tracking best model when capacity reaches maximum
            if epoch >= self.warmup_epochs:
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.patience_counter = 0
                    # Deep copy of model parameters using tree_map
                    from mlx.utils import tree_map
                    self.best_model_weights = tree_map(lambda x: mx.array(x), self.model.parameters())
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
            else:
                print(f"  Warmup phase: not tracking best model yet")
            
            
            # =====================================================================
            # Regular Checkpointing
            # =====================================================================
            
            # Save checkpoint at end of warmup and every 10 epochs
            if epoch == self.warmup_epochs or epoch % 10 == 0:
                self.save_checkpoint(epoch)
            
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
        self.plot_training_curves(
            os.path.join(self.save_dir, 'training_curves.png'),
            show_plot=show_plot
        )
        
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
    parser.add_argument('--resume', type=str, nargs='?', const='auto', default=None,
                        help='Resume from checkpoint. Use "auto" or "latest" for most recent epoch, or specify checkpoint path')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4 for stable training)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable training curve plots (useful for servers/headless environments)')
    
    args = parser.parse_args()
    
    # Load Dataset
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
    
    # Create Data Loaders
    batch_size = args.batch_size
    train_loader = Dataloader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = Dataloader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = Dataloader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # Initialize Model
    model_config = {
        'node_dim': 24,      # Atom features: 5 atom types (H,C,N,O,F) + 6 degree + 5 charge + 6 hybridization + 1 aromatic + 1 ring
        'edge_dim': 6,       # Bond features
        'latent_dim': 32,
        'hidden_dim': 64,
        'num_properties': 1, # Single property (BBBP)
        'num_layers': 2,
        'heads': 4,
        'max_nodes': 20,
        'beta': 1.0,         # Deprecated: kept for backwards compatibility
        'gamma': 1.0,        # Capacity-controlled KL weight (Loss = Recon + Prop + γ⋅|KL-Ct|)
        'dropout': 0.1
    }
    
    model = MLXMGCVAE(**model_config)
    
    # Initialize Trainer
    trainer = MLXMGCVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        save_dir='checkpoints/mlx_mgcvae'
    )
    
    # Resume from Checkpoint (if specified)
    start_epoch = 1
    if args.resume:
        # Load best model weights for continued training
        if args.resume in ['auto', 'latest']:
            print(f"\nAuto-resuming training...")
            
            # Load best model weights (silently to avoid confusing messages)
            best_model_path = os.path.join(trainer.save_dir, 'best_model.npz')
            best_epoch = trainer.load_checkpoint(best_model_path, silent=True)
            
            # Get the actual epoch number to resume from
            latest_epoch_path = os.path.join(trainer.save_dir, 'latest_epoch.txt')
            if os.path.exists(latest_epoch_path):
                with open(latest_epoch_path, 'r') as f:
                    loaded_epoch = int(f.read().strip())
                print(f"  ✓ Loaded best model weights (from epoch {best_epoch}, val_loss: {trainer.best_val_loss:.4f})")
                print(f"  ✓ Resuming training from epoch {loaded_epoch + 1}")
            else:
                loaded_epoch = 0
                print(f"  ⚠ latest_epoch.txt not found, starting from epoch 1")
        else:
            # Load from specific checkpoint path
            loaded_epoch = trainer.load_checkpoint(args.resume)
        
        start_epoch = loaded_epoch + 1
    
    train_metrics, val_metrics = trainer.train(
        num_epochs=args.epochs,
        start_epoch=start_epoch,
        show_plot=not args.no_plot
    )
    
    # =========================================================================
    # Restore Best Model and Evaluate
    # =========================================================================
    
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70)
    
    # Restore best model weights
    if trainer.best_model_weights is not None:
        print("\n✓ Restoring best model weights...")
        model.update(trainer.best_model_weights)
        print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Test Set Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    batch = next(iter(test_loader))
    output = model(batch)
    test_losses = model.compute_loss(batch, output)
    test_losses = {k: v.item() for k, v in test_losses.items()}
    
    print("\nFinal Test Results:")
    print(f"  Total Loss:         {test_losses['total_loss']:.4f}")
    print(f"  Reconstruction:     {test_losses['reconstruction_loss']:.4f}")
    print(f"  KL Loss:            {test_losses['kl_loss']:.4f}")
    print(f"  Property Loss:      {test_losses['property_loss']:.4f}")
    
    print("\nEvaluating metrics...")
    _ = evaluate_property_prediction(model, test_loader)
    _ = evaluate_reconstruction_and_kl(model, test_loader)
    _ = evaluate_conditioning_latent(model, target=[0.9], num_samples=100, tolerance=0.1)
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)