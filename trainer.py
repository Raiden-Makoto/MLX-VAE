import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class MGCVAETrainer:
    """Training class for MGCVAE with comprehensive logging and checkpointing"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        lr=1e-3,
        device='cpu',
        save_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        # =====================================================================
        # Setup Directories
        # =====================================================================
        
        os.makedirs(save_dir, exist_ok=True)
        
        # =====================================================================
        # Optimizer and Scheduler
        # =====================================================================
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=15
        )
        
        # =====================================================================
        # Training History
        # =====================================================================
        
        self.train_losses = []
        self.val_losses = []
        
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
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # =====================================================================
            # Forward Pass
            # =====================================================================
            
            self.optimizer.zero_grad()
            model_output = self.model(batch)
            loss_dict = self.model.compute_loss(batch, model_output)
            
            # =====================================================================
            # Backward Pass
            # =====================================================================
            
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
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
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = batch.to(self.device)
                
                # Forward pass (no gradients)
                model_output = self.model(batch)
                loss_dict = self.model.compute_loss(batch, model_output)
                
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
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'node_dim': self.model.node_dim,
                'edge_dim': self.model.edge_dim,
                'latent_dim': self.model.latent_dim,
                'hidden_dim': self.model.encoder.hidden_dim,
                'num_properties': self.model.num_properties,
                'num_layers': self.model.encoder.num_layers,
                'heads': self.model.encoder.heads,
                'max_nodes': self.model.max_nodes,
                'beta': self.model.beta,
                'gamma': self.model.gamma,
                'dropout': self.model.encoder.dropout
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
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
    
    
    def train(self, num_epochs=30):
        """Full training loop"""
        print(f"Starting MGCVAE training for {num_epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
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
            # Learning Rate Scheduling
            # =====================================================================
            
            self.scheduler.step(val_losses['total'])
            
            # =====================================================================
            # Early Stopping Check
            # =====================================================================
            
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch + 1, is_best=True)
            else:
                self.patience_counter += 1
            
            # =====================================================================
            # Regular Checkpointing
            # =====================================================================
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
            
            # =====================================================================
            # Epoch Summary
            # =====================================================================
            
            print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
            print(f"Recon: {train_losses['reconstruction']:.4f} | KL: {train_losses['kl']:.4f} | Prop: {train_losses['property']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | Patience: {self.patience_counter}/{self.max_patience}")
            
            # =====================================================================
            # Early Stopping
            # =====================================================================
            
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # =====================================================================
        # Training Completion
        # =====================================================================
        
        self.save_checkpoint(epoch + 1)
        self.plot_training_curves(os.path.join(self.save_dir, 'training_curves.png'))
        
        print("Training completed!")
        return self.train_metrics, self.val_metrics