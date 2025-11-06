import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import json
from pathlib import Path
from tqdm import tqdm
from complete_vae_loss import complete_vae_loss


class ARCVAETrainerWithLoss:
    """
    Complete trainer for AR-CVAE using modularized loss functions
    
    Integrates with your complete_vae_loss function that uses
    separate loss modules: reconstruction, KL, posterior collapse, property prediction
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        property_predictor: Optional[nn.Module],
        dataset,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        beta_start: float = 0.0,
        beta_end: float = 0.4,
        beta_warmup_epochs: int = 100,
        lambda_prop: float = 0.1,
        lambda_collapse: float = 0.01,
        free_bits: float = 0.5,
        lambda_mi: float = 0.01,
        grad_clip: float = 1.0,
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize trainer
        
        Args:
            encoder: MLXEncoder instance
            decoder: MLXAutoregressiveDecoder instance
            property_predictor: Optional property prediction network
            dataset: MoleculeDataset instance
            learning_rate: Adam learning rate
            batch_size: Training batch size
            beta_start: Initial KL weight
            beta_end: Final KL weight
            beta_warmup_epochs: Epochs to warm up beta
            lambda_prop: Weight for property prediction loss
            lambda_collapse: Weight for posterior collapse penalty
            grad_clip: Gradient clipping norm
            checkpoint_dir: Directory for saving checkpoints
        """
        self.encoder = encoder
        self.decoder = decoder
        self.property_predictor = property_predictor
        self.dataset = dataset
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        
        # Loss weights
        self.lambda_prop = lambda_prop
        self.lambda_collapse = lambda_collapse
        self.free_bits = free_bits
        self.lambda_mi = lambda_mi
        
        # Beta annealing
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_epochs = beta_warmup_epochs
        
        # Optimizers (separate for encoder and decoder to avoid state conflicts)
        self.encoder_optimizer = optim.Adam(learning_rate=learning_rate)
        self.decoder_optimizer = optim.Adam(learning_rate=learning_rate)
        self.learning_rate = learning_rate
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'train_collapse': [],
            'train_prop': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'val_collapse': [],
            'val_prop': [],
            'beta': [],
            'teacher_forcing': [],
            'learning_rate': [],
            'mutual_info': []
        }
    
    def compute_beta(self, epoch: int) -> float:
        """Compute beta value with linear annealing"""
        if epoch < self.beta_warmup_epochs:
            beta = self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.beta_warmup_epochs)
        else:
            beta = self.beta_end
        return float(beta)
    
    def compute_teacher_forcing_ratio(self, epoch: int, total_epochs: int) -> float:
        """Compute teacher forcing ratio with decay"""
        progress = epoch / total_epochs
        ratio = max(0.5, 0.9 - 0.4 * progress)  # Start at 0.9, decay to 0.5 over all epochs
        return float(ratio)
    
    def _compute_true_train_loss(self, epoch: int, num_batches: int = 10) -> Dict[str, float]:
        """
        Compute training loss WITHOUT teacher forcing for fair comparison
        
        This gives the true generative performance, not inflated by TF.
        """
        beta = self.compute_beta(epoch)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_collapse = 0.0
        total_prop = 0.0
        num_batches_actual = 0
        
        for batch_idx, (molecules, conditions) in enumerate(
            self.dataset.to_batches(self.batch_size, shuffle=False)
        ):
            if batch_idx >= num_batches:
                break
            
            mx.eval(molecules, conditions)
            
            # CRITICAL: teacher_forcing_ratio = 0.0
            loss_dict = complete_vae_loss(
                encoder=self.encoder,
                decoder=self.decoder,
                property_predictor=self.property_predictor,
                x=molecules,
                conditions=conditions,
                beta=beta,
                lambda_prop=self.lambda_prop,
                lambda_collapse=self.lambda_collapse,
                teacher_forcing_ratio=0.0,  # No teacher forcing!
                free_bits=self.free_bits,
                lambda_mi=self.lambda_mi,
                target_mi=4.85
            )
            
            mx.eval(
                loss_dict['total_loss'],
                loss_dict['recon_loss'],
                loss_dict['kl_loss'],
                loss_dict['collapse_penalty'],
                loss_dict['prop_loss']
            )
            
            total_loss += float(loss_dict['total_loss'])
            total_recon += float(loss_dict['recon_loss'])
            total_kl += float(loss_dict['kl_loss'])
            total_collapse += float(loss_dict['collapse_penalty'])
            total_prop += float(loss_dict['prop_loss'])
            num_batches_actual += 1
        
        return {
            'loss': total_loss / num_batches_actual if num_batches_actual > 0 else 0.0,
            'recon': total_recon / num_batches_actual if num_batches_actual > 0 else 0.0,
            'kl': total_kl / num_batches_actual if num_batches_actual > 0 else 0.0,
            'collapse': total_collapse / num_batches_actual if num_batches_actual > 0 else 0.0,
            'prop': total_prop / num_batches_actual if num_batches_actual > 0 else 0.0
        }
    
    def train_epoch(
        self,
        epoch: int,
        total_epochs: int,
        val_dataset: Optional = None
    ) -> Dict[str, float]:
        """
        Single training epoch using complete_vae_loss
        
        Args:
            epoch: Current epoch number
            total_epochs: Total epochs for teacher forcing schedule
            val_dataset: Optional validation dataset
        
        Returns:
            Dictionary with epoch metrics
        """
        # Compute schedules
        beta = self.compute_beta(epoch)
        teacher_forcing_ratio = self.compute_teacher_forcing_ratio(epoch, total_epochs)
        
        # Train with teacher forcing (for learning)
        train_metrics = self._train_epoch_batches(beta, teacher_forcing_ratio)
        
        # Compute true training loss (without TF) for fair comparison
        true_train_metrics = self._compute_true_train_loss(epoch, num_batches=20)
        
        # Validation (already no TF)
        if val_dataset is not None:
            val_metrics = self._validate(val_dataset, beta)
        else:
            val_metrics = {
                'loss': 0.0,
                'recon': 0.0,
                'kl': 0.0,
                'collapse': 0.0,
                'prop': 0.0,
                'mutual_info': 0.0
            }
        
        # Compute mutual information for monitoring
        mu, logvar = self._get_latent_stats()
        mi = self._compute_mutual_information(mu, logvar)
        mx.eval(mi)
        mi_value = float(mi)
        
        # Return comparable metrics (both without teacher forcing)
        metrics = {
            'train_loss': true_train_metrics['loss'],  # Now fair!
            'train_recon': true_train_metrics['recon'],
            'train_kl': true_train_metrics['kl'],
            'train_collapse': true_train_metrics['collapse'],
            'train_prop': true_train_metrics['prop'],
            'val_loss': val_metrics.get('loss', 0.0),
            'val_recon': val_metrics.get('recon', 0.0),
            'val_kl': val_metrics.get('kl', 0.0),
            'val_collapse': val_metrics.get('collapse', 0.0),
            'val_prop': val_metrics.get('prop', 0.0),
            'beta': beta,
            'teacher_forcing': teacher_forcing_ratio,
            'mutual_info': mi_value
        }
        
        return metrics
    
    def _train_epoch_batches(
        self,
        beta: float,
        teacher_forcing_ratio: float
    ) -> Dict[str, float]:
        """
        Train on all batches using complete_vae_loss
        
        Args:
            beta: KL divergence weight
            teacher_forcing_ratio: Teacher forcing probability
        
        Returns:
            Average losses for the epoch
        """
        total_loss = 0.0
        num_batches = 0
        # For component tracking (sampled periodically to reduce cost)
        recon_sum = 0.0
        kl_sum = 0.0
        collapse_sum = 0.0
        prop_sum = 0.0
        comp_count = 0
        
        # Create loss function with decoder teacher forcing
        def model_loss_fn(encoder, decoder, x, conditions):
            """
            Loss function using complete_vae_loss
            
            Wraps complete_vae_loss to work with MLX's value_and_grad
            """
            # Get loss dict
            loss_dict = complete_vae_loss(
                encoder=encoder,
                decoder=decoder,
                property_predictor=self.property_predictor,
                x=x,
                conditions=conditions,
                beta=beta,
                lambda_prop=self.lambda_prop,
                lambda_collapse=self.lambda_collapse,
                teacher_forcing_ratio=teacher_forcing_ratio,
                free_bits=self.free_bits,
                lambda_mi=self.lambda_mi,
                target_mi=4.85  # Standard target MI value
            )
            
            return loss_dict['total_loss']
        
        # Create gradient function
        loss_and_grad_fn = mx.value_and_grad(model_loss_fn, argnums=[0, 1])
        
        num_batches_total = len(self.dataset) // self.batch_size
        
        # Use tqdm for progress bar
        pbar = tqdm(
            self.dataset.to_batches(self.batch_size, shuffle=True),
            total=num_batches_total,
            desc="Training batches"
        )
        
        for batch_idx, (molecules, conditions) in enumerate(pbar):
            # Compute loss and gradients (single forward pass)
            loss, grads = loss_and_grad_fn(
                self.encoder,
                self.decoder,
                molecules,
                conditions
            )
            
            # Gradient clipping (if enabled)
            if self.grad_clip > 0:
                clipped_grads = self._clip_gradients(grads, self.grad_clip)
            else:
                clipped_grads = grads
            
            # Update encoder (grads[0])
            if clipped_grads is not None and len(clipped_grads) > 0:
                self.encoder_optimizer.update(self.encoder, clipped_grads[0])
            
            # Update decoder (grads[1])
            if clipped_grads is not None and len(clipped_grads) > 1:
                self.decoder_optimizer.update(self.decoder, clipped_grads[1])
            
            # Evaluate everything at once (more efficient)
            mx.eval(
                loss,
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.encoder_optimizer.state,
                self.decoder_optimizer.state
            )
            
            # Sample loss components periodically for tracking (without printing)
            if batch_idx == 0 or batch_idx % 25 == 0:
                loss_dict = complete_vae_loss(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    property_predictor=self.property_predictor,
                    x=molecules,
                    conditions=conditions,
                    beta=beta,
                    lambda_prop=self.lambda_prop,
                    lambda_collapse=self.lambda_collapse,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    free_bits=self.free_bits,
                    lambda_mi=self.lambda_mi,
                    target_mi=4.85
                )
                mx.eval(loss_dict.values())
                
                recon = float(loss_dict['recon_loss'])
                kl = float(loss_dict['kl_loss'])
                collapse = float(loss_dict['collapse_penalty'])
                prop = float(loss_dict['prop_loss'])
                
                # Accumulate sampled components
                recon_sum += recon
                kl_sum += kl
                collapse_sum += collapse
                prop_sum += prop
                comp_count += 1
            
            # Accumulate loss
            loss_val = float(loss)
            
            # SANITY CHECK: Catch explosions and negative losses
            if not np.isfinite(loss_val) or loss_val > 2000.0 or loss_val < -10.0:
                print(f"\n⚠️  WARNING: Loss explosion detected at batch {batch_idx}!")
                print(f"   Loss: {loss_val:.2e}")
                # Get full breakdown for diagnostics
                loss_dict = complete_vae_loss(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    property_predictor=self.property_predictor,
                    x=molecules,
                    conditions=conditions,
                    beta=beta,
                    lambda_prop=self.lambda_prop,
                    lambda_collapse=self.lambda_collapse,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    free_bits=self.free_bits,
                    lambda_mi=self.lambda_mi,
                    target_mi=4.85
                )
                mx.eval(loss_dict.values())
                
                # Check latent parameter bounds
                mu_max = float(mx.max(mx.abs(loss_dict['mu'])))
                logvar_max = float(mx.max(loss_dict['logvar']))
                logvar_min = float(mx.min(loss_dict['logvar']))
                
                print(f"   Components: recon={float(loss_dict['recon_loss']):.2f}, "
                      f"kl={float(loss_dict['kl_loss']):.2f}, "
                      f"weighted_kl={float(beta * loss_dict['kl_loss']):.2f}, "
                      f"collapse={float(loss_dict['collapse_penalty']):.2f}")
                print(f"   Latent bounds: |μ|_max={mu_max:.3f} (expected ≤2), "
                      f"logvar=[{logvar_min:.3f}, {logvar_max:.3f}] (expected [-5, 2])")
                print(f"   Skipping this batch...")
                continue  # Skip this batch
            
            total_loss += loss_val
            num_batches += 1
            
            # Update progress bar less frequently
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        return {
            'loss': total_loss / max(1, num_batches),
            'recon': (recon_sum / comp_count) if comp_count > 0 else 0.0,
            'kl': (kl_sum / comp_count) if comp_count > 0 else 0.0,
            'collapse': (collapse_sum / comp_count) if comp_count > 0 else 0.0,
            'prop': (prop_sum / comp_count) if comp_count > 0 else 0.0
        }
    
    def _validate(
        self,
        val_dataset,
        beta: float
    ) -> Dict[str, float]:
        """
        Validation step using complete_vae_loss
        
        Args:
            val_dataset: Validation dataset
            beta: KL weight
        
        Returns:
            Validation metrics
        """
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_collapse = 0.0
        total_prop = 0.0
        num_batches = 0
        
        val_batches = len(val_dataset) // self.batch_size
        
        pbar = tqdm(
            val_dataset.to_batches(self.batch_size, shuffle=False),
            total=val_batches,
            desc="Validating"
        )
        
        for batch_idx, (molecules, conditions) in enumerate(pbar):
            # Get loss dict (no teacher forcing during validation)
            loss_dict = complete_vae_loss(
                encoder=self.encoder,
                decoder=self.decoder,
                property_predictor=self.property_predictor,
                x=molecules,
                conditions=conditions,
                beta=beta,
                lambda_prop=self.lambda_prop,
                lambda_collapse=self.lambda_collapse,
                teacher_forcing_ratio=0.0,
                free_bits=self.free_bits,
                lambda_mi=self.lambda_mi,
                target_mi=4.85
            )
            
            # Evaluate all at once
            mx.eval(
                loss_dict['total_loss'],
                loss_dict['recon_loss'],
                loss_dict['kl_loss'],
                loss_dict['collapse_penalty'],
                loss_dict['prop_loss']
            )
            
            total_loss += float(loss_dict['total_loss'])
            total_recon += float(loss_dict['recon_loss'])
            total_kl += float(loss_dict['kl_loss'])
            total_collapse += float(loss_dict['collapse_penalty'])
            total_prop += float(loss_dict['prop_loss'])
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches,
            'collapse': total_collapse / num_batches,
            'prop': total_prop / num_batches
        }
    
    @staticmethod
    def _clip_gradients(grads, max_norm: float = 1.0) -> Tuple:
        """
        Clip gradients by global norm (fast implementation)
        
        Args:
            grads: Gradients tuple
            max_norm: Maximum norm
        
        Returns:
            Clipped gradients
        """
        # Fast implementation: compute norm in one pass without intermediate evals
        total_norm_sq = mx.array(0.0)
        for g in grads:
            if isinstance(g, dict):
                for v in g.values():
                    if isinstance(v, mx.array):
                        total_norm_sq = total_norm_sq + mx.sum(mx.square(v))
        
        total_norm = mx.sqrt(total_norm_sq)
        mx.eval(total_norm)
        norm_val = float(total_norm)
        
        # Only scale if norm exceeds threshold
        if norm_val > max_norm:
            scale = max_norm / (norm_val + 1e-8)
            def scale_grads(g):
                if isinstance(g, dict):
                    return {k: v * scale if isinstance(v, mx.array) else v for k, v in g.items()}
                return g
            return tuple(scale_grads(g) for g in grads)
        
        return grads
    
    def _get_latent_stats(self) -> Tuple[mx.array, mx.array]:
        """Get latent statistics on small batch for monitoring"""
        molecules, conditions = next(iter(self.dataset.to_batches(64, shuffle=False)))
        mx.eval(molecules, conditions)
        
        mu, logvar = self.encoder(molecules, conditions)
        mx.eval(mu, logvar)
        
        # Log statistics for debugging
        mu_mean = float(mx.mean(mu))
        mu_std = float(mx.std(mu))
        mu_max = float(mx.max(mu))
        mu_min = float(mx.min(mu))
        logvar_mean = float(mx.mean(logvar))
        logvar_std = float(mx.std(logvar))
        logvar_max = float(mx.max(logvar))
        logvar_min = float(mx.min(logvar))
        
        print(f"   Latent Stats: μ=[{mu_min:.3f}, {mu_max:.3f}] (mean={mu_mean:.3f}, std={mu_std:.3f}), "
              f"logvar=[{logvar_min:.3f}, {logvar_max:.3f}] (mean={logvar_mean:.3f}, std={logvar_std:.3f})")
        
        return mu, logvar
    
    @staticmethod
    def _compute_mutual_information(mu: mx.array, logvar: mx.array) -> mx.array:
        """
        Compute mutual information between q(z) and p(z)
        
        High MI (~4-5) indicates effective latent space usage
        Low MI (<1) indicates posterior collapse
        """
        batch_size = mu.shape[0]
        
        # Note: mu should be bounded to [-2, 2] via tanh, logvar to [-5, 2] via tanh by encoder
        # Emergency clipping as defensive safeguard
        mu = mx.clip(mu, -3.0, 3.0)
        logvar = mx.clip(logvar, -6.0, 3.0)
        
        var = mx.exp(logvar)
        kl_per_sample = -0.5 * mx.sum(1.0 + logvar - mx.square(mu) - var, axis=1)
        mean_kl = mx.mean(kl_per_sample)
        
        mean_mu = mx.mean(mu, axis=0)
        mean_var = mx.mean(var, axis=0)
        mean_logvar = mx.log(mean_var + 1e-8)
        
        agg_kl = -0.5 * mx.sum(1.0 + mean_logvar - mx.square(mean_mu) - mean_var)
        
        mi = mean_kl - agg_kl
        mi = mx.maximum(mi, 0.0)
        
        return mi
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'encoder_weights': self.encoder.parameters(),
            'decoder_weights': self.decoder.parameters(),
            'encoder_optimizer_state': self.encoder_optimizer.state,
            'decoder_optimizer_state': self.decoder_optimizer.state,
            'history': self.history
        }
        
        if self.property_predictor is not None:
            checkpoint['predictor_weights'] = self.property_predictor.parameters()
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.npz'
        
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.npz'
            self._save_checkpoint(checkpoint, best_path)
        
        self._save_checkpoint(checkpoint, checkpoint_path)
    
    @staticmethod
    def _save_checkpoint(checkpoint: dict, path: Path):
        """Save checkpoint to file"""
        np.savez(str(path), **checkpoint)
        print(f"    Saved checkpoint: {path}")
    
    def save_history(self, path: str):
        """Save training history to JSON"""
        history_path = Path(path) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"    Saved history: {history_path}")
    
    def plot_history(self, save_path: str = None):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("    matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Total Loss
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction vs KL
        axes[0, 1].plot(self.history['epoch'], self.history['train_recon'], label='Recon')
        axes[0, 1].plot(self.history['epoch'], self.history['train_kl'], label='KL')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Posterior Collapse
        axes[0, 2].plot(self.history['epoch'], self.history['train_collapse'], label='Collapse Penalty')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Penalty')
        axes[0, 2].legend()
        axes[0, 2].set_title('Posterior Collapse Penalty')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Property Prediction
        axes[1, 0].plot(self.history['epoch'], self.history['train_prop'], label='Train')
        axes[1, 0].plot(self.history['epoch'], self.history['val_prop'], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Property Loss')
        axes[1, 0].legend()
        axes[1, 0].set_title('Property Prediction Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Annealing Schedules
        axes[1, 1].plot(self.history['epoch'], self.history['beta'], label='Beta')
        ax2 = axes[1, 1].twinx()
        ax2.plot(self.history['epoch'], self.history['teacher_forcing'], label='TF Ratio', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta', color='blue')
        ax2.set_ylabel('TF Ratio', color='orange')
        axes[1, 1].set_title('Annealing Schedules')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mutual Information
        axes[1, 2].plot(self.history['epoch'], self.history['mutual_info'], label='MI')
        axes[1, 2].axhline(y=4.85, color='r', linestyle='--', label='Target')
        axes[1, 2].axhline(y=1.0, color='orange', linestyle='--', label='Collapse')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Mutual Information')
        axes[1, 2].legend()
        axes[1, 2].set_title('Latent Space Health')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"    Saved plot: {save_path}")
        else:
            plt.show()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        
        # Load encoder weights
        if 'encoder_weights' in checkpoint:
            encoder_weights = checkpoint['encoder_weights'].item()
            self._load_module_weights(self.encoder, encoder_weights)
        
        # Load decoder weights
        if 'decoder_weights' in checkpoint:
            decoder_weights = checkpoint['decoder_weights'].item()
            self._load_module_weights(self.decoder, decoder_weights)
        
        # Load optimizer states
        if 'encoder_optimizer_state' in checkpoint:
            optimizer_state = checkpoint['encoder_optimizer_state'].item()
            self.encoder_optimizer.state = self._convert_to_mlx(optimizer_state)
        if 'decoder_optimizer_state' in checkpoint:
            optimizer_state = checkpoint['decoder_optimizer_state'].item()
            self.decoder_optimizer.state = self._convert_to_mlx(optimizer_state)
        
        # Load training history
        if 'history' in checkpoint:
            self.history = checkpoint['history'].item()
        
        epoch = int(checkpoint.get('epoch', 0))
        return epoch
    
    @staticmethod
    def _load_module_weights(module, weights):
        """Load weights into a module"""
        for key, value in weights.items():
            if hasattr(module, key):
                if isinstance(value, dict):
                    # Nested module weights
                    submodule = getattr(module, key)
                    ARCVAETrainerWithLoss._load_module_weights(submodule, value)
                else:
                    # Convert numpy array to MLX array
                    mlx_value = mx.array(value) if not isinstance(value, mx.array) else value
                    setattr(module, key, mlx_value)
    
    @staticmethod
    def _convert_to_mlx(obj):
        """Recursively convert numpy arrays to MLX arrays"""
        if isinstance(obj, dict):
            return {k: ARCVAETrainerWithLoss._convert_to_mlx(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return mx.array(obj)
        else:
            return obj
