import torch
import sys
import os

# Add parent directory to path to import MGCVAE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mgcvae import MGCVAE

def load_from_checkpoint(checkpoint_path, device='mps'):
    """
    Load MGCVAE model from checkpoint with optimizer and scheduler states
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        device: Device to load model on
    
    Returns:
        tuple: (model, optimizer_state, scheduler_state, checkpoint)
            - model: Loaded MGCVAE model
            - optimizer_state: Optimizer state dict
            - scheduler_state: Scheduler state dict
            - checkpoint: Full checkpoint dictionary (for metrics, epoch, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint
    config = checkpoint['model_config']
    
    # Create model with saved configuration
    model = MGCVAE(
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'],
        latent_dim=config['latent_dim'],
        hidden_dim=config.get('hidden_dim', 64),  # Default if not in old checkpoints
        num_properties=config['num_properties'],
        num_layers=config.get('num_layers', 3),   # Default if not in old checkpoints
        heads=config.get('heads', 4),             # Default if not in old checkpoints
        max_nodes=config['max_nodes'],
        beta=config['beta'],
        gamma=config['gamma'],
        dropout=config.get('dropout', 0.1)        # Default if not in old checkpoints
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, checkpoint['optimizer_state_dict'], checkpoint['scheduler_state_dict'], checkpoint