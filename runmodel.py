import torch
import argparse
import warnings

from data.dataset import QM9GraphDataset
from utils.datautils import create_data_loaders
from mgcvae import MGCVAE
from trainer import MGCVAETrainer
from utils.trainutils import load_from_checkpoint
from utils.metrics import (
    evaluate_property_prediction,
    evaluate_reconstruction_and_kl,
    evaluate_conditioning_latent
)

def run_model(
    device='mps',
    path_to_dataset='./data/qm9_bbbp2.csv',
    loaded=False,
    path_to_checkpoint='checkpoints/mgcvae/best_model.pth',
    epochs=4
):
    """
    Train or evaluate MGCVAE model
    
    Args:
        device: Device to run on ('mps', 'cuda', or 'cpu')
        path_to_dataset: Path to dataset CSV file
        loaded: Whether to load from checkpoint
        path_to_checkpoint: Path to checkpoint file
        epochs: Number of epochs to train
    """
    # =========================================================================
    # Setup Device and Data
    # =========================================================================
    
    device = torch.device(device)
    dataset = QM9GraphDataset(csv_path=path_to_dataset)
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, 
        batch_size=4
    )
    
    # =========================================================================
    # Load Model
    # =========================================================================
    
    if loaded:
        model, optimizer_state, scheduler_state, checkpoint = load_from_checkpoint(
            path_to_checkpoint,
            device=device
        )
        print(f"\nLoaded checkpoint from epoch: {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model_config = {
            'node_dim': 29,
            'edge_dim': 6,
            'latent_dim': 32,
            'hidden_dim': 64,
            'num_properties': 1,
            'num_layers': 2,
            'heads': 4,
            'max_nodes': 20,
            'beta': 0.01,       # Start with low KL weight
            'gamma': 1.0,       # Property prediction weight
            'dropout': 0.1
        }
        model = MGCVAE(**model_config).to(device)
    
    # =========================================================================
    # Setup Trainer
    # =========================================================================
    
    trainer = MGCVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=1e-3,
        device=device,
        save_dir='checkpoints/mgcvae'
    )
    
    if loaded:
        trainer.load_optimizer_scheduler(optimizer_state, scheduler_state)
    
    # =========================================================================
    # Train Model
    # =========================================================================
    
    start_epoch = 1 if not loaded else checkpoint['epoch'] + 1
    _, _ = trainer.train(num_epochs=epochs, start_epoch=start_epoch)
    
    # =========================================================================
    # Evaluate Model
    # =========================================================================
    
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70 + "\n")
    
    evaluate_property_prediction(model, val_loader, device)
    evaluate_reconstruction_and_kl(model, val_loader, device)
    evaluate_conditioning_latent(
        model,
        target=[0.67],
        num_samples=50,
        tolerance=0.15,
        device=device
    )

if __name__ == '__main__':
    # =========================================================================
    # Command Line Arguments
    # =========================================================================
    
    parser = argparse.ArgumentParser(
        description='Train or evaluate MGCVAE model'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        help='Device to use (mps, cuda, or cpu)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='./data/qm9_bbbp2.csv',
        help='Path to dataset CSV file'
    )
    
    parser.add_argument(
        '--loaded',
        action='store_true',
        help='Load from checkpoint'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/mgcvae/best_model.pth',
        help='Path to checkpoint file'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of epochs to train'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # Run Model
    # =========================================================================
    
    run_model(
        device=args.device,
        path_to_dataset=args.dataset,
        loaded=args.loaded,
        path_to_checkpoint=args.checkpoint,
        epochs=args.epochs
    )