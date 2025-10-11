import torch

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
    epochs: int = 4
):
    dataset = QM9GraphDataset(csv_path=path_to_dataset)
    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=4)
    if loaded:
        model, optimizer_state, scheduler_state, checkpoint = load_from_checkpoint(
            path_to_checkpoint,
            device=device
        )
        print(f"\nTraining stopped at epoch: {checkpoint['epoch']}")
        print(f"Final validation loss: {checkpoint['best_val_loss']:.4f}")
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
            'beta': 0.01,      # Start with low KL weight
            'gamma': 1.0,      # Property prediction weight
            'dropout': 0.1
        }
        model = MGCVAE(**model_config).to(device)
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
    _, _ = trainer.train(
        num_epochs=epochs,
        start_epoch=0 if not loaded else checkpoint['epoch']+1
    )

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    _ = evaluate_property_prediction(model, val_loader, device)
    _ = evaluate_reconstruction_and_kl(model, val_loader, device)
    evaluate_conditioning_latent(
        model,
        target=[0.67],
        num_samples=50,
        tolerance=0.15,
        device=device
    )

if __name__ == '__main__':
    run_model('mps', loaded=True)