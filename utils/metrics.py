import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity


# =============================================================================
# Property Prediction Evaluation
# =============================================================================

def evaluate_property_prediction(model, data_loader, device='mps'):
    """
    Evaluate property prediction accuracy
    
    Args:
        model: Trained MGCVAE model
        data_loader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        float: Average MSE across all predictions
    """
    model.eval()
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            preds = output['predicted_properties']
            targets = batch.y
            mse = F.mse_loss(preds, targets, reduction='sum').item()
            total_mse += mse
            total_samples += targets.size(0)

    avg_mse = total_mse / total_samples
    print(f"Property Prediction MSE: {avg_mse:.4f}")
    return avg_mse


# =============================================================================
# Reconstruction and KL Divergence Evaluation
# =============================================================================

def evaluate_reconstruction_and_kl(model, data_loader, device='mps'):
    """
    Evaluate reconstruction quality and KL divergence
    
    Args:
        model: Trained MGCVAE model
        data_loader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        tuple: (avg_reconstruction_loss, avg_kl_divergence)
    """
    model.eval()
    recon_loss = 0.0
    kl_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)
            losses = model.compute_loss(batch, out)
            recon_loss += losses['reconstruction_loss'].item()
            kl_loss += losses['kl_loss'].item()
            total_batches += 1

    print(f"Avg Reconstruction Loss: {recon_loss/total_batches:.4f}")
    print(f"Avg KL Divergence: {kl_loss/total_batches:.4f}")
    return recon_loss/total_batches, kl_loss/total_batches


# =============================================================================
# Novelty and Diversity Evaluation
# =============================================================================

def evaluate_novelty_diversity(generated_smiles, train_smiles):
    """
    Evaluate novelty and diversity of generated molecules
    
    Args:
        generated_smiles: List of generated SMILES strings
        train_smiles: List of training set SMILES strings
        
    Returns:
        tuple: (novelty_percentage, diversity_percentage)
    """
    
    # =========================================================================
    # Compute Molecular Fingerprints
    # =========================================================================
    
    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
        for s in train_smiles
    ]
    gen_fps = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
        for s in generated_smiles
    ]

    # =========================================================================
    # Calculate Novelty
    # =========================================================================
    
    # Novelty: fraction with max similarity < 0.4
    novel = sum(
        max(TanimotoSimilarity(g, t) for t in train_fps) < 0.4
        for g in gen_fps
    ) / len(gen_fps) * 100

    # =========================================================================
    # Calculate Diversity
    # =========================================================================
    
    # Diversity: average pairwise dissimilarity among generated
    sims = []
    for i in range(len(gen_fps)):
        for j in range(i+1, len(gen_fps)):
            sims.append(TanimotoSimilarity(gen_fps[i], gen_fps[j]))
    diversity = (1 - sum(sims)/len(sims)) * 100

    print(f"Novelty:   {novel:.1f}%")
    print(f"Diversity: {diversity:.1f}%")
    return novel, diversity

def evaluate_conditioning_latent(model, target, num_samples=100, tolerance=0.1, device='mps'):
    """
    Simplified conditioning evaluation using latent-space predictions.

    Steps:
      1. Sample latent codes z ~ N(0, I).
      2. Predict properties via the PropertyPredictor.
      3. Compute fraction within ±tolerance of target.

    Args:
        model: Trained MGCVAE model.
        target: List or array [BBBP, toxicity] target values.
        num_samples: Number of latent samples to draw.
        tolerance: Acceptable deviation from target.
        device: 'mps' or 'cpu'.

    Returns:
        dict with keys:
          - success_rate: % predictions within tolerance of target.
          - mae: mean absolute error to target.
          - mean_pred: mean predicted properties.
          - std_pred: std dev of predicted properties.
    """
    model.eval()
    target = np.array(target, dtype=float)

    # 1. Sample latent codes
    z = torch.randn(num_samples, model.latent_dim, device=device)

    # 2. Predict properties
    with torch.no_grad():
        preds = model.property_predictor(z).cpu().numpy()

    # 3. Compute metrics
    diffs = np.abs(preds - target)
    within = np.logical_and(diffs[:, 0] <= tolerance, diffs[:, 1] <= tolerance)
    success_rate = within.sum() / num_samples * 100
    mae = diffs.mean()

    mean_pred = preds.mean(axis=0).tolist()
    std_pred = preds.std(axis=0).tolist()

    print(f"Conditioning Evaluation (latent):")
    print(f"Target:            {target.tolist()}")
    print(f"Success rate:      {success_rate:.1f}% within ±{tolerance}")
    print(f"Mean absolute err: {mae:.4f}")
    print(f"Predicted mean:    {mean_pred}")
    print(f"  Predicted std:     {std_pred}")

    return {
        'success_rate': success_rate,
        'mae': mae,
        'mean_pred': mean_pred,
        'std_pred': std_pred,
    }

# Usage example:
# results = evaluate_conditioning_latent(model, target=[0.8, 0.2], num_samples=200, tolerance=0.1, device=device)
