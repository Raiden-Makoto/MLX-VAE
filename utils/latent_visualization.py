"""Visualize latent space to diagnose LogP vs TPSA conditioning"""

import mlx.core as mx
import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_vae import SelfiesTransformerVAE


def load_model_and_data():
    """Load the best model and training data"""
    print("Loading model...")
    
    # Load model architecture from checkpoint
    norm_file = Path("checkpoints/property_norm.json")
    if not norm_file.exists():
        raise FileNotFoundError("property_norm.json not found")
    
    with open(norm_file, 'r') as f:
        norm_stats = json.load(f)
    
    # Load tokenized sequences
    print("Loading data...")
    tokenized = np.load('mlx_data/chembl_cns_tokenized.npy')
    
    # Load vocabulary from JSON file
    meta_file = Path("mlx_data/chembl_cns_selfies.json")
    with open(meta_file, 'r') as f:
        data = json.load(f)
    
    vocab = data['idx_to_token']
    molecules = data['molecules']
    properties = np.array([[mol['logp'], mol['tpsa']] for mol in molecules], dtype=np.float32)
    
    # Initialize model with saved architecture
    model = SelfiesTransformerVAE(
        vocab_size=len(vocab),
        embedding_dim=norm_stats.get('embedding_dim', 128),
        hidden_dim=norm_stats.get('hidden_dim', 256),
        latent_dim=norm_stats.get('latent_dim', 256),
        num_heads=norm_stats.get('num_heads', 4),
        num_layers=norm_stats.get('num_layers', 4),
        dropout=0.1
    )
    
    # Set normalization
    model.set_property_normalization(
        norm_stats['logp_mean'],
        norm_stats['logp_std'],
        norm_stats['tpsa_mean'],
        norm_stats['tpsa_std']
    )
    
    # Load weights
    best_model_path = "checkpoints/best_model.npz"
    if Path(best_model_path).exists():
        print(f"Loading weights from {best_model_path}")
        model.load_weights(best_model_path)
    else:
        raise FileNotFoundError(f"Model weights not found at {best_model_path}")
    
    return model, tokenized, properties, vocab


def extract_latents(model, sequences, properties, batch_size=128):
    """Extract latent codes z from encoder"""
    print("Extracting latent codes...")
    
    latents = []
    properties_list = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_props = properties[i:i+batch_size]
        
        # Convert to MLX
        batch_mx = mx.array(batch)
        batch_props_mx = mx.array(batch_props)
        
        # Get property embedding
        prop_emb = model.property_encoder(batch_props_mx)
        
        # Get latent from encoder
        mu, logvar = model.encoder(batch_mx, prop_emb)
        
        # Reparameterize to get z
        z = model.reparameterize(mu, logvar)
        
        # Store
        latents.append(np.array(z))
        properties_list.append(batch_props)
    
    latents = np.concatenate(latents, axis=0)
    properties_array = np.concatenate(properties_list, axis=0)
    
    print(f"Extracted {latents.shape[0]} latent codes of dimension {latents.shape[1]}")
    
    # Clean latents (remove NaN/Inf)
    latents = np.nan_to_num(latents, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Remove any remaining outliers
    latents = np.clip(latents, -10, 10)
    
    return latents, properties_array


def visualize_with_pca(latents, properties):
    """Visualize latent space with PCA, colored by LogP and TPSA"""
    print("Computing PCA...")
    
    # Project to 2D with PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latents)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: colored by LogP
    scatter1 = axes[0].scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=properties[:, 0],  # LogP
        cmap='RdBu_r',
        alpha=0.6,
        s=20
    )
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].set_title('Latent Space - Colored by LogP')
    plt.colorbar(scatter1, ax=axes[0], label='LogP')
    
    # Right plot: colored by TPSA
    scatter2 = axes[1].scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=properties[:, 1],  # TPSA
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1].set_title('Latent Space - Colored by TPSA')
    plt.colorbar(scatter2, ax=axes[1], label='TPSA')
    
    plt.tight_layout()
    plt.savefig('output/latent_space_pca.png', dpi=150, bbox_inches='tight')
    print("Saved: output/latent_space_pca.png")
    
    return latent_2d


def visualize_with_tsne(latents, properties):
    """Visualize latent space with t-SNE, colored by LogP and TPSA"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Skipping t-SNE (sklearn not available)")
        return None
    
    print("Computing t-SNE (this may take a while)...")
    
    # Project to 2D with t-SNE
    tsne_reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne_reducer.fit_transform(latents)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: colored by LogP
    scatter1 = axes[0].scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=properties[:, 0],  # LogP
        cmap='RdBu_r',
        alpha=0.6,
        s=20
    )
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].set_title('Latent Space (t-SNE) - Colored by LogP')
    plt.colorbar(scatter1, ax=axes[0], label='LogP')
    
    # Right plot: colored by TPSA
    scatter2 = axes[1].scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=properties[:, 1],  # TPSA
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('Latent Space (t-SNE) - Colored by TPSA')
    plt.colorbar(scatter2, ax=axes[1], label='TPSA')
    
    plt.tight_layout()
    plt.savefig('output/latent_space_tsne.png', dpi=150, bbox_inches='tight')
    print("Saved: output/latent_space_tsne.png")
    
    return latent_2d


def analyze_property_distributions(latents, properties):
    """Analyze how well LogP and TPSA are distributed in latent space"""
    print("\n=== Property Distribution Analysis ===")
    
    # Check for separation
    logp_range = properties[:, 0].max() - properties[:, 0].min()
    tpsa_range = properties[:, 1].max() - properties[:, 1].min()
    
    print(f"LogP range: {properties[:, 0].min():.2f} to {properties[:, 0].max():.2f} (range: {logp_range:.2f})")
    print(f"TPSA range: {properties[:, 1].min():.0f} to {properties[:, 1].max():.0f} (range: {tpsa_range:.0f})")
    
    # Check if properties are clustered or spread out
    logp_std = properties[:, 0].std()
    tpsa_std = properties[:, 1].std()
    
    print(f"LogP std: {logp_std:.2f}")
    print(f"TPSA std: {tpsa_std:.2f}")
    
    # Visualize property distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # LogP distribution
    axes[0].hist(properties[:, 0], bins=50, alpha=0.7, color='blue')
    axes[0].set_xlabel('LogP')
    axes[0].set_ylabel('Count')
    axes[0].set_title('LogP Distribution')
    axes[0].axvline(properties[:, 0].mean(), color='red', linestyle='--', label=f'Mean: {properties[:, 0].mean():.2f}')
    axes[0].legend()
    
    # TPSA distribution
    axes[1].hist(properties[:, 1], bins=50, alpha=0.7, color='green')
    axes[1].set_xlabel('TPSA')
    axes[1].set_ylabel('Count')
    axes[1].set_title('TPSA Distribution')
    axes[1].axvline(properties[:, 1].mean(), color='red', linestyle='--', label=f'Mean: {properties[:, 1].mean():.0f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('output/property_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: output/property_distributions.png")


def compute_property_prediction_correlation(latents, properties, model):
    """Check if latent codes encode property information"""
    print("\n=== Checking if Latent Encodes Properties ===")
    
    # Use property predictor
    latents_mx = mx.array(latents)
    pred_properties = model.property_predictor(latents_mx)
    
    # Convert to numpy and denormalize
    pred_properties_np = np.array(pred_properties)
    pred_logp = pred_properties_np[:, 0] * model.logp_std + model.logp_mean
    pred_tpsa = pred_properties_np[:, 1] * model.tpsa_std + model.tpsa_mean
    
    true_logp = properties[:, 0]
    true_tpsa = properties[:, 1]
    
    # Compute correlations
    try:
        from scipy.stats import pearsonr
        logp_corr, _ = pearsonr(true_logp, pred_logp)
        tpsa_corr, _ = pearsonr(true_tpsa, pred_tpsa)
    except ImportError:
        # Fallback to numpy correlation
        logp_corr = np.corrcoef(true_logp, pred_logp)[0, 1]
        tpsa_corr = np.corrcoef(true_tpsa, pred_tpsa)[0, 1]
    
    print(f"LogP prediction correlation: {logp_corr:.3f}")
    print(f"TPSA prediction correlation: {tpsa_corr:.3f}")
    
    return logp_corr, tpsa_corr


def main():
    """Main visualization pipeline"""
    print("=" * 60)
    print("Latent Space Visualization")
    print("=" * 60)
    
    # Load model and data
    model, sequences, properties, vocab = load_model_and_data()
    
    # Sample subset for visualization (too many points slow down UMAP)
    print(f"\nTotal samples: {len(sequences)}")
    n_samples = min(5000, len(sequences))
    indices = np.random.choice(len(sequences), n_samples, replace=False)
    sequences_sample = sequences[indices]
    properties_sample = properties[indices]
    
    # Extract latents
    latents, properties_array = extract_latents(model, sequences_sample, properties_sample)
    
    # Visualize with PCA
    latent_2d_pca = visualize_with_pca(latents, properties_array)
    
    # Visualize with t-SNE (optional, may be slow)
    visualize_with_tsne(latents, properties_array)
    
    # Analyze property distributions
    analyze_property_distributions(latents, properties_array)
    
    # Check if latents encode properties
    compute_property_prediction_correlation(latents, properties_array, model)
    
    print("\n" + "=" * 60)
    print("Visualization complete! Check output/ for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()

