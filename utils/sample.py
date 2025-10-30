import mlx.core as mx
import numpy as np
import json
import os
import argparse
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models.transformer_vae import SelfiesTransformerVAE
from utils.validate import batch_validate_selfies

with open(os.path.join(project_root, 'mlx_data/chembl_cns_selfies.json')) as f:
    meta = json.load(f)

token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
START = token_to_idx.get('[START]', 1)  # Use [START] token
max_length = meta['max_length']
vocab_size = meta['vocab_size']

def top_k_sampling(logits, k=10):
    """Apply top-k sampling to logits"""
    batch_size, vocab_size = logits.shape
    
    # Ensure k doesn't exceed vocab size
    k = min(k, vocab_size)
    
    # Get top-k values
    top_k_values = mx.topk(logits, k, axis=-1)
    
    # Get the threshold (minimum value in top-k)
    threshold = top_k_values[:, -1:]  # [B, 1]
    
    # Create mask for tokens above threshold
    keep_mask = logits >= threshold
    
    # Apply mask to logits (set excluded tokens to very negative value)
    masked_logits = mx.where(keep_mask, logits, mx.full(logits.shape, -1e9))
    
    return masked_logits

def load_best_model(checkpoint_dir='checkpoints', **model_kwargs):
    """Load the best model from checkpoint directory"""
    # If checkpoint_dir is relative, make it relative to project root
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.npz')
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    # Load normalization stats and architecture params if available
    norm_file = os.path.join(checkpoint_dir, 'property_norm.json')
    if os.path.exists(norm_file):
        import json
        with open(norm_file, 'r') as f:
            norm_stats = json.load(f)
        
        # Use architecture params from checkpoint if available
        model_kwargs_from_checkpoint = {
            'vocab_size': vocab_size,
            'embedding_dim': norm_stats.get('embedding_dim', 128),
            'hidden_dim': norm_stats.get('hidden_dim', 256),
            'latent_dim': norm_stats.get('latent_dim', 64),
            'num_heads': norm_stats.get('num_heads', 4),
            'num_layers': norm_stats.get('num_layers', 4),
            'dropout': norm_stats.get('dropout', 0.1)
        }
        # Override with any provided kwargs
        model_kwargs_from_checkpoint.update(model_kwargs)
        
        model = SelfiesTransformerVAE(**model_kwargs_from_checkpoint)
        model.set_property_normalization(
            norm_stats.get('logp_mean', 0.0),
            norm_stats.get('logp_std', 1.0),
            norm_stats['tpsa_mean'],
            norm_stats['tpsa_std']
        )
        print(f" Loaded property normalization stats")
        print(f" Architecture: {norm_stats.get('num_layers', 4)} layers, {norm_stats.get('num_heads', 4)} heads")
    else:
        # Fallback to defaults if no metadata file
        default_kwargs = {
            'vocab_size': vocab_size,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'latent_dim': 64,
            'num_heads': 4,
            'num_layers': 4
        }
        default_kwargs.update(model_kwargs)
        model = SelfiesTransformerVAE(**default_kwargs)
        # Use dummy normalization when no checkpoint stats are available.
        # The model expects normalized properties; setting mean=0 and std=1
        # makes normalization a no-op so raw inputs pass through unchanged,
        # while still satisfying the interface.
        model.set_property_normalization(0.0, 1.0, 1.0, 1.0)
    
    # Load weights; if new modules were added after the checkpoint was saved,
    # allow missing parameters to be randomly initialized.
    try:
        model.load_weights(best_model_path)
    except Exception as e:
        print(f" Warning: Partial weight load. Proceeding with randomly initialized new parameters. Details: {e}")
    
    print(f" Loaded best model from: {best_model_path}")
    return model

def sample_from_vae(
    model: SelfiesTransformerVAE,
    num_samples: int,
    temperature: float=1.0,
    top_k: int=50
) -> np.ndarray:
    """
    Sample sequences from VAE decoder using top-k sampling
    
    Args:
        model: Trained SelfiesTransformerVAE model
        num_samples: Number of sequences to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling threshold (keep top k most likely tokens)
    
    Returns:
        Generated sequences as numpy array [num_samples, max_length]
    """
    # Sample latent codes
    z = mx.random.normal((num_samples, model.latent_dim))
    
    # Initialize with START token
    seq = mx.full((num_samples, 1), START, dtype=mx.int32)
    samples = seq
    
    # Generate sequence token by token
    for _ in range(max_length - 1):
        # Get logits for next token using decoder directly
        logits = model.decoder(z, seq)[:, -1, :]
        
        # Apply top-k sampling
        logits = top_k_sampling(logits, k=top_k)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Sample next token
        next_token = mx.random.categorical(probs, num_samples=1)
        
        # Append to sequence
        samples = mx.concatenate([samples, next_token], axis=1)
    
    return mx.array(samples)

def tokens_to_selfies(tokens):
    """Convert token indices back to SELFIES strings"""
    selfies_strings = []
    
    # Handle both numpy arrays and MLX arrays
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    
    for i, seq in enumerate(tokens):
        tokens_list = []
        try:
            for token_idx in seq:
                # Convert to int and then string for lookup
                if hasattr(token_idx, 'item'):
                    token_idx = token_idx.item()
                elif hasattr(token_idx, 'tolist'):
                    token_idx = token_idx.tolist()
                
                token_key = str(int(token_idx))
                if token_key in idx_to_token:
                    token = idx_to_token[token_key]
                    if token == '<END>' or token == '[END]' or token == '<PAD>' or token == '[PAD]':
                        break
                    # Skip <START> token in output
                    if token != '<START>' and token != '[START]':
                        tokens_list.append(token)
        except (IndexError, ValueError) as e:
            print(f"Warning: Error processing sequence {i}: {e}")
            continue
        
        selfies_str = ''.join(tokens_list)
        selfies_strings.append(selfies_str)
    
    return selfies_strings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample molecules from trained VAE')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of molecules to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k sampling threshold (keep top k most likely tokens)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing best_model.npz')
    parser.add_argument('--output_file', type=str, default='output/generated_molecules.txt', help='Output file for generated SELFIES')
    
    args = parser.parse_args()
    
    # Load best model
    model = load_best_model(args.checkpoint_dir)
    
    # Generate samples
    print(f" Generating {args.num_samples} molecules with temperature {args.temperature} and top_k {args.top_k}...")
    samples = sample_from_vae(model, args.num_samples, args.temperature, args.top_k)
    
    # Convert to SELFIES strings
    selfies_strings = tokens_to_selfies(samples)
    
    # Save to file
    with open(args.output_file, 'w') as f:
        for selfies in selfies_strings:
            f.write(selfies + '\n')
    
    print(f" Saved {len(selfies_strings)} molecules to {args.output_file}")
    print(f" First few samples:")
    for i, selfies in enumerate(selfies_strings[:5]):
        print(f"  {i+1}: {selfies}")

def generate_conditional_molecules(model, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
    """Generate molecules conditioned via FiLM on LogP and TPSA targets"""
    print(f" Generating {num_samples} molecules with:")
    print(f"   LogP: {target_logp}")
    print(f"   TPSA: {target_tpsa}")
    
    # Generate conditional samples (FiLM conditioning on both properties)
    samples = model.generate_conditional(target_logp, target_tpsa, num_samples, temperature, top_k)
    
    # Convert to SELFIES
    selfies_list = tokens_to_selfies(samples)
    print(f" Generated {len(selfies_list)} SELFIES sequences")
    
    # Validate molecules
    print(" Validating generated molecules...")
    validation_results = batch_validate_selfies(selfies_list, verbose=False)
    
    # Filter valid molecules
    valid_molecules = []
    for i, result in enumerate(validation_results):
        if result and isinstance(result, dict):
            valid_molecules.append({
                'selfies': selfies_list[i],
                'smiles': result.get('smiles', ''),
                'logp': result.get('logp', 0),
                'tpsa': result.get('tpsa', 0),
                'mw': result.get('mw', 0),
                'qed': result.get('qed', 0),
                'sas': result.get('sas', 0)
            })
    
    print(f" {len(valid_molecules)} valid molecules generated")
    return valid_molecules

def analyze_logp_tpsa_accuracy(molecules, target_logp, target_tpsa):
    """Analyze how well generated molecules match the TPSA target with robust stats (TPSA-only)."""
    if not molecules:
        return {}
    
    tpsa_values = [m.get('tpsa', 0) for m in molecules if 'tpsa' in m and m['tpsa'] > 0]
    
    if not tpsa_values:
        print(" No valid TPSA values to analyze")
        return {}
    
    import numpy as np
    tpsa_array = np.array(tpsa_values)
    
    # Robust and descriptive stats
    tpsa_mean = np.mean(tpsa_array)
    tpsa_std = np.std(tpsa_array)
    tpsa_median = float(np.median(tpsa_array))
    p10 = float(np.percentile(tpsa_array, 10))
    p90 = float(np.percentile(tpsa_array, 90))
    
    # Errors
    abs_err = np.abs(tpsa_array - target_tpsa)
    mae = float(np.mean(abs_err))
    med_ae = float(np.median(abs_err))
    
    # Hit rates within tolerance bands
    def hit_rate(tol):
        return float((np.sum(abs_err <= tol) / len(tpsa_array)) * 100.0)
    tol20 = 20.0
    tol10 = 10.0
    acc20 = hit_rate(tol20)
    acc10 = hit_rate(tol10)
    within20 = int(np.sum(abs_err <= tol20))
    within10 = int(np.sum(abs_err <= tol10))
    
    # Best k matches (preview)
    k_preview = min(5, len(tpsa_array))
    best_idx = np.argsort(abs_err)[:k_preview]
    best_tpsa = [float(tpsa_array[i]) for i in best_idx]
    best_errs = [float(abs_err[i]) for i in best_idx]
    
    print(f"\n CONDITIONAL GENERATION ANALYSIS")
    print(f"=" * 40)
    print(f"Target TPSA: {target_tpsa:.2f}")
    print(f"Mean +/- SD: {tpsa_mean:.2f} +/- {tpsa_std:.2f}")
    print(f"Median [P10, P90]: {tpsa_median:.2f} [{p10:.2f}, {p90:.2f}]")
    print(f"MAE / Median AE: {mae:.2f} / {med_ae:.2f}")
    print(f"Accuracy (+/-{tol20:.0f}): {acc20:.1f}% ({within20}/{len(tpsa_array)})")
    print(f"Accuracy (+/-{tol10:.0f}): {acc10:.1f}% ({within10}/{len(tpsa_array)})")
    if k_preview > 0:
        print(f"Best {k_preview} TPSA matches (value | abs err):")
        for v, e in zip(best_tpsa, best_errs):
            print(f"  {v:.2f} | {e:.2f}")
    
    return {
        'tpsa_mean': float(tpsa_mean),
        'tpsa_std': float(tpsa_std),
        'tpsa_median': tpsa_median,
        'p10': p10,
        'p90': p90,
        'mae': mae,
        'median_ae': med_ae,
        'accuracy_pct_20': acc20,
        'accuracy_pct_10': acc10,
        'best_values': best_tpsa,
        'best_abs_errors': best_errs,
    }


def generate_conditional_molecules_with_search(
    model: SelfiesTransformerVAE,
    target_logp: float,
    target_tpsa: float,
    num_candidates: int = 1000,
    top_k: int = 100,
    temperature: float = 1.0,
    token_top_k: int = 10
):
    """
    Latent-space search for conditional generation.
    1) Sample many z ~ N(0, I)
    2) Decode each with FiLM conditioning on target properties
    3) Validate and measure properties
    4) Select top_k closest to target in property space
    Returns a list of dicts with SMILES and measured properties.
    """
    # Normalize targets (if stats available)
    if model.logp_mean is not None and model.logp_std is not None:
        norm_logp = (target_logp - model.logp_mean) / model.logp_std
    else:
        norm_logp = target_logp
    if model.tpsa_mean is not None and model.tpsa_std is not None:
        norm_tpsa = (target_tpsa - model.tpsa_mean) / model.tpsa_std
    else:
        norm_tpsa = target_tpsa

    # Sample candidate z vectors
    z_candidates = mx.random.normal((num_candidates, model.latent_dim))

    # Build FiLM embedding for the targets (shared across candidates)
    logp_arr = mx.array([[norm_logp]] * num_candidates)
    tpsa_arr = mx.array([[norm_tpsa]] * num_candidates)
    logp_emb = model.property_encoder_logp(logp_arr)
    tpsa_emb = model.property_encoder_tpsa(tpsa_arr)
    film_emb = mx.concatenate([logp_emb, tpsa_emb], axis=-1)

    # Decode candidates
    selfies_list = []
    valid_indices = []
    for i in range(num_candidates):
        z_i = z_candidates[i:i+1]
        cond_i = film_emb[i:i+1]
        seq = model._decode_conditional(z_i, cond_i, temperature, token_top_k)
        # Convert tokens to SELFIES (batch size 1)
        seq_list = tokens_to_selfies(seq)
        if seq_list and isinstance(seq_list[0], str) and len(seq_list[0]) > 0:
            selfies_list.append(seq_list[0])
            valid_indices.append(i)

    if not selfies_list:
        return []

    # Validate and compute properties for all generated SELFIES
    results = batch_validate_selfies(selfies_list, verbose=False)

    # Collect valid molecules and their properties
    molecules = []
    properties = []
    for res in results:
        if res and isinstance(res, dict) and 'smiles' in res:
            molecules.append(res)
            properties.append([res.get('logp', 0.0), res.get('tpsa', 0.0)])

    if not properties:
        return []

    props_array = mx.array(properties)
    target = mx.array([[target_logp, target_tpsa]])
    dists = mx.sum((props_array - target) ** 2, axis=1)

    # Select top_k closest
    k = min(top_k, len(molecules))
    order = mx.argsort(dists)[:k]
    top = [molecules[int(i.item())] for i in order]
    return top