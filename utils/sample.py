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

with open(os.path.join(project_root, 'mlx_data/cns_metadata.json')) as f:
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
    
    # Initialize model with default or provided parameters
    default_kwargs = {
        'vocab_size': vocab_size,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'latent_dim': 64
    }
    default_kwargs.update(model_kwargs)
    
    model = SelfiesTransformerVAE(**default_kwargs)
    model.load_weights(best_model_path)
    
    print(f"✅ Loaded best model from: {best_model_path}")
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
                    if token == '<END>' or token == '<PAD>':
                        break
                    # Skip <START> token in output
                    if token != '<START>':
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
    print(f"🎲 Generating {args.num_samples} molecules with temperature {args.temperature} and top_k {args.top_k}...")
    samples = sample_from_vae(model, args.num_samples, args.temperature, args.top_k)
    
    # Convert to SELFIES strings
    selfies_strings = tokens_to_selfies(samples)
    
    # Save to file
    with open(args.output_file, 'w') as f:
        for selfies in selfies_strings:
            f.write(selfies + '\n')
    
    print(f"💾 Saved {len(selfies_strings)} molecules to {args.output_file}")
    print(f"📝 First few samples:")
    for i, selfies in enumerate(selfies_strings[:5]):
        print(f"  {i+1}: {selfies}")

def generate_conditional_molecules(model, target_logp, target_tpsa, num_samples=100, temperature=1.0, top_k=10):
    """Generate molecules with target LogP and TPSA values"""
    print(f"🧬 Generating {num_samples} molecules with:")
    print(f"   LogP: {target_logp}")
    print(f"   TPSA: {target_tpsa}")
    
    # Generate conditional samples
    samples = model.generate_conditional(target_logp, target_tpsa, num_samples, temperature, top_k)
    
    # Convert to SELFIES
    selfies_list = tokens_to_selfies(samples)
    print(f"✅ Generated {len(selfies_list)} SELFIES sequences")
    
    # Validate molecules
    print("🔍 Validating generated molecules...")
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
    
    print(f"✅ {len(valid_molecules)} valid molecules generated")
    return valid_molecules

def analyze_logp_tpsa_accuracy(molecules, target_logp, target_tpsa):
    """Analyze how well generated molecules match LogP and TPSA targets"""
    if not molecules:
        return {}
    
    logp_values = [m.get('logp', 0) for m in molecules if 'logp' in m and m['logp'] > 0]
    tpsa_values = [m.get('tpsa', 0) for m in molecules if 'tpsa' in m and m['tpsa'] > 0]
    
    if not logp_values or not tpsa_values:
        print("❌ No valid property values to analyze")
        return {}
    
    logp_mean = sum(logp_values) / len(logp_values)
    tpsa_mean = sum(tpsa_values) / len(tpsa_values)
    
    logp_error = abs(logp_mean - target_logp)
    tpsa_error = abs(tpsa_mean - target_tpsa)
    
    print(f"\n📊 CONDITIONAL GENERATION ANALYSIS")
    print(f"=" * 40)
    print(f"Target LogP: {target_logp:.2f}")
    print(f"Generated LogP: {logp_mean:.2f} ± {logp_error:.2f}")
    print(f"Target TPSA: {target_tpsa:.2f}")
    print(f"Generated TPSA: {tpsa_mean:.2f} ± {tpsa_error:.2f}")
    print(f"LogP Accuracy: {max(0, 1 - logp_error/2):.1%}")
    print(f"TPSA Accuracy: {max(0, 1 - tpsa_error/50):.1%}")
    
    return {
        'logp_mean': logp_mean,
        'tpsa_mean': tpsa_mean,
        'logp_error': logp_error,
        'tpsa_error': tpsa_error
    }