import mlx.core as mx
import numpy as np
import json
import os
import argparse
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models.vae import SelfiesVAE

with open(os.path.join(project_root, 'mlx_data/qm9_cns_selfies.json')) as f:
    meta = json.load(f)

token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
START = token_to_idx['<START>']
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
    
    model = SelfiesVAE(**default_kwargs)
    model.load_weights(best_model_path)
    
    print(f"✅ Loaded best model from: {best_model_path}")
    return model

def sample_from_vae(
    model: SelfiesVAE,
    num_samples: int,
    temperature: float=1.0,
    top_k: int=50
) -> np.ndarray:
    """
    Sample sequences from VAE decoder using top-k sampling
    
    Args:
        model: Trained SelfiesVAE model
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
        # Get logits for next token
        logits = model.D(z, seq)[:, -1, :]
        
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
    for i, seq in enumerate(tokens):
        tokens_list = []
        for token_idx in seq:
            # Convert integer index to string key for lookup
            token_key = str(token_idx.item())
            if token_key in idx_to_token:
                token = idx_to_token[token_key]
                if token == '<END>' or token == '<PAD>':
                    break
                # Skip <START> token in output
                if token != '<START>':
                    tokens_list.append(token)
        
        selfies_str = ''.join(tokens_list)
        selfies_strings.append(selfies_str)
    
    return selfies_strings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample molecules from trained VAE')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of molecules to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k sampling threshold (keep top k most likely tokens)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing best_model.npz')
    parser.add_argument('--output_file', type=str, default='generated_molecules.txt', help='Output file for generated SELFIES')
    
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