#!/usr/bin/env python3
"""
Data loading utilities for the cleaned CNS dataset
"""

import json
import numpy as np
from selfies import encoder, decoder
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import mlx.core as mx

def load_cns_dataset():
    """Load the cleaned CNS dataset"""
    with open('mlx_data/cns_final_dataset.json', 'r') as f:
        data = json.load(f)
    
    print(f"📊 Loaded {len(data)} CNS molecules")
    return data

def create_vocabulary(data):
    """Create vocabulary from SELFIES strings"""
    print("🔤 Creating vocabulary...")
    
    all_tokens = set()
    max_length = 0
    
    for entry in data:
        selfies_str = entry['selfies']
        tokens = selfies_str.split('][')
        # Clean up tokens
        tokens = [token.strip('[]') for token in tokens]
        tokens = ['[' + token + ']' for token in tokens if token]
        
        all_tokens.update(tokens)
        max_length = max(max_length, len(tokens))
    
    # Add special tokens
    all_tokens.add('[PAD]')
    all_tokens.add('[START]')
    all_tokens.add('[END]')
    
    # Create mappings
    token_to_idx = {token: idx for idx, token in enumerate(sorted(all_tokens))}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    vocab_size = len(all_tokens)
    
    print(f"📚 Vocabulary size: {vocab_size}")
    print(f"📏 Max sequence length: {max_length}")
    
    return token_to_idx, idx_to_token, vocab_size, max_length

def tokenize_selfies(data, token_to_idx, max_length):
    """Tokenize SELFIES strings"""
    print("🔢 Tokenizing SELFIES...")
    
    tokenized = []
    properties = []
    
    for entry in data:
        selfies_str = entry['selfies']
        tokens = selfies_str.split('][')
        tokens = [token.strip('[]') for token in tokens]
        tokens = ['[' + token + ']' for token in tokens if token]
        
        # Convert to indices
        indices = [token_to_idx.get(token, token_to_idx['[PAD]']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices.extend([token_to_idx['[PAD]']] * (max_length - len(indices)))
        
        tokenized.append(indices)
        
        # Extract properties
        logp = entry['logp']
        tpsa = entry['tpsa']
        properties.append([logp, tpsa])
    
    return np.array(tokenized), np.array(properties)

def create_batches(tokenized_mx, properties_mx, batch_size, shuffle=True):
    """Create batches for training"""
    num_samples = len(tokenized_mx)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        # Convert numpy array to list for indexing mlx arrays
        batch_indices_list = batch_indices.tolist()
        batch_tokens = tokenized_mx[batch_indices_list]
        batch_properties = properties_mx[batch_indices_list]
        batches.append((batch_tokens, batch_properties))
    
    return batches

def save_metadata(token_to_idx, idx_to_token, vocab_size, max_length, properties):
    """Save metadata for inference"""
    metadata = {
        'token_to_idx': token_to_idx,
        'idx_to_token': idx_to_token,
        'vocab_size': vocab_size,
        'max_length': max_length,
        'logp_values': properties[:, 0].tolist(),
        'tpsa_values': properties[:, 1].tolist()
    }
    
    with open('mlx_data/cns_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("💾 Metadata saved to mlx_data/cns_metadata.json")

def main():
    """Main data processing function"""
    print("🧬 PROCESSING CNS DATASET")
    print("========================")
    
    # Load dataset
    data = load_cns_dataset()
    
    # Create vocabulary
    token_to_idx, idx_to_token, vocab_size, max_length = create_vocabulary(data)
    
    # Tokenize
    tokenized, properties = tokenize_selfies(data, token_to_idx, max_length)
    
    # Save tokenized data
    np.save('mlx_data/cns_tokenized.npy', tokenized)
    print("💾 Tokenized data saved to mlx_data/cns_tokenized.npy")
    
    # Save metadata
    save_metadata(token_to_idx, idx_to_token, vocab_size, max_length, properties)
    
    print("✅ Data processing complete!")
    print(f"📊 Ready for training with {len(data)} molecules")

if __name__ == "__main__":
    main()
