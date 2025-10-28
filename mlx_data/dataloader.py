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

with open('mlx_data/chembl_cns_selfies.json', 'r') as f:
    meta = json.load(f)

tokenized = np.load('mlx_data/chembl_cns_tokenized.npy')
token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
vocab_size = meta['vocab_size']
max_length = meta['max_length']

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
