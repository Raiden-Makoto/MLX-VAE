import mlx.core as mx
import json
import numpy as np

with open('qm9_cns_selfies.json', 'r') as f:
    meta = json.load(f)

tokenized = np.load('qm9_cns_tokenized.npy')
token_to_idx = meta['token_to_idx']
idx_to_token = meta['idx_to_token']
vocab_size = meta['vocab_size']
max_length = meta['max_length']

# SPECIAL TOKENS
PAD = token_to_idx['<PAD>']
START = token_to_idx['<START>']
END = token_to_idx['<END>']

BATCH_SIZE = 128

# Convert numpy array to MLX array
tokenized_mx = mx.array(tokenized)

def create_batches(data, batch_size, shuffle=True):
    """Create batches from MLX array data"""
    n_samples = data.shape[0]
    
    if shuffle:
        # Create random permutation of indices
        indices = mx.random.permutation(mx.arange(n_samples))
        data = data[indices]
    
    # Calculate number of batches
    n_batches = n_samples // batch_size
    
    # Create batches
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = data[start_idx:end_idx]
        batches.append(batch)
    
    return batches

# Create batches
batches = create_batches(tokenized_mx, BATCH_SIZE, shuffle=True)
print(f"Created {len(batches)} batches of size {BATCH_SIZE}")
print(f"Batch shape: {batches[0].shape}")