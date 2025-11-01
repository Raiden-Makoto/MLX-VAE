import numpy as np
import mlx.core as mx

class MoleculeDataset:
    """
    Dataset for tokenized molecules with properties
    Designed for MLX with lazy evaluation
    """
    
    def __init__(
        self,
        tokenized_molecules: list,
        properties: np.ndarray,
        max_length: int = 120,
        pad_token: int = 0
    ):
        """
        Args:
            tokenized_molecules: List of tokenized SMILES sequences
            properties: Array of shape [n_samples, num_properties]
            max_length: Maximum sequence length for padding
            pad_token: Padding token index
        """
        self.molecules = tokenized_molecules
        self.max_length = max_length
        self.pad_token = pad_token
        
        # Convert to numpy arrays
        self.properties = np.array(properties, dtype=np.float32)
        
        # Compute normalization statistics
        self.properties_mean = self.properties.mean(axis=0, keepdims=True)
        self.properties_std = self.properties.std(axis=0, keepdims=True)
        
        # Avoid division by zero
        self.properties_std = np.where(
            self.properties_std < 1e-8,
            1.0,
            self.properties_std
        )
        
        # Normalize properties
        self.properties_normalized = (
            (self.properties - self.properties_mean) / self.properties_std
        )
    
    def __len__(self) -> int:
        return len(self.molecules)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample"""
        mol = list(self.molecules[idx])  # Copy to avoid modification
        props = self.properties_normalized[idx]
        
        # Pad or truncate to max_length
        if len(mol) < self.max_length:
            mol = mol + [self.pad_token] * (self.max_length - len(mol))
        else:
            mol = mol[:self.max_length]
        
        return {
            'molecule': mx.array(mol, dtype=mx.uint32),
            'properties': mx.array(props, dtype=mx.float32)
        }
    
    def to_batches(self, batch_size: int, shuffle: bool = True):
        """
        Generate batches of data (generator)
        Lazy evaluation - only materializes when accessed
        """
        indices = np.arange(len(self))
        
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(self), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            molecules_batch = []
            properties_batch = []
            
            for idx in batch_indices:
                sample = self[int(idx)]
                molecules_batch.append(sample['molecule'])
                properties_batch.append(sample['properties'])
            
            # Stack arrays - still lazy until evaluated
            molecules = mx.stack(molecules_batch)
            properties = mx.stack(properties_batch)
            
            yield molecules, properties
