import pandas as pd
import selfies as sf
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

# Load QM9 neutral dataset
df = pd.read_csv('mlx_data/qm9_cns_neutral.csv', usecols=['smiles'])
print(f"Original dataset size: {len(df)}")

# Set SELFIES constraints for QM9
sf.set_semantic_constraints()
constraints = sf.get_semantic_constraints()
constraints['?'] = 3
sf.set_semantic_constraints(constraints)

# Convert SMILES to SELFIES
def smiles_to_selfies(smiles_str):
    try:
        selfies_str = sf.encoder(smiles_str)
        recovered_smiles = sf.decoder(selfies_str)
        if recovered_smiles is not None:
            return selfies_str
    except Exception as e:
        print(f"Failed to convert {smiles_str}: {e}")
    return None

def calculate_properties(smiles_str):
    """Calculate LogP and TPSA for a SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None, None
        
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        return logp, tpsa
    except:
        return None, None

print("Converting SMILES to SELFIES and calculating properties...")
selfies_list = []
logp_list = []
tpsa_list = []
valid_indices = []

for i, smiles in enumerate(df['smiles']):
    if i % 10000 == 0:
        print(f"Processed {i}/{len(df)} molecules")
    
    selfies = smiles_to_selfies(smiles)
    if selfies is not None:
        logp, tpsa = calculate_properties(smiles)
        if logp is not None and tpsa is not None:
            selfies_list.append(selfies)
            logp_list.append(logp)
            tpsa_list.append(tpsa)
            valid_indices.append(i)

print(f"Successfully converted {len(selfies_list)}/{len(df)} molecules with properties")

# Build vocabulary
def get_selfies_vocab(selfies_list):
    alphabet = set()
    for selfies in selfies_list:
        tokens = list(sf.split_selfies(selfies))
        alphabet.update(tokens)
    return sorted(list(alphabet))

vocab = get_selfies_vocab(selfies_list)

# Add special tokens
special_tokens = ['<PAD>', '<START>', '<END>']
all_tokens = special_tokens + vocab

token_to_idx = {token: i for i, token in enumerate(all_tokens)}
idx_to_token = {i: token for token, i in token_to_idx.items()}

print(f"Vocabulary size: {len(token_to_idx)}")

# Tokenize sequences
def tokenize_selfies(selfies_str, max_len):
    tokens = ['<START>'] + list(sf.split_selfies(selfies_str)) + ['<END>']
    indices = [token_to_idx[token] for token in tokens]
    
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices.extend([token_to_idx['<PAD>']] * (max_len - len(indices)))
    
    return indices

# Calculate max length (95th percentile)
lengths = [len(list(sf.split_selfies(selfies))) + 2 for selfies in selfies_list]
max_len = int(np.percentile(lengths, 95))
print(f"Using max_length: {max_len}")

# Tokenize all sequences
print("Tokenizing sequences...")
tokenized_data = []
for selfies in selfies_list:
    tokens = tokenize_selfies(selfies, max_len)
    tokenized_data.append(tokens)

tokenized_data = np.array(tokenized_data, dtype=np.int32)

# Save only essential data
output_data = {
    'tokenized_sequences': tokenized_data.tolist(),
    'token_to_idx': token_to_idx,
    'idx_to_token': idx_to_token,
    'vocab_size': len(token_to_idx),
    'max_length': max_len,
    'logp_values': logp_list,
    'tpsa_values': tpsa_list
}

with open('mlx_data/qm9_cns_selfies.json', 'w') as f:
    json.dump(output_data, f)


np.save('mlx_data/qm9_cns_tokenized.npy', tokenized_data)

print("Minimal data saved!")
print(f"Tokenized data shape: {tokenized_data.shape}")