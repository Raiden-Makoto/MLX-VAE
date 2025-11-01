import pandas as pd
import json
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors
from chembl_webresource_client.new_client import new_client
import numpy as np
import tqdm
import os

print("Downloading ChEMBL molecules with CNS properties...")

# Set SELFIES constraints
sf.set_semantic_constraints()
constraints = sf.get_semantic_constraints()
constraints['?'] = 3
sf.set_semantic_constraints(constraints)

# Output file
output_path = 'mlx_data/chembl_cns_selfies.json'
output_npy = 'mlx_data/chembl_cns_tokenized.npy'

# Fresh start: initialize empty accumulator (wipe any prior state)
    output_data = []
    existing_smiles = set()

def smiles_to_selfies(smiles_str):
    try:
        selfies_str = sf.encoder(smiles_str)
        recovered_smiles = sf.decoder(selfies_str)
        if recovered_smiles is not None:
            return selfies_str
    except:
        pass
    return None

def calculate_properties(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        
        tpsa = Descriptors.TPSA(mol)
        return tpsa
    except:
        return None

def save_intermediate():
    """Save current data to JSON"""
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

# Connect to ChEMBL
molecule = new_client.molecule

print("Querying ChEMBL for CNS molecules...")
try:
    # Get molecules with CNS flag = 1 - fetch full records
    results = molecule.filter(
        molecule_type='Small molecule',
        cns_flag=1
    )
    print(f"Fetching CNS molecules from ChEMBL...")
except Exception as e:
    print(f"Error querying ChEMBL: {e}")
    results = []

# Process molecules - no LogP filter
print("Processing CNS molecules...")
target_count = 20000
processed = len(output_data)
skipped = 0

print(f"Starting from {processed} molecules already collected")

for i, record in enumerate(tqdm.tqdm(results, desc="Processing molecules")):
    if processed >= target_count:
        break
    
    try:
        # Extract SMILES from molecule_structures
        structures = record.get('molecule_structures', {})
        smiles = structures.get('canonical_smiles') if isinstance(structures, dict) else None
        
        if not smiles:
            skipped += 1
            continue
        
        # Skip if we've already seen this SMILES
        if smiles in existing_smiles:
            continue
        
        # Calculate properties (TPSA only)
        tpsa = calculate_properties(smiles)
        if tpsa is None:
            skipped += 1
            continue
        
        # Convert to SELFIES
        selfies = smiles_to_selfies(smiles)
        if selfies is None:
            skipped += 1
            continue
        
        # Append to output data (store only SELFIES and TPSA)
        molecule_entry = {
            'selfies': selfies,
            'tpsa': tpsa
        }
        output_data.append(molecule_entry)
        existing_smiles.add(smiles)  # Track as seen to avoid duplicates
        processed += 1
        # Write to JSON immediately every molecule
        save_intermediate()
        
    except Exception as e:
        skipped += 1
        continue

print(f"\n Processed {processed} CNS molecules")
print(f"   Total in file: {len(output_data)}")

if processed == 0:
    print("No molecules found matching criteria.")
    exit(1)

# Final processing: build vocab, tokenize, and compute normalization stats
print("\nBuilding vocabulary and tokenizing...")

def get_selfies_vocab(molecules):
    alphabet = set()
    for mol in molecules:
        tokens = list(sf.split_selfies(mol['selfies']))
        alphabet.update(tokens)
    return sorted(list(alphabet))

vocab = get_selfies_vocab(output_data)

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
lengths = [len(list(sf.split_selfies(mol['selfies']))) + 2 for mol in output_data]
max_len = int(np.percentile(lengths, 95))
print(f"Using max_length: {max_len}")

# Tokenize all sequences
print("Tokenizing sequences...")
tokenized_data = []
for mol in output_data:
    tokens = tokenize_selfies(mol['selfies'], max_len)
    tokenized_data.append(tokens)

tokenized_data = np.array(tokenized_data, dtype=np.int32)

# Final save with all metadata
tpsa_values = [mol['tpsa'] for mol in output_data]
tpsa_mean = float(np.mean(tpsa_values)) if tpsa_values else 0.0
tpsa_std = float(np.std(tpsa_values)) if tpsa_values else 1.0

final_output = {
    'tokenized_sequences': tokenized_data.tolist(),
    'token_to_idx': token_to_idx,
    'idx_to_token': idx_to_token,
    'vocab_size': len(token_to_idx),
    'max_length': max_len,
    'molecules': output_data,
    'tpsa_mean': tpsa_mean,
    'tpsa_std': tpsa_std
}

with open(output_path, 'w') as f:
    json.dump(final_output, f, indent=2)

np.save(output_npy, tokenized_data)

print(f"\n Saved {len(output_data)} ChEMBL CNS molecules to {output_path}")
    print(f"   Average TPSA: {np.mean(tpsa_values):.2f}")
