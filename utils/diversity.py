from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import matplotlib.pyplot as plt

def fingerprints(smiles_list, radius: int=2, nBits: int=1024):
    """Generate Morgan fingerprints for a list of SMILES"""
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        fps.append(fp)
    return fps

def calculate_diversity(input_list, target_list):
    """Calculate median pairwise similarity between two sets of molecules"""
    fingerprints_A = fingerprints(input_list)
    fingerprints_B = fingerprints(target_list)
    
    if not fingerprints_A or not fingerprints_B:
        return 0.0
    
    # Calculate similarity matrix
    similarity_matrix = [
        [TanimotoSimilarity(fp1, fp2) for fp2 in fingerprints_B] for fp1 in fingerprints_A 
    ]
    
    # Flatten similarity matrix to get all pairwise similarities
    similarities = [sim for row in similarity_matrix for sim in row]
    
    # Return median similarity
    return np.median(similarities)

def plot_similarity_distribution(similarities, title="Tanimoto Similarity Distribution"):
    """Plot histogram of Tanimoto similarities"""
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.axvline(np.mean(similarities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(similarities):.3f}')
    plt.axvline(np.median(similarities), color='green', linestyle='--', 
                label=f'Median: {np.median(similarities):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()
