from rdkit import Chem
from rdkit.Chem import Descriptors
import selfies as sf

import os

def validate_selfies(selfies_str):
    if not selfies_str: return None
    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return {
            'smiles': smiles,
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'mw' : Descriptors.ExactMolWt(mol),
        }
    except Exception as e:
        print(f"Error decoding SELFIES '{selfies_str}': {e}")
        return None

def batch_validate_selfies(selfies_list):
    results = []
    for selfies in selfies_list:
        result = validate_selfies(selfies)
        if result is not None:
            results.append(result)
    return results

if __name__ == "__main__":
    with open('generated_molecules.txt', 'r') as f:
        selfies_list = [line.strip() for line in f.readlines()]
    results = batch_validate_selfies(selfies_list)
    print(f"Valid molecules: {len(results)}/{len(selfies_list)}, {len(results)/len(selfies_list)*100:.2f}% success rate")
    for i, data in enumerate(results):
        print(f"Mol {i+1}: {data['smiles']}. LogP: {data['logp']:.2f}, TPSA: {data['tpsa']:.2f}, MW: {data['mw']:.2f}")