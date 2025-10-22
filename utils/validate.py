from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import selfies as sf
import pandas as pd

from utils.smarts import preliminary_filter
from utils.sascorer import calculateScore

def validate_selfies(selfies_str):
    if not selfies_str: return None
    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        # Get canonical SMILES to avoid duplicates
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        return {
            'smiles': canonical_smiles,
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'mw': Descriptors.ExactMolWt(mol),
            'sas': calculateScore(mol),
            'qed': QED.qed(mol)
        }
    except Exception as e:
        print(f"Error decoding SELFIES '{selfies_str}': {e}")
        return None

def batch_validate_selfies(selfies_list):
    results = []
    seen_smiles = set()  # Track unique SMILES
    
    for selfies in selfies_list:
        result = validate_selfies(selfies)
        if result is not None:
            # Only add if we haven't seen this canonical SMILES before
            if result['smiles'] not in seen_smiles:
                results.append(result)
                seen_smiles.add(result['smiles'])
    
    # Apply preliminary filtering to remove unstable molecules
    smiles_list = [result['smiles'] for result in results]
    filtered_smiles = preliminary_filter(smiles_list)
    
    # Filter results to only include molecules that passed the preliminary filter
    filtered_results = [result for result in results if result['smiles'] in filtered_smiles]
    
    return filtered_results

if __name__ == "__main__":
    with open('output/generated_molecules.txt', 'r') as f:
        selfies_list = [line.strip() for line in f.readlines()]
    
    results = batch_validate_selfies(selfies_list)
    
    print(f"Total generated: {len(selfies_list)}")
    print(f"Unique valid molecules: {len(results)}")
    print(f"Success rate: {len(results)/len(selfies_list)*100:.2f}% (unique/total)")
    
    # Print results
    for i, data in enumerate(results):
        print(f"Mol {i+1}: {data['smiles']}. LogP: {data['logp']:.2f}, TPSA: {data['tpsa']:.2f}, MW: {data['mw']:.2f}, SAS: {data['sas']:.2f}, QED: {data['qed']:.2f}")
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = 'output/validation_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved validation results to {csv_path}")
        print(f"📊 CSV contains {len(df)} valid molecules with columns: {list(df.columns)}")