from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import selfies as sf
import pandas as pd

from utils.smarts import preliminary_filter
from utils.sascorer import calculateScore
from utils.geomopt import strain_filter

def validate_selfies(selfies_str, verbose=False):
    if not selfies_str: 
        if verbose: print(f" EMPTY SELFIES: '{selfies_str}'")
        return None
    
    try:
        smiles = sf.decoder(selfies_str)
        if not smiles or smiles.strip() == '': 
            if verbose: print(f" EMPTY SMILES: SELFIES '{selfies_str}' -> empty SMILES")
            return None
        
        # Filter out charged molecules (only actual charges outside brackets)
        # Brackets can contain formal charges like [NH3+] which are valid
        if '+' in smiles.replace('[', '').replace(']', '') or '-' in smiles.replace('[', '').replace(']', ''):
            if verbose: print(f" CHARGED MOLECULE: '{smiles}' (contains charges outside brackets)")
            return None
        
        # Filter out radicals (only actual radicals with dots)
        # Valid bracketed atoms like [NH1], [nH], [cH] should pass
        if '[.' in smiles or '.]' in smiles:
            if verbose: print(f" RADICAL: '{smiles}' (contains radical dots)")
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            if verbose: print(f" INVALID STRUCTURE: '{smiles}' (RDKit cannot parse)")
            return None
        
        # Get canonical SMILES to avoid duplicates
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        # Don't print successful validations - only failures
        return {
            'smiles': canonical_smiles,
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'mw': Descriptors.ExactMolWt(mol),
            'sas': calculateScore(mol),
            'qed': QED.qed(mol)
        }
    except Exception as e:
        if verbose: print(f" DECODE ERROR: SELFIES '{selfies_str}' -> {e}")
        return None

def batch_validate_selfies(selfies_list, verbose=True):
    results = []
    seen_smiles = set()  # Track unique SMILES
    
    # Counters for detailed reporting
    stats = {
        'total_input': len(selfies_list),
        'empty_selfies': 0,
        'decode_errors': 0,
        'empty_smiles': 0,
        'charged_molecules': 0,
        'radicals': 0,
        'invalid_structures': 0,
        'duplicates': 0,
        'preliminary_filtered': 0,
        'strain_filtered': 0,
        'final_valid': 0
    }
    
    if verbose:
        print(f" DETAILED VALIDATION REPORT")
        print(f"=" * 60)
        print(f"Input: {stats['total_input']} SELFIES sequences")
        print()
    
    # Step 1: Basic validation
    if verbose: print("STEP 1: Basic SELFIES → SMILES validation")
    basic_valid = []
    
    for i, selfies in enumerate(selfies_list):
        if not selfies or selfies.strip() == '':
            stats['empty_selfies'] += 1
            if verbose: print(f" EMPTY SELFIES: '{selfies}'")
            continue
            
        result = validate_selfies(selfies, verbose=verbose)
        if result is not None:
            # Check for duplicates
            if result['smiles'] in seen_smiles:
                stats['duplicates'] += 1
                if verbose: print(f" DUPLICATE: '{result['smiles']}' (already seen)")
            else:
                basic_valid.append(result)
                seen_smiles.add(result['smiles'])
        else:
            # Count specific failure types
            try:
                smiles = sf.decoder(selfies)
                if not smiles or smiles.strip() == '':
                    stats['empty_smiles'] += 1
                    if verbose: print(f" EMPTY SMILES: '{smiles}'")
                elif '+' in smiles.replace('[', '').replace(']', '') or '-' in smiles.replace('[', '').replace(']', ''):
                    stats['charged_molecules'] += 1
                    if verbose: print(f" CHARGED: '{smiles}'")
                elif '[.' in smiles or '.]' in smiles:
                    stats['radicals'] += 1
                    if verbose: print(f" RADICAL: '{smiles}'")
                else:
                    stats['invalid_structures'] += 1
                    if verbose: print(f" INVALID: '{smiles}'")
            except:
                stats['decode_errors'] += 1
                if verbose: print(f" DECODE ERROR: '{selfies}'")
    
    stats['basic_valid'] = len(basic_valid)
    
    if verbose:
        print(f"\n Basic Validation Summary:")
        print(f"    Valid molecules: {stats['basic_valid']}")
        print(f"    Empty SELFIES: {stats['empty_selfies']}")
        print(f"    Decode errors: {stats['decode_errors']}")
        print(f"    Empty SMILES: {stats['empty_smiles']}")
        print(f"    Charged molecules: {stats['charged_molecules']}")
        print(f"    Radicals: {stats['radicals']}")
        print(f"    Invalid structures: {stats['invalid_structures']}")
        print(f"    Duplicates: {stats['duplicates']}")
        print()
    
    if not basic_valid:
        if verbose: print(" No valid molecules after basic validation!")
        return results
    
    # Step 2: Preliminary filtering
    if verbose: print("STEP 2: Preliminary filtering (unstable molecules)")
    smiles_list = [result['smiles'] for result in basic_valid]
    filtered_smiles = preliminary_filter(smiles_list)
    stats['preliminary_filtered'] = len(smiles_list) - len(filtered_smiles)
    
    if verbose:
        print(f"    Filtered out: {stats['preliminary_filtered']} molecules")
        print()
    
    # Step 3: Strain filtering
    if verbose: print("STEP 3: Strain filtering (high energy conformations)")
    final_filtered_smiles = strain_filter(filtered_smiles)
    stats['strain_filtered'] = len(filtered_smiles) - len(final_filtered_smiles)
    
    if verbose:
        print(f"    Filtered out: {stats['strain_filtered']} molecules")
        print()
    
    # Filter results to only include final valid molecules
    filtered_smiles_set = set(final_filtered_smiles)
    final_results = [result for result in basic_valid if result['smiles'] in filtered_smiles_set]
    stats['final_valid'] = len(final_results)
    
    if verbose:
        print(f" FINAL SUMMARY:")
        print(f"=" * 60)
        print(f"Input molecules:     {stats['total_input']:3d}")
        print(f"Basic validation:    {stats['basic_valid']:3d} ({stats['basic_valid']/stats['total_input']*100:.1f}%)")
        print(f"Preliminary filter:  {len(filtered_smiles):3d} ({len(filtered_smiles)/stats['total_input']*100:.1f}%)")
        print(f"Strain filter:       {stats['final_valid']:3d} ({stats['final_valid']/stats['total_input']*100:.1f}%)")
        print(f"=" * 60)
        print(f"Total loss: {stats['total_input'] - stats['final_valid']:3d} molecules")
        print(f"Success rate: {stats['final_valid']/stats['total_input']*100:.1f}%")
        print()
    
    return final_results

if __name__ == "__main__":
    with open('output/generated_molecules.txt', 'r') as f:
        selfies_list = [line.strip() for line in f.readlines()]
    
    results = batch_validate_selfies(selfies_list)
    
    print(f"Total unique valid, stable, low strain molecules: {len(results)}")
    print(f"Success rate: {len(results)/len(selfies_list)*100:.2f}% (unique/total)")
    
    # Print results
    for i, data in enumerate(results):
        print(f"Mol {i+1}: {data['smiles']}. LogP: {data['logp']:.2f}, TPSA: {data['tpsa']:.2f}, MW: {data['mw']:.2f}, SAS: {data['sas']:.2f}, QED: {data['qed']:.2f}")
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = 'output/validation_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved validation results to {csv_path}")
        print(f"CSV contains {len(df)} valid molecules with columns: {list(df.columns)}")