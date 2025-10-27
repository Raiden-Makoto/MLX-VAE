from rdkit import Chem
from rdkit.Chem import AllChem

def compute_strain(smiles_str, ff='MMFF94') -> float:
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol: return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) != 0: return None
    if ff == 'MMFF94':
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None: return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    elif ff == 'UFF':
        ff = AllChem.UFFGetMoleculeForceField(mol)
    else: raise ValueError(f"Unsupported force field: {ff}")
    if ff is None: return None
    ff.Minimize(maxIts=100)
    return ff.CalcEnergy()

def strain_filter(smiles_list, ff='MMFF94', verbose=True):
    filtered_smiles = []
    THRESHOLD = 100.0 # kcal/mol
    
    stats = {
        'high_strain': 0,
        'embedding_failed': 0,
        'force_field_failed': 0,
        'passed': 0
    }
    
    for smiles in smiles_list:
        try:
            strain = compute_strain(smiles, ff)
            if strain is not None:
                if strain <= THRESHOLD:
                    filtered_smiles.append(smiles)
                    stats['passed'] += 1
                else:
                    stats['high_strain'] += 1
                    if verbose: print(f" HIGH STRAIN: '{smiles}' ({strain:.1f} kcal/mol)")
            else:
                stats['embedding_failed'] += 1
                if verbose: print(f" EMBEDDING FAILED: '{smiles}'")
        except Exception as e:
            stats['force_field_failed'] += 1
            if verbose: print(f" FORCE FIELD ERROR: '{smiles}'")
    
    if verbose:
        print(f" Strain Filter Summary:")
        print(f"    High strain (>={THRESHOLD} kcal/mol): {stats['high_strain']} molecules")
        print(f"    Embedding failed: {stats['embedding_failed']} molecules")
        print(f"    Force field failed: {stats['force_field_failed']} molecules")
        print(f"    Passed: {stats['passed']} molecules")
        print()
    
    return filtered_smiles