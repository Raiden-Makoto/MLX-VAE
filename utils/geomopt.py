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

def strain_filter(smiles_list, ff='MMFF94'):
    filtered_smiles = []
    THRESHOLD = 100.0 # kcal/mol
    for smiles in smiles_list:
        try:
            strain = compute_strain(smiles, ff)
            if strain is not None and strain <= THRESHOLD:
                filtered_smiles.append(smiles)
        except Exception:
            # Skip molecules that cause errors
            print(f"Error computing strain for {smiles}")
            continue
    return filtered_smiles