from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import argparse
import os

def cns_validity(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    # === Molecular Weight ===
    mw = Descriptors.ExactMolWt(mol)
    print(f"Molecular weight: {mw:.4f}. CNS target: ≤ 450 Da")
    # === Lipophilicity ===
    logp = Descriptors.MolLogP(mol)
    print(f"Lipophilicity: {logp:.4f}. CNS target: 2.0-5.0")
    # === Polar Surface Area ===
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    print(f"Polar Surface Area: {tpsa:.4f}. CNS target: 40-90 Å²")
    # === Hydrogen Bond Donors ===
    try:
        hbd = Lipinski.NumHDonors(mol)  # Newer RDKit versions
    except AttributeError:
        hbd = Lipinski.NumHBD(mol)  # Older RDKit versions
    print(f"Hydrogen Bond Donors: {hbd:.4f}. CNS target: ≤ 1")
    # === Hydrogen Bond Acceptors ===
    try:
        hba = Lipinski.NumHAcceptors(mol)  # Newer RDKit versions
    except AttributeError:
        hba = Lipinski.NumHBA(mol)  # Older RDKit versions
    print(f"Hydrogen Bond Acceptors: {hba:.4f}. CNS target: 2-5 N/O atoms")
    # === Rotatable Bonds ===
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    print(f"Rotatable Bonds: {rotatable_bonds:.4f}. CNS target: ≤ 10")
    # === Aromatic Rings ===
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    print(f"Aromatic Rings: {aromatic_rings:.4f}. CNS target: 2 to 3")
    cns_valid = all([
        mw <= 450,  # Molecular weight
        40 <= tpsa <= 90,  # Polar surface area
        2 <= logp <= 5,    # Lipophilicity
        hbd <= 1,          # Hydrogen bond donors
        2 <= hba <= 5,     # Hydrogen bond acceptors
        rotatable_bonds <= 10,  # Rotatable bonds
        2 <= aromatic_rings <= 3  # Aromatic rings
    ])
    return cns_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check CNS validity of a SMILES string")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to check")
    parser.add_argument("--input-file", type=str, default="generated_smiles.txt", help="Input file containing SMILES strings")
    args = parser.parse_args()
    with open(args.input_file, "r") as f:
        smiles_list = f.readlines()
    cns_valid_count = 0
    total_checked = 0
    for smiles in smiles_list[:args.num_samples]:
        smiles = smiles.strip()
        if not smiles:  # Skip empty lines
            continue
        print(f"Checking: {smiles}")
        is_valid = cns_validity(smiles)
        print(f"CNS valid: {is_valid}")
        if is_valid:
            cns_valid_count += 1
        total_checked += 1
        print("-"*70)
    
    if total_checked > 0:
        percentage = (cns_valid_count / total_checked) * 100
        print(f"Percentage CNS valid: {percentage:.1f}% ({cns_valid_count}/{total_checked})")
    else:
        print("No valid SMILES found to check")
    print("="*70)