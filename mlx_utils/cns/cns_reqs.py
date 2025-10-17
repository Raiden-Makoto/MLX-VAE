from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import argparse
import os

def cns_validity(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    # === Molecular Weight ===
    mw = Descriptors.ExactMolWt(mol)
    print(f"Molecular weight: {mw}. CNS target: \u2264 400-450 Da")
    # === Lipophilicity ===
    logp = Descriptors.MolLogP(mol)
    print(f"Lipophilicity: {logp}. CNS target: \u2248 2.0")
    # === Polar Surface Area ===
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    print(f"Polar Surface Area: {tpsa}. CNS target: \u2264 60-90 Å\u00B2")
    # === Hydrogen Bond Donors ===
    hbd = Lipinski.NumHBD(mol)
    print(f"Hydrogen Bond Donors: {hbd}. CNS target: \u2264 1")
    # === Hydrogen Bond Acceptors ===
    hba = Lipinski.NumHBA(mol)
    print(f"Hydrogen Bond Acceptors: {hba}. CNS target: 2-5 N/O atoms")
    # === Rotatable Bonds ===
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    print(f"Rotatable Bonds: {rotatable_bonds}. CNS target: \u2264 10")
    # === Aromatic Rings ===
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    print(f"Aromatic Rings: {aromatic_rings}. CNS target: 2 to 3")
    cns_valid = all([
        400 <= mw <= 450,
        60 <= tpsa <= 90,
        2 <= logp <= 5,
        hbd <= 1,
        2 <= hba <= 5,
        rotatable_bonds <= 10,
        2 <= aromatic_rings <= 3
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
    for smiles in smiles_list[:args.num_samples]:
        smiles = smiles.strip() 
        print(f"Checking: {smiles}")
        print(f"CNS valid: {cns_validity(smiles)}")
        cns_valid_count += 1 if cns_validity(smiles) else 0
        print("-"*70)
    print(f"Percentage CNS valid: {cns_valid_count/args.num_samples*100}%")
    print("="*70)