from rdkit import Chem

def preliminary_filter(smiles_list):
    INSTABILITY_SMARTS = {
        "peroxide": "[OX2]–[OX2]",                    # –O–O– linkage
        "ozonide": "[OX2]–[OX2]–[OX2]",               # –O–O–O– linkage
        "azide": "[N]=[N+]=[N-]",                     # azide group
        "diazonium": "[NX3+]=[N]",                    # diazonium species
        "small_ring_3": "C1CC1",                      # cyclopropane
        "small_ring_4": "C1CCC1",                     # cyclobutane
        "hypervalent": "[!#1;!#6;!#7;!#8;!#9]",        # atoms heavier than typical valence
    }
    compiled_patterns = {
        name: Chem.MolFromSmarts(pattern.replace("–", "-"))
        for name, pattern in INSTABILITY_SMARTS.items()
    }
    filtered_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue # an error occured
        should_filter = False
        for name, pattern in compiled_patterns.items():
            if mol.HasSubstructMatch(pattern):
                print(f"{smiles} filtered out due to {name}.")
                should_filter = True
                break
        if not should_filter:
            filtered_smiles.append(smiles)
    return filtered_smiles