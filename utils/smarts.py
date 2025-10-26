from rdkit import Chem

def preliminary_filter(smiles_list, verbose=True):
    INSTABILITY_SMARTS = {
        "peroxide": "[OX2]–[OX2]",                    # –O–O– linkage
        "ozonide": "[OX2]–[OX2]–[OX2]",               # –O–O–O– linkage
        "azide": "[N]=[N+]=[N-]",                     # azide group
        "diazonium": "[NX3+]=[N]",                    # diazonium species
        # Removed: "small_ring_3": "C1CC1",                      # cyclopropane - too common
        # Removed: "small_ring_4": "C1CCC1",                     # cyclobutane - too common
        # Removed: "hypervalent": "[!#1;!#6;!#7;!#8;!#9]",        # atoms heavier than typical valence
    }
    compiled_patterns = {
        name: Chem.MolFromSmarts(pattern.replace("–", "-"))
        for name, pattern in INSTABILITY_SMARTS.items()
    }
    
    filtered_smiles = []
    filter_counts = {name: 0 for name in INSTABILITY_SMARTS.keys()}
    filter_counts['invalid_mol'] = 0
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: 
            filter_counts['invalid_mol'] += 1
            if verbose: print(f"❌ INVALID MOLECULE: '{smiles}' (RDKit parsing failed)")
            continue
            
        should_filter = False
        for name, pattern in compiled_patterns.items():
            if mol.HasSubstructMatch(pattern):
                filter_counts[name] += 1
                if verbose: print(f"❌ UNSTABLE: '{smiles}' ({name})")
                should_filter = True
                break
                
        if not should_filter:
            filtered_smiles.append(smiles)
    
    if verbose:
        print(f"📊 Preliminary Filter Summary:")
        for reason, count in filter_counts.items():
            if count > 0:
                print(f"   ❌ {reason}: {count} molecules")
        print(f"   ✅ Passed: {len(filtered_smiles)} molecules")
        print()
    
    return filtered_smiles