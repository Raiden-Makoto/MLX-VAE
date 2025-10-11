import torch
import numpy as np
from rdkit import Chem


# =============================================================================
# Molecule Generation from Model Outputs
# =============================================================================

def logits_to_molecule(sampled_graphs, graph_idx=0, validity_check=True):
    """
    Convert sampled graph logits to RDKit molecule and SMILES
    
    Args:
        sampled_graphs: Output from decoder.sample_graph()
        graph_idx: Which graph in the batch to convert (default: 0)
        validity_check: Whether to perform RDKit validity checks
        
    Returns:
        dict: {'smiles': str, 'mol': rdkit.Mol, 'valid': bool}
    """
    
    # =========================================================================
    # Extract Graph Data
    # =========================================================================
    
    # Extract data for specific graph
    graph_size = sampled_graphs['graph_sizes'][graph_idx].item()
    node_probs = sampled_graphs['node_probs'][graph_idx]  # [max_nodes, node_dim]
    edge_exist_probs = sampled_graphs['edge_exist_probs'][graph_idx]  # [num_edges]
    edge_type_probs = sampled_graphs['edge_type_probs'][graph_idx]  # [num_edges, edge_dim]
    edge_indices = sampled_graphs['edge_indices']  # [num_edges, 2]
    
    # =========================================================================
    # Step 1: Sample Discrete Atoms
    # =========================================================================
    
    atoms = []
    
    # Atom type mapping (matches your dataset's node features)
    atom_types = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
    
    for i in range(min(graph_size, node_probs.size(0))):
        # Get atom type probabilities (first 10 dimensions of node features)
        atom_type_probs = node_probs[i, :len(atom_types)]
        atom_type_idx = torch.argmax(atom_type_probs).item()
        atomic_num = atom_types[atom_type_idx]
        atoms.append(atomic_num)
    
    if len(atoms) == 0:
        return {'smiles': '', 'mol': None, 'valid': False}
    
    # =========================================================================
    # Step 2: Create RDKit Molecule
    # =========================================================================
    
    mol = Chem.RWMol()
    
    # Add atoms
    atom_idx_map = {}
    for i, atomic_num in enumerate(atoms):
        atom = Chem.Atom(atomic_num)
        idx = mol.AddAtom(atom)
        atom_idx_map[i] = idx
    
    # =========================================================================
    # Step 3: Sample and Add Bonds
    # =========================================================================
    
    bond_type_mapping = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE, 
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC
    }
    
    edge_threshold = 0.5  # Threshold for bond existence
    
    for edge_idx, (i, j) in enumerate(edge_indices):
        i, j = i.item(), j.item()
        
        # Only consider edges within the actual graph size
        if i >= graph_size or j >= graph_size:
            continue
            
        # Check if edge should exist
        if edge_exist_probs[edge_idx].item() > edge_threshold:
            # Determine bond type
            bond_type_probs = edge_type_probs[edge_idx]
            bond_type_idx = torch.argmax(bond_type_probs).item()
            
            # Map to RDKit bond type
            if bond_type_idx in bond_type_mapping:
                bond_type = bond_type_mapping[bond_type_idx]
                
                try:
                    mol.AddBond(atom_idx_map[i], atom_idx_map[j], bond_type)
                except:
                    # Skip invalid bonds
                    continue
    
    # =========================================================================
    # Step 4: Sanitize and Validate Molecule
    # =========================================================================
    
    try:
        # Convert to Mol object
        mol = mol.GetMol()
        
        if validity_check:
            # Sanitize molecule (adds hydrogens, checks valences, etc.)
            Chem.SanitizeMol(mol)
            
            # Additional validity checks
            if mol is None:
                return {'smiles': '', 'mol': None, 'valid': False}
            
            # Check for reasonable number of atoms
            if mol.GetNumAtoms() == 0:
                return {'smiles': '', 'mol': None, 'valid': False}
        
        # Generate SMILES
        smiles = Chem.MolToSmiles(mol, canonical=True)
        
        return {
            'smiles': smiles,
            'mol': mol,
            'valid': True,
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds()
        }
        
    except Exception as e:
        # Molecule is invalid
        return {
            'smiles': '',
            'mol': None, 
            'valid': False,
            'error': str(e)
        }


def batch_logits_to_molecules(sampled_graphs, validity_check=True):
    """
    Convert a batch of sampled graphs to molecules
    
    Args:
        sampled_graphs: Output from decoder.sample_graph()
        validity_check: Whether to perform RDKit validity checks
        
    Returns:
        list: List of molecule dictionaries
    """
    batch_size = len(sampled_graphs['graph_sizes'])
    molecules = []
    
    for i in range(batch_size):
        mol_dict = logits_to_molecule(sampled_graphs, graph_idx=i, validity_check=validity_check)
        molecules.append(mol_dict)
    
    return molecules


# =============================================================================
# Generation Quality Evaluation
# =============================================================================

def evaluate_generation_quality(
    model,
    target_properties,
    num_samples=100,
    device='mps'
):
    """
    Comprehensive evaluation of generated molecules
    
    Args:
        model: Trained MGCVAE model
        target_properties: Target [BBBP, toxicity] values
        num_samples: Number of molecules to generate
        device: Device to run on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # =========================================================================
    # Generate Molecules
    # =========================================================================
    
    with torch.no_grad():
        sampled_graphs = model.generate(
            target_properties=target_properties,
            num_samples=num_samples,
            temperature=0.8,
            device=device
        )
    
    # =========================================================================
    # Convert to Molecules
    # =========================================================================
    
    molecules = batch_logits_to_molecules(sampled_graphs, validity_check=True)
    
    # =========================================================================
    # Calculate Metrics
    # =========================================================================
    
    valid_molecules = [m for m in molecules if m['valid']]
    valid_smiles = [m['smiles'] for m in valid_molecules]
    
    validity = len(valid_molecules) / len(molecules) * 100
    uniqueness = len(set(valid_smiles)) / len(valid_smiles) * 100 if valid_smiles else 0
    
    # Molecule size statistics
    sizes = [m['num_atoms'] for m in valid_molecules]
    avg_size = np.mean(sizes) if sizes else 0
    
    results = {
        'validity': validity,
        'uniqueness': uniqueness,
        'num_generated': len(molecules),
        'num_valid': len(valid_molecules),
        'valid_smiles': valid_smiles,
        'avg_molecule_size': avg_size,
        'target_properties': target_properties
    }
    
    print(f"Generation Results for target {target_properties}:")
    print(f"Validity: {validity:.1f}%")
    print(f"Uniqueness: {uniqueness:.1f}%") 
    print(f"Avg molecule size: {avg_size:.1f} atoms")
    print(f"Valid SMILES examples: {valid_smiles[:5]}")
    
    return results