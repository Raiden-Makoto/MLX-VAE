import mlx.core as mx
import numpy as np

# Optional RDKit import
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =============================================================================
# Molecule Generation from Model Outputs
# =============================================================================

def logits_to_molecule(sampled_graphs, graph_idx=0, validity_check=True):
    """
    Convert sampled graph logits to RDKit molecule and SMILES (MLX version)
    
    Args:
        sampled_graphs: Output from decoder.sample_graph()
        graph_idx: Which graph in the batch to convert (default: 0)
        validity_check: Whether to perform RDKit validity checks
        
    Returns:
        dict: {'smiles': str, 'mol': rdkit.Mol, 'valid': bool}
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule generation. Install with: pip install rdkit")
    
    # =========================================================================
    # Extract Graph Data
    # =========================================================================
    
    # Extract data for specific graph
    graph_size = int(sampled_graphs['graph_sizes'][graph_idx].item())
    node_probs = sampled_graphs['node_probs'][graph_idx]  # [max_nodes, node_dim]
    edge_exist_probs = sampled_graphs['edge_exist_probs'][graph_idx]  # [num_edges]
    edge_type_probs = sampled_graphs['edge_type_probs'][graph_idx]  # [num_edges, edge_dim]
    edge_indices = sampled_graphs['edge_indices']  # [num_edges, 2]
    
    # =========================================================================
    # Step 1: Sample Discrete Atoms
    # =========================================================================
    
    atoms = []
    
    # Atom type mapping (matches your dataset's node features)
    atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
    
    for i in range(min(graph_size, node_probs.shape[0])):
        # Get atom type probabilities (first 5 dimensions for atom types)
        atom_type_probs = node_probs[i, :len(atom_types)]
        atom_type_idx = int(mx.argmax(atom_type_probs).item())
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
    
    for edge_idx in range(edge_indices.shape[0]):
        i = int(edge_indices[edge_idx, 0].item())
        j = int(edge_indices[edge_idx, 1].item())
        
        # Only consider edges within the actual graph size
        if i >= graph_size or j >= graph_size:
            continue
            
        # Check if edge should exist
        if float(edge_exist_probs[edge_idx].item()) > edge_threshold:
            # Determine bond type
            bond_type_probs = edge_type_probs[edge_idx]
            bond_type_idx = int(mx.argmax(bond_type_probs).item())
            
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
    Convert a batch of sampled graphs to molecules (MLX version)
    
    Args:
        sampled_graphs: Output from decoder.sample_graph()
        validity_check: Whether to perform RDKit validity checks
        
    Returns:
        list: List of molecule dictionaries
    """
    batch_size = sampled_graphs['graph_sizes'].shape[0]
    molecules = []
    
    for i in range(batch_size):
        mol_dict = logits_to_molecule(sampled_graphs, graph_idx=i, validity_check=validity_check)
        molecules.append(mol_dict)
    
    return molecules