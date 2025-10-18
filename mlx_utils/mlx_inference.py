import mlx.core as mx
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for package imports when running as a script
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from mlx_models.mlx_mgcvae import MLXMGCVAE  # type: ignore

# Optional RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
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
        # Use probabilistic sampling instead of argmax to allow diversity
        atom_type_idx = int(mx.random.categorical(atom_type_probs, num_samples=1).item())
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
            
            # Connectivity check: must be a single connected component
            try:
                frags = Chem.GetMolFrags(mol, asMols=False)
                if isinstance(frags, tuple) and len(frags) > 1:
                    return {'smiles': '', 'mol': None, 'valid': False}
            except Exception:
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


## Removed batch helper to keep script minimal

def main():
    parser = argparse.ArgumentParser(description='MLX-MGCVAE inference')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of molecules to sample')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--condition', action='store_true',
                        help='Use property conditioning at inference')
    parser.add_argument('--target', type=float, default=None,
                        help='Target property value (required if --condition)')
    # Minimal flags only

    args = parser.parse_args()
    checkpoint_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'mlx_mgcvae', 'best_model.npz')
    if not os.path.exists(checkpoint_path):
        # Auto-discover best_model.npz anywhere under checkpoints/
        candidates = []
        checkpoints_root = os.path.join(PROJECT_ROOT, 'checkpoints')
        for root, _, files in os.walk(checkpoints_root):
            for fname in files:
                if fname == 'best_model.npz':
                    full = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(full)
                    except Exception:
                        mtime = 0
                    candidates.append((mtime, full))
        if candidates:
            candidates.sort(reverse=True)
            checkpoint_path = candidates[0][1]
            print(f"Using discovered checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found under {checkpoints_root} (looked for best_model.npz)")

    # Build model (mirror training defaults)
    model = MLXMGCVAE(
        node_dim=24,
        edge_dim=6,
        latent_dim=32,
        hidden_dim=64,
        num_properties=1,
        num_layers=2,
        heads=4,
        max_nodes=20,
        beta=1.0,
        gamma=0.3,
        dropout=0.1,
        condition=args.condition,
    )
    model.load_weights(checkpoint_path)

    # Sample latent z ~ N(0, I)
    z = mx.random.normal(shape=(args.num_samples, model.latent_dim))
    # Build properties tensor
    if args.condition:
        if args.target is None:
            raise ValueError("--target is required when --condition is set")
        target_props = mx.full((args.num_samples, model.num_properties), args.target)
    else:
        target_props = mx.zeros((args.num_samples, model.num_properties))
    # Decode and discretize
    decoder_out = model.decode(z, target_props)
    sampled = model.decoder.sample_graph(decoder_out, temperature=args.temperature)

    # Convert to molecules/SMILES (inline loop)
    valid = 0
    smiles_list = []
    valid_mols = []
    seen_signatures = set()
    unique_attempts = 0
    edge_threshold = 0.5
    for i in range(args.num_samples):
        # Build a discrete graph signature (size, node types, edges) to dedupe attempts
        gsize = int(sampled['graph_sizes'][i].item())
        node_probs_i = sampled['node_probs'][i]
        node_types = []
        for n in range(min(gsize, node_probs_i.shape[0])):
            atom_type_probs = node_probs_i[n, :5]
            # Use probabilistic sampling instead of argmax
            node_types.append(int(mx.random.categorical(atom_type_probs, num_samples=1).item()))
        edges = []
        edge_indices = sampled['edge_indices']
        edge_exist_probs = sampled['edge_exist_probs'][i]
        edge_type_probs = sampled['edge_type_probs'][i]
        for e in range(edge_indices.shape[0]):
            a = int(edge_indices[e, 0].item())
            b = int(edge_indices[e, 1].item())
            if a >= gsize or b >= gsize:
                continue
            if float(edge_exist_probs[e].item()) > edge_threshold:
                bt = int(mx.argmax(edge_type_probs[e]).item())
                ia, ib = (a, b) if a <= b else (b, a)
                edges.append((ia, ib, bt))
        signature = (gsize, tuple(node_types), tuple(edges))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_attempts += 1

        # RDKit validity and SMILES
        mol_dict = logits_to_molecule(sampled, graph_idx=i, validity_check=True)
        if mol_dict.get('valid', False):
            valid += 1
            smiles_list.append(mol_dict['smiles'])
            valid_mols.append(mol_dict['mol'])
    print(f"Generated {valid}/{unique_attempts} valid unique molecules")
    # Always save SMILES list to file
    smiles_path = os.path.join(PROJECT_ROOT, 'generated_smiles.txt')
    try:
        with open(smiles_path, 'a') as f:
            for smi in smiles_list:
                f.write(smi + '\n')
        print(f"Saved {len(smiles_list)} unique SMILES to {smiles_path}")
    except Exception as e:
        print(f"Failed to save SMILES: {e}")


if __name__ == '__main__':
    main()