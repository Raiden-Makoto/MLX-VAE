# Message-Passing Neural Network (MPNN) for Blood-Brain Barrier Permeability Prediction

A PyTorch implementation of a Message-Passing Neural Network for predicting blood-brain barrier (BBB) permeability of small molecules from their SMILES representations.

## Overview

This module implements an MPNN that learns molecular graph representations to predict whether a molecule can cross the blood-brain barrier. The model achieves **~0.92 ROC-AUC** on the BBBP (Blood-Brain Barrier Penetration) test set.

### Key Features

- **Graph-based molecular representation**: Converts SMILES strings to molecular graphs
- **Edge-conditioned message passing**: Uses bond features to compute edge-specific message transformations
- **GRU state updates**: Maintains and updates node representations across message passing steps
- **GPU acceleration**: Supports MPS (Apple Silicon), CUDA, and CPU devices
- **Easy inference API**: Simple function calls for single molecule prediction

## Architecture

### Model Components

```
Input SMILES → Molecular Graph → MPNN → BBB Permeability Score
```

1. **Node Features (12-dim per atom)**:
   - Element one-hot encoding (C, N, O, F, P, S, Cl, Br, I)
   - Atom degree
   - Total valence
   - Number of hydrogens

2. **Edge Features (4-dim per bond)**:
   - Bond type (single, double, triple)
   - Conjugation status

3. **Message Passing**:
   - Edge Network: Computes edge-specific weight matrices from bond features
   - 4 message passing steps with GRU updates
   - Global mean pooling for graph-level representation

4. **Readout Network**:
   - 2-layer MLP (32 → 512 → 1)
   - Sigmoid activation for binary classification

### Network Diagram

```
Node Features (12) → Linear → Hidden (32)
                                    ↓
                     ┌──────────────┴──────────────┐
                     │   Message Passing (×4)      │
                     │                             │
                     │  Edge Net → Messages        │
                     │  GRU → Updated Hidden       │
                     └──────────────┬──────────────┘
                                    ↓
                     Global Mean Pooling
                                    ↓
                     MLP (512) → Sigmoid → Score
```

## Installation

### Prerequisites

```bash
# Python 3.11+ recommended
# Install PyTorch with appropriate backend
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch-geometric

# Install RDKit for molecular processing
pip install rdkit

# Install other dependencies
pip install pandas scikit-learn tqdm
```

### Environment Setup

```bash
# Clone or navigate to project directory
cd QVAE

# Install from requirements.txt (recommended)
pip install -r requirements.txt
```

## Dataset

### BBBP Dataset

The **Blood-Brain Barrier Penetration (BBBP)** dataset contains 2,039 molecules with binary labels:
- **Positive (1)**: Molecule crosses the BBB
- **Negative (0)**: Molecule does not cross the BBB

**Source**: [MoleculeNet](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv)

**Data Format**:
```csv
num,name,smiles,p_np
1,Propanolol,C1=CC=C(C=C1)OCCNCC(COC2=CC=CC=C2)O,1
2,Clonidine,C1CN=C(N1)NC2=C(C=CC=C2Cl)Cl,1
...
```

## Usage

### Training a Model

Use the `mpnn.ipynb` notebook for training:

```python
# 1. Load dataset
dataset = BBBPDataset("BBBP.csv")

# 2. Split into train/test
train_ds, test_ds = train_test_split(dataset, test_size=0.2, random_state=67)

# 3. Create data loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 4. Initialize model
model = MPNN(
    node_dim=12,  # Atom features
    edge_dim=4,   # Bond features
    message_dim=32,
    num_steps=4,
    hidden_dim=512
).to(device)

# 5. Train
optimizer = Adam(model.parameters(), lr=5e-4)
criterion = nn.BCELoss()

for epoch in range(40):
    # Training loop...
    pass
```

### Inference on Single Molecules

```python
from mpnn import load_checkpoint, predict_single_molecule
import torch

# Load trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, _ = load_checkpoint("../checkpoints/mpnn_bbbp.pt", device=device)

# Predict BBB permeability
smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
score = predict_single_molecule(model, smiles, device=device)

print(f"BBB Permeability Score: {score:.4f}")
print(f"Prediction: {'Permeable' if score > 0.5 else 'Non-permeable'}")
```

### Batch Inference

```python
import pandas as pd

# Load molecules
df = pd.read_csv("molecules.csv")

# Predict for all molecules
df['bbbp_score'] = df['smiles'].apply(
    lambda smiles: predict_single_molecule(model, smiles, device=device)
)

# Filter permeable molecules
permeable = df[df['bbbp_score'] > 0.5]
```

## API Reference

### Classes

#### `MPNN(node_dim, edge_dim, message_dim, num_steps, hidden_dim)`

Message-Passing Neural Network model.

**Parameters**:
- `node_dim` (int): Dimension of node features (default: 12)
- `edge_dim` (int): Dimension of edge features (default: 4)
- `message_dim` (int): Hidden dimension for message passing (default: 32)
- `num_steps` (int): Number of message passing iterations (default: 4)
- `hidden_dim` (int): Hidden dimension for readout MLP (default: 512)

**Methods**:
- `forward(data)`: Forward pass returning BBB permeability scores

#### `EdgeNetwork(edge_dim, message_dim)`

Edge-conditioned message passing layer.

**Parameters**:
- `edge_dim` (int): Dimension of edge features
- `message_dim` (int): Hidden dimension for messages

### Functions

#### `smiles_to_graph(smiles)`

Converts a SMILES string to a PyTorch Geometric Data object.

**Parameters**:
- `smiles` (str): SMILES representation of molecule

**Returns**:
- `Data`: PyTorch Geometric graph object with node/edge features, or `None` if conversion fails

**Example**:
```python
from mpnn import smiles_to_graph

graph = smiles_to_graph("CCO")  # Ethanol
print(f"Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")
```

#### `predict_single_molecule(model, smiles, device='mps')`

Runs inference on a single molecule.

**Parameters**:
- `model` (MPNN): Trained model
- `smiles` (str): SMILES string of molecule
- `device` (str): Device to run on ('cpu', 'cuda', 'mps')

**Returns**:
- `float`: BBB permeability score [0, 1], or `None` if prediction fails

**Example**:
```python
score = predict_single_molecule(model, "c1ccccc1", device="cpu")  # Benzene
```

#### `load_checkpoint(checkpoint_path, device='mps')`

Loads a saved model checkpoint.

**Parameters**:
- `checkpoint_path` (str): Path to `.pt` checkpoint file
- `device` (str): Device to load model on

**Returns**:
- `tuple`: (model, checkpoint_dict)

**Example**:
```python
model, checkpoint = load_checkpoint("../checkpoints/mpnn_bbbp.pt")
```

## Model Performance

### Results on BBBP Test Set

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9161 |
| Accuracy | ~85% |
| Training Time | ~10 min (40 epochs on MPS) |

### Example Predictions

| Molecule | SMILES | Score | Prediction |
|----------|--------|-------|------------|
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(O)=O` | 0.17 | Non-permeable |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | 0.41 | Non-permeable |
| Benzene | `c1ccccc1` | 0.95 | Permeable |
| Caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | 0.89 | Permeable |

## Files

```
mpnn/
├── readme.md           # This file
├── mpnn.py             # Model implementation and utilities
├── mpnn.ipynb          # Training notebook
├── BBBP.csv            # Training dataset
└── ../checkpoints/
    └── mpnn_bbbp.pt    # Trained model weights
```

## Implementation Details

### Molecular Featurization

**Why no 3D coordinates?**
- MPNN only requires 2D graph structure (connectivity)
- 3D conformer generation is slow and often fails for complex molecules
- Bond types and atom properties capture sufficient information

**Atom Features**:
```python
# One-hot encoding for common elements
elements = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

# Structural properties
features = [
    *one_hot_element,  # 9 dimensions
    degree,            # Number of bonds
    valence,           # Total valence
    num_hydrogens      # Implicit + explicit H
]  # Total: 12 dimensions
```

**Edge Features**:
```python
features = [
    is_single,      # Single bond
    is_double,      # Double bond
    is_triple,      # Triple bond
    is_conjugated   # Aromatic/conjugated
]  # Total: 4 dimensions
```

### Training Configuration

```python
# Optimizer
Adam(lr=5e-4)

# Loss
BCELoss()  # Binary Cross-Entropy

# Training
batch_size = 32
epochs = 40
train/test split = 80/20

# Hardware
Device: MPS (Apple Silicon GPU)
Training time: ~10 minutes
```

## Troubleshooting

### Common Issues

**1. RDKit can't parse SMILES**
```python
# Issue: Invalid SMILES string
Failed to parse SMILES: invalid_string

# Solution: Check SMILES validity
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print("Invalid SMILES")
```

**2. Out of Memory**
```python
# Issue: Large batch size on limited GPU memory

# Solution: Reduce batch size
loader = DataLoader(dataset, batch_size=16)  # Instead of 32
```

**3. Device compatibility**
```python
# Issue: MPS not available

# Solution: Use CPU fallback
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## References

1. **Gilmer et al. (2017)** - "Neural Message Passing for Quantum Chemistry"  
   https://arxiv.org/abs/1704.01212

2. **MoleculeNet** - BBBP Dataset  
   https://moleculenet.org/datasets-1

3. **PyTorch Geometric** - Graph Neural Networks in PyTorch  
   https://pytorch-geometric.readthedocs.io/

## Citation

If you use this code, please cite:

```bibtex
@software{mpnn_bbbp,
  title={MPNN for Blood-Brain Barrier Permeability Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/QVAE}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

**Last Updated**: October 9, 2024  
**PyTorch Version**: 2.8.0  
**Python Version**: 3.11+

