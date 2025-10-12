# MLX Data - Dataset for Apple Silicon

This directory contains the MLX implementation of the QM9 molecular graph dataset.

## QM9GraphDataset (MLX Version)

**File:** `mlx-dataset.py`

An in-memory dataset that converts SMILES strings to MLX graph representations with BBB permeability labels.

### Features

- **In-Memory Storage**: All graphs loaded into memory (faster but requires more RAM)
- **MLX-Native**: Uses `mlx.core` arrays and `mlx_graphs.data.GraphData`
- **Feature Extraction**: Same as PyTorch version (29D nodes, 6D edges)
- **Configurable**: Custom SMILES and label column names

### Architecture

```python
class QM9GraphDataset:
    def __init__(
        csv_path: str,           # Path to CSV file
        smiles_col: str = "smiles",     # SMILES column name
        label_col: str = "p_np",        # Label column name
        transform=None,          # Optional transform
        force_reload: bool = False      # Force reload from CSV
    )
```

### Node Features (29D)

Same as PyTorch implementation:

1. **Atomic Number** (10D): One-hot encoding for [H, C, N, O, F, P, S, Cl, Br, I]
2. **Degree** (6D): One-hot for degrees [0, 1, 2, 3, 4, 5]
3. **Formal Charge** (5D): One-hot for [-2, -1, 0, 1, 2]
4. **Hybridization** (6D): One-hot for [S, SP, SP2, SP3, SP3D, SP3D2]
5. **Aromaticity** (1D): Binary flag
6. **Ring Membership** (1D): Binary flag

**Total: 29 dimensions**

### Edge Features (6D)

1. **Bond Type** (4D): One-hot for [SINGLE, DOUBLE, TRIPLE, AROMATIC]
2. **Conjugation** (1D): Binary flag
3. **Ring Membership** (1D): Binary flag

**Total: 6 dimensions**

### Output Format

Each graph is returned as `mlx_graphs.data.GraphData` with:
- `x`: Node features `[num_nodes, 29]`
- `edge_index`: Edge connectivity `[2, num_edges]`
- `edge_attr`: Edge features `[num_edges, 6]`
- `y`: BBB permeability label `[1]`
- `batch`: Batch indices `[num_nodes]` (all zeros for single graph)
- `smiles`: Original SMILES string

### Usage

```python
from mlx_data.mlx_dataset import QM9GraphDataset
from mlx_graphs.loaders import Dataloader

# Load dataset
dataset = QM9GraphDataset(
    csv_path='data/qm9_bbbp2.csv',
    smiles_col='smiles',
    label_col='p_np'  # or 'bbbp' depending on your CSV
)

print(f"Dataset size: {len(dataset)}")

# Access individual graphs
graph = dataset[0]
print(f"Nodes: {graph.x.shape[0]}")
print(f"Edges: {graph.edge_index.shape[1]}")
print(f"Label: {graph.y[0].item()}")

# Create data loader for batching
loader = Dataloader(
    dataset._graphs,
    batch_size=32,
    shuffle=True
)

# Iterate over batches
for batch in loader:
    # batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
    pass
```

### Differences from PyTorch Version

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Storage** | Saves to disk (`.pt` files) | In-memory list |
| **Loading** | Lazy loading from disk | All loaded at init |
| **Class** | Inherits from `torch_geometric.data.Dataset` | Standalone class |
| **Data Format** | `torch.tensor` | `mx.array` |
| **Graph Object** | `torch_geometric.data.Data` | `mlx_graphs.data.GraphData` |
| **Batch Field** | Not included in single graphs | Included (all zeros) |
| **Edge Format** | `.t().contiguous()` | `.T` |

### Molecular Processing

Both versions use **identical RDKit processing**:

1. Parse SMILES → RDKit Mol object
2. Add explicit hydrogens (`Chem.AddHs()`)
3. Extract atom features (element, degree, charge, hybridization, etc.)
4. Extract bond features (type, conjugation, ring)
5. Create bidirectional edges (i→j and j→i for each bond)
6. Convert to graph format

**Result**: Same 29D node features and 6D edge features for any given molecule!

### Performance

- **Fast**: All graphs in memory, no disk I/O during training
- **Memory**: Requires ~100MB RAM for 2,142 molecules
- **Apple Silicon**: Optimized for M-series chips
- **Batching**: Uses `mlx_graphs.loaders.Dataloader` for efficient batching

### Files

- `mlx-dataset.py` - Main dataset implementation
- Compatible with: `data/qm9_bbbp.csv`, `data/qm9_bbbp2.csv`

### Column Names

**Important**: The default label column is `"p_np"`. If your CSV uses a different name (e.g., `"bbbp"`), specify it:

```python
dataset = QM9GraphDataset(
    csv_path='data/qm9_bbbp2.csv',
    label_col='bbbp'  # Specify your column name
)
```

### Validation

✅ **Feature extraction verified** - Produces identical features to PyTorch  
✅ **Graph structure verified** - Same molecular representations  
✅ **Data types verified** - Correct `mx.float32` and `mx.int64`  
✅ **Edge handling verified** - Bidirectional edges created properly  

---

## Requirements

```
mlx
mlx-graphs
pandas
rdkit
```

---

## Notes

- Syntax fixes applied: `Dataloader` (not `DataLoader`), `edge_feat_dim` constant
- Works with same CSV files as PyTorch version
- Can be used interchangeably - just change import statements

