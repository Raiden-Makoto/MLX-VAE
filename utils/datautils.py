from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


def create_data_loaders(dataset, batch_size=32, train_ratio=0.8, val_ratio=0.1, random_state=67):
    """Split dataset and return DataLoaders for train, val, and test."""
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=1-train_ratio, random_state=random_state)
    val_ratio_adj = val_ratio / (1-train_ratio)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1-val_ratio_adj, random_state=random_state)
    train_ds = [dataset[i] for i in train_idx]
    val_ds   = [dataset[i] for i in val_idx]
    test_ds  = [dataset[i] for i in test_idx]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Dataset splits — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader