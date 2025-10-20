# script to visualize the generated molecules
# load molecules from the csv file
# display molecules in image with molecular properties as caption

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid converter issues
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from PIL import Image
import io

def load_molecules_from_csv(csv_path='output/validation_results.csv'):
    """Load validated molecules from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def create_molecule_grid(df, max_molecules=20, figsize=(15, 10)):
    """Create a grid visualization of molecules with properties as captions"""
    
    # Try to get working molecules, skip errors
    working_molecules = []
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                # Test if we can draw it
                img = Draw.MolToImage(mol, size=(200, 200))
                working_molecules.append((idx, row))
                if len(working_molecules) >= max_molecules:
                    break
        except Exception as e:
            print(f"Skipping molecule {idx+1} due to error: {str(e)[:50]}...")
            continue
    
    n_molecules = len(working_molecules)
    if n_molecules == 0:
        print("No molecules could be visualized!")
        return None
    
    # Calculate grid dimensions
    cols = 5
    rows = (n_molecules + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, (idx, row) in enumerate(working_molecules):
        ax = axes_flat[i]
        
        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(row['smiles'])
            
            # Draw molecule using PIL to avoid converter issues
            img = Draw.MolToImage(mol, size=(200, 200))
            ax.imshow(img)
            
            # Create caption with properties
            caption = f"LogP: {row['logp']:.2f}\nTPSA: {row['tpsa']:.1f}\nMW: {row['mw']:.1f}"
            ax.text(0.5, -0.1, caption, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f"Molecule {i+1}", fontsize=10)
            
        except Exception as e:
            print(f"Error visualizing molecule {i+1}: {str(e)[:50]}...")
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='red')
            ax.set_title(f"Molecule {i+1}", fontsize=10)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Hide unused subplots
    for i in range(n_molecules, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f"Generated Molecules ({n_molecules} successfully visualized)", 
                 fontsize=16, y=0.98)
    
    return fig

def create_property_distributions(df):
    """Create distribution plots for molecular properties"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # LogP distribution
    axes[0, 0].hist(df['logp'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('LogP Distribution')
    axes[0, 0].set_xlabel('LogP')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(df['logp'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["logp"].mean():.2f}')
    axes[0, 0].legend()
    
    # TPSA distribution
    axes[0, 1].hist(df['tpsa'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('TPSA Distribution')
    axes[0, 1].set_xlabel('TPSA')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(df['tpsa'].mean(), color='red', linestyle='--',
                      label=f'Mean: {df["tpsa"].mean():.1f}')
    axes[0, 1].legend()
    
    # MW distribution
    axes[1, 0].hist(df['mw'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Molecular Weight Distribution')
    axes[1, 0].set_xlabel('MW (Da)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].axvline(df['mw'].mean(), color='red', linestyle='--',
                      label=f'Mean: {df["mw"].mean():.1f}')
    axes[1, 0].legend()
    
    # LogP vs TPSA scatter
    scatter = axes[1, 1].scatter(df['logp'], df['tpsa'], c=df['mw'], 
                                cmap='viridis', alpha=0.7, s=50)
    axes[1, 1].set_title('LogP vs TPSA (colored by MW)')
    axes[1, 1].set_xlabel('LogP')
    axes[1, 1].set_ylabel('TPSA')
    plt.colorbar(scatter, ax=axes[1, 1], label='MW (Da)')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load molecules
    print("📊 Loading molecules from CSV...")
    df = load_molecules_from_csv()
    print(f"✅ Loaded {len(df)} valid molecules")
    
    # Create molecule grid visualization
    print("🎨 Creating molecule grid visualization...")
    fig1 = create_molecule_grid(df, max_molecules=20)
    if fig1 is not None:
        fig1.savefig('output/molecule_grid.png', dpi=300, bbox_inches='tight')
        print("💾 Saved molecule grid to output/molecule_grid.png")
    else:
        print("❌ Could not create molecule grid visualization")
   
    # plt.show()  # Commented out to avoid hanging terminal