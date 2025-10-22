#!/usr/bin/env python3
"""
Comprehensive molecule generation and analysis pipeline.
Combines sampling, validation, and visualization into one script.
"""

import argparse
import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from PIL import Image
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vae import SelfiesVAE
from utils.sample import sample_from_vae, tokens_to_selfies, load_best_model
from utils.validate import validate_selfies, batch_validate_selfies
from utils.visualize import create_molecule_grid

def load_data():
    """Load vocabulary and metadata"""
    with open('mlx_data/qm9_cns_selfies.json', 'r') as f:
        meta = json.load(f)
    return meta['token_to_idx'], meta['idx_to_token'], meta['vocab_size']

def generate_molecules(model, num_samples, temperature=1.0, top_k=10):
    """Generate molecules using the VAE"""
    print(f"🎲 Generating {num_samples} molecules with temperature {temperature} and top_k {top_k}...")
    
    # Generate sequences
    samples = sample_from_vae(model, num_samples, temperature, top_k)
    
    # Convert to SELFIES
    token_to_idx, idx_to_token, vocab_size = load_data()
    selfies_list = tokens_to_selfies(samples)
    
    # Filter out empty sequences
    selfies_list = [s for s in selfies_list if s]
    
    print(f"✅ Generated {len(selfies_list)} non-empty SELFIES sequences")
    return selfies_list

def validate_molecules(selfies_list):
    """Validate generated molecules and convert to SMILES"""
    print(f"🔍 Validating {len(selfies_list)} molecules...")
    
    # Validate and convert to SMILES
    results = batch_validate_selfies(selfies_list)
    
    if not results:
        print("❌ No valid molecules generated!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    total_generated = len(selfies_list)
    total_valid = len(results)
    unique_valid = len(df['smiles'].unique())
    
    print(f"📊 Validation Results:")
    print(f"   Total generated: {total_generated}")
    print(f"   Valid molecules: {total_valid}")
    print(f"   Unique valid: {unique_valid}")
    print(f"   Success rate: {unique_valid/total_generated:.1%}")
    
    return df

def visualize_molecules(df, max_molecules=20):
    """Create visualizations of the generated molecules"""
    print(f"🎨 Creating visualizations for {len(df)} molecules...")
    
    # Create molecule grid
    fig = create_molecule_grid(df, max_molecules=max_molecules)
    if fig is not None:
        fig.savefig('output/molecule_grid.png', dpi=300, bbox_inches='tight')
        print("💾 Saved molecule grid to output/molecule_grid.png")
    else:
        print("❌ Could not create molecule grid visualization")

def save_results(df, output_file='output/generation_results.csv'):
    """Save validation results to CSV"""
    if df is not None:
        df.to_csv(output_file, index=False)
        print(f"💾 Saved results to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate and analyze molecules with VAE')
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of molecules to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Top-k sampling threshold')
    parser.add_argument('--max_visualize', type=int, default=50,
                       help='Maximum number of molecules to visualize')
    parser.add_argument('--output_file', type=str, default='output/generation_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--checkpoint', type=str, default='checkpoints',
                       help='Model checkpoint directory to load from')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    print("🚀 Starting molecule generation and analysis pipeline...")
    print("="*60)
    
    # Load model
    print("📥 Loading model...")
    model = load_best_model(args.checkpoint)
    print(f"✅ Loaded model from {args.checkpoint}")
    
    # Generate molecules
    selfies_list = generate_molecules(model, args.num_samples, args.temperature, args.top_k)
    
    if not selfies_list:
        print("❌ No molecules generated. Exiting.")
        return
    
    # Validate molecules
    df = validate_molecules(selfies_list)
    
    if df is None:
        print("❌ No valid molecules found. Exiting.")
        return
    
    # Save results
    save_results(df, args.output_file)
    
    # Visualize molecules
    visualize_molecules(df, args.max_visualize)
    
    print("="*60)
    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
