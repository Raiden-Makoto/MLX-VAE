#!/usr/bin/env python3
"""
Comprehensive molecule generation and analysis pipeline.
Combines sampling, validation, and visualization into one script.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd

from utils.sample import load_best_model, sample_from_vae, tokens_to_selfies, generate_conditional_molecules, analyze_logp_tpsa_accuracy
from utils.validate import batch_validate_selfies
from utils.visualize import create_molecule_grid, create_property_distributions
from utils.diversity import calculate_diversity


# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def load_data():
    """Load vocabulary and metadata"""
    with open('mlx_data/qm9_cns_selfies.json', 'r') as f:
        meta = json.load(f)
    return meta['token_to_idx'], meta['idx_to_token'], meta['vocab_size']

def load_dataset_smiles():
    """Load SMILES from the training dataset"""
    try:
        # Load from the CNS dataset
        df = pd.read_csv('mlx_data/qm9_cns.csv')
        if 'smiles' in df.columns:
            return df['smiles'].tolist()
        elif 'SMILES' in df.columns:
            return df['SMILES'].tolist()
        else:
            print(f"No SMILES column found. Available columns: {df.columns.tolist()}")
            return None
    except FileNotFoundError:
        print("CNS dataset not found, skipping diversity calculation")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def calculate_diversity_metrics(df):
    """Calculate diversity between generated molecules and dataset"""
    dataset_smiles = load_dataset_smiles()
    if dataset_smiles is None:
        return
    
    generated_smiles = df['smiles'].tolist()
    
    print(f"Calculating diversity metrics...")
    print(f"   Generated molecules: {len(generated_smiles)}")
    print(f"   Dataset molecules: {len(dataset_smiles)}")
    
    # Calculate median similarity
    median_similarity = calculate_diversity(generated_smiles, dataset_smiles)
    
    print(f"Diversity Results:")
    print(f"   Median similarity to dataset: {median_similarity:.3f}")
    print(f"   Diversity score: {1 - median_similarity:.3f} (higher = more diverse)")
    
    # Generate and save diversity histogram
    print(f"Generating diversity histogram...")
    from utils.diversity import fingerprints, plot_similarity_distribution
    from rdkit.DataStructs import TanimotoSimilarity
    
    # Calculate all similarities for histogram
    fps_generated = fingerprints(generated_smiles)
    fps_dataset = fingerprints(dataset_smiles)
    
    similarities = []
    for fp1 in fps_generated:
        for fp2 in fps_dataset:
            sim = TanimotoSimilarity(fp1, fp2)
            similarities.append(sim)
    
    # Create and save histogram
    fig = plot_similarity_distribution(similarities, "Generated vs Dataset Similarity Distribution")
    fig.savefig('output/diversity_histogram.png', dpi=300, bbox_inches='tight')
    print(f"Saved diversity histogram to output/diversity_histogram.png")
    
    return median_similarity


def validate_molecules(selfies_list):
    """Validate generated molecules and convert to SMILES"""
    print(f"🔍 Validating {len(selfies_list)} molecules...")
    
    # Validate and convert to SMILES (includes filtering)
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
    
    return df

def visualize_molecules(df, max_molecules=20):
    """Create visualizations of the generated molecules"""
    print(f"🎨 Creating visualizations for {len(df)} molecules...")
    
    # Create molecule grid
    fig = create_molecule_grid(df, max_molecules=max_molecules)
    if fig is not None:
        fig.savefig('output/molecule_grid.png', dpi=300, bbox_inches='tight')
        print("Saved molecule grid to output/molecule_grid.png")
    else:
        print("❌ Could not create molecule grid visualization")

    # Create property distributions
    try:
        prop_fig = create_property_distributions(df)
        if prop_fig is not None:
            prop_fig.savefig('output/property_distributions.png', dpi=300, bbox_inches='tight')
            print("Saved property distributions to output/property_distributions.png")
        else:
            print("❌ Could not create property distributions visualization")
    except Exception as e:
        print(f"❌ Error creating property distributions: {e}")

def save_results(df, output_file='output/generation_results.csv'):
    """Save validation results to CSV"""
    if df is not None:
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate and analyze molecules with VAE')
    parser.add_argument('--num_samples', type=int, default=128, 
                       help='Number of molecules to generate (default 128)')
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
    
    # Conditional generation arguments (DEFAULT)
    parser.add_argument('--regular', action='store_true',
                       help='Use regular generation instead of conditional')
    parser.add_argument('--logp', type=float, default=1.0,
                       help='Target LogP value for conditional generation (default: 1.0)')
    parser.add_argument('--tpsa', type=float, default=40.0,
                       help='Target TPSA value for conditional generation (default: 40.0)')
    parser.add_argument('--analyze', action='store_true', default=True,
                       help='Analyze conditional generation accuracy (default: True)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Check if regular generation is requested (conditional is DEFAULT)
    if args.regular:
        print("🚀 Starting molecule generation and analysis pipeline...")
        print("="*60)
        
        # Load model
        print("📥 Loading model...")
        model = load_best_model(args.checkpoint)
        print(f"✅ Loaded model from {args.checkpoint}")
        
        # Generate molecules
        samples = sample_from_vae(model, args.num_samples, args.temperature, args.top_k)
        selfies_list = tokens_to_selfies(samples)
    
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
        
        # Calculate diversity metrics
        calculate_diversity_metrics(df)
        
        # Visualize molecules
        visualize_molecules(df, args.max_visualize)
        
        print("="*60)
        print("🎉 Pipeline completed successfully!")
        
    else:
        # CONDITIONAL GENERATION (DEFAULT)
        print("🧬 CONDITIONAL MOLECULAR GENERATION")
        print("="*50)
        print(f"Target LogP: {args.logp}")
        print(f"Target TPSA: {args.tpsa}")
        print(f"Number of samples: {args.num_samples}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-k: {args.top_k}")
        print()
        
        # Load model
        print("📥 Loading model...")
        model = load_best_model(args.checkpoint)
        print("✅ Model loaded successfully!")
        print()
        
        # Generate conditional molecules
        molecules = generate_conditional_molecules(
            model, args.logp, args.tpsa, args.num_samples, 
            args.temperature, args.top_k
        )
        
        if not molecules:
            print("❌ No valid molecules generated!")
            return
        
        print(f"✅ Generated {len(molecules)} valid molecules")
        
        # Analyze accuracy (default behavior)
        if args.analyze:
            print("\n📊 Analyzing conditional generation accuracy...")
            analyze_logp_tpsa_accuracy(molecules, args.logp, args.tpsa)
        
        # Convert to DataFrame for consistent processing
        df = pd.DataFrame(molecules)
        
        # Save results
        save_results(df, args.output_file)
        
        # Calculate diversity metrics
        calculate_diversity_metrics(df)
        
        # Visualize molecules
        visualize_molecules(df, args.max_visualize)
        
        print("="*60)
        print("🎉 Conditional generation completed!")

if __name__ == "__main__":
    main()
