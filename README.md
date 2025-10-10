# Conditional Graph Variational Autoencoder for Molecular Generation

## Project Overview

This project implements a Conditional Graph Variational Autoencoder (CGVAE) for generating molecular structures with desired properties. The specific application is designing molecules with controlled blood-brain barrier (BBB) permeability - a critical factor in developing drugs that target the central nervous system.

## The Problem

The blood-brain barrier prevents most molecules from entering the brain, which is a major obstacle in treating neurological diseases. Only about 2% of small molecule drugs can cross the BBB. Traditional approaches rely on trial-and-error synthesis and testing, which is expensive and time-consuming. Computational methods can predict BBB permeability but cannot generate new molecular designs optimized for this property.

## Approach

This project combines **variational autoencoders** (VAEs) with **graph neural networks** (GNNs) to create a generative model that:

1. **Learns continuous representations of molecular graphs** - The encoder maps molecules to a smooth 64-dimensional latent space where similar molecules are close together

2. **Conditions generation on target properties** - The decoder takes both a latent code and a target BBB permeability score to generate molecular structures with desired characteristics

3. **Jointly optimizes structure and properties** - The model simultaneously learns to reconstruct molecular graphs and predict BBB permeability, ensuring the latent space captures property-relevant information

## Key Technical Components

**Graph Encoder**: A 4-layer Graph Attention Network (GAT) that processes molecular graphs (atoms as nodes, bonds as edges) and outputs parameters of a latent distribution. Uses multi-head attention to learn which parts of the molecule are important, residual connections for stable training, and multi-pooling (mean + max + sum) for robust graph-level representations.

**Graph Decoder**: Takes a latent vector z and a target BBB permeability value, then generates a molecular graph by predicting: (1) how many atoms the molecule should have, (2) what type each atom is, and (3) which bonds exist between atoms. The property conditioning allows control over the generated molecules' characteristics.

**Property Predictor**: A multi-layer perceptron that predicts BBB permeability directly from the latent code. This component serves two purposes: it validates that the latent space encodes property information, and it regularizes the representation during training.

**MPNN Baseline**: A pre-trained Message-Passing Neural Network that achieves 0.92 ROC-AUC on BBB permeability prediction. This model is used to generate training labels for the QM9 dataset and serves as a benchmark for property prediction.

## Dataset

The training data consists of 2,142 molecules sampled from the QM9 dataset. Each molecule is:
- Represented as a graph with 29-dimensional atom features and 6-dimensional bond features
- Labeled with BBB permeability scores predicted by the pre-trained MPNN model
- Preprocessed into PyTorch Geometric format for efficient batching

Atom features include element type, degree, formal charge, hybridization, aromaticity, and ring membership. Bond features include bond type, conjugation, and ring membership.

## Training Objective

The model is trained with a composite loss function:

$L = L_\text{reconstruction} + \beta \times L_{KL} + \gamma \times L_\text{property}$

- **Reconstruction loss**: Measures how accurately the model can recreate input molecular graphs
- **KL divergence**: Regularizes the latent distribution to be close to a standard normal, ensuring smooth interpolation
- **Property loss**: Measures prediction accuracy for BBB permeability from the latent code

The hyperparameters β and γ balance these objectives. β controls how much the latent space is regularized (β-VAE framework), and γ controls how much the latent space is optimized for property prediction.

## Why Graph-Based Generation?

Many molecular generation methods use SMILES strings (text representations of molecules). The problem is that randomly generated or modified SMILES strings are often invalid - they don't correspond to real molecules. This project directly generates molecular graphs, which better preserves chemical validity. The graph structure explicitly represents atoms and bonds, making it easier to enforce chemical constraints.

## Why Conditional Generation?

Most molecular VAEs only learn to compress and reconstruct molecules. This project adds property conditioning, which means you can guide the generation process. Instead of generating random molecules from the latent space, you specify target properties (e.g., "BBB permeability = 0.9") and the model generates molecules optimized for those properties. This is crucial for drug discovery applications where you want molecules with specific characteristics.

## Current Status

The project includes:
- Complete implementation of encoder, decoder, and property predictor
- Pre-trained MPNN model for BBB permeability prediction (0.92 ROC-AUC)
- Dataset of 2,142 QM9 molecules with predicted BBB labels
- Training infrastructure with data loaders and caching

The CGVAE model architecture is implemented and ready for training. Full training and evaluation results are pending.

## Project Structure

```
QVAE/
├── models/              # Neural network implementations
│   ├── graphencoder.py      # GAT-based encoder
│   ├── graphdecoder.py      # Conditional decoder
│   ├── propertypredictor.py # BBB prediction head
│   └── readme.md            # Detailed architecture docs
│
├── mpnn/                # Pre-trained baseline model
│   ├── mpnn.py              # MPNN implementation
│   └── mpnn.ipynb           # MPNN training
│
├── data/                # Dataset and loading utilities
│   ├── dataset.py           # QM9GraphDataset class
│   └── qm9_bbbp.csv        # 2,142 molecules with BBB labels
│
├── cgvae.ipynb          # Main training notebook
├── checkpoints/         # Saved model weights
└── processed/           # Cached graph data
```