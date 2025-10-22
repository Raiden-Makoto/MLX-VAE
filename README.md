# MLX-VAE: Variational Autoencoder for Molecular Generation

A state-of-the-art Variational Autoencoder (VAE) for molecular generation using SELFIES representation, built with Apple's MLX framework for efficient training on Apple Silicon.

## 🧬 Overview

This project implements a sophisticated VAE that learns to generate novel molecular structures by:
- **Encoding** SELFIES sequences into a continuous latent space
- **Decoding** latent representations back into valid SELFIES molecules
- **Generating** chemically valid molecules with controlled properties
- **Filtering** molecules for stability, synthetic accessibility, and conformational strain

## ✨ Key Features

### Core Architecture
- **SELFIES Representation**: Uses SELFIES (Self-Referencing Embedded Strings) for 100% chemically valid molecular generation
- **Bidirectional LSTM Encoder**: Captures sequential dependencies in both directions
- **Custom LSTM Decoder**: Explicit latent state initialization for better generation control
- **Top-K Sampling**: Prevents early termination and improves generation quality
- **MLX Framework**: Optimized for Apple Silicon with efficient memory usage

### Advanced Training Features
- **β-Annealing**: Gradual KL divergence warm-up for stable training
- **Free Bits**: Prevents posterior collapse with configurable KL thresholds
- **Information Regularization**: Encourages diverse latent representations
- **Diversity Loss**: Promotes molecular diversity in generated samples
- **Dropout Regularization**: Prevents overfitting with configurable dropout rates

### Molecular Validation & Filtering
- **Chemical Stability Filtering**: Removes peroxides, small rings, azides, and hypervalent atoms
- **Synthetic Accessibility (SA_Score)**: Filters molecules based on synthetic feasibility
- **Drug-likeness (QED)**: Evaluates molecules for drug-like properties
- **Conformational Strain**: Removes molecules with high strain energy (>100 kcal/mol)
- **Comprehensive Properties**: LogP, TPSA, Molecular Weight, Heavy Atoms, Rotatable Bonds

## 📊 Performance

- **High Success Rate**: Generates chemically valid molecules with comprehensive filtering
- **Diverse Properties**: Wide range of LogP, TPSA, MW, QED, and SAS values
- **Fast Training**: Efficient MLX implementation with gradient clipping and layer normalization
- **Stable Convergence**: Advanced regularization prevents KL collapse and posterior collapse
- **Quality Control**: Multi-stage filtering ensures generated molecules are stable and synthesizable

## 🏗️ Architecture

```
Input SELFIES → Bidirectional LSTM Encoder → Latent Space (μ, σ) → Custom LSTM Decoder → Generated SELFIES
                                                      ↓
                                              Comprehensive Filtering
                                                      ↓
                                    Stable, Synthesizable Molecules
```

### Mathematical Formulation

The VAE learns to model the molecular distribution $p(x)$ by maximizing the Evidence Lower Bound (ELBO):

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) || p(z)) + \lambda_{div} \cdot \mathcal{L}_{div} + \lambda_{info} \cdot \mathcal{L}_{info}$$

Where:
- $q_\phi(z|x)$: Encoder (approximate posterior)
- $p_\theta(x|z)$: Decoder (likelihood)
- $p(z) = \mathcal{N}(0, I)$: Prior distribution
- $\beta$: KL annealing weight with free bits
- $\mathcal{L}_{div}$: Diversity loss for molecular variety
- $\mathcal{L}_{info}$: Information regularization loss

### Components

- **Encoder**: $h = \text{BidirectionalLSTM}(x) \rightarrow \mu, \log\sigma = \text{Linear}(\text{LayerNorm}(\text{Dropout}(h)))$
- **Reparameterization**: $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$
- **Decoder**: $\hat{x} = \text{CustomLSTM}(z, x) \rightarrow \text{VocabProjection}(\text{LayerNorm}(\text{Dropout}(\hat{x})))$
- **Loss**: $\mathcal{L} = \text{CrossEntropy}(x, \hat{x}) + \beta \cdot \max(D_{KL} - \tau, 0) + \lambda_{div} \cdot \mathcal{L}_{div} + \lambda_{info} \cdot \mathcal{L}_{info}$
- **Sampling**: Top-K filtering with temperature scaling

## 🔬 Molecular Analysis Pipeline

### Validation Process
1. **SELFIES → SMILES**: Convert and validate molecular structure
2. **Deduplication**: Remove duplicate canonical SMILES
3. **Chemical Filtering**: Remove unstable molecules (peroxides, small rings, etc.)
4. **Strain Filtering**: Remove molecules with high conformational strain
5. **Property Calculation**: Compute LogP, TPSA, MW, QED, SAS, and structural properties

### Visualization
- **Molecule Grid**: Visual display of generated molecules with property captions
- **Property Distributions**: Histograms and scatter plots of molecular properties
- **Quality Metrics**: Success rates, uniqueness, and property statistics

## 🚀 Usage

### Training
```bash
python train.py --num_epochs 100 --batch_size 32 --learning_rate 1e-4 \
                --max_beta 0.1 --free_bits 0.05 --diversity_weight 0.1 \
                --info_reg_weight 0.1 --beta_annealing linear
```

### Generation & Analysis
```bash
python inference.py --num_samples 128 --temperature 1.0 --top_k 10 \
                   --max_visualize 50 --checkpoint checkpoints/mlx_mgcvae
```

### Validation Only
```bash
python utils/validate.py  # Uses output/generated_molecules.txt
```

### Visualization Only
```bash
python utils/visualize.py  # Uses output/validation_results.csv
```

## 📁 Project Structure

```
QVAE/
├── models/                 # VAE architecture components
│   ├── encoder.py         # Bidirectional LSTM encoder
│   ├── decoder.py         # Custom LSTM decoder
│   ├── custom_lstm.py     # Custom LSTM implementation
│   └── vae.py            # Main VAE model
├── utils/                 # Utility functions
│   ├── loss.py           # Loss functions with free bits
│   ├── sample.py         # Sampling and generation
│   ├── validate.py       # Molecular validation pipeline
│   ├── visualize.py      # Visualization tools
│   ├── smarts.py         # Chemical stability filtering
│   ├── geomopt.py        # Conformational strain analysis
│   └── sascorer.py       # Synthetic accessibility scoring
├── mlx_data/             # Data processing and vocabulary
├── checkpoints/          # Model checkpoints and metadata
├── output/               # Generated molecules and visualizations
├── train.py              # Training script
└── inference.py          # Generation and analysis pipeline
```

## 🎯 Key Innovations

1. **Free Bits Implementation**: Prevents posterior collapse with configurable KL thresholds
2. **Multi-Stage Filtering**: Comprehensive molecular validation pipeline
3. **Conformational Analysis**: Strain energy filtering for realistic molecules
4. **Advanced Regularization**: Information regularization and diversity loss
5. **Comprehensive Metrics**: QED, SA_Score, and structural property analysis
6. **Efficient MLX Implementation**: Optimized for Apple Silicon performance
