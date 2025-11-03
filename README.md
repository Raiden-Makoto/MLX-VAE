# Transformer-VAE: Advanced Molecular Generation with FiLM-Conditioned Property Control

A cutting-edge Variational Autoencoder (VAE) for molecular generation using SELFIES representation, built with Apple's MLX framework and powered by Transformer architecture with **FiLM-conditioned property-guided generation**.

## 🧬 Overview

This project implements a sophisticated VAE that learns to generate novel molecular structures by:
- **Encoding** SELFIES sequences into a continuous latent space
- **Decoding** latent representations back into valid SELFIES molecules
- **Generating** chemically valid molecules with **TPSA-targeted control** via FiLM and inverse mapping (TPSA→z)
- **Filtering** molecules for stability, synthetic accessibility, and conformational strain
- **FiLM-Conditioned Generation**: Advanced feature-wise linear modulation for sophisticated property control

## ✨ Key Features

### Core Architecture
- **SELFIES Representation**: Uses SELFIES (Self-Referencing Embedded Strings) for 100% chemically valid molecular generation
- **Transformer Architecture**: State-of-the-art multi-head self-attention mechanism with positional encoding
- **Self-Attention Decoder**: Uses self-attention (not cross-attention) for proper sequence learning
- **Parallel Processing**: Processes all sequence positions simultaneously for faster training
- **Causal Masking**: Ensures autoregressive generation with proper attention patterns
- **MLX Framework**: Optimized for Apple Silicon with efficient memory usage
- **FiLM Layers**: Feature-wise Linear Modulation for sophisticated property conditioning

### FiLM-Conditioned Generation (DEFAULT)
- **Feature-wise Linear Modulation**: Advanced conditioning using γ (scaling) and β (shifting) parameters
- **Property-Guided Sampling**: Generate molecules targeting TPSA (primary) and optionally LogP
- **Default Targets**: TPSA default can be set to dataset median/mode
- **Sophisticated Control**: Each latent dimension gets property-specific transformations
- **Accuracy Analysis**: Real-time evaluation of how well generated molecules match targets
- **Flexible Targeting**: Customize LogP and TPSA values for specific applications
- **Backward Compatibility**: Regular generation available with `--regular` flag

### Advanced Training Features
- **β-Annealing**: Gradual KL divergence warm-up for stable training
- **Free Bits**: Prevents posterior collapse with configurable KL thresholds
- **Information Regularization**: Encourages diverse latent representations
- **Diversity Loss**: Promotes molecular diversity in generated samples
- **Dropout Regularization**: Prevents overfitting with configurable dropout rates
- **FiLM-Conditioned Training**: Properties fed to decoder via FiLM; latent is regularized only by VAE loss
- **TPSA→z Inverse Mapping**: Separate predictor learns mapping from normalized TPSA to latent z for stronger control
- **Stratified Sampling (Mandatory)**: Training uses TPSA-binned, equal-per-bin sampling to de-skew the dataset

### Molecular Validation & Filtering
- **Chemical Stability Filtering**: Removes peroxides, small rings, and azides
- **Synthetic Accessibility (SA_Score)**: Filters molecules based on synthetic feasibility
- **Drug-likeness (QED)**: Evaluates molecules for drug-like properties
- **Conformational Strain**: Removes molecules with high strain energy (>100 kcal/mol)
- **Comprehensive Properties**: LogP, TPSA, Molecular Weight, Heavy Atoms, Rotatable Bonds

## 📊 Performance

- **High Success Rate**: Generates chemically valid molecules with comprehensive filtering
- **FiLM-Conditioned Accuracy**: Enhanced LogP and TPSA accuracy through sophisticated feature-wise modulation
- **Diverse Properties**: Wide range of LogP, TPSA, MW, QED, and SAS values
- **Fast Training**: Efficient MLX implementation with gradient clipping and layer normalization
- **Stable Convergence**: Advanced regularization prevents KL collapse and posterior collapse
- **Quality Control**: Multi-stage filtering ensures generated molecules are stable and synthesizable
- **Property Guidance**: Successfully generates molecules targeting specific LogP and TPSA values

## 🏗️ Architecture

```
Input SELFIES → Transformer Encoder → Latent (μ, σ) → Reparameterization → Decoder (Self-Attention) → Generated SELFIES
                    ↑                                                                  ↑
             Properties (LogP, TPSA)  →   Property Encoder → FILM Layers (γ, β) ─────────┘
                    ↓
       Stable, Synthesizable, Property-Controlled Molecules
```

### Conditional Generation Path

The model trains a VAE (reconstruction + β·KL). Properties modulate the decoder via FiLM. At inference, TPSA control uses an inverse mapping (TPSA→z) to select latent codes near the target.

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - \beta \cdot D_{KL}(q_\phi(z|x,c) || p(z|c)) + \lambda_{div} \cdot \mathcal{L}_{div}$$

Where:
- $q_\phi(z|x,c)$: Encoder (conditional posterior)
- $p_\theta(x|z,c)$: Decoder (conditional likelihood)  
- $p(z|c)$: Conditional prior distribution
- $c$: Properties (normalized LogP, TPSA)

### Components

- **Transformer Encoder**: Multi-head self-attention → sequence pooling → $\mu, \log\sigma$
- **Property Encoder**: 2-layer MLP: `Linear(properties) → ReLU → Linear` → property embedding
- **Property-Conditioned Latent**: $\mu_p(c), \sigma_p(c)$ learned from properties alone
- **Reparameterization**: $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$
- **Transformer Decoder**: Self-attention with FILM conditioning (γ, β modulation)
- **Conditional Generation**: Sample $z \sim p(z|c)$, apply FILM conditioning, decode

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

### Data Preparation

Download ChEMBL CNS dataset (20,000 molecules, SELFIES + TPSA only) and tokenize:
```bash
python mlx_data/download_chembl_cns.py
```

### Training

**Train VAE (with mandatory TPSA stratification):**
```bash
python scripts/train.py --epochs 50 --batch_size 128 --learning_rate 1e-4 \
                --num_heads 4 --num_layers 4 --dropout 0.1 \
                --tpsa_bins 5 --per_bin 2000
```

**Train TPSA→z Predictor (after VAE):**
```bash
python scripts/train_predictor.py --epochs 50 --batch_size 128 --learning_rate 1e-4
```

### Conditional Generation & Analysis (TPSA-targeted)
```bash
# Generate molecules with a specific TPSA target (inverse mapping + FiLM)
python scripts/inference.py --num_samples 128 --tpsa 72.0

# Or use default targets (median values from dataset)
python scripts/inference.py
```

### Unconditional Generation
```bash
python scripts/inference.py --regular --num_samples 128
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
├── models/                 # Transformer VAE architecture components
│   ├── transformer_encoder.py    # Transformer encoder with masked pooling
│   ├── transformer_decoder.py   # Transformer decoder with self-attention
│   ├── transformer_vae.py       # Main CVAE model with property conditioning
│   └── layers/                   # Layer implementations
│       ├── multi_head_attention.py
│       ├── feed_forward.py
│       ├── positional_encoding.py
│       ├── transformer_encoder_layer.py
│       ├── transformer_decoder_layer.py
│       └── film.py
├── utils/                 # Utility functions
│   ├── loss.py           # Loss functions with free bits
│   ├── sample.py         # Sampling and conditional generation
│   ├── validate.py       # Molecular validation pipeline
│   ├── visualize.py      # Visualization tools
│   ├── smarts.py         # Chemical stability filtering
│   ├── geomopt.py        # Conformational strain analysis
│   ├── sascorer.py       # Synthetic accessibility scoring
│   └── diversity.py      # Molecular diversity metrics
├── mlx_data/             # Data processing and vocabulary
│   ├── convert.py        # Dataset conversion with property calculation
│   ├── dataloader.py     # Training data loader
│   ├── qm9_cns_selfies.json  # SELFIES data with LogP/TPSA properties
│   └── qm9_cns_tokenized.npy  # Tokenized sequences
├── scripts/              # Training and inference scripts
│   ├── train.py          # Training script
│   ├── train_predictor.py # TPSA→z predictor training
│   └── inference.py      # Conditional generation and analysis pipeline
├── checkpoints/          # Model checkpoints and metadata
└── output/               # Generated molecules and visualizations
```

## 🎯 Key Innovations

1. **Conditional VAE Architecture**: Property-controlled generation with LogP and TPSA targets
2. **Self-Attention Decoder**: Uses self-attention (not cross-attention) for proper sequence learning
3. **Transformer-Based**: Multi-head self-attention for sequence modeling
4. **Property Conditioning**: Deep property encoder with FILM layers for sophisticated conditioning
5. **Normalized Properties**: Automatic property normalization (handles 25x scale differences)
6. **Multi-Stage Filtering**: Comprehensive molecular validation pipeline
7. **Conformational Analysis**: Strain energy filtering for realistic molecules
8. **Advanced Regularization**: KL annealing, diversity loss, and property prediction loss
9. **Efficient MLX Implementation**: Optimized for Apple Silicon performance

## 🧪 Conditional Generation

The model supports property-conditioned generation:
- **LogP range**: -10.15 to 44.97 (mean: 3.25, std: 2.21)
- **TPSA range**: 0 to 810 (mean: 81.55, std: 54.86)
- **Median values**: LogP=3.21, TPSA=72.03 (optimal for CNS penetration)
- **Accuracy**: Property conditioning with CVAE architecture
