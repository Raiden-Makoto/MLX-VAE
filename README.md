# AR-CVAE: Autoregressive Conditional VAE for Molecular Generation

A Variational Autoencoder (VAE) for molecular generation using SELFIES representation, built with Apple's MLX framework and powered by LSTM-based autoregressive architecture with property conditioning.

## 🧬 Overview

This project implements an Autoregressive Conditional VAE (AR-CVAE) that learns to generate novel molecular structures by:
- **Encoding** SELFIES sequences into a continuous latent space using stacked LSTM layers
- **Decoding** latent representations autoregressively with property conditioning
- **Generating** chemically valid molecules with TPSA-targeted control
- **Training** with advanced regularization to prevent posterior collapse

## ✨ Key Features

### Core Architecture
- **SELFIES Representation**: Uses SELFIES (Self-Referencing Embedded Strings) for chemically valid molecular generation
- **LSTM-Based**: Stacked LSTM layers for sequence encoding and autoregressive decoding
- **Autoregressive Decoder**: Token-by-token generation with teacher forcing during training
- **Property Conditioning**: TPSA properties concatenated with embeddings for conditional generation
- **MLX Framework**: Optimized for Apple Silicon with efficient memory usage

### Advanced Training Features
- **β-Annealing**: Gradual KL divergence warm-up (default: 20 epochs)
- **Free Bits Constraint**: Minimum KL per dimension (default: 0.5) to prevent posterior collapse
- **Mutual Information Penalty**: Encourages diverse latent representations
- **Posterior Collapse Penalty**: Additional regularization to prevent latent collapse
- **Teacher Forcing Decay**: Starts at 0.7, decays to 0.3 over training
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpoint Management**: Automatic clearing (unless resuming), checkpoint saving

### Training Optimizations
- **Optimized Training Loop**: Combined eval calls, efficient gradient clipping
- **Progress Bars**: tqdm progress bars for training and validation
- **Fixed Sequence Length**: 80 tokens (matched to dataset)
- **80/10/10 Split**: Fixed train/val/test data split
- **Reproducible**: Fixed random seed (67) for reproducibility

## 🏗️ Architecture

```
Input SELFIES → LSTM Encoder → Latent (μ, σ) → Reparameterization → LSTM Decoder (Autoregressive) → Generated SELFIES
                    ↑                                                                  ↑
             Properties (TPSA) ───────────────────────────────────────────── Concatenated ─────────┘
```

### Loss Function

$$\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL} + \lambda_{collapse} \cdot \mathcal{L}_{collapse} + \lambda_{MI} \cdot (-\mathcal{MI}) + \lambda_{prop} \cdot \mathcal{L}_{prop}$$

Where:
- $\mathcal{L}_{recon}$: Reconstruction loss (cross-entropy)
- $\mathcal{L}_{KL}$: KL divergence with free bits constraint
- $\mathcal{L}_{collapse}$: Posterior collapse penalty
- $\mathcal{MI}$: Mutual information (maximized)
- $\mathcal{L}_{prop}$: Property prediction loss (if predictor available)

### Components

- **LSTM Encoder**: Stacked LSTM layers → final hidden state → $\mu, \log\sigma$
- **Property Conditioning**: TPSA properties concatenated with token embeddings
- **Reparameterization**: $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$
- **LSTM Decoder**: Autoregressive token generation with property conditioning
- **Teacher Forcing**: Uses ground truth tokens during training for stability

## 🚀 Usage

### Training

**Basic training:**
```bash
./train.sh --epochs 50 --batch_size 128 --learning_rate 1e-4
```

**Resume from checkpoint:**
```bash
./train.sh --resume --epochs 100
```

**With custom parameters:**
```bash
./train.sh --epochs 100 --batch_size 256 --learning_rate 5e-5 \
           --free_bits 0.5 --lambda_mi 0.01 --lambda_collapse 0.1
```

### Key Training Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 128)
- `--learning_rate`: Adam learning rate (default: 1e-4)
- `--beta_start`: Initial KL weight (default: 0.0)
- `--beta_end`: Final KL weight (default: 0.4)
- `--beta_warmup_epochs`: Epochs to warm up beta (default: 20)
- `--free_bits`: Minimum KL per dimension (default: 0.5)
- `--lambda_collapse`: Posterior collapse penalty weight (default: 0.1)
- `--lambda_mi`: Mutual information penalty weight (default: 0.01)
- `--grad_clip`: Gradient clipping norm (default: 1.0)
- `--resume`: Resume from checkpoint_best.npz
- `--verbose`: Print detailed epoch summaries

### Model Architecture Parameters

- `--embedding_dim`: Embedding dimension (default: 128)
- `--hidden_dim`: LSTM hidden dimension (default: 256)
- `--latent_dim`: Latent space dimension (default: 128)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.2)

## 📁 Project Structure

```
QVAE/
├── models/                 # Model architecture components
│   ├── encoder.py          # LSTM encoder with property conditioning
│   ├── decoder.py          # Autoregressive LSTM decoder
│   ├── decoder_sampling.py # Temperature-based sampling decoder
│   └── vae.py              # Complete AR-CVAE model
├── losses/                 # Loss function modules
│   ├── recon.py            # Reconstruction loss (cross-entropy)
│   ├── kl.py               # KL divergence with free bits
│   ├── info.py             # Mutual information and posterior collapse
│   ├── prop.py             # Property prediction loss
│   ├── enc.py              # Encoder-only loss
│   └── dec.py              # Decoder-only loss
├── mlx_data/               # Data processing
│   ├── dataloader.py       # MoleculeDataset with padding/truncation
│   └── chembl_cns_selfies.json  # Tokenized dataset (20K molecules)
├── complete_vae_loss.py    # Complete training loss function
├── trainer.py              # Training loop with checkpointing
├── train.py                # Training script with CLI
├── train.sh                # Training script wrapper
└── checkpoints/            # Model checkpoints and history
```

## 🎯 Key Features

1. **Autoregressive Generation**: Token-by-token generation with teacher forcing
2. **Property Conditioning**: TPSA properties guide molecule generation
3. **Posterior Collapse Prevention**: Free bits constraint + MI penalty + collapse penalty
4. **Efficient MLX Implementation**: Optimized training loop for Apple Silicon
5. **Modular Loss Functions**: Separate modules for different loss components
6. **Checkpoint Management**: Automatic clearing, resume support
7. **Fixed Data Split**: 80/10/10 train/val/test split
8. **Reproducible**: Fixed random seed for consistent results

## 📊 Dataset

- **Source**: ChEMBL CNS dataset
- **Size**: 20,000 molecules
- **Representation**: SELFIES (tokenized)
- **Properties**: TPSA (Topological Polar Surface Area)
- **Sequence Length**: 80 tokens (fixed)
- **Vocabulary Size**: 95 tokens

## 🔧 Development

### Installation

```bash
# Create virtual environment
python3 -m venv qvae
source qvae/bin/activate

# Install MLX (see MLX documentation for installation)
# Install other dependencies
pip install numpy tqdm
```

### Testing

Individual components can be tested:
- Encoder: `models/encoder.py`
- Decoder: `models/decoder.py`
- VAE: `models/vae.py`
- Loss functions: `losses/*.py`

## 📈 Training Metrics

The trainer tracks:
- Total loss
- Reconstruction loss
- KL divergence loss
- Posterior collapse penalty
- Mutual information
- Beta (KL weight) schedule
- Teacher forcing ratio

Training history is saved to `checkpoints/training_history.json` and plotted to `checkpoints/training_history.png`.

## 🎓 Technical Details

- **Framework**: MLX (Apple's Machine Learning framework)
- **Architecture**: Stacked LSTM layers (not Transformer)
- **Conditioning**: Property concatenation (not FiLM)
- **Sequence Length**: 80 tokens
- **Batch Size**: 128 (default, configurable)
- **Optimizer**: Adam
- **Random Seed**: Fixed to 67

## 📝 Notes

- The autoregressive decoder loop is the main computational bottleneck
- Training speed: ~4-6 iterations/second (depends on hardware)
- Model size optimized for speed: embedding=128, hidden=256, latent=128, layers=2
- All sequences are preprocessed to exactly 80 tokens

## 🤝 Contributing

This is a research project. For questions or improvements, please open an issue or submit a pull request.

## 📄 License

[Add your license here]
