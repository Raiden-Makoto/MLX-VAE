# AR-CVAE: Autoregressive Conditional VAE for Molecular Generation

An Autoregressive Conditional Variational Autoencoder (AR-CVAE) for molecular generation using SELFIES representation, built with Apple's MLX framework for efficient training on Apple Silicon.

## 🧬 Overview

This project implements an AR-CVAE that learns to generate novel molecular structures by:
- **Encoding** SELFIES sequences into a continuous latent space using LSTM encoder
- **Decoding** latent representations back into valid SELFIES molecules using autoregressive LSTM decoder
- **Conditioning** generation on molecular properties (TPSA) for property-targeted generation
- **Training** with teacher forcing and KL annealing for stable learning

## ✨ Key Features

### Core Architecture
- **SELFIES Representation**: Uses SELFIES (Self-Referencing Embedded Strings) for chemically valid molecular generation
- **LSTM Encoder**: Bi-directional LSTM encoder learns sequence representations
- **Autoregressive LSTM Decoder**: Generates molecules token-by-token with property conditioning
- **Property Conditioning**: Conditions generation on molecular properties (TPSA)
- **MLX Framework**: Optimized for Apple Silicon with efficient memory usage

### Training Features
- **β-Annealing**: Gradual KL divergence warm-up for stable training
- **Free Bits**: Prevents posterior collapse with configurable KL thresholds
- **Teacher Forcing**: Uses teacher forcing during training with decay schedule
- **Mutual Information Regularization**: Encourages diverse latent representations
- **Posterior Collapse Prevention**: Penalty term to prevent degenerate latent space
- **True Training Loss**: Reports training loss without teacher forcing for fair comparison with validation

### Loss Components
- **Reconstruction Loss**: Cross-entropy loss for sequence reconstruction
- **KL Divergence**: Regularizes latent space with β-annealing
- **Posterior Collapse Penalty**: Prevents collapse to prior
- **Mutual Information Penalty**: Encourages informative latent codes

## 🏗️ Architecture

```
Input SELFIES → LSTM Encoder → Latent (μ, σ) → Reparameterization → Autoregressive LSTM Decoder → Generated SELFIES
                    ↑                                                                  ↑
             Properties (TPSA) ─────────────────────────────────────────────────────────┘
```

### Components

- **LSTM Encoder**: Multi-layer LSTM encoder → sequence pooling → $\mu, \log\sigma$
- **Property Conditioning**: Properties concatenated with embeddings at decoder input
- **Reparameterization**: $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$
- **Autoregressive Decoder**: LSTM decoder generates tokens sequentially with teacher forcing during training

## 📊 Training

### Basic Usage

```bash
python train.py
```

### With Custom Options

```bash
python train.py --epochs 50 --batch_size 64 --learning_rate 5e-5 \
                --beta_start 0.0 --beta_end 0.05 --beta_warmup_epochs 35 \
                --lambda_collapse 0.001 --free_bits 1.0 \
                --lambda_mi 0.01 --grad_clip 1.0
```

### Resume Training

```bash
python train.py --resume
```

### Arguments

- `--data`: Path to dataset JSON file (default: `mlx_data/chembl_cns_selfies.json`)
- `--vocab_size`: Vocabulary size (default: 80)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--hidden_dim`: Hidden dimension (default: 256)
- `--latent_dim`: Latent dimension (default: 128)
- `--num_conditions`: Number of property conditions (default: 1 for TPSA)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.2)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--beta_start`: Initial KL weight (default: 0.0)
- `--beta_end`: Final KL weight (default: 0.05)
- `--beta_warmup_epochs`: Epochs for beta warmup (default: 35)
- `--lambda_prop`: Property loss weight (default: 0.1)
- `--lambda_collapse`: Posterior collapse penalty weight (default: 0.001)
- `--free_bits`: Free bits constraint (default: 1.0)
- `--lambda_mi`: Mutual information penalty weight (default: 0.01)
- `--grad_clip`: Gradient clipping norm (default: 1.0)
- `--checkpoint_dir`: Checkpoint directory (default: `./checkpoints`)
- `--checkpoint_freq`: Checkpoint frequency in epochs (default: 10)
- `--resume`: Resume from checkpoint_best.npz
- `--verbose`: Print detailed epoch summaries

## 📁 Project Structure

```
QVAE/
├── models/                 # AR-CVAE architecture components
│   ├── encoder.py          # LSTM encoder
│   ├── decoder.py          # Autoregressive LSTM decoder
│   ├── decoder_sampling.py # Decoder for inference/sampling
│   └── vae.py              # Main AR-CVAE model
├── losses/                 # Loss function modules
│   ├── recon.py           # Reconstruction loss
│   ├── kl.py              # KL divergence
│   ├── info.py             # Mutual information and collapse penalty
│   ├── prop.py            # Property prediction loss
│   ├── enc.py             # Encoder loss
│   ├── dec.py             # Decoder loss
│   └── stable.py          # Stability checks
├── mlx_data/               # Data processing
│   ├── dataloader.py      # Dataset loader with normalization
│   └── chembl_cns_selfies.json  # ChEMBL CNS dataset
├── complete_vae_loss.py   # Complete loss function
├── trainer.py             # Training loop with true loss computation
├── train.py               # Training script
├── data_diagnostic.py     # Diagnostic tool for train/val analysis
└── checkpoints/           # Model checkpoints and training history
```

## 🎯 Key Features

1. **Autoregressive Generation**: Token-by-token generation with LSTM decoder
2. **Property Conditioning**: Conditional generation based on TPSA
3. **Teacher Forcing**: Uses teacher forcing during training with decay schedule
4. **True Loss Reporting**: Training loss computed without teacher forcing for fair comparison
5. **KL Annealing**: Gradual warmup of KL divergence weight
6. **Posterior Collapse Prevention**: Multiple mechanisms to prevent collapse
7. **Efficient MLX Implementation**: Optimized for Apple Silicon

## 📈 Training Monitoring

Training history is saved to `checkpoints/training_history.json` and includes:
- Train/validation losses (both without teacher forcing for fair comparison)
- Reconstruction, KL, collapse penalty, and property losses
- Beta and teacher forcing schedules
- Mutual information metrics

Training plots are saved to `checkpoints/training_history.png`.

## 🔍 Troubleshooting

### Train-Val Divergence

The model now reports "true" training loss (without teacher forcing) to match validation loss computation. This ensures fair comparison and prevents misleading divergence metrics.

### Posterior Collapse

The model includes multiple mechanisms to prevent posterior collapse:
- Free bits constraint
- Mutual information penalty
- Posterior collapse penalty

Monitor mutual information in training history - target is ~4.85.

## 📝 Requirements

See `requirements.txt` for full dependencies. Key packages:
- `mlx` and `mlx.nn`: Apple MLX framework
- `numpy`: Numerical operations
- `selfies`: SELFIES molecular representation
- `rdkit`: Molecular validation (optional)
- `matplotlib`: Training visualization (optional)

## 🚀 Quick Start

1. **Prepare Data**: Ensure `mlx_data/chembl_cns_selfies.json` exists
2. **Train Model**: Run `python train.py`
3. **Monitor Training**: Check `checkpoints/training_history.json` and `.png`
4. **Resume Training**: Use `python train.py --resume` to continue from best checkpoint

## 📄 License

See LICENSE file for details.
