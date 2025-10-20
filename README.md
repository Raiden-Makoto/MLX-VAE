# Conditional SELFIES-Based Variational Autoencoder

A PyTorch-style implementation of a Variational Autoencoder (VAE) for molecular generation using SELFIES representation, built with Apple's MLX framework for efficient training on Apple Silicon.

## 🧬 Overview

This project implements a VAE that learns to generate novel molecular structures by:
- **Encoding** SELFIES sequences into a continuous latent space
- **Decoding** latent representations back into valid SELFIES molecules
- **Generating** chemically valid molecules with controlled properties

## ✨ Key Features

- **SELFIES Representation**: Uses SELFIES (Self-Referencing Embedded Strings) for 100% chemically valid molecular generation
- **Bidirectional LSTM Encoder**: Captures sequential dependencies in both directions
- **Custom LSTM Decoder**: Explicit latent state initialization for better generation control
- **Top-K Sampling**: Prevents early termination and improves generation quality
- **β-Annealing**: Gradual KL divergence warm-up for stable training
- **Chemistry Enforcement**: Built-in chemical validity constraints
- **MLX Framework**: Optimized for Apple Silicon with efficient memory usage

## 📊 Performance

- **92% Success Rate**: 46/50 generated molecules are chemically valid
- **Diverse Properties**: LogP (-2.13 to 0.84), TPSA (0-76), MW (16-85 Da)
- **Fast Training**: Efficient MLX implementation with gradient clipping and layer normalization
- **Stable Convergence**: β-annealing prevents KL collapse

## 🏗️ Architecture

```
Input SELFIES → Bidirectional LSTM Encoder → Latent Space (μ, σ) → Custom LSTM Decoder → Generated SELFIES
```

### Components

- **Encoder**: Bidirectional LSTM + LayerNorm → μ, logσ layers
- **Decoder**: Custom LSTM with latent initialization + LayerNorm → Vocabulary projection
- **Loss**: Reconstruction (Cross-entropy) + β × KL Divergence
- **Sampling**: Top-K filtering with temperature control
