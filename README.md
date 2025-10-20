# Conditional SELFIES-Based Variational Autoencoder

An implementation of a Variational Autoencoder (VAE) for molecular generation using SELFIES representation, built with Apple's MLX framework for efficient training on Apple Silicon.

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

### Mathematical Formulation

The VAE learns to model the molecular distribution $p(x)$ by maximizing the Evidence Lower Bound (ELBO):

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) || p(z))$$

Where:
- $q_\phi(z|x)$: Encoder (approximate posterior)
- $p_\theta(x|z)$: Decoder (likelihood)
- $p(z) = \mathcal{N}(0, I)$: Prior distribution
- $\beta$: KL annealing weight

### Components

- **Encoder**: $h = \text{BidirectionalLSTM}(x) \rightarrow \mu, \log\sigma = \text{Linear}(\text{LayerNorm}(h))$
- **Reparameterization**: $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$
- **Decoder**: $\hat{x} = \text{CustomLSTM}(z, x) \rightarrow \text{VocabProjection}(\text{LayerNorm}(\hat{x}))$
- **Loss**: $\mathcal{L} = \text{CrossEntropy}(x, \hat{x}) + \beta \cdot D_{KL}(q_\phi(z|x) || \mathcal{N}(0, I))$
- **Sampling**: Top-K filtering with temperature scaling
