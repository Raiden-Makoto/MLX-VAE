# Models Directory

This directory contains the core model components for the Selfies-based Variational Autoencoder (VAE).

## File Structure

```
models/
├── encoder.py          # Bidirectional LSTM encoder
├── decoder.py          # Custom LSTM decoder with latent initialization
├── custom_lstm.py      # Custom LSTM implementation
└── README.md          # This file
```

## Components

### 1. `encoder.py` - SelfiesEncoder

**Purpose**: Encodes SELFIES token sequences into latent space distributions.

**Architecture**:
- **Embedding Layer**: Maps token indices to dense vectors
- **Bidirectional LSTM**: Two separate LSTMs (forward + backward)
- **Output Projection**: Maps concatenated hidden states to μ and σ

**Key Features**:
- Bidirectional processing for better context understanding
- Outputs both mean (μ) and log-variance (σ) for latent distribution
- Handles variable-length sequences

**Input/Output**:
- **Input**: `[batch_size, sequence_length]` token indices
- **Output**: `(μ, σ)` where both are `[batch_size, latent_dim]`

### 2. `decoder.py` - SelfiesDecoder

**Purpose**: Decodes latent codes back into SELFIES token sequences.

**Architecture**:
- **Latent Projection**: Maps latent codes to LSTM initial states
- **Custom LSTM**: Processes sequence with latent-initialized hidden states
- **Output Projection**: Maps LSTM outputs to vocabulary logits

**Key Features**:
- Latent code directly initializes LSTM hidden and cell states
- Custom LSTM implementation for proper initialization control
- Generates probability distributions over vocabulary

**Input/Output**:
- **Input**: `z [batch_size, latent_dim]`, `seq [batch_size, sequence_length]`
- **Output**: `logits [batch_size, sequence_length, vocab_size]`

### 3. `custom_lstm.py` - CustomLSTM

**Purpose**: Custom LSTM implementation that accepts initial hidden states.

**Architecture**:
- **Four Gates**: Input, Forget, Cell, Output gates
- **Manual Implementation**: Step-by-step LSTM computation
- **Initial State Support**: Accepts `(h₀, c₀)` initialization

**Key Features**:
- Full control over LSTM computation
- Proper initial state handling
- MLX-compatible implementation
- Supports both training and inference modes

**Why Custom?**
MLX's built-in LSTM doesn't support initial hidden state initialization the way PyTorch does. This custom implementation provides the flexibility needed for VAE decoders.