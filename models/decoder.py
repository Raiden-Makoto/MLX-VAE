import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, Optional


class MLXAutoregressiveDecoder(nn.Module):
    """Autoregressive decoder for AR-CVAE molecule generation using MLX
    
    Generates molecules token-by-token, conditioning on:
    - Latent vector z
    - Property conditions
    - Previous tokens (during training with teacher forcing)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 200,
        num_conditions: int = 6,
        num_layers: int = 3,
        pad_token: int = 0,
        end_token: int = 2
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent space dimension
            num_conditions: Number of property conditions
            num_layers: Number of LSTM layers
            pad_token: Token index for padding
            end_token: Token index for end-of-sequence
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.num_layers = num_layers
        self.pad_token = pad_token
        self.end_token = end_token
        
        # Project latent vector to initial hidden state
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Project conditions to hidden state component
        self.condition_to_hidden = nn.Linear(num_conditions, hidden_dim)
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM decoder with combined input (embedding + conditions)
        # Input size = embedding_dim + num_conditions
        # MLX only supports single-layer LSTM, so we stack them manually
        for i in range(num_layers):
            setattr(
                self,
                f'lstm_layer_{i}',
                nn.LSTM(
                    input_size=embedding_dim + num_conditions if i == 0 else hidden_dim,
                    hidden_size=hidden_dim
                )
            )
        
        # Output projection to vocabulary
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def initialize_hidden_state(
        self,
        z: mx.array,
        conditions: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Initialize LSTM hidden and cell states from latent vector and conditions
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            conditions: Normalized properties [batch_size, num_conditions]
        
        Returns:
            hidden: Initial hidden state [num_layers, batch_size, hidden_dim]
            cell: Initial cell state [num_layers, batch_size, hidden_dim]
        """
        batch_size = z.shape[0]
        
        # Project z and conditions to hidden dimension
        hidden_z = self.z_to_hidden(z)  # [batch_size, hidden_dim]
        hidden_c = self.condition_to_hidden(conditions)  # [batch_size, hidden_dim]
        
        # Combine them
        hidden_init = (hidden_z + hidden_c) / 2.0  # [batch_size, hidden_dim]
        
        # Replicate for all LSTM layers
        hidden = mx.repeat(
            mx.expand_dims(hidden_init, 0),
            self.num_layers,
            axis=0
        )  # [num_layers, batch_size, hidden_dim]
        
        # Initialize cell state to zeros
        cell = mx.zeros_like(hidden)  # [num_layers, batch_size, hidden_dim]
        
        return hidden, cell
    
    def __call__(
        self,
        z: mx.array,
        conditions: mx.array,
        target_seq: Optional[mx.array] = None,
        max_length: int = 80,
        teacher_forcing_ratio: float = 0.5
    ) -> mx.array:
        """
        Autoregressive decoding
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            conditions: Property conditions [batch_size, num_conditions]
            target_seq: Ground truth sequences for teacher forcing [batch_size, seq_length]
            max_length: Maximum sequence length
            teacher_forcing_ratio: Probability of using teacher forcing (during training)
        
        Returns:
            output_logits: Predicted logits [batch_size, seq_length, vocab_size]
        """
        batch_size = z.shape[0]
        
        # Determine sequence length
        if target_seq is not None:
            seq_length = target_seq.shape[1]
        else:
            seq_length = max_length
        
        # Initialize hidden and cell states
        hidden, cell = self.initialize_hidden_state(z, conditions)
        
        # Start token (usually 0 or special START token)
        current_token = mx.zeros((batch_size,), dtype=mx.uint32)
        
        # Collect output logits for all timesteps
        output_logits_list = []
        
        # Autoregressive generation loop
        for t in range(seq_length):
            # Embed current token: [batch_size] -> [batch_size, embedding_dim]
            embedded = self.embedding(current_token)
            
            # Concatenate with conditions: [batch_size, embedding_dim + num_conditions]
            lstm_input = mx.concatenate([embedded, conditions], axis=1)
            
            # Add sequence dimension for LSTM: [batch_size, 1, embedding_dim + num_conditions]
            lstm_input = mx.expand_dims(lstm_input, 1)
            
            # LSTM step through stacked layers
            # MLX LSTM returns: (output, hidden)
            # With input [batch, 1, features], output is [batch, 1, hidden_dim]
            output = lstm_input
            for i in range(self.num_layers):
                lstm_layer = getattr(self, f'lstm_layer_{i}')
                output, hidden = lstm_layer(output)
            
            # Extract output from last timestep
            # output shape: [batch, 1, hidden_dim]
            lstm_output = output[:, 0, :]  # [batch_size, hidden_dim]
            
            # Project to vocabulary: [batch_size, vocab_size]
            logits = self.fc_out(lstm_output)
            output_logits_list.append(logits)
            
            # Decide next token
            # During training: use teacher forcing; during inference: use predicted token
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth token
                current_token = target_seq[:, t]
            else:
                # Autoregressive: use predicted token (argmax)
                current_token = mx.argmax(logits, axis=1, keepdims=False).astype(mx.uint32)
        
        # Stack all timestep outputs: [batch_size, seq_length, vocab_size]
        output_logits = mx.stack(output_logits_list, axis=1)
        
        return output_logits
