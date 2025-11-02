import mlx.core as mx
import mlx.nn as nn
from .decoder import MLXAutoregressiveDecoder


class MLXAutoregressiveDecoderSampling(nn.Module):
    """
    Enhanced autoregressive decoder with temperature-based sampling
    
    This wrapper extends the base decoder with sampling capabilities
    for inference-time generation with temperature control.
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
        """Initialize decoder with sampling capabilities"""
        super().__init__()
        
        # Use the standard decoder as base
        self.decoder = MLXAutoregressiveDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_conditions=num_conditions,
            num_layers=num_layers,
            pad_token=pad_token,
            end_token=end_token
        )
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.pad_token = pad_token
        self.end_token = end_token
    
    def generate_with_temperature(
        self,
        z: mx.array,
        conditions: mx.array,
        max_length: int = 120,
        temperature: float = 1.0,
        early_stopping: bool = True
    ) -> mx.array:
        """
        Generate molecules using temperature-based sampling
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            conditions: Property conditions [batch_size, num_conditions]
            max_length: Maximum sequence length
            temperature: Sampling temperature
                - < 1.0: More deterministic
                - = 1.0: Standard softmax
                - > 1.0: More diverse
            early_stopping: Stop when end token is encountered
        
        Returns:
            generated_tokens: Token sequences [batch_size, seq_length]
        """
        batch_size = z.shape[0]
        
        # Initialize hidden and cell states
        hidden, cell = self.decoder.initialize_hidden_state(z, conditions)
        
        # Start token
        current_token = mx.zeros((batch_size,), dtype=mx.uint32)
        
        # Track which sequences have ended
        has_ended = mx.zeros((batch_size,), dtype=mx.bool_)
        
        generated_tokens = []
        
        for t in range(max_length):
            # Check early stopping
            if early_stopping and mx.all(has_ended):
                break
            
            # Embed current token
            embedded = self.decoder.embedding(current_token)
            
            # Concatenate with conditions
            lstm_input = mx.concatenate([embedded, conditions], axis=1)
            lstm_input = mx.expand_dims(lstm_input, 1)
            
            # LSTM step through stacked layers
            output = lstm_input
            for i in range(self.decoder.num_layers):
                lstm_layer = getattr(self.decoder, f'lstm_layer_{i}')
                output, hidden = lstm_layer(output)
            
            # Extract output from last timestep
            lstm_output = output[:, 0, :]  # [batch_size, hidden_dim]
            
            # Get logits
            logits = self.decoder.fc_out(lstm_output)
            
            # Apply temperature
            logits_scaled = logits / temperature
            
            # Compute probabilities
            probs = mx.softmax(logits_scaled, axis=1)
            
            # Sample from distribution using argmax for now
            # TODO: implement proper categorical sampling
            current_token = mx.argmax(probs, axis=1).astype(mx.uint32)
            
            generated_tokens.append(current_token)
            
            # Track ended sequences
            ended_now = current_token == self.end_token
            has_ended = mx.logical_or(has_ended, ended_now)
        
        # Stack tokens: [batch_size, seq_length]
        generated_tokens = mx.stack(generated_tokens, axis=1)
        
        return generated_tokens

