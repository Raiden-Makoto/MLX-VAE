import mlx.core as mx
import mlx.nn as nn

class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input gate weights
        self.W_ii = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Forget gate weights
        self.W_if = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Cell gate weights
        self.W_ig = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Output gate weights
        self.W_io = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, bias=True)
    
    def __call__(self, x, initial_state=None):
        # x: [B, T, input_dim]
        batch_size, seq_len, _ = x.shape
        
        if initial_state is None:
            h = mx.zeros((batch_size, self.hidden_dim))
            c = mx.zeros((batch_size, self.hidden_dim))
        else:
            h, c = initial_state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, input_dim]
            
            # Input gate
            i_t = mx.sigmoid(self.W_ii(x_t) + self.W_hi(h))
            
            # Forget gate
            f_t = mx.sigmoid(self.W_if(x_t) + self.W_hf(h))
            
            # Cell gate
            g_t = mx.tanh(self.W_ig(x_t) + self.W_hg(h))
            
            # Output gate
            o_t = mx.sigmoid(self.W_io(x_t) + self.W_ho(h))
            
            # Update cell state
            c = f_t * c + i_t * g_t
            
            # Update hidden state
            h = o_t * mx.tanh(c)
            
            outputs.append(h)
        
        # Stack outputs: [B, T, hidden_dim]
        outputs = mx.stack(outputs, axis=1)
        return outputs, (h, c)
