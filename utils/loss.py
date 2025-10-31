import mlx.core as mx
import os
import numpy as np

# Cache for PAD token (loaded once, reused)
_pad_token_cache = None

def get_pad_token():
    """Lazy load and cache PAD token (only load once)."""
    global _pad_token_cache
    if _pad_token_cache is None:
        try:
            import json
            with open('mlx_data/chembl_cns_selfies.json') as f:
                meta = json.load(f)
            _pad_token_cache = meta['token_to_idx']['<PAD>']
        except:
            _pad_token_cache = 0  # Default PAD token
    return _pad_token_cache

PAD = get_pad_token()

# Cache for token mappings (loaded once, reused)
_token_mappings_cache = None

def get_token_mappings():
    """Lazy load and cache token mappings (only load once)."""
    global _token_mappings_cache
    if _token_mappings_cache is None:
        try:
            import json
            with open('mlx_data/chembl_cns_selfies.json') as f:
                meta = json.load(f)
            _token_mappings_cache = (meta['idx_to_token'], meta['token_to_idx'])
        except:
            _token_mappings_cache = (None, None)
    return _token_mappings_cache

# Cache for TPSA normalization stats (loaded once, reused)
_tpsa_norm_cache = None

def get_tpsa_normalization():
    """Lazy load and cache TPSA normalization stats (mean, std)."""
    global _tpsa_norm_cache
    if _tpsa_norm_cache is None:
        try:
            import json
            # Try to load from saved file first
            norm_file = 'checkpoints/property_norm.json'
            if os.path.exists(norm_file):
                with open(norm_file, 'r') as f:
                    norm_stats = json.load(f)
                _tpsa_norm_cache = (norm_stats.get('tpsa_mean', 82.0), norm_stats.get('tpsa_std', 54.86))
            else:
                # Fallback: compute from dataset
                with open('mlx_data/chembl_cns_selfies.json') as f:
                    meta = json.load(f)
                molecules = meta.get('molecules', [])
                if molecules:
                    tpsa_values = np.array([mol.get('tpsa', 82.0) for mol in molecules], dtype=np.float32)
                    _tpsa_norm_cache = (float(np.mean(tpsa_values)), float(np.std(tpsa_values)))
                else:
                    _tpsa_norm_cache = (82.0, 54.86)  # Default fallback
        except:
            _tpsa_norm_cache = (82.0, 54.86)  # Default fallback
    return _tpsa_norm_cache

def latent_diversity_loss(z):
    """
    Compute a differentiable diversity penalty on a batch of latent vectors.
    Encourages latent codes to be dissimilar (low average cosine similarity).

    Args:
        z: MLX array of shape (batch_size, latent_dim)

    Returns:
        diversity_loss: scalar (average off-diagonal cosine similarity)
    """
    batch_size = z.shape[0]
    
    # 1. Normalize each latent vector to unit norm (with epsilon for stability)
    z_norm = z / mx.sqrt(mx.sum(z * z, axis=1, keepdims=True) + 1e-8)  # shape: (B, D)
    
    # 2. Compute cosine similarity matrix: (B, D) @ (D, B) -> (B, B)
    sim_matrix = mx.matmul(z_norm, z_norm.T)
    
    # 3. Create mask to exclude diagonal self-similarities
    mask = mx.ones_like(sim_matrix)
    mask = mx.where(mx.eye(batch_size), mx.zeros_like(mask), mask)
    
    # 4. Sum and average only off-diagonal entries
    # sum of similarities
    sim_sum = mx.sum(sim_matrix * mask)
    # number of off-diagonal pairs = B*(B-1)
    pair_count = batch_size * (batch_size - 1)
    
    # 5. Diversity loss = average pairwise similarity
    diversity_loss = sim_sum / pair_count
    return diversity_loss

def _decode_logits_to_tokens(logits):
    """
    Greedily decode logits to token sequences.
    
    Args:
        logits: [B, T-1, V] model output logits
    
    Returns:
        decoded_tokens: [B, T-1] token indices (numpy array)
    """
    # Greedy decode: argmax at each position
    # logits: [B, T-1, V] -> [B, T-1] after argmax
    if isinstance(logits, mx.array):
        logits_np = np.array(logits)
    else:
        logits_np = logits
    
    decoded = np.argmax(logits_np, axis=-1)  # [B, T-1]
    return decoded

def _compute_tpsa_from_logits(logits, target_tpsa_raw=None):
    """
    Decode logits → tokens → SELFIES → SMILES → TPSA.
    Returns computed TPSA values and loss.
    
    Args:
        logits: [B, T-1, V] model output
        target_tpsa_raw: [B] raw (unnormalized) target TPSA values
    
    Returns:
        property_loss: MSE loss between computed and target TPSA (or 0.0 if computation fails)
        computed_tpsa: list of computed TPSA values (for monitoring)
    """
    # Lazy load token mappings only when needed
    idx_to_token, _ = get_token_mappings()
    if idx_to_token is None or target_tpsa_raw is None:
        return mx.array(0.0), []
    
    try:
        from selfies import decoder as sf_decoder
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit import RDLogger
        
        # Suppress RDKit warnings (stereochemistry conflicts are automatically resolved)
        RDLogger.DisableLog('rdApp.*')
        
        # Decode logits to tokens
        decoded_tokens = _decode_logits_to_tokens(logits)  # [B, T-1]
        batch_size = decoded_tokens.shape[0]
        
        # Convert tokens to SELFIES strings
        selfies_list = []
        for b in range(batch_size):
            tokens_list = []
            for token_idx in decoded_tokens[b]:
                token_key = str(int(token_idx))
                if token_key in idx_to_token:
                    token = idx_to_token[token_key]
                    if token in ['<END>', '[END]', '<PAD>', '[PAD]']:
                        break
                    if token not in ['<START>', '[START]']:
                        tokens_list.append(token)
            selfies_str = ''.join(tokens_list)
            selfies_list.append(selfies_str)
        
        # Convert SELFIES to SMILES and compute TPSA
        computed_tpsa = []
        valid_indices = []
        for i, selfies in enumerate(selfies_list):
            try:
                smiles = sf_decoder(selfies)
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        tpsa = Descriptors.TPSA(mol)
                        computed_tpsa.append(tpsa)
                        valid_indices.append(i)
            except:
                continue
        
        if len(valid_indices) == 0:
            return mx.array(0.0), []
        
        # Extract target TPSA for valid samples
        if isinstance(target_tpsa_raw, mx.array):
            target_np = np.array(target_tpsa_raw)
        else:
            target_np = target_tpsa_raw
        
        target_tpsa_valid = target_np[valid_indices]  # [N] where N = num valid
        computed_tpsa_arr = np.array(computed_tpsa, dtype=np.float32)  # [N]
        
        # Normalize TPSA values before computing MSE (same normalization as model uses)
        tpsa_mean, tpsa_std = get_tpsa_normalization()
        if tpsa_std > 0:
            target_norm = (target_tpsa_valid - tpsa_mean) / tpsa_std
            computed_norm = (computed_tpsa_arr - tpsa_mean) / tpsa_std
            # Compute MSE loss on normalized values (much smaller scale)
            mse = np.mean((computed_norm - target_norm) ** 2)
        else:
            # Fallback: compute on raw values if std is zero
            mse = np.mean((computed_tpsa_arr - target_tpsa_valid) ** 2)
        
        # Clip property loss to reasonable range to prevent explosion (normalized MSE should be ~0.01-1.0)
        mse = np.clip(mse, 0.0, 100.0)
        
        return mx.array(mse), computed_tpsa
        
    except Exception as e:
        # If anything fails, return zero loss (non-blocking)
        return mx.array(0.0), []

def compute_loss(x, logits, mu, logvar, beta: float=1.0, diversity_weight: float=0.01, 
                 target_tpsa_raw=None, property_weight: float=0.0):
    """
    Compute VAE loss: reconstruction + β * KL divergence + diversity penalty + property loss
    
    Args:
        x: Input sequences [B, T]
        logits: Model predictions [B, T-1, V]
        mu: Latent mean [B, L]
        logvar: Latent log-variance [B, L]
        beta: KL weight
        diversity_weight: Weight for diversity loss
        target_tpsa_raw: Target TPSA (raw, unnormalized) [B] (optional)
        property_weight: Weight for property reconstruction loss
    
    Returns:
        total_loss, recon_loss, kl_loss, diversity_loss, property_loss
    """
    target = x[:, 1:]  # Shift for next token prediction
    
    # Reconstruction loss (cross-entropy) with masking and numeric stability
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + 1e-9)
    per_token_nll = -mx.take_along_axis(log_probs, mx.expand_dims(target, axis=-1), axis=-1).squeeze(-1)
    mask = (target != PAD).astype(mx.float32)
    # Sum over valid tokens, normalize by count of valid tokens
    valid_count = mx.maximum(mx.sum(mask), 1.0)
    recon_loss = mx.sum(per_token_nll * mask) / valid_count
    
    # KL divergence loss
    kl_loss = -0.5 * mx.mean(1 + logvar - mx.square(mu) - mx.exp(logvar))
    
    # Clip KL loss to prevent numerical instability (reasonable range)
    kl_loss = mx.clip(kl_loss, -10.0, 10.0)
    
    # Diversity loss (encourage diverse latent representations)
    diversity_loss = latent_diversity_loss(mu)
    
    # Property reconstruction loss: decode molecules and compute actual TPSA
    property_loss = mx.array(0.0)
    if property_weight > 0 and target_tpsa_raw is not None:
        property_loss, _ = _compute_tpsa_from_logits(logits, target_tpsa_raw)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss + diversity_weight * diversity_loss + property_weight * property_loss
    
    return total_loss, recon_loss, kl_loss, diversity_loss, property_loss