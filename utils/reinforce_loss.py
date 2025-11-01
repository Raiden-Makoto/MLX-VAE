# REINFORCE-based policy gradient for molecular property optimization
# Trains decoder to generate molecules with target TPSA

import mlx.core as mx
import mlx.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import numpy as np

# Suppress RDKit warnings (stereochemistry conflicts are automatically resolved)
RDLogger.DisableLog('rdApp.*')


def _sample_tokens_from_logits(logits, temperature: float = 1.0, top_k: int = 0):
    """
    Stochastically sample token ids from logits with optional temperature and top-k.
    Returns [B, T] int token ids.
    """
    if temperature != 1.0:
        logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        top_vals = mx.sort(logits, axis=-1)[:, :, -k:]
        kth = top_vals[:, :, :1]
        logits = mx.where(logits < kth, -1e9, logits)
    probs = mx.softmax(logits, axis=-1)
    # Sample per position
    B, T, V = probs.shape
    sampled = []
    for t in range(T):
        p_t = probs[:, t, :]
        tok_t = mx.random.categorical(p_t, axis=-1).astype(mx.int32)
        sampled.append(tok_t)
    return mx.stack(sampled, axis=1)


def compute_tpsa_batch(logits, vocab_to_selfies, selfies_to_smiles, temperature: float = 1.0, top_k: int = 0, use_sampling: bool = True):
    """
    Compute TPSA from logits without argmax (for batch processing)
    
    Args:
        logits: [B, seq_len, vocab_size]
        vocab_to_selfies: dict mapping token id to SELFIES symbol
        selfies_to_smiles: function to convert SELFIES to SMILES
    
    Returns:
        tpsa_values: [B] array of TPSA values
        valid_mask: [B] boolean mask of valid molecules
    """
    B, seq_len, vocab_size = logits.shape
    tpsa_values = []
    valid_mask = []
    
    # Decode tokens for property computation (non-differentiable path)
    if use_sampling:
        token_ids = _sample_tokens_from_logits(logits, temperature=temperature, top_k=top_k)
    else:
        token_ids = mx.argmax(logits, axis=-1)  # greedy
    token_ids_np = np.array(token_ids)  # Convert to numpy for easier processing
    
    for b in range(B):
        try:
            # Convert tokens to SELFIES
            tokens = token_ids_np[b].tolist()
            
            # Build SELFIES string, skipping special tokens
            selfies_tokens = []
            for token_id in tokens:
                token_str = vocab_to_selfies.get(int(token_id), "")
                # Stop on END/PAD tokens
                if token_str in ['<END>', '[END]', '<PAD>', '[PAD]']:
                    break
                # Skip START tokens
                if token_str not in ['<START>', '[START]', '', None]:
                    selfies_tokens.append(token_str)
            
            selfies_str = ''.join(selfies_tokens)
            
            if not selfies_str:
                tpsa_values.append(0.0)
                valid_mask.append(False)
                continue
            
            # Convert to SMILES
            smiles = selfies_to_smiles(selfies_str)
            if not smiles:
                tpsa_values.append(0.0)
                valid_mask.append(False)
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                tpsa = Descriptors.TPSA(mol)
                tpsa_values.append(tpsa)
                valid_mask.append(True)
            else:
                tpsa_values.append(0.0)
                valid_mask.append(False)
        except Exception as e:
            tpsa_values.append(0.0)
            valid_mask.append(False)
    
    return mx.array(tpsa_values), mx.array(valid_mask)


def compute_reinforce_loss(
    logits,
    target_tpsa,
    vocab_to_selfies,
    selfies_to_smiles,
    baseline=None,
    gamma=0.9999,
    temperature: float = 1.0,
    top_k: int = 0,
    use_sampling: bool = True,
    tol_band: float = 0.0,
    reward_scale: float = 1.0,
):
    """
    Compute REINFORCE policy gradient loss for TPSA targeting
    
    Args:
        logits: [B, seq_len, vocab_size] decoder output
        target_tpsa: [B] target TPSA values
        vocab_to_selfies: dict mapping token id to SELFIES
        selfies_to_smiles: conversion function
        baseline: optional value function for variance reduction
        gamma: discount factor (close to 1 for no discounting)
    
    Returns:
        policy_loss: scalar loss to minimize
        reward: [B] reward signal (for monitoring)
        valid_mask: [B] which samples were valid
    """
    B, seq_len, vocab_size = logits.shape
    
    # 1. Compute properties from generated molecules
    pred_tpsa, valid_mask = compute_tpsa_batch(
        logits, vocab_to_selfies, selfies_to_smiles,
        temperature=temperature, top_k=top_k, use_sampling=use_sampling
    )
    
    # 2. Compute reward: reward progress toward target (smoother reward signal)
    # Convert to MLX arrays for computation
    pred_mx = pred_tpsa if isinstance(pred_tpsa, mx.array) else mx.array(pred_tpsa)
    target_mx = target_tpsa if isinstance(target_tpsa, mx.array) else mx.array(target_tpsa)
    valid_mx = valid_mask if isinstance(valid_mask, mx.array) else mx.array(valid_mask)
    
    # Compute absolute error
    tpsa_error = mx.abs(pred_mx - target_mx)
    
    # Initialize reward array with penalty for invalid molecules
    reward_arr = mx.full((B,), -100.0)
    
    if tol_band and tol_band > 0:
        # Normalized reward: -abs(pred_tpsa - target_tpsa) / tol_band
        # This gives reward in [-1, 0] range: closer to target -> higher (less negative) reward
        # Reward any improvement, not just hitting target exactly
        reward_shaped = -tpsa_error / float(tol_band)
        
        # Don't clip yet - preserve signal range
        # Add bonus for being within tolerance band (positive reinforcement for good performance)
        within_tol = tpsa_error <= tol_band
        # Bonus makes reward positive when within tolerance
        reward_shaped = mx.where(within_tol, reward_shaped + 1.0, reward_shaped)
        
        # Apply to valid molecules only
        reward_arr = mx.where(valid_mx, reward_shaped, reward_arr)
    else:
        # No tolerance band: simple negative error penalty
        reward_arr = mx.where(valid_mx, -tpsa_error, reward_arr)
    
    # Scale reward if requested
    if reward_scale != 1.0:
        reward_arr = reward_arr * float(reward_scale)
    
    # Clip reward to preserve signal (allow -167 through, not just -30)
    reward_arr = mx.clip(reward_arr, -167.0, 10.0)  # Allow negative errors up to -167, positive up to 10
    
    reward = reward_arr
    valid_mask_mx = valid_mx
    
    # 3. Baseline subtraction for variance reduction (optional)
    if baseline is not None:
        advantage = reward - baseline
    else:
        # Simple baseline: mean reward (reduces variance)
        advantage = reward - mx.mean(reward)
    
    # 4. Compute log probabilities from logits (differentiable)
    # log_softmax = log(softmax(...))
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + 1e-9)  # [B, seq_len, vocab_size] (add epsilon for stability)
    
    # Get token ids used for property computation (argmax, non-differentiable)
    token_ids = mx.argmax(logits, axis=-1)  # [B, seq_len]
    
    # Gather log probs for the tokens we actually used (differentiable!)
    # This is the key: we use argmax to select which log_prob to use, but log_prob itself is differentiable
    selected_log_probs = mx.take_along_axis(
        log_probs,
        mx.expand_dims(token_ids, axis=-1),
        axis=-1
    )  # [B, seq_len, 1]
    
    selected_log_probs = mx.squeeze(selected_log_probs, axis=-1)  # [B, seq_len]
    
    # Sum log probs over sequence (trajectory log likelihood)
    trajectory_log_prob = mx.sum(selected_log_probs, axis=-1)  # [B]
    
    # 5. REINFORCE: loss = -E[log_prob * advantage]
    # We want to increase log_prob when advantage > 0 (good reward)
    policy_loss = -mx.mean(trajectory_log_prob * advantage)
    
    return policy_loss, reward, valid_mask_mx


class REINFORCELoss(nn.Module):
    """Module wrapper for REINFORCE loss computation with cached mappings"""
    
    def __init__(self, vocab_to_selfies, selfies_to_smiles, temperature: float = 1.0, top_k: int = 0, use_sampling: bool = True, tol_band: float = 0.0, reward_scale: float = 1.0):
        super().__init__()
        self.vocab_to_selfies = vocab_to_selfies
        self.selfies_to_smiles = selfies_to_smiles
        self.baseline = None  # Can be trained value function
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.use_sampling = bool(use_sampling)
        self.tol_band = float(tol_band)
        self.reward_scale = float(reward_scale)
    
    def __call__(self, logits, target_tpsa):
        return compute_reinforce_loss(
            logits,
            target_tpsa,
            self.vocab_to_selfies,
            self.selfies_to_smiles,
            baseline=self.baseline,
            temperature=self.temperature,
            top_k=self.top_k,
            use_sampling=self.use_sampling,
            tol_band=self.tol_band,
            reward_scale=self.reward_scale,
        )


# Integration example for train.py:
"""
# In train.py, replace property loss computation:

from reinforce_loss import REINFORCELoss

# Initialize
reinforce = REINFORCELoss(vocab_to_selfies, selfies_to_smiles)

# In training loop:
logits, mu, logvar, tpsa_pred = model(input_seq, properties=batch_properties)

# Standard VAE losses
recon_loss = cross_entropy(logits, target_seq)
kl_loss = -0.5 * mx.sum(1 + logvar - mu**2 - mx.exp(logvar))

# REINFORCE policy gradient loss
policy_loss, reward, valid_mask = reinforce(logits, batch_properties[:, 1])

# Total loss (curriculum: increase policy_weight over epochs)
epoch_ratio = epoch / total_epochs
policy_weight = min(100.0 * epoch_ratio, 10.0)  # Ramp up gradually

total_loss = recon_loss + beta * kl_loss + policy_weight * policy_loss

# Log metrics
valid_pct = mx.mean(valid_mask.astype(mx.float32))
mean_reward = mx.mean(mx.where(valid_mask, reward, 0))

print(f"Epoch {epoch}: Recon={recon_loss:.4f} KL={kl_loss:.4f} "
      f"Policy={policy_loss:.4f} MeanReward={mean_reward:.4f} Valid={valid_pct*100:.1f}%")
"""
