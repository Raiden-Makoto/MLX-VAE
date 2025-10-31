# REINFORCE-based policy gradient for molecular property optimization
# Trains decoder to generate molecules with target TPSA

import mlx.core as mx
import mlx.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


def compute_tpsa_batch(logits, vocab_to_selfies, selfies_to_smiles):
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
    
    # Argmax decode (only for property computation, not differentiable)
    token_ids = mx.argmax(logits, axis=-1)  # [B, seq_len]
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
    pred_tpsa, valid_mask = compute_tpsa_batch(logits, vocab_to_selfies, selfies_to_smiles)
    
    # 2. Compute reward: negative absolute error (we want to minimize error)
    # Convert to numpy for easier computation
    pred_np = np.array(pred_tpsa)
    target_np = np.array(target_tpsa) if isinstance(target_tpsa, mx.array) else target_tpsa
    valid_np = np.array(valid_mask) if isinstance(valid_mask, mx.array) else valid_mask
    
    tpsa_error = np.abs(pred_np - target_np)
    
    # Only compute reward for valid molecules
    reward_arr = np.full(B, -100.0, dtype=np.float32)  # Default penalty
    reward_arr[valid_np] = -tpsa_error[valid_np]  # Negative error = positive reward for good molecules
    
    reward = mx.array(reward_arr)
    valid_mask_mx = mx.array(valid_np.astype(np.float32))
    
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
    
    def __init__(self, vocab_to_selfies, selfies_to_smiles):
        super().__init__()
        self.vocab_to_selfies = vocab_to_selfies
        self.selfies_to_smiles = selfies_to_smiles
        self.baseline = None  # Can be trained value function
    
    def __call__(self, logits, target_tpsa):
        return compute_reinforce_loss(
            logits,
            target_tpsa,
            self.vocab_to_selfies,
            self.selfies_to_smiles,
            baseline=self.baseline,
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
