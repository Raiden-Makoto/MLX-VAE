#!/bin/bash
# Training script for AR-CVAE molecular generation

# Activate virtual environment
source qvae/bin/activate

# Run training script with arguments
python train.py "$@"

