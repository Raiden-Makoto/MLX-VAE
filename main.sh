#!/usr/bin/env bash
set -euo pipefail

# Train VAE for a specified number of epochs, then train TPSA→z predictor with early stopping,
# then run inference for a specified number of samples.

# Defaults
EPOCHS=50
NUM_SAMPLES=256
# Predictor early stopping (fixed; not configurable via CLI)
PRED_MAX_EPOCHS=100
PATIENCE=5

usage() {
  echo "Usage: $0 --epochs N --num_samples K" >&2
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --num_samples)
      NUM_SAMPLES="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "=== Step 1/3: Train VAE (epochs=$EPOCHS) ==="
python "$ROOT_DIR/train.py" --epochs "$EPOCHS"

echo "=== Step 2/3: Train TPSA→z predictor with early stopping (patience=$PATIENCE, max_epochs=$PRED_MAX_EPOCHS) ==="
BEST_MSE="inf"
NO_IMPROVE=0
for (( ep=1; ep<=PRED_MAX_EPOCHS; ep++ )); do
  echo "-- Predictor epoch $ep --"
  # Train a single epoch to enable shell-level early stopping
  OUT_FILE="/tmp/train_predictor_epoch_${ep}.log"
  python "$ROOT_DIR/train_predictor.py" --epochs 1 --checkpoint_dir "$ROOT_DIR/checkpoints" > "$OUT_FILE" 2>&1 || true
  # Parse last reported MSE from stdout
  MSE_LINE="$(grep -E "^PRED_MSE=" "$OUT_FILE" | tail -n 1 || true)"
  MSE_VAL="${MSE_LINE#PRED_MSE=}"
  if [[ -z "$MSE_VAL" ]]; then
    echo "Could not parse MSE from predictor output; continuing."
    continue
  fi
  echo "Epoch $ep predictor MSE=$MSE_VAL"
  # Compare floating values using python for reliability
  IMPROVED=$(python - <<PY
import math
best = float('inf') if "$BEST_MSE" == "inf" else float("$BEST_MSE")
cur = float("$MSE_VAL")
print(1 if cur < best - 1e-8 else 0)
PY
)
  if [[ "$BEST_MSE" == "inf" || "$IMPROVED" == "1" ]]; then
    BEST_MSE="$MSE_VAL"
    NO_IMPROVE=0
    echo "New best predictor MSE: $BEST_MSE"
  else
    NO_IMPROVE=$((NO_IMPROVE+1))
    echo "No improvement ($NO_IMPROVE/$PATIENCE)"
    if [[ $NO_IMPROVE -ge $PATIENCE ]]; then
      echo "Early stopping predictor training. Best MSE=$BEST_MSE"
      break
    fi
  fi
done

echo "=== Step 3/3: Run inference (num_samples=$NUM_SAMPLES) ==="
python "$ROOT_DIR/inference.py" --num_samples "$NUM_SAMPLES" --checkpoint "$ROOT_DIR/checkpoints"

echo "All done. Results in output/ and checkpoints/."


