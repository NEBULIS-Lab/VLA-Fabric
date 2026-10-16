#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CHECKPOINT BASE_CHECKPOINT DATASET TASK OUTPUT" >&2
  exit 2
fi

PYTHON=${PYTHON:-python}
"$PYTHON" scripts/evaluate_native_paired.py \
  --checkpoint "$1" \
  --expected-stage pi_native_v2_residual_action_stage2 \
  --expected-step 40000 \
  --base-checkpoint "$2" \
  --dataset "$3" \
  --task "$4" \
  --output-dir "$5" \
  --conditions fresh \
  --seed 20260824 \
  --num-denoising-steps 10 \
  --inference-mode full \
  --execute-horizon 25 \
  --early-gpu-claim

