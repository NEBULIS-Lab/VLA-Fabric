#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 DATASET BASE_CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

DATASET=$1
BASE_CHECKPOINT=$2
OUTPUT_ROOT=$3
GPUS=${GPUS:-0,1,2,3}
COORDINATOR_PORT=${COORDINATOR_PORT:-29620}
PYTHON=${PYTHON:-python}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ ${#GPU_LIST[@]} -ne 4 ]]; then
  echo "the paper protocol requires exactly four GPU IDs" >&2
  exit 2
fi

run_stage() {
  local stage=$1
  local steps=$2
  local warmup=$3
  local seed=$4
  local output=$5
  local stage1=${6:-}
  local coordinator="127.0.0.1:${COORDINATOR_PORT}"
  local pids=()

  for rank in 0 1 2 3; do
    local extra=()
    if [[ -n "$stage1" ]]; then
      extra+=(--stage1-checkpoint "$stage1")
    fi
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$rank]} \
      "$PYTHON" scripts/train_dual_pi05.py \
        --stage "$stage" \
        --dataset "$DATASET" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --output "$output" \
        --steps "$steps" \
        --batch-size 1 \
        --device-count 4 \
        --gradient-accumulation 3 \
        --learning-rate 5e-5 \
        --final-learning-rate 5e-6 \
        --warmup-steps "$warmup" \
        --checkpoint-every 10000 \
        --model-seed 20260824 \
        --training-seed "$seed" \
        --distributed-process-count 4 \
        --distributed-process-id "$rank" \
        --distributed-coordinator-address "$coordinator" \
        --freeze-action-expert-ffw \
        --preload-dataset \
        --execute \
        "${extra[@]}" &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ $failed -eq 0 ]]
}

STAGE1_OUTPUT="$OUTPUT_ROOT/stage1"
STAGE2_OUTPUT="$OUTPUT_ROOT/stage2"
run_stage pi_native_v2_raw_common_stage1 10000 500 20260828 "$STAGE1_OUTPUT"
run_stage pi_native_v2_residual_action_stage2 40000 2000 20260829 \
  "$STAGE2_OUTPUT" "$STAGE1_OUTPUT/checkpoint_10000"

