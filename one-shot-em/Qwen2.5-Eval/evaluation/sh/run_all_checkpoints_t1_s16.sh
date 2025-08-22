#!/usr/bin/env bash
set -euo pipefail
set -x   # echo commands for easier debugging

# ROOT_DIR="/homes/gws/lxh22/rl-sft/one-shot-em/checkpoints/Qwen2.5-Math-1.5B/one_shot_1.5b_t0.5"
ROOT_DIR="/homes/gws/lxh22/rl-sft/DFT/verl/checkpoints/numina-cot-sft-qwen-2.5-math-1.5b"
EVAL_SCRIPT="sh/eval_lxh_t1_s16.sh"   # path to your eval script

STEPS=(0 100 200 300 400 500 600 700 781)

for STEP in "${STEPS[@]}"; do
    CKPT="${ROOT_DIR}/global_step_${STEP}"
    export MODEL_NAME_OR_PATH="$CKPT"
    export OUTPUT_DIR="${CKPT}/temp10"
    mkdir -p "$OUTPUT_DIR"
    bash "$EVAL_SCRIPT"
done

echo "✅  Finished submitting selected checkpoints to eval"

