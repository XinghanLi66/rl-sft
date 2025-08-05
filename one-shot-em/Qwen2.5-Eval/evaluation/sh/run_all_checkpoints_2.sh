#!/usr/bin/env bash
set -euo pipefail
set -x   # echo commands for easier debugging

# ROOT_DIR="/homes/gws/lxh22/rl-sft/one-shot-em/checkpoints/Qwen2.5-Math-1.5B/one_shot_1.5b_t0.5"
ROOT_DIR="/local1/lxh/save/em/Qwen2.5-Math-7B/prompted_one_shot_7b_t0.5"
EVAL_SCRIPT="sh/eval_lxh_2.sh"   # path to your eval script

STEPS=(20 40 60 80) 

for STEP in "${STEPS[@]}"; do
    CKPT="${ROOT_DIR}/step_${STEP}"
    export MODEL_NAME_OR_PATH="$CKPT"
    export OUTPUT_DIR="${CKPT}/temp06"
    mkdir -p "$OUTPUT_DIR"
    bash "$EVAL_SCRIPT"
done

echo "✅  Finished submitting selected checkpoints to eval"