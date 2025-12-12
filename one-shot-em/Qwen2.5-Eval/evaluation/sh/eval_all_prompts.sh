#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="4,5,6,7"
TOKENIZERS_PARALLELISM=false
MODEL_TAG="1.5B" # 1.5B, 7B-Instruct

MODEL_NAME_OR_PATH="/homes/gws/lxh22/models/Qwen2.5-Math-${MODEL_TAG}"
RESULTS_BASE="/homes/gws/lxh22/rl-sft/one-shot-em/results"
EVAL_PY="/homes/gws/lxh22/rl-sft/one-shot-em/Qwen2.5-Eval/evaluation/math_eval.py"

DATA_NAMES="minerva_math,olympiadbench,math500"
SPLIT="test"
NUM_TEST_SAMPLE=-1
MAX_TOKENS_PER_CALL="3072"
TEMPERATURE=0.0

# Prompts to evaluate
PROMPTS=(
  "to"
  "pi1"
  "to+pi1"
  "to+pi1+qr"
  "to+pi1+ci"
  "pi1+ci"
  "nb"
  "to+pi1+nb"
  "to+pi1+qr+nb"
)
# ----------------------------------------------------------------

echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Data:  ${DATA_NAMES} (${SPLIT})"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo

for PROMPT_TYPE in "${PROMPTS[@]}"; do
  OUTPUT_DIR="${RESULTS_BASE}/${MODEL_TAG}-eval-${PROMPT_TYPE}"
  echo "============================================================"
  echo "Running prompt_type='${PROMPT_TYPE}'"
  echo "Output dir          = '${OUTPUT_DIR}'"
  echo "============================================================"

  mkdir -p "${OUTPUT_DIR}"

  python3 -u "${EVAL_PY}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_name "${DATA_NAMES}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 0 \
    --temperature ${TEMPERATURE} \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --max_tokens_per_call "${MAX_TOKENS_PER_CALL}" \
    --overwrite

  echo "Done: ${PROMPT_TYPE}"
  echo
done

echo "All prompts finished."
