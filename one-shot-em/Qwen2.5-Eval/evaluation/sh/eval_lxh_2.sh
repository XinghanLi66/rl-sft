set -x
export CUDA_VISIBLE_DEVICES="4,5,6,7"
MODEL_NAME_OR_PATH="/local1/lxh/save/offline_grpo/Qwen2.5-Math-7B_pi1_r128_offline_0815-1526/global_step_20/actor"
# /homes/gws/lxh22/models/Qwen2.5-Math-1.5B
# /homes/gws/lxh22/models/Qwen2.5-Math-7B
# /homes/gws/lxh22/models/Qwen2.5-Math-7B-Instruct

OUTPUT_DIR="/homes/gws/lxh22/rl-sft/one-shot-em/results/7B-eval-offRLVR-20"

mkdir -p $OUTPUT_DIR
PROMPT_TYPE="qwen25-math-cot"

# qwen25-math-cot (vanilla)
# to
# pi1
# to+pi1
# to+pi1+qr
# to+pi1+ci
# pi1+ci
# nb
# to+pi1+nb
# qwen25-math-cot-enhanced (to+pi1+qr+nb)

MAX_TOKENS_PER_CALL="3072"
SPLIT="test"
NUM_TEST_SAMPLE=-1
DATA_NAMES="amc23x8,aime25x8"
IFS=',' read -ra DATASETS <<< "$DATA_NAMES"
ALL_EXIST=true

TOKENIZERS_PARALLELISM=false \
python3 -u /homes/gws/lxh22/rl-sft/one-shot-em/Qwen2.5-Eval/evaluation/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAMES} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
    --overwrite 
