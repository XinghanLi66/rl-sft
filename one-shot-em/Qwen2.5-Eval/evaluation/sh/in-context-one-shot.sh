
set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MAX_TOKENS_PER_CALL=$3
OUTPUT_DIR=$4
NUM_SHOTS=$5
SELF_DEFINED_EXAMPLES_TYPE=${6:-"official4"}  # default: official4

SPLIT="test"
NUM_TEST_SAMPLE=-1




DATA_NAMES="math500,minerva_math,olympiadbench"
IFS=',' read -ra DATASETS <<< "$DATA_NAMES"
ALL_EXIST=true

for DATASET in "${DATASETS[@]}"; do
    if [ ! -d "${OUTPUT_DIR}/${DATASET}" ] || [ -z "$(find ${OUTPUT_DIR}/${DATASET} -name '*metrics.json' -print -quit)" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "============ All datasets in ${DATA_NAMES} already evaluated ============="
    for DATASET in "${DATASETS[@]}"; do
        METRICS_FILE=$(find ${OUTPUT_DIR}/${DATASET} -name '*metrics.json' -print -quit)
        echo "============ ${DATASET} ============="
        # cat top 15 lines
        cat ${METRICS_FILE} | head -n 10
    done
else
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAMES} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
        --overwrite \
        --num_shots=${NUM_SHOTS} \
        --self_defined_examples_type=${SELF_DEFINED_EXAMPLES_TYPE}
fi



DATA_NAMES="aime24x8_2,aime25x8,amc23x8"
IFS=',' read -ra DATASETS <<< "$DATA_NAMES"
ALL_EXIST=true

# Check if all datasets in ${DATA_NAMES} already evaluated, if so, skip evaluation
for DATASET in "${DATASETS[@]}"; do
    if [ ! -d "${OUTPUT_DIR}/${DATASET}" ] || [ -z "$(find ${OUTPUT_DIR}/${DATASET} -name '*metrics.json' -print -quit)" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "============ All datasets in ${DATA_NAMES} already evaluated ============="
    for DATASET in "${DATASETS[@]}"; do
        METRICS_FILE=$(find ${OUTPUT_DIR}/${DATASET} -name '*metrics.json' -print -quit)
        echo "============ ${DATASET} ============="
        # cat top 15 lines
        cat ${METRICS_FILE} | head -n 10
    done
else
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
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
        --overwrite \
        --num_shots=${NUM_SHOTS} \
        --self_defined_examples_type=${SELF_DEFINED_EXAMPLES_TYPE}
fi



