# rm -rf sh/test_in_context_yiping.sh; vim sh/test_in_context_yiping.sh

PROMPT_TYPE="qwen25-math-cot-tool"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MAX_TOKENS="3072"

MODEL_NAME_OR_PATH=/mmfs1/gscratch/simondu/lxh/models/Qwen2.5-Math-7B
OUTPUT_DIR_ROOT=/mmfs1/gscratch/simondu/lxh/results
PROJECT_NAME="in-context-debug"

run_experiment () {
    local NUM_SHOTS=$1
    local MODEL_NAME=$3
    local EXAMPLE_TYPE=$4
    
    local EXPERIMENT_NAME="${MODEL_NAME}_${NUM_SHOTS}shot_${EXAMPLE_TYPE}"
    local BASE_OUTPUT_DIR="${OUTPUT_DIR_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

    echo "============ ${EXPERIMENT_NAME} ============="
    echo "NUM_SHOTS: ${NUM_SHOTS}"
    echo "MODEL_NAME: ${MODEL_NAME}"
    echo "EXAMPLE_TYPE: ${EXAMPLE_TYPE}"
    echo "MODEL_NAME_OR_PATH: ${MODEL_NAME_OR_PATH}"
    
    ## multiple runs
    for i in {1..5}; do
        RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/pi1to/eval-${i}"
        echo "OUTPUT_DIR: ${RUN_OUTPUT_DIR}"
        bash sh/in-context-one-shot-to.sh \
             "$PROMPT_TYPE" "$MODEL_NAME_OR_PATH" "$MAX_TOKENS" \
             "$RUN_OUTPUT_DIR" "$NUM_SHOTS" "$EXAMPLE_TYPE"
    done
}

# define the parameters to run the experiment
# format: [num_shots model_master model_name example_type]
#                                                 minerva_math 	olympiadbench	math500    aime25x8  	amc23x8   	aime24x8_2
declare -a experiments=(
    
    # 72B model
    # "1 Qwen Qwen2.5-Math-72B pi1"                   # 28.3         	26.1         	57.0        5.4       	38.8      	22.1
    # "1 Qwen Qwen2.5-Math-72B pi1_step500_2"         # 23.9         	23.9         	52.4        7.9       	39.7      	19.6
    # "4 Qwen Qwen2.5-Math-72B official4"             # 19.5         	24.0         	51.4        4.2       	37.2      	10.8
    # "1 Qwen Qwen2.5-Math-72B official4"

    
    # 7B model
    # "1 Qwen Qwen2.5-Math-7B pi1_step0"            # 17.6         	25.9         	57.6        9.6       	36.9      	10.0 
    # "1 Qwen Qwen2.5-Math-7B pi1_step0_correct"      # 19.5         	32.6         	68.2        6.2       	41.6      	13.8 
    # "1 Qwen Qwen2.5-Math-7B pi1_step500_2"        # 29.0         	38.7         	69.6        6.7       	48.8      	15.8 
    # "1 Qwen Qwen2.5-Math-7B pi1_step500_longest"  # 25.7         	38.1         	66.8        8.8       	46.2      	15.8
    # "1 Qwen Qwen2.5-Math-7B pi1_step1300"         # 15.4         	23.7         	51.4        5.0       	31.2      	9.6
    # "1 Qwen Qwen2.5-Math-7B pi13_step500"         # 19.9         	20.6            45.0        5.0       	31.9      	12.9
    # "1 Qwen Qwen2.5-Math-7B pi13_step0_correct"     # 13.6         	19.3         	44.6        5.0       	27.8      	8.8
    # "4 Qwen Qwen2.5-Math-7B minerva_math"         # 10.3          17.0            46.8        1.2      	15.0      	1.3
    # "1 Qwen Qwen2.5-Math-7B minerva_math"         # 13.6          19.9            49.0        1.7      	16.6      	2.5

    
    # base model Qwen2.5-7B
    # "1 Qwen Qwen2.5-7B pi1"                         # 8.1          	11.1         	31.8        0.8       	15.0      	1.7
    # "1 Qwen Qwen2.5-7B pi1_step500_2"               # 7.0          	11.6         	30.8        0.4       	14.7      	1.7
    # "4 Qwen Qwen2.5-7B official4"                   # 12.9         	12.1         	41.2        0.0       	7.2       	0.0
    # "1 Qwen Qwen2.5-7B official4"                   # 8.8          	9.9          	39.2        0.4       	11.2      	0.8

    # base model Qwen2.5-1.5B
    # "1 Qwen Qwen2.5-1.5B pi1"                       # 3.7          	4.1          	20.0        0.8       	2.2       	0.0
    # "1 Qwen Qwen2.5-1.5B pi1_step500_2"             # 3.7          	3.1          	10.8        0.8       	2.8       	0.0
    # "4 Qwen Qwen2.5-1.5B official4"                 # 2.6          	2.1          	22.8        0.0       	3.4       	0.0
    # "1 Qwen Qwen2.5-1.5B official4"

    # main experiments
    "1 Qwen Qwen2.5-Math-7B pi1"                # 30.1         	41.3         	75.4        13.3       	48.4      	15.8
    # "4 Qwen Qwen2.5-Math-7B official4"
    # "1 Qwen Qwen2.5-Math-7B official4"
    
    # 1.5B model
    # "1 Qwen Qwen2.5-Math-1.5B pi1_step0"            # 17.6         	25.9         	57.6        9.6       	36.9      	10.0 
    # "1 Qwen Qwen2.5-Math-1.5B pi1_step0_correct"    # 
    # "1 Qwen Qwen2.5-Math-1.5B pi1_step500_2"        # 29.0         	38.7         	69.6        6.7       	48.8      	15.8 
    # "1 Qwen Qwen2.5-Math-1.5B pi1_step500_longest"  # 25.7         	38.1         	66.8        8.8       	46.2      	15.8
    # "1 Qwen Qwen2.5-Math-1.5B pi1_step1300"         # 15.4         	23.7         	51.4        5.0       	31.2      	9.6
    # "1 Qwen Qwen2.5-Math-1.5B pi13_step500"         # 19.9         	20.6            45.0        5.0       	31.9      	12.9
    # "1 Qwen Qwen2.5-Math-1.5B pi13_step0_correct"   # 
    # "4 Qwen Qwen2.5-Math-1.5B minerva_math"         # 10.3            17.0            46.8        1.2      	15.0      	1.3
    # "1 Qwen Qwen2.5-Math-1.5B minerva_math"         # 13.6            19.9            49.0        1.7      	16.6      	2.5
    
    # "1 Qwen Qwen2.5-Math-1.5B pi1"
    # "4 Qwen Qwen2.5-Math-1.5B official4"
    # "1 Qwen Qwen2.5-Math-1.5B official4"
    # "4 Qwen Qwen2.5-Math-1.5B minerva_math"
    # "1 Qwen Qwen2.5-Math-1.5B minerva_math"
)

# run all experiments
for exp in "${experiments[@]}"; do
    # split the experiment parameters into an array
    IFS=' ' read -r -a params <<< "$exp"
    
    # run the experiment
    run_experiment "${params[0]}" "${params[1]}" "${params[2]}" "${params[3]}"
done