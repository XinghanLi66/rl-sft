#!/usr/bin/env bash
set -euo pipefail
set -x

# --- 默认值定义 ---
PROJECT_NAME="offline_grpo"
EXPERIMENT_NAME="7b_pi1_ofrl"
ROOT_DIR="/local1/lxh/save"
STEPS_STR="160 120 80 40 200 240"
EVAL_DIV="temp00"

# --- 使用帮助 ---
usage() {
    echo "Usage: $0 [-d <root_dir>] [-s <steps_string>]"
    echo "  -p  Project name."
    echo "  -e  Experiment name."
    echo "  -d  Path to the root directory of checkpoints."
    echo "  -s  A space-separated string of steps to evaluate (e.g., \"1000 2000\")."
    echo "  -v  Eval division."
    exit 1
}

# --- 解析命令行参数 ---
while getopts ":p:e:d:s:v:h" opt; do
    case ${opt} in
        p) PROJECT_NAME="$OPTARG";;
        e) EXPERIMENT_NAME="$OPTARG";;
        d) ROOT_DIR="$OPTARG";;
        s) STEPS_STR="$OPTARG";;
        v) EVAL_DIV="$OPTARG";;
        h) usage;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage;;
    esac
done

# --- 脚本主逻辑 ---
# 将 STEPS 字符串转换为数组
case $EVAL_DIV in
    temp00) EVAL_SCRIPT="sh/eval_lxh.sh";;
    temp06) EVAL_SCRIPT="sh/eval_lxh_2.sh";;
    *) echo "Unknown eval division: $EVAL_DIV"; exit 1;;
esac

IFS=' ' read -r -a STEPS <<< "$STEPS_STR"
ROOT_DIR="${ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

echo "Running evaluation with:"
echo "  ROOT_DIR = $ROOT_DIR"
echo "  STEPS    = ${STEPS[@]}"

for STEP in "${STEPS[@]}"; do
    CKPT="${ROOT_DIR}/global_step_${STEP}"
    export MODEL_NAME_OR_PATH="$CKPT"
    export OUTPUT_DIR="${CKPT}/${EVAL_DIV}"
    mkdir -p "$OUTPUT_DIR"
    bash "$EVAL_SCRIPT"
done

echo "✅ Finished submitting selected checkpoints to eval"