#!/bin/bash
# Prediction script with configurable parameters
# Usage: ./predict.sh <language> <dataset> [test_suffix] [gold_file]
# Example: ./predict.sh eng restaurant dev_task1
#          ./predict.sh zho laptop test_task1 ../task-dataset/track_a/subtask_1/zho/zho_laptop_test_task1.jsonl
#          ./predict.sh eng laptop eval_20   # Special mode for 80/20 split evaluation
#          ./predict.sh eng laptop test      # Special mode for test set prediction using traindev model
#          ./predict.sh all test             # Predict all datasets with test mode

set -e  # Exit on error

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <language> <dataset> [test_suffix] [gold_file]"
    echo "Example: $0 eng restaurant dev_task1"
    echo "         $0 zho laptop test_task1 ../task-dataset/track_a/subtask_1/zho/zho_laptop_test_task1.jsonl"
    echo "         $0 eng laptop eval_20   # Special mode for 80/20 split evaluation"
    echo "         $0 eng laptop test      # Special mode for test set prediction using traindev model"
    echo "         $0 all test             # Predict all datasets with test mode"
    exit 1
fi

# Special handling for "all" mode - predict all datasets
if [ "$1" == "all" ]; then
    MODE=$2
    echo "=========================================="
    echo "Predicting ALL datasets with mode: ${MODE}"
    echo "=========================================="
    
    DATASETS=(
        "eng laptop"
        "eng restaurant"
        "zho laptop"
        "zho restaurant"
        "zho finance"
    )
    
    for dataset in "${DATASETS[@]}"; do
        read -r lang name <<< "${dataset}"
        echo ""
        echo ">>> Predicting: ${lang}_${name} (${MODE} mode)"
        echo "------------------------------------------"
        "$0" "${lang}" "${name}" "${MODE}"
        echo ""
    done
    
    echo "=========================================="
    echo "All predictions completed!"
    if [ "${MODE}" == "test" ]; then
        echo "Output files saved to: ${OUTPUT_DIR}/test_prediction/"
    fi
    echo "=========================================="
    exit 0
fi

LANG_CODE=$1
DATASET=$2
TEST_SUFFIX=${3:-"dev_task1"}  # Default to dev_task1
GOLD_FILE=$4  # Optional gold file for evaluation

# Special handling for test mode (use traindev model for test set prediction)
if [ "${TEST_SUFFIX}" == "test" ]; then
    TEST_FILE="${DATA_DIR}/${LANG_CODE}/${LANG_CODE}_${DATASET}_test_task1.jsonl"
    CHECKPOINT="${CHECKPOINT_TRAINDEV}/${LANG_CODE}_${DATASET}/${LANG_CODE}_${DATASET}_best_model.pt"
    OUTPUT_FILE="${OUTPUT_DIR}/test_prediction/${LANG_CODE}/pred_${LANG_CODE}_${DATASET}.jsonl"
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}/test_prediction/${LANG_CODE}"
    
    echo "=========================================="
    echo "Prediction Configuration (Test Mode)"
    echo "=========================================="
    echo "Mode: test (using train+dev merged model)"
    echo "Language: ${LANG_CODE}"
    echo "Dataset: ${DATASET}"
    echo "Test file: ${TEST_FILE}"
    echo "Checkpoint: ${CHECKPOINT}"
    echo "Model: ${MODEL_NAME}"
    echo "Max length: ${MAX_LENGTH}"
    echo "Output file: ${OUTPUT_FILE}"
    echo "=========================================="
    
    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "Error: Checkpoint not found: ${CHECKPOINT}"
        echo "Please train the model first using: ./train.sh ${LANG_CODE} ${DATASET} traindev"
        exit 1
    fi
    
    # Check if test file exists
    if [ ! -f "${TEST_FILE}" ]; then
        echo "Error: Test file not found: ${TEST_FILE}"
        exit 1
    fi
    
    # Run prediction
    python3 predict.py \
        --test_file "${TEST_FILE}" \
        --checkpoint "${CHECKPOINT}" \
        --output_file "${OUTPUT_FILE}" \
        --model_name "${MODEL_NAME}" \
        --max_length ${MAX_LENGTH} \
        --batch_size ${PREDICT_BATCH_SIZE}
    
    echo ""
    echo "=========================================="
    echo "Prediction completed!"
    echo "Output file: ${OUTPUT_FILE}"
    echo "=========================================="
    exit 0
fi

# Special handling for eval_20 mode (80% train, 20% eval)
# Two modes are supported:
#   1. Standard mode (100% training): ./predict.sh eng laptop dev_task1
#      - Uses DATA_DIR (original full dataset)
#      - Uses checkpoints/{lang}_{dataset}/best_model.pt
#   2. eval_20 mode (80% training): ./predict.sh eng laptop eval_20
#      - Uses SPLIT_DATA_DIR (80/20 split dataset)
#      - Uses checkpoints/{lang}_{dataset}_80/best_model.pt
if [ "${TEST_SUFFIX}" == "eval_20" ]; then
    # Determine task suffix based on dataset
    if [ "${DATASET}" == "finance" ]; then
        TASK_TYPE="train_task1"
    else
        TASK_TYPE="train_alltasks"
    fi
    
    # Use SPLIT_DATA_DIR for 80/20 split data
    TEST_FILE="${SPLIT_DATA_DIR}/${LANG_CODE}/${LANG_CODE}_${DATASET}_${TASK_TYPE}_20_without_va.jsonl"
    GOLD_FILE="${SPLIT_DATA_DIR}/${LANG_CODE}/${LANG_CODE}_${DATASET}_${TASK_TYPE}_20_with_va.jsonl"
    CHECKPOINT="${CHECKPOINT_SPLIT}/${LANG_CODE}_${DATASET}/${LANG_CODE}_${DATASET}_80_best_model.pt"
    OUTPUT_FILE="${EVAL_20_DIR}/pred_${LANG_CODE}_${DATASET}_20_finetuning.jsonl"
    
    # Create eval_20 output directory
    mkdir -p "${EVAL_20_DIR}"
else
    # Standard prediction mode (dev set, using 100% train model)
    TEST_FILE="${DATA_DIR}/${LANG_CODE}/${LANG_CODE}_${DATASET}_${TEST_SUFFIX}.jsonl"
    CHECKPOINT="${CHECKPOINT_TRAIN}/${LANG_CODE}_${DATASET}/${LANG_CODE}_${DATASET}_best_model.pt"
    OUTPUT_FILE="${OUTPUT_DIR}/dev_prediction/${LANG_CODE}/pred_${LANG_CODE}_${DATASET}.jsonl"
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}/dev_prediction/${LANG_CODE}"
fi

echo "=========================================="
echo "Prediction Configuration"
echo "=========================================="
echo "Mode: ${TEST_SUFFIX}"
echo "Language: ${LANG_CODE}"
echo "Dataset: ${DATASET}"
echo "Test file: ${TEST_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Max length: ${MAX_LENGTH}"
echo "Output file: ${OUTPUT_FILE}"
if [ -n "${GOLD_FILE}" ]; then
    echo "Gold file: ${GOLD_FILE}"
fi
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT}"
    if [ "${TEST_SUFFIX}" == "eval_20" ]; then
        echo "Please train the model first using: ./train.sh ${LANG_CODE} ${DATASET} train_alltasks_80"
    else
        echo "Please train the model first using: ./train.sh ${LANG_CODE} ${DATASET}"
    fi
    exit 1
fi

# Check if test file exists
if [ ! -f "${TEST_FILE}" ]; then
    echo "Error: Test file not found: ${TEST_FILE}"
    exit 1
fi

# Build prediction command
PREDICT_CMD="python3 predict.py \
    --test_file \"${TEST_FILE}\" \
    --checkpoint \"${CHECKPOINT}\" \
    --output_file \"${OUTPUT_FILE}\" \
    --model_name \"${MODEL_NAME}\" \
    --max_length ${MAX_LENGTH} \
    --batch_size ${PREDICT_BATCH_SIZE}"

# Add gold file if provided
if [ -n "${GOLD_FILE}" ]; then
    PREDICT_CMD="${PREDICT_CMD} --gold_file \"${GOLD_FILE}\""
fi

# Run prediction
eval ${PREDICT_CMD}

echo ""
echo "=========================================="
echo "Prediction completed!"
echo "Output file: ${OUTPUT_FILE}"
echo "=========================================="
