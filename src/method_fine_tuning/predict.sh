#!/bin/bash
# Prediction script with configurable parameters
# Usage: ./predict.sh <language> <dataset> [test_suffix] [gold_file]
# Example: ./predict.sh eng restaurant dev_task1
#          ./predict.sh zho laptop test_task1 ../task-dataset/track_a/subtask_1/zho/zho_laptop_test_task1.jsonl

set -e  # Exit on error

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <language> <dataset> [test_suffix] [gold_file]"
    echo "Example: $0 eng restaurant dev_task1"
    echo "         $0 zho laptop test_task1 ../task-dataset/track_a/subtask_1/zho/zho_laptop_test_task1.jsonl"
    exit 1
fi

LANG_CODE=$1
DATASET=$2
TEST_SUFFIX=${3:-"dev_task1"}  # Default to dev_task1
GOLD_FILE=$4  # Optional gold file for evaluation

# Build file names
TEST_FILE="${DATA_DIR}/${LANG_CODE}/${LANG_CODE}_${DATASET}_${TEST_SUFFIX}.jsonl"
CHECKPOINT="${CHECKPOINT_DIR}/${LANG_CODE}_${DATASET}/best_model.pt"
OUTPUT_FILE="${OUTPUT_DIR}/${LANG_CODE}/pred_${LANG_CODE}_${DATASET}.jsonl"

echo "=========================================="
echo "Prediction Configuration"
echo "=========================================="
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
    echo "Please train the model first using: ./train.sh ${LANG_CODE} ${DATASET}"
    exit 1
fi

# Check if test file exists
if [ ! -f "${TEST_FILE}" ]; then
    echo "Error: Test file not found: ${TEST_FILE}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}/${LANG_CODE}"

# Build prediction command
PREDICT_CMD="python predict.py \
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
