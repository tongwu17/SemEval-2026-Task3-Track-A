#!/bin/bash
# Training script with configurable parameters
# Usage: ./train.sh <language> <dataset> [task_suffix]
# Example: ./train.sh eng restaurant train_alltasks
#          ./train.sh zho laptop train_alltasks
#          ./train.sh rus restaurant train_task1

set -e  # Exit on error

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <language> <dataset> [task_suffix]"
    echo "Example: $0 eng restaurant train_alltasks"
    echo "         $0 zho laptop train_task1"
    exit 1
fi

LANG_CODE=$1
DATASET=$2
TASK_SUFFIX=${3:-"train_alltasks"}  # Default to train_alltasks

# Build file names
TRAIN_FILE="${LANG_CODE}_${DATASET}_${TASK_SUFFIX}.jsonl"
OUTPUT_NAME="${LANG_CODE}_${DATASET}"
MODEL_OUTPUT_DIR="${CHECKPOINT_DIR}/${OUTPUT_NAME}"

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Language: ${LANG_CODE}"
echo "Dataset: ${DATASET}"
echo "Train file: ${TRAIN_FILE}"
echo "Model: ${MODEL_NAME}"
echo "Max length: ${MAX_LENGTH}"
echo "Output directory: ${MODEL_OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p "${MODEL_OUTPUT_DIR}"

# Run training
python train.py \
    --data_dir "${DATA_DIR}/${LANG_CODE}" \
    --train_file "${TRAIN_FILE}" \
    --model_name "${MODEL_NAME}" \
    --max_length ${MAX_LENGTH} \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --dropout ${DROPOUT} \
    --patience ${PATIENCE} \
    --val_split ${VAL_SPLIT} \
    --seed ${SEED}

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${MODEL_OUTPUT_DIR}/best_model.pt"
echo "=========================================="
