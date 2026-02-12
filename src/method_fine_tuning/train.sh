#!/bin/bash
# Training script with configurable parameters
# Usage: ./train.sh <language> <dataset> [task_suffix]
# Example: ./train.sh eng restaurant train_alltasks
#          ./train.sh zho laptop train_alltasks
#          ./train.sh rus restaurant train_task1
#          ./train.sh eng laptop traindev   # Merge train+dev for test prediction
#          ./train.sh all traindev          # Train all datasets with traindev mode

set -e  # Exit on error

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <language> <dataset> [task_suffix]"
    echo "Example: $0 eng restaurant train_alltasks"
    echo "         $0 zho laptop train_task1"
    echo "         $0 eng laptop traindev  # Merge train+dev for test prediction"
    echo "         $0 all traindev         # Train all datasets with traindev mode"
    exit 1
fi

# Special handling for "all" mode - train all datasets
if [ "$1" == "all" ]; then
    MODE=$2
    echo "=========================================="
    echo "Training ALL datasets with mode: ${MODE}"
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
        echo ">>> Training: ${lang}_${name} (${MODE} mode)"
        echo "------------------------------------------"
        "$0" "${lang}" "${name}" "${MODE}"
        echo ""
    done
    
    echo "=========================================="
    echo "All training completed!"
    echo "=========================================="
    exit 0
fi

LANG_CODE=$1
DATASET=$2
TASK_SUFFIX=${3:-"train_alltasks"}  # Default to train_alltasks

# Special handling for traindev mode (merge train+dev for test prediction)
if [ "${TASK_SUFFIX}" == "traindev" ]; then
    # Determine task type based on dataset
    if [ "${DATASET}" == "finance" ]; then
        TASK_TYPE="task1"
        TRAIN_FILE="${LANG_CODE}_${DATASET}_train_task1.jsonl"
    else
        TASK_TYPE="alltasks"
        TRAIN_FILE="${LANG_CODE}_${DATASET}_train_alltasks.jsonl"
    fi
    
    DEV_FILE="${LANG_CODE}_${DATASET}_dev_task1_with_va.jsonl"
    OUTPUT_NAME="${LANG_CODE}_${DATASET}"
    MODEL_OUTPUT_DIR="${CHECKPOINT_TRAINDEV}/${OUTPUT_NAME}"
    TRAIN_DATA_DIR="${DATA_DIR}/${LANG_CODE}"
    
    echo "=========================================="
    echo "Training Configuration (Train+Dev Merged)"
    echo "=========================================="
    echo "Language: ${LANG_CODE}"
    echo "Dataset: ${DATASET}"
    echo "Mode: traindev (merge train + dev for test prediction)"
    echo "Train file: ${TRAIN_FILE}"
    echo "Dev file: ${DEV_FILE}"
    echo "Data directory: ${TRAIN_DATA_DIR}"
    echo "Model: ${MODEL_NAME}"
    echo "Max length: ${MAX_LENGTH}"
    echo "Output directory: ${MODEL_OUTPUT_DIR}"
    echo "=========================================="
    
    # Create output directory
    mkdir -p "${MODEL_OUTPUT_DIR}"
    
    # Build log file name
    LOG_DIR="./logs/train_dev"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/training_log_${LANG_CODE}_${DATASET}_traindev.txt"
    echo "Log file: ${LOG_FILE}"
    
    # Run training with dev file
    python3 train.py \
        --data_dir "${TRAIN_DATA_DIR}" \
        --train_file "${TRAIN_FILE}" \
        --dev_file "${DEV_FILE}" \
        --model_name "${MODEL_NAME}" \
        --max_length ${MAX_LENGTH} \
        --output_dir "${MODEL_OUTPUT_DIR}" \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --dropout ${DROPOUT} \
        --patience ${PATIENCE} \
        --val_split ${VAL_SPLIT} \
        --seed ${SEED} \
        --log_file "${LOG_FILE}"
    
    echo ""
    echo "=========================================="
    echo "Training completed!"
    echo "Model saved to: ${MODEL_OUTPUT_DIR}/${OUTPUT_NAME}_best_model.pt"
    echo "=========================================="
    exit 0
fi

# Build file names
TRAIN_FILE="${LANG_CODE}_${DATASET}_${TASK_SUFFIX}.jsonl"
OUTPUT_NAME="${LANG_CODE}_${DATASET}"

# Determine checkpoint directory based on task suffix
if [[ "$TASK_SUFFIX" == *"_80"* ]]; then
    # 80% split training → train_split/
    MODEL_OUTPUT_DIR="${CHECKPOINT_SPLIT}/${OUTPUT_NAME}"
    TRAIN_DATA_DIR="${SPLIT_DATA_DIR}/${LANG_CODE}"
else
    # 100% train data → train/
    MODEL_OUTPUT_DIR="${CHECKPOINT_TRAIN}/${OUTPUT_NAME}"
    TRAIN_DATA_DIR="${DATA_DIR}/${LANG_CODE}"
fi

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Language: ${LANG_CODE}"
echo "Dataset: ${DATASET}"
echo "Task suffix: ${TASK_SUFFIX}"
echo "Train file: ${TRAIN_FILE}"
echo "Data directory: ${TRAIN_DATA_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Max length: ${MAX_LENGTH}"
echo "Output directory: ${MODEL_OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p "${MODEL_OUTPUT_DIR}"

# Build log file name and determine log directory
if [[ "$TASK_SUFFIX" == *"_80"* ]]; then
    LOG_DIR="./logs/train_split"
else
    LOG_DIR="./logs/train"
fi
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/training_log_${LANG_CODE}_${DATASET}_${TASK_SUFFIX}.txt"

echo "Log file: ${LOG_FILE}"

# Run training
# Note: TRAIN_DATA_DIR is determined automatically:
#   - For _80 suffix: uses SPLIT_DATA_DIR (../task-dataset-split)
#   - For others: uses DATA_DIR (../../task-dataset/track_a/subtask_1)
python3 train.py \
    --data_dir "${TRAIN_DATA_DIR}" \
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
    --seed ${SEED} \
    --log_file "${LOG_FILE}"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${MODEL_OUTPUT_DIR}/${OUTPUT_NAME}_best_model.pt"
echo "=========================================="
