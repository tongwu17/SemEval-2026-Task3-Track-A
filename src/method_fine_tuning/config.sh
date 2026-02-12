#!/bin/bash
# Shared configuration for training and inference
# This ensures consistency between train.py and predict.py

# ===== Model Configuration =====
# Model name: can switch to xlm-roberta-large for better performance
MODEL_NAME="xlm-roberta-base"
# MODEL_NAME="xlm-roberta-large"  # Uncomment to use large version

# Maximum sequence length
MAX_LENGTH=256

# ===== Training Configuration =====
BATCH_SIZE=16
NUM_EPOCHS=10
LEARNING_RATE=2e-5
DROPOUT=0.1
PATIENCE=3
VAL_SPLIT=0.1
SEED=42

# ===== Inference Configuration =====
PREDICT_BATCH_SIZE=32

# ===== Paths =====
# Data directory (relative to src/method_fine_tuning/)
# Option 1: Original full dataset (100% training)
DATA_DIR="../../task-dataset/track_a/subtask_1"
# Option 2: 80/20 split dataset (for eval_20 comparison)
# DATA_DIR="../task-dataset-split"

# Data directory for 80/20 split (used when training with _80 suffix)
SPLIT_DATA_DIR="../task-dataset-split"

# Checkpoint directory (relative to src/method_fine_tuning/)
CHECKPOINT_DIR="./checkpoints"
# Subdirectories for different training modes
CHECKPOINT_TRAIN="${CHECKPOINT_DIR}/train"           # 100% train data
CHECKPOINT_SPLIT="${CHECKPOINT_DIR}/train_split"     # 80% train data (for LLM comparison)
CHECKPOINT_TRAINDEV="${CHECKPOINT_DIR}/train_dev"    # train+dev merged (for test prediction)

# Output directory (relative to src/method_fine_tuning/)
OUTPUT_DIR="./outputs"

# Eval 20 output directory (for fine-tuning vs LLM comparison)
EVAL_20_DIR="./eval_20"
