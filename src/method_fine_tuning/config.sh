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
DATA_DIR="../../task-dataset/track_a/subtask_1"

# Checkpoint directory (relative to src/method_fine_tuning/)
CHECKPOINT_DIR="./checkpoints"

# Output directory (relative to src/method_fine_tuning/)
OUTPUT_DIR="./outputs"
