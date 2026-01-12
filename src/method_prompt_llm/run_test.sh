#!/bin/bash
# LLM Prediction Script
# Usage: ./run_test.sh [method] [lang] [domain] [dataset] [samples]
# Examples:
#   ./run_test.sh ollama eng laptop train 50    # Test on 50 train samples
#   ./run_test.sh gpt-4o eng laptop dev          # Full dev set prediction
#   ./run_test.sh gpt-4o-mini zho restaurant dev 100  # 100 dev samples

set -e

# Parse arguments with defaults
METHOD=${1:-"ollama"}
LANG=${2:-"eng"}
DOMAIN=${3:-"laptop"}
DATASET=${4:-"train"}
MAX_SAMPLES=${5:-""}

# Determine file suffix
if [ "$DATASET" = "train" ]; then
    FILE_SUFFIX="train_alltasks"
else
    FILE_SUFFIX="dev_task1"
fi

INPUT_FILE="../../task-dataset/track_a/subtask_1/${LANG}/${LANG}_${DOMAIN}_${FILE_SUFFIX}.jsonl"

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "[ERROR] Input file not found: $INPUT_FILE"
    exit 1
fi

echo "LLM Prediction"
echo "=================================="
echo "Method:   $METHOD"
echo "Language: $LANG"
echo "Domain:   $DOMAIN"
echo "Dataset:  $DATASET"
echo "Samples:  ${MAX_SAMPLES:-All}"
echo "Input:    $INPUT_FILE"
echo ""

# Build command based on method
if [[ "$METHOD" == "ollama" ]] || [[ "$METHOD" == "llama3.2" ]]; then
    # Check Ollama
    if ! command -v ollama &> /dev/null; then
        echo "[ERROR] Ollama not installed. Install with: brew install ollama"
        exit 1
    fi
    
    CMD="python3 ollama.py --input $INPUT_FILE --model llama3.2"
    
elif [[ "$METHOD" == gpt* ]]; then
    # Check API key
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "[ERROR] OPENAI_API_KEY not set"
        echo "Set it with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    
    CMD="python3 method_openai.py --input $INPUT_FILE --model $METHOD"
    
else
    echo "[ERROR] Unknown method: $METHOD"
    echo "Supported: ollama, llama3.2, gpt-4, gpt-4o, gpt-4o-mini, gpt-5.2"
    exit 1
fi

# Add max_samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Run prediction
echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "[SUCCESS] Prediction complete!"
echo "Check outputs/ directory for results"
