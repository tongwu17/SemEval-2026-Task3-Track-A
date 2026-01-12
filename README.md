# SemEval-2026-Task3-Track-A

## Project Overview
This project provides solutions for SemEval 2026 Task - Dimensional ABSA, Track A Subtask 1 (DimASR).

## Methodologies

This project implements two complementary approaches for Valence-Arousal prediction:

### Method 1: Fine-tuning 
Fine-tuning approach based on pretrained language model (XLM-RoBERTa).

### Method 2: Prompt-based LLM 
Large Language Models with prompting - no training required.

## Project Structure
```
src/
├── method_fine_tuning/     # Fine-tuning method implementation
│   ├── config.sh           # Shared configuration
│   ├── train.sh            # Training shell script 
│   ├── predict.sh          # Prediction shell script 
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script
│   ├── model.py            # Model definition
│   ├── data_loader.py      # Data loading
│   └── requirements.txt    # Dependencies
├── method_prompt_llm/      # Prompt-based LLM method
│   ├── method_openai.py    # OpenAI API implementation
│   ├── ollama.py           # Ollama local implementation
│   ├── run_test.sh         # Simplified test script
│   └── README.md           # Method documentation
└── task-dataset-split/     # Split training data 
```

---

# Method 1: Fine-tuning Approach

## Model Architecture
```
Input: [CLS] Text [SEP] Aspect [SEP]
  ↓
XLM-RoBERTa Encoder
  ↓
[CLS] Representation
  ↓
├─ Valence Head (Linear + Sigmoid*8 + 1)
└─ Arousal Head (Linear + Sigmoid*8 + 1)
```

## Quick Start

### 1. Install Dependencies
```bash
cd src/method_fine_tuning
pip install -r requirements.txt
```

### 2. Configure Model Settings (Optional)
Edit [config.sh](src/method_fine_tuning/config.sh) to change model or parameters:
```bash
# Switch to large model for better performance
MODEL_NAME="xlm-roberta-large"  # Change from xlm-roberta-base

# Adjust max sequence length
MAX_LENGTH=512  # Increase if needed
```

### 3. Train Model

**Recommended: Use Shell Script (Ensures Consistency)**
```bash
cd src/method_fine_tuning
./train.sh <language> <dataset> [task_suffix]

# Examples:
./train.sh eng restaurant train_alltasks    
./train.sh zho laptop train_alltasks        
./train.sh zho finance train_task1          
```

**Alternative: Direct Python Command**
```bash
python train.py \
    --data_dir ../task-dataset/track_a/subtask_1/<LANG> \
    --train_file <LANG>_<DOMAIN>_train_<task_type>.jsonl \
    --model_name xlm-roberta-base \
    --max_length 256 \
    --output_dir ./checkpoints/<LANG>_<DOMAIN> \
    --batch_size 16 \
    --num_epochs 10 \
    --val_split 0.1
```

### 4. Generate Predictions

**Recommended: Use Shell Script (Auto-matches Training Config)**
```bash
cd src/method_fine_tuning
./predict.sh <language> <dataset> [test_suffix] [gold_file]

# Examples:
./predict.sh eng restaurant dev_task1                           # Predict only
./predict.sh zho laptop test_task1 <path_to_gold_file>         # Predict + Evaluate
```

**Alternative: Direct Python Command**
```bash
python predict.py \
    --checkpoint ./checkpoints/<LANG>_<DOMAIN>/best_model.pt \
    --test_file ../task-dataset/track_a/subtask_1/<LANG>/<LANG>_<DOMAIN>_dev_task1.jsonl \
    --model_name xlm-roberta-base \
    --max_length 256 \
    --output_file ./outputs/<LANG>/pred_<LANG>_<DOMAIN>.jsonl
```

## Experimental Settings
- **Pretrained Model**: xlm-roberta-base
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Training Epochs**: 10
- **Optimizer**: AdamW
- **Loss Function**: MSE Loss

**Note**: Val RMSE_VA is the result on validation set during training, Codabench RMSE is the official score after submission.

---

# Method 2: Prompt-based LLM Approach

## Overview

Use Large Language Models (LLM) with prompting for Valence-Arousal prediction, no training required.

### Two Implementations

- **Ollama**: Local LLM 
- **OpenAI API**: Cloud LLM 

## Quick Start

### Option 1: Simplified Script (Recommended)

```bash
cd src/method_prompt_llm

# Test with Ollama (free, local)
./run_test.sh ollama eng laptop dev

# Test with GPT-4o-mini (cheap)
./run_test.sh gpt-4o-mini eng laptop dev 50

# Full prediction with GPT-4o
./run_test.sh gpt-4o eng laptop dev

# Other examples
./run_test.sh gpt-4o zho restaurant dev       # Chinese restaurant
./run_test.sh ollama jpn hotel train 100      # Japanese hotel, 100 samples
```

### Option 2: Direct Python Command

#### Method 1: Ollama Local Test

```bash
# 1. Install Ollama
brew install ollama
ollama pull llama3.2

# 2. Run prediction (output filename auto-generated)
cd src/method_prompt_llm

# Simplified version (auto-generate output filename)
python3 ollama.py \
  --input ../../task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl
# Output: ./outputs/pred_eng_laptop_ollama.jsonl

# Full version (manual specification)
python3 ollama.py \
  --input ../../task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl \
  --output ./outputs/pred_eng_laptop_ollama.jsonl \
  --model llama3.2 \
  --log_file ./log_ollama_eng_laptop.txt
```

#### Method 2: OpenAI API

```bash
# 1. Set API Key
export OPENAI_API_KEY='sk-your-key-here'

# 2. Run prediction (auto-add method suffix)
python3 method_openai.py \
  --input ../../task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl \
  --model gpt-4o-mini
# Output: ./outputs/pred_eng_laptop_gpt_4o_mini.jsonl

# Use GPT-4o
python3 method_openai.py \
  --input ../../task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl \
  --model gpt-4o
# Output: ./outputs/pred_eng_laptop_gpt_4o.jsonl
```

### Custom Method Name

```bash
# Custom method suffix
python3 ollama.py \
  --input ../../task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl \
  --method_suffix my_custom_method
# Output: ./outputs/pred_eng_laptop_my_custom_method.jsonl
```

## Output Files

All prediction results saved in `./outputs/` directory:
- `pred_eng_laptop_ollama.jsonl` - Ollama predictions
- `pred_eng_laptop_gpt_4o.jsonl` - GPT-4o predictions
- `pred_eng_laptop_gpt_4o_mini.jsonl` - GPT-4o-mini predictions

