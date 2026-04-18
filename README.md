# SemEval-2026-Task3-Track-A
[![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://x.com/nicolayr_/status/2045478369488175317)


## Project Overview
This project provides solutions for [SemEval 2026 Task 3](https://github.com/DimABSA/DimABSA2026) - Dimensional ABSA, Track A Subtask 1 (DimASR).

## Methodologies

### Main Method: Fine-tuning
Fine-tuning approach based on pretrained language model (XLM-RoBERTa).

### Prompt-based LLM
Large Language Models with prompting for comparison.

## Project Structure
```
├── evaluation_script/          # Evaluation metrics
├── task-dataset/               # Task dataset
│   └── track_a/subtask_1/     
├── src/
│   ├── method_fine_tuning/     # Fine-tuning method
│   │   ├── config.sh           # Shared configuration
│   │   ├── train.sh            # Training shell script
│   │   ├── train.py            
│   │   ├── predict.sh          # Prediction shell script
│   │   ├── predict.py          
│   │   ├── model.py            # Model definition
│   │   ├── data_loader.py      # Data loading
│   │   ├── outputs/            # Full predictions (dev / test)
│   │   └── requirements.txt
│   ├── method_prompt_llm/      # Prompt-based LLM (for comparison)
│   │   ├── method_openai.py    # OpenAI API implementation
│   │   ├── method_bulkchain.py # Bulk chain LLM implementation
│   │   ├── utils_data.py       # Data utilities
│   │   ├── utils_fewshot.py    # Few-shot utilities
│   │   ├── providers/          # LLM provider adapters
│   │   ├── run_*.sh            # Run scripts (GPT-5.2, LLaMA variants)
│   │   └── requirements.txt
│   ├── compare_methods.py      # Compare fine-tuning vs LLM results
│   └── task-dataset-split/     # Split training data
└── README.md
```

## Quick Start

### Fine-tuning

```bash
cd src/method_fine_tuning
pip install -r requirements.txt

# Train
./train.sh <language> <dataset> [task_suffix]
# e.g. ./train.sh eng restaurant train_alltasks

# Predict
./predict.sh <language> <dataset> [test_suffix] [gold_file]
# e.g. ./predict.sh eng restaurant dev_task1
```

### Prompt-based LLM 

Models used: GPT-5.2, LLaMA-3-70B, LLaMA-3.3-70B, LLaMA-4-Maverick.

```bash
cd src/method_prompt_llm
pip install -r requirements.txt

# OpenAI API
export OPENAI_API_KEY='your-api-key-here'
python3 method_openai.py \
  --input ../../task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl \
  --model gpt-5.2

# Bulk Chain (LLaMA)
./run_llama3_70B_instruct.sh
```

