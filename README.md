# SemEval-2026-Task3-Track-A

## Publications
- Paper: [NCL-BU at SemEval-2026 Task 3: Fine-tuning XLM-RoBERTa for Multilingual Dimensional Sentiment Regression](https://aclanthology.org/2026.semeval-1.240.pdf)
  - ACL Anthology: https://aclanthology.org/2026.semeval-1.240/
  - DOI: [10.18653/v1/2026.semeval-1.240](https://doi.org/10.18653/v1/2026.semeval-1.240)
- Accepted at the Proceedings of the 20th International Workshop on Semantic Evaluation (2026)
- Publisher: Association for Computational Linguistics

## Citation
If you use this work, please cite:

```bibtex
@inproceedings{wu-etal-2026-ncl,
    title = "{NCL}-{BU} at {S}em{E}val-2026 Task 3: Fine-tuning {XLM}-{R}o{BERT}a for Multilingual Dimensional Sentiment Regression",
    author = "Wu, Tong and Rusnachenko, Nicolay and Liang, Huizhi(elly)",
    booktitle = "Proceedings of the 20th {I}nternational {W}orkshop on {S}emantic {E}valuation (2026)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.semeval-1.240/",
    doi = "10.18653/v1/2026.semeval-1.240",
    pages = "1911--1918",
    isbn = "979-8-89176-414-9"
}
```

## Project Overview
This project provides solutions for [SemEval 2026 Task 3](https://github.com/DimABSA/DimABSA2026) - Dimensional ABSA, Track A Subtask 1 (DimASR).

## Methodologies

### Main Method: Fine-tuning XLM-RoBERTa
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
│   │   ├── outputs/            
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

Models used: GPT-5.2, LLaMA-3-70B, LLaMA-4-Maverick.

```bash
cd src/method_prompt_llm
pip install -r requirements.txt

# OpenAI API
export OPENAI_API_KEY='your-api-key-here'
python3 method_openai.py \
  --input ../../task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl \
  --model gpt-5.2
```

