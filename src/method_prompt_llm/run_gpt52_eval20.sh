#!/bin/bash
# Run GPT-5.2 on 20% evaluation set
# Usage: bash run_gpt52_eval20.sh
# Requires: export OPENAI_API_KEY='sk-...'

python method_openai.py \
    --input ../task-dataset-split/eng/eng_laptop_train_alltasks_20_without_va.jsonl \
    --model gpt-5.2

python method_openai.py \
    --input ../task-dataset-split/eng/eng_restaurant_train_alltasks_20_without_va.jsonl \
    --model gpt-5.2

python method_openai.py \
    --input ../task-dataset-split/zho/zho_laptop_train_alltasks_20_without_va.jsonl \
    --model gpt-5.2

python method_openai.py \
    --input ../task-dataset-split/zho/zho_restaurant_train_alltasks_20_without_va.jsonl \
    --model gpt-5.2

python method_openai.py \
    --input ../task-dataset-split/zho/zho_finance_train_task1_20_without_va.jsonl \
    --model gpt-5.2
