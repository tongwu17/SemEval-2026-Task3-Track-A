#!/bin/bash
python3 method_bulkchain.py  \
    --model_name meta/llama-4-maverick-instruct  \
    --provider_path providers/replicate_104.py  \
    --langs eng zho \
    --dataset_name eval_20 \
    --api_token <API_KEY>