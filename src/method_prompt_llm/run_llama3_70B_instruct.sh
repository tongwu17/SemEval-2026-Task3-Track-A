#!/bin/bash
python3 method_bulkchain.py  \
    --model_name meta/meta-llama-3-70b-instruct  \
    --provider_path providers/replicate_104.py  \
    --langs eng zho \
    --dataset_name eval_20 \
    --api_token <API_KEY>