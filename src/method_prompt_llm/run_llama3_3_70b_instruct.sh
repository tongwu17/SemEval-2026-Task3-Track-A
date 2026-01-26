#!/bin/bash
python3 method_bulkchain.py  \
    --base_url https://api.novita.ai/openai \
    --model_name meta-llama/llama-3.3-70b-instruct  \
    --provider_path providers/openai_156.py  \
    --langs eng zho \
    --api_token API_KEY \
    --batch_size 20 \
    --dataset_name eval_20 \
    --sleep_time 61