"""
Prediction Script
Used to generate submission files
"""

import os
import json
import argparse
import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import sys

from model import create_model
from data_loader import format_va_string


def load_test_data(test_path):
    """
    Load test data
    
    Format:
    {
        "ID": "R001",
        "Text": "average to good thai food, but terrible delivery.",
        "Aspect": ["thai food", "delivery"]
    }
    """
    data = []
    with jsonlines.open(test_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def predict(model, tokenizer, test_data, device, max_length=256, batch_size=32):
    """
    Batch prediction
    
    Returns:
        predictions: List[Dict] - Format similar to input, but with added Aspect_VA field
    """
    model.eval()
    predictions = []
    
    # Build input for each aspect of each sample
    samples = []
    sample_info = []  # Store (ID, aspect) information
    
    for item in test_data:
        text = item['Text']
        for aspect in item['Aspect']:
            # Tokenize
            encoding = tokenizer(
                text,
                aspect,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            samples.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
            sample_info.append((item['ID'], aspect, text))
    
    # Batch prediction
    all_valences = []
    all_arousals = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Predicting"):
            batch_samples = samples[i:i+batch_size]
            
            # Build batch
            input_ids = torch.stack([s['input_ids'] for s in batch_samples]).to(device)
            attention_mask = torch.stack([s['attention_mask'] for s in batch_samples]).to(device)
            
            # Predict
            valence, arousal = model(input_ids, attention_mask)
            
            all_valences.extend(valence.cpu().numpy().tolist())
            all_arousals.extend(arousal.cpu().numpy().tolist())
    
    # Organize prediction results
    # Group by ID
    id_to_predictions = {}
    for idx, (item_id, aspect, text) in enumerate(sample_info):
        valence = all_valences[idx]
        arousal = all_arousals[idx]
        va_str = format_va_string(valence, arousal)
        
        if item_id not in id_to_predictions:
            id_to_predictions[item_id] = {
                'ID': item_id,
                'Aspect_VA': []
            }
        
        id_to_predictions[item_id]['Aspect_VA'].append({
            'Aspect': aspect,
            'VA': va_str
        })
    
    # Convert to list (maintain original order)
    predictions = []
    for item in test_data:
        if item['ID'] in id_to_predictions:
            predictions.append(id_to_predictions[item['ID']])
    
    return predictions


def save_predictions(predictions, output_path):
    """
    Save prediction results in JSONL format
    
    Output format:
    {
        "ID": "R001",
        "Aspect_VA": [
            {"Aspect": "thai food", "VA": "6.75#6.38"},
            {"Aspect": "delivery", "VA": "2.88#6.62"}
        ]
    }
    """
    with jsonlines.open(output_path, mode='w') as writer:
        for pred in predictions:
            writer.write(pred)
    print(f"Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./outputs/predictions.jsonl',
                       help='Test file path')
    parser.add_argument('--checkpoint', type=str, default='./outputs/predictions.jsonl',
                       help='Model checkpoint path')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path (if not specified, auto-generate based on test_file: outputs/{lang}/pred_{lang}_{dataset}.jsonl)')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                       help='Pretrained model name (must match training)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gold_file', type=str, default=None,
                       help='Gold label file path (optional). If provided, will automatically run official evaluation script after prediction')
    
    args = parser.parse_args()
    
    # Auto-generate output file path based on test file if not specified
    if args.output_file is None:
        # Extract language code and dataset name from test_file
        # Example: eng_laptop_dev_task1.jsonl -> lang=eng, dataset=laptop
        test_basename = os.path.basename(args.test_file)
        parts = test_basename.split('_')
        if len(parts) >= 2:
            lang_code = parts[0]  # eng, zho, rus, etc.
            dataset_name = parts[1]  # laptop, restaurant, etc.
            args.output_file = f'./outputs/{lang_code}/pred_{lang_code}_{dataset_name}.jsonl'
            print(f"Auto-generated output path: {args.output_file}")
        else:
            args.output_file = './outputs/predictions.jsonl'
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = create_model(args.model_name, version='v1')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (Best RMSE_VA: {checkpoint.get('rmse_va', 'N/A')})")
    
    # Load test data
    print(f"Loading test data from: {args.test_file}")
    test_data = load_test_data(args.test_file)
    print(f"Test samples: {len(test_data)}")
    
    # Predict
    print("Starting prediction...")
    predictions = predict(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # Save results
    # Automatically create output directory (including language subfolders like eng/, zho/)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_predictions(predictions, args.output_file)
    
    print("\n" + "="*50)
    print("Prediction completed!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Output file: {args.output_file}")
    print("="*50)
    
    # If gold file provided, automatically run official evaluation
    if args.gold_file:
        print("\n" + "="*50)
        print("Running official evaluation...")
        print("="*50)
        
        # Directly call official evaluation script
        import subprocess
        eval_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'evaluation_script', 'metrics_subtask_1_2_3.py'
        )
        
        try:
            result = subprocess.run(
                [sys.executable, eval_script, 
                 '-t', '1',
                 '-p', args.output_file,
                 '-g', args.gold_file],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"\nEvaluation failed:")
            print(e.stdout)
            print(e.stderr)
        except FileNotFoundError:
            print(f"\nOfficial evaluation script not found: {eval_script}")
            print("Please run manually:")
            print(f"python ../evaluation_script/metrics_subtask_1_2_3.py -t 1 -p {args.output_file} -g {args.gold_file}")


if __name__ == "__main__":
    main()
