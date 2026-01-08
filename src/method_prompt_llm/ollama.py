"""
Prompt-based LLM for DimASR using Ollama
"""

import json
import jsonlines
import re
from tqdm import tqdm
import subprocess
import time

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def call_ollama(prompt, model='llama3.2'):
    """Call Ollama local model"""
    try:
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def extract_va_scores(response):
    """Extract VA scores from LLM response"""
    # Match formats: 7.50#6.80 or valence: 7.5, arousal: 6.8
    patterns = [
        r'(\d+\.?\d*)\s*#\s*(\d+\.?\d*)',  # 7.50#6.80
        r'valence[:\s]+(\d+\.?\d*)[,\s]+arousal[:\s]+(\d+\.?\d*)',  # valence: 7.5, arousal: 6.8
        r'(\d+\.?\d*)[,\s]+(\d+\.?\d*)',  # 7.5, 6.8
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            v, a = float(match.group(1)), float(match.group(2))
            # Limit to [1,9] range
            v = max(1.0, min(9.0, v))
            a = max(1.0, min(9.0, a))
            return v, a
    
    # Return neutral values by default
    print(f"Warning: Could not extract VA from: {response[:100]}")
    return 5.0, 5.0

def create_prompt_few_shot(text, aspect, examples=None):
    """Create Few-shot prompt"""
    
    base_prompt = """Task: Predict Valence and Arousal scores for an aspect in a sentence.

Definitions:
- Valence: emotional positivity/negativity (1.0 = very negative, 9.0 = very positive)
- Arousal: emotional intensity (1.0 = very calm, 9.0 = very excited)

Examples:"""

    # Add examples (if provided)
    if examples:
        for ex in examples:
            base_prompt += f"""
Text: "{ex['text']}"
Aspect: "{ex['aspect']}"
Answer: {ex['valence']:.2f}#{ex['arousal']:.2f}
"""
    else:
        # Default examples
        base_prompt += """
Text: "The salads are fantastic."
Aspect: "salads"
Answer: 7.88#7.75

Text: "The service was terrible and slow."
Aspect: "service"
Answer: 2.10#7.50

Text: "It's okay, nothing special."
Aspect: "general"
Answer: 5.20#4.30
"""

    base_prompt += f"""
Now predict for:
Text: "{text}"
Aspect: "{aspect}"

Output ONLY in format: valence#arousal (e.g., 7.50#6.80)
Answer: """

    return base_prompt

def predict_va_with_llm(data_path, model='llama3.2', use_few_shot=True, max_samples=None):
    """Predict VA scores using LLM"""
    
    # Check Ollama
    if not check_ollama():
        print("[ERROR] Ollama not installed!")
        print("Install with: brew install ollama")
        print("Then run: ollama pull llama3.2")
        return None
    
    print(f"[INFO] Using Ollama model: {model}")
    
    # Load data
    samples = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            text = obj['Text']
            id_val = obj['ID']
            
            # Extract aspects (support multiple formats)
            if 'Quadruplet' in obj:
                # train_alltasks format
                for quad in obj['Quadruplet']:
                    samples.append({
                        'id': id_val,
                        'text': text,
                        'aspect': quad['Aspect'] if quad['Aspect'] != 'NULL' else 'general',
                        'va': quad.get('VA')
                    })
            elif 'Aspect_VA' in obj:
                # task1 format (with labels)
                for aspect_va in obj['Aspect_VA']:
                    samples.append({
                        'id': id_val,
                        'text': text,
                        'aspect': aspect_va['Aspect'],
                        'va': aspect_va.get('VA')
                    })
            elif 'Aspect' in obj:
                # task1 format (no labels, test set)
                for aspect in obj['Aspect']:
                    samples.append({
                        'id': id_val,
                        'text': text,
                        'aspect': aspect,
                        'va': None
                    })
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Total samples: {len(samples)}")
    
    # Predict
    results = []
    for sample in tqdm(samples, desc="Predicting"):
        prompt = create_prompt_few_shot(sample['text'], sample['aspect'])
        
        response = call_ollama(prompt, model)
        if response:
            valence, arousal = extract_va_scores(response)
        else:
            valence, arousal = 5.0, 5.0  # Default values
        
        results.append({
            'id': sample['id'],
            'text': sample['text'],
            'aspect': sample['aspect'],
            'pred_valence': valence,
            'pred_arousal': arousal,
            'gold_va': sample['va']
        })
        
        # Avoid requests too fast
        time.sleep(0.1)
    
    return results

def evaluate_results(results):
    """Evaluate results (if gold labels available)"""
    from scipy.stats import pearsonr
    import numpy as np
    import math
    
    # Filter samples with gold labels
    valid_results = [r for r in results if r['gold_va'] is not None]
    
    if not valid_results:
        print("No gold labels available for evaluation")
        return None
    
    pred_v = [r['pred_valence'] for r in valid_results]
    pred_a = [r['pred_arousal'] for r in valid_results]
    
    gold_v = []
    gold_a = []
    for r in valid_results:
        v, a = r['gold_va'].split('#')
        gold_v.append(float(v))
        gold_a.append(float(a))
    
    # Calculate metrics
    pcc_v = pearsonr(pred_v, gold_v)[0]
    pcc_a = pearsonr(pred_a, gold_a)[0]
    
    # RMSE_VA
    gold_va = gold_v + gold_a
    pred_va = pred_v + pred_a
    rmse_va = math.sqrt(sum((a - b)**2 for a, b in zip(gold_va, pred_va)) / len(gold_v))
    
    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
        'n_samples': len(valid_results)
    }

def save_predictions(results, output_path):
    """Save prediction results as JSONL"""
    from collections import defaultdict
    
    # Group by ID
    grouped = defaultdict(list)
    for r in results:
        grouped[r['id']].append({
            'Aspect': r['aspect'],
            'VA': f"{r['pred_valence']:.2f}#{r['pred_arousal']:.2f}"
        })
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for id_val, aspect_va_list in grouped.items():
            record = {
                'ID': id_val,
                'Aspect_VA': aspect_va_list
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[INFO] Saved predictions to: {output_path}")

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL file (only for dev set)')
    parser.add_argument('--model', type=str, default='llama3.2', help='Ollama model name')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for testing')
    parser.add_argument('--log_file', type=str, default=None, help='Log file path')
    parser.add_argument('--method_suffix', type=str, default='ollama', help='Method name suffix for output file')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation mode: only compute RMSE, do not save predictions (for train set)')
    args = parser.parse_args()
    
    # Detect if train or dev dataset
    input_basename = os.path.basename(args.input)
    is_train = 'train' in input_basename
    
    # Auto-set eval_only (if train dataset)
    if is_train and not args.eval_only:
        args.eval_only = True
        print(f"[INFO] Detected train dataset, auto-enabled --eval_only mode")
    
    # Auto-generate output filename (only for dev dataset)
    if args.output is None and not args.eval_only:
        parts = input_basename.replace('.jsonl', '').split('_')
        if len(parts) >= 2:
            lang = parts[0]
            domain = parts[1]
            args.output = f"./outputs/pred_{lang}_{domain}_{args.method_suffix}.jsonl"
        else:
            args.output = f"./outputs/pred_output_{args.method_suffix}.jsonl"
        print(f"[INFO] Auto-generated output file: {args.output}")
    
    # Generate log filename
    if args.log_file is None:
        # Extract info from input filename: eng_laptop_train_alltasks.jsonl -> eng_laptop_train
        parts = input_basename.replace('.jsonl', '').split('_')
        if len(parts) >= 2:
            lang = parts[0]  # eng
            domain = parts[1]  # laptop
            dataset_type = 'train' if is_train else 'dev'
            sample_suffix = f"_{args.max_samples}" if args.max_samples else "_full"
            args.log_file = f"log_ollama_{lang}_{domain}_{dataset_type}{sample_suffix}.txt"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.log_file = f"log_ollama_{timestamp}.txt"
    
    # Create log file
    import sys
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(args.log_file)
    
    print("=" * 50)
    print(f"Prompt-based LLM {'Evaluation' if args.eval_only else 'Prediction'} Log")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output if not args.eval_only else 'N/A (eval only)'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Mode: {'Evaluation only (train set)' if args.eval_only else 'Prediction (dev set)'}")
    print(f"Log file: {args.log_file}")
    print("=" * 50)
    print()
    
    # Predict
    results = predict_va_with_llm(
        args.input,
        model=args.model,
        max_samples=args.max_samples
    )
    
    if results:
        # Evaluate (if gold labels available)
        metrics = evaluate_results(results)
        if metrics:
            print("\n" + "=" * 50)
            print("Evaluation Results:")
            print("=" * 50)
            print(f"  PCC_V (Valence Correlation):  {metrics['PCC_V']:.4f}")
            print(f"  PCC_A (Arousal Correlation):  {metrics['PCC_A']:.4f}")
            print(f"  RMSE_VA:                       {metrics['RMSE_VA']:.4f}")
            print(f"  Samples evaluated:             {metrics['n_samples']}")
            print("=" * 50)
        
        # Save predictions (only in non-eval_only mode)
        if not args.eval_only:
            save_predictions(results, args.output)
            print(f"\n[SUCCESS] Prediction completed!")
            print(f"[SUCCESS] Results saved to: {args.output}")
        else:
            print(f"\n[SUCCESS] Evaluation completed!")
        
        print(f"[SUCCESS] Log saved to: {args.log_file}")
        print("\n" + "=" * 50)
