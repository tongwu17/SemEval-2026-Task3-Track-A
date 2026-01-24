import json
import jsonlines
import re
import os
from tqdm import tqdm
import subprocess
import time

from utils_fewshot import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

# Change model here if needed
# MODEL_NAME = "llama3.2"
MODEL_NAME = "llama4"
# Model name for output files (e.g., llama4 -> llama_4)
MODEL_NAME_FOR_FILE = re.sub(r'(\D)(\d)', r'\1_\2', MODEL_NAME).replace('.', '_')
# External drive path for large models (set to None to use default ~/.ollama)
OLLAMA_MODELS_PATH = "/Volumes/Data/ollama_models"

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

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


def load_samples(data_path, max_samples=None):
    """Load samples from JSONL file"""
    samples = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            text = obj['Text']
            id_val = obj['ID']
            
            if 'Quadruplet' in obj:
                for quad in obj['Quadruplet']:
                    samples.append({
                        'id': id_val,
                        'text': text,
                        'aspect': quad['Aspect'] if quad['Aspect'] != 'NULL' else 'general',
                        'va': quad.get('VA')
                    })
            elif 'Aspect_VA' in obj:
                for aspect_va in obj['Aspect_VA']:
                    samples.append({
                        'id': id_val,
                        'text': text,
                        'aspect': aspect_va['Aspect'],
                        'va': aspect_va.get('VA')
                    })
            elif 'Aspect' in obj:
                for aspect in obj['Aspect']:
                    samples.append({
                        'id': id_val,
                        'text': text,
                        'aspect': aspect,
                        'va': None
                    })
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"[INFO] Loaded {len(samples)} samples from {data_path}")
    return samples

def predict_va_with_llm(samples, model=MODEL_NAME, max_retries=3):
    """Predict VA scores using Ollama with few-shot learning (includes retry)."""
    
    if not check_ollama():
        print("[ERROR] Ollama not installed!")
        print("Install with: brew install ollama")
        print("Then run: ollama pull llama3.2")
        return None
    
    if model is None:
        model = MODEL_NAME

    print(f"[INFO] Using Ollama model: {model}")
    print(f"[INFO] Total samples to predict: {len(samples)}")
    
    # Prepare env for Ollama calls
    env = os.environ.copy()
    if OLLAMA_MODELS_PATH:
        env['OLLAMA_MODELS'] = OLLAMA_MODELS_PATH

    # Predict
    results = []
    for sample in tqdm(samples, desc="Predicting"):
        user_prompt = f"""{FEW_SHOT_EXAMPLES}

Now predict:
Text: "{sample['text']}"
Aspect: "{sample['aspect']}"

Output ONLY the scores in format valence#arousal:"""

        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        
        response_text = ""
        last_error = None

        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ['ollama', 'run', model],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env
                )
                response_text = result.stdout.strip()
                if response_text:
                    break
            except subprocess.TimeoutExpired:
                last_error = "Timeout"
            except Exception as e:
                last_error = e

            if attempt < max_retries - 1:
                print(f"\n[WARN] Ollama call failed, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(1)

        if response_text:
            valence, arousal = extract_va_scores(response_text)
        else:
            if last_error:
                print(f"\nError: {last_error}")
            valence, arousal = 5.0, 5.0
        
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
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='Ollama model name')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for testing')
    parser.add_argument('--log_file', type=str, default=None, help='Log file path')
    parser.add_argument('--method_suffix', type=str, default=MODEL_NAME_FOR_FILE, help='Method name suffix for output file')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation mode: only compute RMSE, do not save predictions (for train set)')
    args = parser.parse_args()
    
    # Detect dataset type: eval_20 (20% validation), train (80% full), or dev/test
    input_basename = os.path.basename(args.input)
    is_eval_20 = '_20_without_va' in input_basename  # 20% validation set for method comparison
    is_train = 'train' in input_basename and not is_eval_20
    
    # Auto-set eval_only for 80% train dataset (not for 20% eval)
    if is_train and not args.eval_only:
        args.eval_only = True
        print(f"[INFO] Detected 80% train dataset, auto-enabled --eval_only mode")
    
    # Auto-generate output filename based on dataset type
    if args.output is None:
        parts = input_basename.replace('.jsonl', '').replace('_20_without_va', '').replace('_alltasks', '').replace('_task1', '').split('_')
        if len(parts) >= 2:
            lang = parts[0]
            domain = parts[1]
            if is_eval_20:
                # 20% validation set → save to eval_20/
                args.output = f"./eval_20/pred_{lang}_{domain}_20_{args.method_suffix}.jsonl"
            elif not args.eval_only:
                # dev/test set → save to outputs/
                args.output = f"./outputs/pred_{lang}_{domain}_{args.method_suffix}.jsonl"
        else:
            if is_eval_20:
                args.output = f"./eval_20/pred_output_20_{args.method_suffix}.jsonl"
            elif not args.eval_only:
                args.output = f"./outputs/pred_output_{args.method_suffix}.jsonl"
        if args.output:
            print(f"[INFO] Auto-generated output file: {args.output}")
    
    # Generate log filename (only for train set)
    if is_train:
        if args.log_file is None:
            parts = input_basename.replace('.jsonl', '').split('_')
            if len(parts) >= 2:
                lang = parts[0]  # eng
                domain = parts[1]  # laptop
                sample_suffix = f"_{args.max_samples}" if args.max_samples else "_full"
                args.log_file = f"log_ollama_{lang}_{domain}_train{sample_suffix}.txt"
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                args.log_file = f"log_ollama_{timestamp}.txt"
        
        # Create log file (only for train)
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
    if is_eval_20:
        print("Prompt-based LLM Evaluation on 20% Validation Set")
    elif args.eval_only:
        print("Prompt-based LLM Evaluation (80% train set)")
    else:
        print("Prompt-based LLM Prediction (dev/test set)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output if args.output else 'N/A (eval only)'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    if is_eval_20:
        print(f"Mode: 20% validation (for method comparison)")
    elif args.eval_only:
        print(f"Mode: Evaluation only (80% train set)")
    else:
        print(f"Mode: Prediction (dev/test set)")
    if is_train and args.log_file:
        print(f"Log file: {args.log_file}")
    print("=" * 50)
    print()
    
    # Load data
    # TODO. Use load_samples_flat from utils_fewshot.py
    samples = load_samples(args.input, max_samples=args.max_samples)

    # Predict
    results = predict_va_with_llm(
        samples,
        model=args.model,
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
        
        # Save predictions (for eval_20 and dev/test, not for 80% train)
        if is_eval_20 or not args.eval_only:
            save_predictions(results, args.output)
            print(f"\n[SUCCESS] Prediction completed!")
            print(f"[SUCCESS] Results saved to: {args.output}")
        else:
            print(f"\n[SUCCESS] Evaluation completed!")
        
        if is_train and args.log_file:
            print(f"[SUCCESS] Log saved to: {args.log_file}")
        print("\n" + "=" * 50)
