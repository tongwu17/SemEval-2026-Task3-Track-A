"""
Prompt-based LLMusing OpenAI API
"""

import json
import jsonlines
import re
from tqdm import tqdm
import time
from openai import OpenAI
import os

def extract_va_scores(response):
    """Extract VA scores from LLM response"""
    patterns = [
        r'(\d+\.?\d*)\s*#\s*(\d+\.?\d*)',  # 7.50#6.80
        r'valence[:\s]+(\d+\.?\d*)[,\s]+arousal[:\s]+(\d+\.?\d*)',
        r'(\d+\.?\d*)[,\s]+(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            v, a = float(match.group(1)), float(match.group(2))
            v = max(1.0, min(9.0, v))
            a = max(1.0, min(9.0, a))
            return v, a
    
    print(f"Warning: Could not extract VA from: {response[:100]}")
    return 5.0, 5.0

def create_prompt_few_shot(text, aspect, examples=None):
    system_message = """You are an expert in sentiment analysis. Your task is to predict Valence and Arousal scores for aspects in sentences.

Definitions:
- Valence: emotional positivity/negativity (1.0 = very negative, 5.0 = neutral, 9.0 = very positive)
- Arousal: emotional intensity/excitement (1.0 = very calm/sluggish, 5.0 = moderate, 9.0 = very excited)

Output format: valence#arousal (e.g., 7.50#6.80)"""

    # Few-shot examples
    user_examples = """Examples:

1. Text: "The salads are fantastic."
   Aspect: "salads"
   Answer: 7.88#7.75

2. Text: "The service was terrible and slow."
   Aspect: "service"
   Answer: 2.10#7.50

3. Text: "It's okay, nothing special."
   Aspect: "general"
   Answer: 5.20#4.30

4. Text: "The battery life is amazing!"
   Aspect: "battery"
   Answer: 8.50#8.00

5. Text: "Keyboard feels cheap and flimsy."
   Aspect: "keyboard"
   Answer: 2.80#6.20"""

    user_query = f"""
Now predict:
Text: "{text}"
Aspect: "{aspect}"

Output ONLY the scores in format valence#arousal:"""

    return system_message, user_examples + user_query

def predict_va_with_openai(data_path, model='gpt-3.5-turbo', max_samples=None, api_key=None):
    # Initialize OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        # Read from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("[ERROR] No API key found!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            return None
        client = OpenAI(api_key=api_key)
    
    print(f"[INFO] Using OpenAI model: {model}")
    
    # Load data
    samples = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            text = obj['Text']
            id_val = obj['ID']
            
            if 'Aspect_VA' in obj:
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
    
    print(f"Total samples: {len(samples)}")
    
    # Predict
    results = []
    total_tokens = 0
    
    for sample in tqdm(samples, desc="Predicting"):
        system_msg, user_msg = create_prompt_few_shot(sample['text'], sample['aspect'])
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3,  # Reduce randomness
                max_tokens=50
            )
            
            answer = response.choices[0].message.content.strip()
            total_tokens += response.usage.total_tokens
            
            valence, arousal = extract_va_scores(answer)
            
        except Exception as e:
            print(f"\nError: {e}")
            valence, arousal = 5.0, 5.0
            time.sleep(1)
        
        results.append({
            'id': sample['id'],
            'text': sample['text'],
            'aspect': sample['aspect'],
            'pred_valence': valence,
            'pred_arousal': arousal,
            'gold_va': sample['va']
        })
        
        # API rate limiting
        time.sleep(0.05)
    
    print(f"\n[INFO] Total tokens used: {total_tokens}")
    
    return results

def evaluate_results(results):
    """Evaluate results"""
    from scipy.stats import pearsonr
    import math
    
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
    
    pcc_v = pearsonr(pred_v, gold_v)[0]
    pcc_a = pearsonr(pred_a, gold_a)[0]
    
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
    """Save prediction results"""
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for r in results:
        grouped[r['id']].append({
            'Aspect': r['aspect'],
            'VA': f"{r['pred_valence']:.2f}#{r['pred_arousal']:.2f}"
        })
    
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
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL file (only for dev set)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', 
                       choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini'],
                       help='OpenAI model')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for testing')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--method_suffix', type=str, default=None, help='Method name suffix for output file')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation mode: only compute RMSE, do not save predictions (for train set)')
    args = parser.parse_args()
    
    # Detect if train or dev dataset
    input_basename = os.path.basename(args.input)
    is_train = 'train' in input_basename
    
    # Auto-set eval_only (if train dataset)
    if is_train and not args.eval_only:
        args.eval_only = True
        print(f"[INFO] Detected train dataset, auto-enabled --eval_only mode")
    
    # Auto-determine method suffix
    if args.method_suffix is None:
        args.method_suffix = args.model.replace('-', '_')  # gpt-3.5-turbo -> gpt_3_5_turbo
    
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
    
    # Generate log filename (if needed)
    log_file = None
    if hasattr(args, 'log_file') and args.log_file:
        log_file = args.log_file
    else:
        # Auto-generate log filename
        parts = input_basename.replace('.jsonl', '').split('_')
        if len(parts) >= 2:
            lang = parts[0]
            domain = parts[1]
            dataset_type = 'train' if is_train else 'dev'
            sample_suffix = f"_{args.max_samples}" if args.max_samples else "_full"
            model_name = args.method_suffix
            log_file = f"log_{model_name}_{lang}_{domain}_{dataset_type}{sample_suffix}.txt"
    
    print(f"Starting LLM {'evaluation' if args.eval_only else 'prediction'} with OpenAI...")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output if not args.eval_only else 'N/A (eval only)'}")
    print(f"Mode: {'Evaluation only (train set)' if args.eval_only else 'Prediction (dev set)'}")
    if log_file:
        print(f"Log file: {log_file}")
    
    # Predict
    results = predict_va_with_openai(
        args.input,
        model=args.model,
        max_samples=args.max_samples,
        api_key=args.api_key
    )
    
    if results:
        # Evaluate (if gold labels available)
        metrics = evaluate_results(results)
        if metrics:
            print("\nEvaluation Results:")
            print("=" * 50)
            print(f"  PCC_V (Valence Correlation):  {metrics['PCC_V']:.4f}")
            print(f"  PCC_A (Arousal Correlation):  {metrics['PCC_A']:.4f}")
            print(f"  RMSE_VA:                       {metrics['RMSE_VA']:.4f}")
            print(f"  Samples evaluated:             {metrics['n_samples']}")
            print("=" * 50)
        
        # Save predictions (only in non-eval_only mode)
        if not args.eval_only:
            save_predictions(results, args.output)
            print("\n[SUCCESS] Prediction completed!")
            print(f"[SUCCESS] Results saved to: {args.output}")
        else:
            print("\n[SUCCESS] Evaluation completed!")
