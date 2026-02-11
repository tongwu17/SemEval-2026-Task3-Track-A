import json
import jsonlines
import re
from tqdm import tqdm
import time
from openai import OpenAI
import os
from utils_fewshot import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

# Change model here if needed
MODEL_NAME = "gpt-5.2"

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

def load_samples(data_path, max_samples=None):
    """
    Load samples from JSONL file
    
    Args:
        data_path: Path to input JSONL file
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of sample dicts with keys: id, text, aspect, va
    """
    samples = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            text = obj['Text']
            id_val = obj['ID']
            
            if 'Quadruplet' in obj:
                # Format from split dataset (with or without VA)
                for quad in obj['Quadruplet']:
                    samples.append({
                        'id': id_val,
                        'text': text,
                        'aspect': quad['Aspect'],
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

def predict_va_with_openai(samples, model=MODEL_NAME, api_key=None, max_retries=3):
    """
    Predict VA scores using OpenAI API with few-shot learning
    
    Args:
        samples: List of sample dicts (from load_samples)
        model: OpenAI model name
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        max_retries: Max retries per sample when API call fails
    
    Returns:
        List of prediction results with VA scores
    """
    # Initialize OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("[ERROR] No API key found!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            return None
        client = OpenAI(api_key=api_key)
    
    if model is None:
        model = MODEL_NAME

    print(f"[INFO] Using OpenAI model: {model}")
    print(f"[INFO] Total samples to predict: {len(samples)}")
    
    # Predict
    results = []
    total_tokens = 0
    use_responses_api = model.startswith("gpt-5")
    
    for sample in tqdm(samples, desc="Predicting"):
        user_prompt = f"""{FEW_SHOT_EXAMPLES}

Now predict:
Text: "{sample['text']}"
Aspect: "{sample['aspect']}"

Output ONLY the scores in format valence#arousal:"""
        
        answer = ""
        last_error = None

        for attempt in range(max_retries):
            try:
                if use_responses_api:
                    # Responses API (for gpt-5.x)
                    response = client.responses.create(
                        model=model,
                        input=[
                            {
                                "role": "system",
                                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": user_prompt}],
                            },
                        ],
                        temperature=0.1,
                        max_output_tokens=50,
                    )

                    # Extract text
                    if hasattr(response, "output_text") and response.output_text:
                        answer = response.output_text.strip()
                    elif getattr(response, "output", None):
                        for item in response.output:
                            content = getattr(item, "content", None)
                            if not content:
                                continue
                            for part in content:
                                text_val = getattr(part, "text", None)
                                if text_val:
                                    answer = text_val.strip()
                                    break
                            if answer:
                                break

                    if answer and hasattr(response, "usage"):
                        usage = response.usage
                        tokens = getattr(usage, "total_tokens", None)
                        if tokens is None:
                            tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
                        total_tokens += tokens
                else:
                    # Chat Completions API (gpt-4 family)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=50
                    )
                    
                    answer = response.choices[0].message.content.strip()
                    if answer:
                        total_tokens += response.usage.total_tokens
                
                if answer:
                    break
            except Exception as e:
                last_error = e

            if attempt < max_retries - 1:
                print(f"\n[WARN] OpenAI call failed, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(1)

        if answer:
            valence, arousal = extract_va_scores(answer)
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
    parser.add_argument('--model', type=str, default=MODEL_NAME, 
                       choices=['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-5.2'],
                       help='OpenAI model')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for testing')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--method_suffix', type=str, default=None, help='Method name suffix for output file')
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
    
    # Auto-determine method suffix
    if args.method_suffix is None:
        args.method_suffix = args.model.replace('-', '_')  
    
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
            dataset_type = 'train' if is_train else ('eval_20' if is_eval_20 else 'dev')
            sample_suffix = f"_{args.max_samples}" if args.max_samples else "_full"
            model_name = args.method_suffix
            log_file = f"log_{model_name}_{lang}_{domain}_{dataset_type}{sample_suffix}.txt"
    
    print("=" * 50)
    if is_eval_20:
        print("OpenAI LLM Evaluation on 20% Validation Set")
    elif args.eval_only:
        print("OpenAI LLM Evaluation (80% train set)")
    else:
        print("OpenAI LLM Prediction (dev/test set)")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output if args.output else 'N/A (eval only)'}")
    if is_eval_20:
        print(f"Mode: 20% validation (for method comparison)")
    elif args.eval_only:
        print(f"Mode: Evaluation only (80% train set)")
    else:
        print(f"Mode: Prediction (dev/test set)")
    if log_file:
        print(f"Log file: {log_file}")
    print("=" * 50)
    
    # Load data
    samples = load_samples(args.input, max_samples=args.max_samples)
    
    # Predict
    results = predict_va_with_openai(
        samples,
        model=args.model,
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
        
        # Save predictions (for eval_20 and dev/test, not for 80% train)
        if is_eval_20 or not args.eval_only:
            save_predictions(results, args.output)
            print("\n[SUCCESS] Prediction completed!")
            print(f"[SUCCESS] Results saved to: {args.output}")
        else:
            print("\n[SUCCESS] Evaluation completed!")
