#!/usr/bin/env python3
"""
Compare Fine-tuning vs LLM-based Methods

This script compares RMSE scores between fine-tuning and LLM-based methods
for different language-domain combinations on the 20% evaluation set.

Usage:
    python src/compare_methods.py -t 1  # for subtask 1
    python src/compare_methods.py -t 1 -o results  # save results to results/
"""

import os
import sys
import subprocess
import re
import json
import tempfile
import argparse
import pandas as pd
from pathlib import Path


def _adapt_gold_to_official_format(gold_file):
    """Convert a gold file to the Aspect_VA format expected by the official
    evaluator and write it to a temp file. Returns the temp file path.

    Some splits store annotations under 'Quadruplet' (with 'Aspect' and 'VA');
    the official script only reads 'Aspect_VA'. This adapter is purely a
    field rename so the official metric computation is unchanged.
    """
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    try:
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if 'Aspect_VA' not in d:
                    if 'Quadruplet' in d:
                        d['Aspect_VA'] = [
                            {'Aspect': q['Aspect'], 'VA': q['VA']}
                            for q in d['Quadruplet']
                            if 'Aspect' in q and 'VA' in q
                        ]
                    else:
                        raise RuntimeError(
                            f"Gold file {gold_file} has neither 'Aspect_VA' nor 'Quadruplet' fields."
                        )
                tmp.write(json.dumps(d, ensure_ascii=False) + '\n')
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise


def run_evaluation(pred_file, gold_file, task=1):
    """
    Evaluate prediction file against gold file using official metrics via subprocess
    
    Returns:
        dict: {'overall_rmse': float, 'pcc_v': float, 'pcc_a': float} for task 1
              or {'cF1': float, 'cPrecision': float, 'cRecall': float} for task 2/3
    """
    # Get path to official evaluation script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, '..', 'evaluation_script', 'metrics_subtask_1_2_3.py')

    if not os.path.exists(eval_script):
        raise FileNotFoundError(f"Official evaluation script not found: {eval_script}")

    # Adapt gold file format to what the official evaluator expects
    adapted_gold = _adapt_gold_to_official_format(gold_file)

    # Call official script via subprocess
    cmd = [
        sys.executable, eval_script,
        '-p', pred_file,
        '-g', adapted_gold,
        '-t', str(task)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        try:
            os.unlink(adapted_gold)
        except OSError:
            pass
    output = (result.stdout or '') + (result.stderr or '')

    if result.returncode != 0:
        raise RuntimeError(
            f"Official evaluation failed for {pred_file} vs {gold_file} (task={task}).\n"
            f"Exit code: {result.returncode}\n"
            f"Output:\n{output}"
        )

    # Parse output for task 1
    if task == 1:
        rmse_match = re.search(r"'RMSE_VA':\s*([\d.]+)", output)
        pcc_v_match = re.search(r"'PCC_V':\s*([\d.-]+)", output)
        pcc_a_match = re.search(r"'PCC_A':\s*([\d.-]+)", output)

        if not rmse_match:
            raise RuntimeError(
                f"Official evaluation output does not contain RMSE_VA for {pred_file}.\n"
                f"Output:\n{output}"
            )

        return {
            'overall_rmse': float(rmse_match.group(1)),
            'pcc_v': float(pcc_v_match.group(1)) if pcc_v_match else None,
            'pcc_a': float(pcc_a_match.group(1)) if pcc_a_match else None,
        }

    # Parse output for task 2/3
    cf1_match = re.search(r"'cF1':\s*([\d.]+)", output)
    cprec_match = re.search(r"'cPrecision':\s*([\d.]+)", output)
    crec_match = re.search(r"'cRecall':\s*([\d.]+)", output)

    if not cf1_match:
        raise RuntimeError(
            f"Official evaluation output does not contain cF1 for {pred_file}.\n"
            f"Output:\n{output}"
        )

    return {
        'cF1': float(cf1_match.group(1)),
        'cPrecision': float(cprec_match.group(1)) if cprec_match else None,
        'cRecall': float(crec_match.group(1)) if crec_match else None,
    }
def find_finetuning_predictions(lang, domain, eval_dir='src/method_fine_tuning/eval_20'):
    """Find fine-tuning prediction files"""
    pattern = f"pred_{lang}_{domain}_*.jsonl"
    pred_files = list(Path(eval_dir).glob(pattern))
    return pred_files


def find_llm_predictions(lang, domain):
    """Find LLM prediction files from eval_20/ directory"""
    pred_files = []
    
    # Search in eval_20/ (all LLM models)
    eval_dir = Path('src/method_prompt_llm/eval_20')
    pattern = f"pred_{lang}_{domain}_20_*.jsonl"
    for f in eval_dir.glob(pattern):
        if f.is_file():  # Exclude directories
            pred_files.append(f)
    
    return pred_files


def compare_methods(task=1, output_dir=None, llm_only=False):
    """
    Compare fine-tuning and LLM-based methods
    
    Args:
        task: Task number (1, 2, or 3)
        output_dir: Directory to save results
        llm_only: If True, only compare LLM methods (no fine-tuning)
    """
    # Define language-domain combinations
    combinations = [
        ('eng', 'laptop'),
        ('eng', 'restaurant'),
        ('zho', 'laptop'),
        ('zho', 'restaurant'),
        ('zho', 'finance') if task == 1 else None
    ]
    combinations = [c for c in combinations if c is not None]
    
    results = []
    # Store all LLM results for detailed summary
    all_llm_results = {}
    
    for lang, domain in combinations:
        print(f"\n{'='*60}")
        print(f"Processing: {lang}_{domain}")
        print(f"{'='*60}")
        
        # Find gold file (20_with_va.jsonl)
        gold_file = f"src/task-dataset-split/{lang}/{lang}_{domain}_train_alltasks_20_with_va.jsonl"
        if task == 1 and domain == 'finance':
            gold_file = f"src/task-dataset-split/{lang}/{lang}_{domain}_train_task1_20_with_va.jsonl"
        
        if not os.path.exists(gold_file):
            print(f"Warning: Gold file not found: {gold_file}")
            continue
        
        print(f"Gold file: {gold_file}")
        
        # Find fine-tuning predictions (skip if llm_only)
        ft_pred_files = [] if llm_only else find_finetuning_predictions(lang, domain)
        
        # Find LLM predictions
        llm_pred_files = find_llm_predictions(lang, domain)
        
        if not ft_pred_files and not llm_pred_files:
            print(f"Warning: No prediction files found for {lang}_{domain}")
            continue
        
        # Evaluate fine-tuning results
        ft_results = {}
        if not llm_only:
            for ft_file in ft_pred_files:
                print(f"\nEvaluating fine-tuning: {ft_file.name}")
                scores = run_evaluation(str(ft_file), gold_file, task)
                if scores:
                    ft_results[ft_file.stem] = scores
                    print(f"   Overall RMSE: {scores.get('overall_rmse', 'N/A')}")
        
        # Evaluate LLM results
        llm_results = {}
        dataset_key = f"{lang}_{domain}"
        all_llm_results[dataset_key] = {}
        
        for llm_file in llm_pred_files:
            print(f"\nEvaluating LLM: {llm_file.name}")
            scores = run_evaluation(str(llm_file), gold_file, task)
            if scores:
                llm_results[llm_file.stem] = scores
                # Extract model name for detailed tracking
                model_name = llm_file.name.replace(f'pred_{lang}_{domain}_20_', '').replace('.jsonl', '')
                all_llm_results[dataset_key][model_name] = scores.get('overall_rmse', float('inf'))
                print(f"   Overall RMSE: {scores.get('overall_rmse', 'N/A')}")
        
        # Find best method for this language-domain
        best_ft = min(ft_results.items(), key=lambda x: x[1].get('overall_rmse', float('inf'))) if ft_results else None
        best_llm = min(llm_results.items(), key=lambda x: x[1].get('overall_rmse', float('inf'))) if llm_results else None
        
        # Compare and select best
        best_method = None
        best_score = float('inf')
        best_model = None
        
        if best_ft and not llm_only:
            ft_rmse = best_ft[1].get('overall_rmse', float('inf'))
            if ft_rmse < best_score:
                best_score = ft_rmse
                best_method = 'Fine-tuning'
                best_model = best_ft[0]
        
        if best_llm:
            llm_rmse = best_llm[1].get('overall_rmse', float('inf'))
            if llm_rmse < best_score or llm_only:
                best_score = llm_rmse
                best_method = 'LLM'
                best_model = best_llm[0]
        
        result_row = {
            'Language': lang,
            'Domain': domain,
            'Best_FT_RMSE': best_ft[1].get('overall_rmse', 'N/A') if best_ft else 'N/A',
            'Best_FT_Model': best_ft[0] if best_ft else 'N/A',
            'Best_LLM_RMSE': best_llm[1].get('overall_rmse', 'N/A') if best_llm else 'N/A',
            'Best_LLM_Model': best_llm[0] if best_llm else 'N/A',
            'Best_Method': best_method,
            'Best_RMSE': best_score if best_score != float('inf') else 'N/A',
            'Best_Model': best_model
        }
        
        results.append(result_row)
        
        print(f"\n{'='*60}")
        if llm_only:
            print(f"Best LLM for {lang}_{domain}: {best_model.replace('pred_', '').split('_20_')[-1] if best_model else 'N/A'} (RMSE: {best_score:.4f})")
        else:
            print(f"Best method for {lang}_{domain}: {best_method} (RMSE: {best_score:.4f})")
            print(f"Model: {best_model}")
        print(f"{'='*60}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Print formatted summary table
    print("\n\n")
    
    if llm_only:
        print("=" * 140)
        print("LLM Comparison Summary")
        print("=" * 140)
        
        # Get all unique LLM model names
        all_models = set()
        for dataset_results in all_llm_results.values():
            all_models.update(dataset_results.keys())
        all_models = sorted(all_models)
        
        # Create header with model names
        header = f"{'Dataset':<17}"
        for model in all_models:
            # Shorten model names for display
            short_name = model.replace('meta-llama-3-70b-instruct', 'LLaMA-3-70B')
            short_name = short_name.replace('llama-3.3-70b-instruct', 'LLaMA-3.3-70B')
            short_name = short_name.replace('llama-4-maverick-instruct', 'LLaMA-4-Maverick')
            short_name = short_name.replace('gpt_5.2', 'GPT-5.2')
            header += f"{short_name:<18}"
        header += f"{'Best Model':<35}{'Best RMSE':<10}"
        print(header)
        print("-" * 140)
        
        # Print each row with all LLM results
        for r in results:
            dataset = f"{r['Language']}_{r['Domain']}"
            row = f"{dataset:<17}"
            
            dataset_key = f"{r['Language']}_{r['Domain']}"
            for model in all_models:
                rmse = all_llm_results.get(dataset_key, {}).get(model, None)
                if rmse:
                    row += f"{rmse:<18.4f}"
                else:
                    row += f"{'N/A':<18}"
            
            # Best model and RMSE
            best_model = r.get('Best_LLM_Model', 'N/A') or 'N/A'
            if best_model and best_model != 'N/A':
                parts = best_model.replace('pred_', '').split('_20_')
                if len(parts) > 1:
                    best_model = parts[1]
            best_rmse = r.get('Best_LLM_RMSE', 'N/A')
            best_rmse_str = f"{best_rmse:.4f}" if isinstance(best_rmse, float) else 'N/A'
            
            row += f"{best_model:<35}{best_rmse_str:<10}"
            print(row)
        
        print("=" * 140)
    else:
        print("Comparison Summary")
        print("=" * 120)
        
        # Print header
        print(f"{'Dataset':<19}{'FT RMSE':<13}{'LLM Best RMSE':<15}{'LLM Best Model':<35}{'Best Method':<15}{'Best Model':<25}")
        print("-" * 120)
        
        # Print each row
        for r in results:
            dataset = f"{r['Language']}_{r['Domain']}"
            ft_rmse = f"{r['Best_FT_RMSE']:.4f}" if isinstance(r['Best_FT_RMSE'], float) else 'N/A'
            llm_rmse = f"{r['Best_LLM_RMSE']:.4f}" if isinstance(r['Best_LLM_RMSE'], float) else 'N/A'
            
            # Get LLM best model name
            llm_best_model = r.get('Best_LLM_Model', 'N/A') or 'N/A'
            if llm_best_model and llm_best_model != 'N/A':
                parts = llm_best_model.replace('pred_', '').split('_20_')
                if len(parts) > 1:
                    llm_best_model = parts[1]
            
            # Get overall best method and model
            best_method = r.get('Best_Method', 'N/A') or 'N/A'
            best_model_raw = r.get('Best_Model', 'N/A') or 'N/A'
            
            # Format best model name
            if best_method == 'Fine-tuning':
                best_model = 'XLM-RoBERTa'
            elif best_model_raw and best_model_raw != 'N/A':
                parts = best_model_raw.replace('pred_', '').split('_20_')
                if len(parts) > 1:
                    best_model = parts[1]
                else:
                    best_model = parts[0]
            else:
                best_model = 'N/A'
            
            print(f"{dataset:<19}{ft_rmse:<13}{llm_rmse:<15}{llm_best_model:<35}{best_method:<15}{best_model:<25}")
        
        print("=" * 120)
    
    # Save to file if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        csv_file = os.path.join(output_dir, f'comparison_subtask_{task}.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
        # Save detailed JSON
        json_file = os.path.join(output_dir, f'comparison_subtask_{task}_detailed.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {json_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compare Fine-tuning vs LLM methods')
    parser.add_argument('-t', '--task', type=int, choices=[1, 2, 3], default=1,
                        help='Task number (1, 2, or 3)')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory to save results')
    parser.add_argument('--llm-only', action='store_true',
                        help='Only compare LLM methods (no fine-tuning)')
    
    args = parser.parse_args()
    
    if args.llm_only:
        print(f"Starting comparison for Subtask {args.task} (LLM Only)...")
    else:
        print(f"Starting comparison for Subtask {args.task}...")
    compare_methods(task=args.task, output_dir=args.output_dir, llm_only=args.llm_only)


if __name__ == '__main__':
    main()
