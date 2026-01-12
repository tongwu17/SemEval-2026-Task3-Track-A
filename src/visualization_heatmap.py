"""
Heatmap Visualization for MSE/RMSE by Aspect Category

This script generates heatmap visualizations showing prediction errors (MSE/RMSE)
for different aspect categories across Valence and Arousal dimensions.

Usage:
    python src/visualization_heatmap.py -g gold.jsonl -p pred.jsonl -o output_dir
    
    # Compare multiple models
    python src/visualization_heatmap.py -g gold.jsonl \
        -p model1.jsonl model2.jsonl \
        -n "Fine-tuned BERT" "Few-shot GPT-4o" \
        -o output_dir
"""

import os
import argparse
import jsonlines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def parse_va_string(va_str):
    """Parse VA string '6.75#6.38' to (valence, arousal) tuple"""
    try:
        parts = va_str.split('#')
        valence = float(parts[0])
        arousal = float(parts[1])
        return valence, arousal
    except:
        return None, None


def extract_category_errors(gold_data, pred_data):
    """
    Extract prediction errors grouped by aspect category
    
    Returns:
        dict: {category: {'valence_errors': [], 'arousal_errors': []}}
    """
    category_errors = defaultdict(lambda: {'valence_errors': [], 'arousal_errors': []})
    
    # Create prediction lookup by ID and Category
    pred_lookup = {}
    for item in pred_data:
        item_id = item['ID']
        
        # Handle different task formats
        if 'Aspect_VA' in item:  # Subtask 1
            for aspect_va in item['Aspect_VA']:
                aspect = aspect_va.get('Aspect', 'NULL')
                va = aspect_va.get('VA', '')
                # For subtask 1, we need to match with gold to get category
                key = (item_id, aspect)
                pred_lookup[key] = va
        elif 'Triplet' in item:  # Subtask 2
            for triplet in item['Triplet']:
                aspect = triplet.get('Aspect', 'NULL')
                va = triplet.get('VA', '')
                key = (item_id, aspect)
                pred_lookup[key] = va
        elif 'Quadruplet' in item:  # Subtask 3
            for quad in item['Quadruplet']:
                aspect = quad.get('Aspect', 'NULL')
                category = quad.get('Category', 'UNKNOWN')
                va = quad.get('VA', '')
                key = (item_id, aspect, category)
                pred_lookup[key] = va
    
    # Match with gold data and calculate errors
    matched = 0
    missing = 0
    
    for item in gold_data:
        item_id = item['ID']
        
        # Handle different task formats
        gold_elements = []
        if 'Quadruplet' in item:
            gold_elements = item['Quadruplet']
        elif 'Triplet' in item:
            gold_elements = item['Triplet']
        elif 'Aspect_VA' in item:
            gold_elements = item['Aspect_VA']
        
        for element in gold_elements:
            aspect = element.get('Aspect', 'NULL')
            category = element.get('Category', 'UNKNOWN')
            gold_va = element.get('VA', None)
            
            if gold_va is None:
                continue
            
            # Parse gold VA
            gold_v, gold_a = parse_va_string(gold_va)
            if gold_v is None or gold_a is None:
                continue
            
            # Find prediction (try different keys for different formats)
            pred_va = None
            for key in [(item_id, aspect, category), (item_id, aspect)]:
                if key in pred_lookup:
                    pred_va = pred_lookup[key]
                    break
            
            if pred_va:
                pred_v, pred_a = parse_va_string(pred_va)
                
                if pred_v is not None and pred_a is not None:
                    # Calculate squared errors
                    v_error = (pred_v - gold_v) ** 2
                    a_error = (pred_a - gold_a) ** 2
                    
                    category_errors[category]['valence_errors'].append(v_error)
                    category_errors[category]['arousal_errors'].append(a_error)
                    matched += 1
                else:
                    missing += 1
            else:
                missing += 1
    
    print(f"Matched samples: {matched}")
    print(f"Missing predictions: {missing}")
    
    return category_errors


def calculate_mse_rmse(category_errors):
    """
    Calculate MSE and RMSE for each category
    
    Returns:
        pd.DataFrame with columns: Category, Valence_MSE, Arousal_MSE, Valence_RMSE, Arousal_RMSE
    """
    results = []
    
    for category, errors in category_errors.items():
        v_errors = errors['valence_errors']
        a_errors = errors['arousal_errors']
        
        if len(v_errors) > 0:
            v_mse = np.mean(v_errors)
            v_rmse = np.sqrt(v_mse)
            a_mse = np.mean(a_errors)
            a_rmse = np.sqrt(a_mse)
            
            results.append({
                'Category': category,
                'Valence_MSE': v_mse,
                'Arousal_MSE': a_mse,
                'Valence_RMSE': v_rmse,
                'Arousal_RMSE': a_rmse,
                'Sample_Count': len(v_errors)
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Category')
    
    return df


def plot_heatmap(df, metric='RMSE', output_path=None, title=None):
    """
    Plot heatmap for MSE or RMSE by category
    
    Args:
        df: DataFrame with category errors
        metric: 'MSE' or 'RMSE'
        output_path: Path to save the figure
        title: Custom title for the plot
    """
    if metric == 'MSE':
        data = df[['Category', 'Valence_MSE', 'Arousal_MSE']].set_index('Category')
        data.columns = ['Valence', 'Arousal']
        cmap = 'YlOrRd'  # Yellow-Orange-Red
        fmt = '.3f'
    else:  # RMSE
        data = df[['Category', 'Valence_RMSE', 'Arousal_RMSE']].set_index('Category')
        data.columns = ['Valence', 'Arousal']
        cmap = 'YlOrRd'
        fmt = '.3f'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(6, len(data) * 0.4)))
    
    # Plot heatmap
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, 
                cbar_kws={'label': f'{metric} Value'},
                linewidths=0.5, linecolor='gray',
                ax=ax)
    
    # Set title and labels
    if title is None:
        title = f'{metric} by Aspect Category and VA Dimension'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Aspect Category', fontsize=12, fontweight='bold')
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")
    
    return fig


def plot_comparison_heatmap(dfs, model_names, metric='RMSE', output_path=None):
    """
    Plot side-by-side comparison heatmaps for multiple models
    
    Args:
        dfs: List of DataFrames for each model
        model_names: List of model names
        metric: 'MSE' or 'RMSE'
        output_path: Path to save the figure
    """
    n_models = len(dfs)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, max(6, len(dfs[0]) * 0.4)))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (df, model_name, ax) in enumerate(zip(dfs, model_names, axes)):
        if metric == 'MSE':
            data = df[['Category', 'Valence_MSE', 'Arousal_MSE']].set_index('Category')
            data.columns = ['Valence', 'Arousal']
            fmt = '.3f'
        else:  # RMSE
            data = df[['Category', 'Valence_RMSE', 'Arousal_RMSE']].set_index('Category')
            data.columns = ['Valence', 'Arousal']
            fmt = '.3f'
        
        # Plot heatmap
        max_val = max([max(df[f'Valence_{metric}'].max(), df[f'Arousal_{metric}'].max()) for df in dfs])
        sns.heatmap(data, annot=True, fmt=fmt, cmap='YlOrRd',
                    cbar_kws={'label': f'{metric} Value'},
                    linewidths=0.5, linecolor='gray',
                    ax=ax, vmin=0, vmax=max_val)
        
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
        
        if idx == 0:
            ax.set_ylabel('Aspect Category', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        plt.sca(ax)
        plt.yticks(rotation=0)
    
    fig.suptitle(f'{metric} Comparison by Aspect Category', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison heatmap to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate heatmap visualization for prediction errors by aspect category')
    
    parser.add_argument('-g', '--gold', required=True,
                        help='Path to gold standard JSONL file')
    parser.add_argument('-p', '--pred', nargs='+', required=True,
                        help='Path(s) to prediction JSONL file(s)')
    parser.add_argument('-n', '--names', nargs='+',
                        help='Model names for comparison (same order as --pred)')
    parser.add_argument('-o', '--output_dir', default='./heatmap_output',
                        help='Output directory for heatmap images (default: ./heatmap_output)')
    parser.add_argument('-m', '--metric', choices=['MSE', 'RMSE', 'both'], default='both',
                        help='Metric to visualize: MSE, RMSE, or both (default: both)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots in addition to saving them')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load gold data
    print(f"\nLoading gold data from: {args.gold}")
    gold_data = load_jsonl(args.gold)
    print(f"Loaded {len(gold_data)} gold samples")
    
    # Process each prediction file
    all_dfs = []
    model_names = args.names if args.names else [f"Model {i+1}" for i in range(len(args.pred))]
    
    if len(model_names) != len(args.pred):
        print(f"Warning: Number of names ({len(model_names)}) doesn't match number of prediction files ({len(args.pred)})")
        model_names = [f"Model {i+1}" for i in range(len(args.pred))]
    
    for pred_file, model_name in zip(args.pred, model_names):
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"Prediction file: {pred_file}")
        print(f"{'='*60}")
        
        # Load predictions
        pred_data = load_jsonl(pred_file)
        print(f"Loaded {len(pred_data)} predictions")
        
        # Extract errors by category
        category_errors = extract_category_errors(gold_data, pred_data)
        print(f"Found {len(category_errors)} unique categories")
        
        # Calculate MSE/RMSE
        df = calculate_mse_rmse(category_errors)
        all_dfs.append(df)
        
        # Print statistics
        print(f"\n{model_name} - Category Statistics:")
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(args.output_dir, f"{model_name.replace(' ', '_')}_category_errors.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved statistics to: {csv_path}")
        
        # Generate individual heatmaps
        if args.metric in ['MSE', 'both']:
            mse_path = os.path.join(args.output_dir, f"{model_name.replace(' ', '_')}_MSE_heatmap.png")
            plot_heatmap(df, metric='MSE', output_path=mse_path, 
                        title=f'{model_name} - MSE by Category')
        
        if args.metric in ['RMSE', 'both']:
            rmse_path = os.path.join(args.output_dir, f"{model_name.replace(' ', '_')}_RMSE_heatmap.png")
            plot_heatmap(df, metric='RMSE', output_path=rmse_path,
                        title=f'{model_name} - RMSE by Category')
    
    # Generate comparison heatmaps if multiple models
    if len(all_dfs) > 1:
        print(f"\n{'='*60}")
        print("Generating comparison heatmaps...")
        print(f"{'='*60}")
        
        if args.metric in ['MSE', 'both']:
            comp_mse_path = os.path.join(args.output_dir, "comparison_MSE_heatmap.png")
            plot_comparison_heatmap(all_dfs, model_names, metric='MSE', output_path=comp_mse_path)
        
        if args.metric in ['RMSE', 'both']:
            comp_rmse_path = os.path.join(args.output_dir, "comparison_RMSE_heatmap.png")
            plot_comparison_heatmap(all_dfs, model_names, metric='RMSE', output_path=comp_rmse_path)
    
    # Show plots if requested
    if args.show:
        plt.show()
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
