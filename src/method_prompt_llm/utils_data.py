import json
import re
from os import makedirs
from os.path import dirname, exists
import jsonlines


def extract_va_score(va_text):
    """Extract clean VA score from LLM response (handles verbose outputs)."""
    if not va_text:
        return "5.00#5.00"
    
    # If already clean format, return as-is
    if re.fullmatch(r'\d+\.?\d*#\d+\.?\d*', va_text.strip()):
        return va_text.strip()
    
    # Try to find valence#arousal pattern in verbose text
    patterns = [
        r'(\d+\.?\d*)\s*#\s*(\d+\.?\d*)',  # 7.50#6.80
        r'\$\\boxed\{(\d+\.?\d*)\s*#\s*(\d+\.?\d*)\}',  # LaTeX boxed format
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, va_text)
        if matches:
            # Take the last match (usually the final answer)
            v, a = float(matches[-1][0]), float(matches[-1][1])
            # Clamp to [1, 9] range
            v = max(1.0, min(9.0, v))
            a = max(1.0, min(9.0, a))
            return f"{v:.2f}#{a:.2f}"
    
    # Default neutral value if extraction fails
    print(f"Warning: Could not extract VA from: {va_text[:80]}...")
    return "5.00#5.00"

def load_samples_flat(data_path, max_samples=None):
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
                for ind, quad in enumerate(obj['Quadruplet']):
                    samples.append({
                        'id': f"{id_val}:{ind}",
                        'text': text,
                        'aspect': quad['Aspect'],
                        #'va': quad.get('VA')
                    })
            elif 'Aspect_VA' in obj:
                for ind, aspect_va in enumerate(obj['Aspect_VA']):
                    samples.append({
                        'id': f"{id_val}:{ind}",
                        'text': text,
                        'aspect': aspect_va['Aspect'],
                        #'va': aspect_va.get('VA')
                    })
            elif 'Aspect' in obj:
                for ind, aspect in enumerate(obj['Aspect']):
                    samples.append({
                        'id': f"{id_val}:{ind}",
                        'text': text,
                        'aspect': aspect,
                        #'va': None
                    })
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"[INFO] Loaded {len(samples)} samples from {data_path}")
    return samples

def flat_to_task_a_format_iter(results):
    """Save prediction results as JSONL"""
    from collections import defaultdict
    
    # Group by ID
    grouped = defaultdict(list)
    for r in results:
        # Remove only the last :index that we added in load_samples_flat
        # e.g., "6834782:S026:0" -> "6834782:S026"
        # e.g., "rest16_quad_dev_3:0" -> "rest16_quad_dev_3"
        id_parts = r['id'].rsplit(':', 1)  
        group_id = id_parts[0]
        grouped[group_id].append({
            'Aspect': r['aspect'],
            'VA': extract_va_score(r['va'])  # Extract clean VA score
        })

    for id_val, aspect_va_list in grouped.items():
        yield {
            'ID': id_val,
            'Aspect_VA': aspect_va_list
        }


def write_jsonl(results_it, output_path):
    makedirs(dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results_it:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def iter_src_filepaths(langs, domains, template):
    for lang in langs:
        for domain in domains:
            file_path = template.format(lang=lang, domain=domain)
            print(f"[INFO] file: {file_path}")
            if exists(file_path):
                yield lang, domain, file_path
            else:
                print(f"Skipping (not exist)")


def iter_jsonl(file_path):
    with open(file_path, 'r') as reader:
        for line in reader:
            yield json.loads(line)