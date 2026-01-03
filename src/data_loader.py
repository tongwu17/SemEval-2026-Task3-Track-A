"""
Data loading and preprocessing module
Process DimABSA Track A Subtask 1 data
Support reading directly from raw alltasks or task1 format
"""

import json
import jsonlines
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DimASRDataset(Dataset):
    """Dimensional Aspect Sentiment Regression Dataset"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        """
        Args:
            data_path: JSONL file path (supports alltasks or task1 format)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_and_flatten_data(data_path)
        
    def _load_and_flatten_data(self, data_path: str) -> List[Dict]:
        """
        Load and flatten JSONL data
        Auto-detect format (Quadruplet from alltasks or Aspect_VA from task1)
        If text has multiple aspects, create multiple samples
        """
        samples = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                text = obj['Text']
                id_val = obj['ID']
                
                # Method 1: Extract from Quadruplet (alltasks format)
                if 'Quadruplet' in obj:
                    for quad in obj['Quadruplet']:
                        aspect = quad['Aspect'] if quad['Aspect'] != 'NULL' else 'general'
                        samples.append({
                            'id': id_val,
                            'text': text,
                            'aspect': aspect,
                            'va': quad['VA']
                        })
                
                # Method 2: Extract from Aspect_VA (task1 format, with labels)
                elif 'Aspect_VA' in obj:
                    for aspect_va in obj['Aspect_VA']:
                        samples.append({
                            'id': id_val,
                            'text': text,
                            'aspect': aspect_va['Aspect'],
                            'va': aspect_va['VA']
                        })
                
                # Method 3: Extract from Aspect (task1 format, test set without labels)
                elif 'Aspect' in obj:
                    for aspect in obj['Aspect']:
                        samples.append({
                            'id': id_val,
                            'text': text,
                            'aspect': aspect
                        })
                
                else:
                    raise ValueError(f"Unknown data format in {data_path}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return processed data for one sample
        """
        sample = self.samples[idx]
        text = sample['text']
        aspect = sample['aspect']
        
        # Construct input: [CLS] text [SEP] aspect [SEP]
        encoding = self.tokenizer(
            text,
            aspect,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'id': sample['id'],
            'aspect': aspect
        }
        
        # If has labels (train/val set)
        if 'va' in sample:
            valence, arousal = map(float, sample['va'].split('#'))
            result['valence'] = valence
            result['arousal'] = arousal
        
        return result


def load_data(data_path: str) -> List[Dict]:
    """
    Load raw JSONL data
    
    Returns:
        List of dictionaries with keys: ID, Text, Aspect_VA/Aspect/Quadruplet
    """
    data = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def parse_va_string(va_str: str) -> Tuple[float, float]:
    """
    Parse VA string
    
    Args:
        va_str: "6.75#6.38" format string
        
    Returns:
        (valence, arousal) tuple
    """
    valence, arousal = va_str.split('#')
    return float(valence), float(arousal)


def format_va_string(valence: float, arousal: float) -> str:
    """
    Format VA string
    
    Args:
        valence: valence value (1.00-9.00)
        arousal: Arousal value (1.00-9.00)
        
    Returns:
        "V#A" format string, rounded to 2 decimals
    """
    # Ensure within valid range
    valence = max(1.0, min(9.0, valence))
    arousal = max(1.0, min(9.0, arousal))
    return f"{valence:.2f}#{arousal:.2f}"


def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2):
    """Create DataLoader"""
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
