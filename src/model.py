"""
DimASRModel Definition
Dual regression head architecture based on pretrained language model
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DimASRModel(nn.Module):
    """
    Dimensional Aspect Sentiment Regression Model
    
    Architecture:
        Input: [CLS] Text [SEP] Aspect [SEP]
        ↓
        Pretrained Encoder (XLM-RoBERTa / BERT / etc.)
        ↓
        [CLS] Token Representation
        ↓
        ├─ Valence Regression Head
        └─ Arousal Regression Head
    """
    
    def __init__(self, 
                 model_name: str = 'xlm-roberta-base',
                 dropout: float = 0.1,
                 hidden_size: int = None):
        """
        Args:
            model_name: HuggingFace pretrained model name
            dropout: Dropout rate
            hidden_size: Hidden size (if None, use model default)
        """
        super().__init__()
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get encoder output dimension
        self.hidden_size = hidden_size or self.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Valence regression head (output range: 1-9)
        self.valence_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Arousal regression head (output range: 1-9)
        self.arousal_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            valence: (batch_size,) range [1, 9]
            arousal: (batch_size,) range [1, 9]
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        cls_output = self.dropout(cls_output)
        
        # Regression prediction
        valence_logits = self.valence_head(cls_output).squeeze(-1)  # (batch_size,)
        arousal_logits = self.arousal_head(cls_output).squeeze(-1)  # (batch_size,)
        
        # Map output to [1, 9] range
        # Use sigmoid to map to [0, 1], then scale to [1, 9]
        valence = torch.sigmoid(valence_logits) * 8.0 + 1.0
        arousal = torch.sigmoid(arousal_logits) * 8.0 + 1.0
        
        return valence, arousal
    
    def predict(self, input_ids, attention_mask):
        """
        Prediction interface (no gradient)
        """
        self.eval()
        with torch.no_grad():
            valence, arousal = self.forward(input_ids, attention_mask)
        return valence, arousal


class DimASRModelV2(nn.Module):
    """
    Improved version: uses deeper regression heads and residual connections
    """
    
    def __init__(self, 
                 model_name: str = 'xlm-roberta-base',
                 dropout: float = 0.1):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        
        # Shared feature extraction layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Valence-specific layer
        self.valence_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Arousal-specific layer
        self.arousal_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Shared features
        shared_features = self.shared_layer(cls_output)
        
        # Predict
        valence_logits = self.valence_head(shared_features).squeeze(-1)
        arousal_logits = self.arousal_head(shared_features).squeeze(-1)
        
        # Map to [1, 9]
        valence = torch.sigmoid(valence_logits) * 8.0 + 1.0
        arousal = torch.sigmoid(arousal_logits) * 8.0 + 1.0
        
        return valence, arousal


def create_model(model_name='xlm-roberta-base', 
                 version='v1',
                 dropout=0.1):
    """
    Factory function: create model instance
    
    Args:
        model_name: Pretrained model name
        version: 'v1' or 'v2'
        dropout: Dropout rate
        
    Returns:
        Model instance
    """
    if version == 'v1':
        return DimASRModel(model_name, dropout)
    elif version == 'v2':
        return DimASRModelV2(model_name, dropout)
    else:
        raise ValueError(f"Unknown version: {version}")


if __name__ == "__main__":
    # Test model
    print("Testing DimASR Model...")
    
    model = create_model('xlm-roberta-base', version='v1')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Forward pass
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    valence, arousal = model(input_ids, attention_mask)
    
    print(f"Valence shape: {valence.shape}, range: [{valence.min():.2f}, {valence.max():.2f}]")
    print(f"Arousal shape: {arousal.shape}, range: [{arousal.min():.2f}, {arousal.max():.2f}]")
