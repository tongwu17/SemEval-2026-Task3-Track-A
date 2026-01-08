import os
import json
import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import sys
from scipy.stats import pearsonr

from model import create_model
from data_loader import DimASRDataset

def calculate_rmse(pred_valence, pred_arousal, gold_valence, gold_arousal):
    """Fast RMSE calculation for quick monitoring during training"""
    pred_valence = np.array(pred_valence)
    pred_arousal = np.array(pred_arousal)
    gold_valence = np.array(gold_valence)
    gold_arousal = np.array(gold_arousal)
    
    N = len(pred_valence)
    valence_error = (pred_valence - gold_valence) ** 2
    arousal_error = (pred_arousal - gold_arousal) ** 2
    rmse_va = np.sqrt(np.sum(valence_error + arousal_error) / N)
    
    return float(rmse_va)


def calculate_individual_rmse(pred_valence, pred_arousal, gold_valence, gold_arousal):
    """Fast individual RMSE calculation for training monitoring"""
    pred_valence = np.array(pred_valence)
    pred_arousal = np.array(pred_arousal)
    gold_valence = np.array(gold_valence)
    gold_arousal = np.array(gold_arousal)
    
    valence_rmse = np.sqrt(np.mean((pred_valence - gold_valence) ** 2))
    arousal_rmse = np.sqrt(np.mean((pred_arousal - gold_arousal) ** 2))
    
    return float(valence_rmse), float(arousal_rmse)


def calculate_pcc(pred_valence, pred_arousal, gold_valence, gold_arousal):
    """Calculate Pearson Correlation Coefficient for Valence and Arousal"""
    pred_valence = np.array(pred_valence)
    pred_arousal = np.array(pred_arousal)
    gold_valence = np.array(gold_valence)
    gold_arousal = np.array(gold_arousal)
    
    pcc_v = pearsonr(pred_valence, gold_valence)[0]
    pcc_a = pearsonr(pred_arousal, gold_arousal)[0]
    
    return float(pcc_v), float(pcc_a)

class Trainer:
    """Trainer"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 scheduler,
                 device,
                 output_dir='./checkpoints',
                 patience=3,
                 model_name='xlm-roberta-base',
                 max_length=256):
        """
        Args:
            model: model
            train_loader: training data loader
            val_loader: validation data loader
            optimizer: optimizer
            scheduler: learning rate scheduler
            device: device
            output_dir: model save directory
            patience: early stopping patience
            model_name: pretrained model name
            max_length: maximum sequence length
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.patience = patience
        self.model_name = model_name
        self.max_length = max_length
        
        # MSE loss function
        self.criterion = nn.MSELoss()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self.best_rmse = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move data to device (MPS requires float32)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            valence_gold = batch['valence'].float().to(self.device)
            arousal_gold = batch['arousal'].float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            valence_pred, arousal_pred = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss_v = self.criterion(valence_pred, valence_gold)
            loss_a = self.criterion(arousal_pred, arousal_gold)
            loss = loss_v + loss_a
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validation"""
        self.model.eval()
        
        pred_valences = []
        pred_arousals = []
        gold_valences = []
        gold_arousals = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                valence_pred, arousal_pred = self.model(input_ids, attention_mask)
                
                pred_valences.extend(valence_pred.cpu().numpy().tolist())
                pred_arousals.extend(arousal_pred.cpu().numpy().tolist())
                gold_valences.extend(batch['valence'].float().cpu().numpy().tolist())
                gold_arousals.extend(batch['arousal'].float().cpu().numpy().tolist())
        
        # Calculate RMSE
        rmse_va = calculate_rmse(pred_valences, pred_arousals,
                                 gold_valences, gold_arousals)
        valence_rmse, arousal_rmse = calculate_individual_rmse(
            pred_valences, pred_arousals, gold_valences, gold_arousals)
        
        # Calculate PCC (Pearson Correlation Coefficient)
        pcc_v, pcc_a = calculate_pcc(pred_valences, pred_arousals,
                                      gold_valences, gold_arousals)
        
        return {
            'rmse_va': rmse_va,
            'valence_rmse': valence_rmse,
            'arousal_rmse': arousal_rmse,
            'pcc_v': pcc_v,
            'pcc_a': pcc_a
        }
    
    def train(self, num_epochs):
        """Main training loop"""
        logging.info(f"Starting training for {num_epochs} epochs...")
        logging.info(f"Device: {self.device}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            logging.info(f"\nEpoch {epoch}:")
            logging.info(f"  Train Loss: {train_loss:.4f}")
            logging.info(f"  Val RMSE_VA: {val_metrics['rmse_va']:.4f}")
            logging.info(f"  Val Valence RMSE: {val_metrics['valence_rmse']:.4f}")
            logging.info(f"  Val Arousal RMSE: {val_metrics['arousal_rmse']:.4f}")
            logging.info(f"  Val PCC_V: {val_metrics['pcc_v']:.4f}")
            logging.info(f"  Val PCC_A: {val_metrics['pcc_a']:.4f}")
            
            # Save best model
            if val_metrics['rmse_va'] < self.best_rmse:
                self.best_rmse = val_metrics['rmse_va']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save model
                model_path = os.path.join(self.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'rmse_va': self.best_rmse,
                    'metrics': val_metrics,
                    'model_name': self.model_name,
                    'max_length': self.max_length
                }, model_path)
                logging.info(f"  → Saved best model (RMSE_VA: {self.best_rmse:.4f})")
            else:
                self.patience_counter += 1
                logging.info(f"  No improvement ({self.patience_counter}/{self.patience})")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logging.info(f"\nEarly stopping at epoch {epoch}")
                logging.info(f"Best model at epoch {self.best_epoch} with RMSE_VA: {self.best_rmse:.4f}")
                break
        
        return self.best_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../task-dataset/track_a/subtask_1/eng',
                       help='Data directory path')
    parser.add_argument('--train_file', type=str, default='eng_restaurant_train_alltasks.jsonl',
                       help='Training file name')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base',
                       help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='model save directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio (split from training set)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path (default: training_log_<train_file>.txt)')
    
    args = parser.parse_args()
    
    # Setup log file
    if args.log_file is None:
        # Extract dataset name from train_file (e.g., zho_restaurant_train_alltasks.jsonl -> zho_restaurant_train_alltasks)
        dataset_name = os.path.splitext(args.train_file)[0]
        args.log_file = f'training_log_{dataset_name}.txt'
    
    # Configure logging: output to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(args.log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Log training configuration
    logging.info("="*50)
    logging.info(f"Training Configuration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*50)
    logging.info(f"Data: {args.data_dir}/{args.train_file}")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Epochs: {args.num_epochs}, Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.learning_rate}, Patience: {args.patience}")
    logging.info(f"Val split: {args.val_split}, Seed: {args.seed}")
    logging.info(f"Log file: {args.log_file}")
    logging.info("="*50 + "\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load tokenizer
    logging.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    logging.info("Loading data...")
    train_path = os.path.join(args.data_dir, args.train_file)
    
    # Load complete training data
    full_train_dataset = DimASRDataset(train_path, tokenizer, args.max_length)
    
    # Split training and validation sets
    from torch.utils.data import random_split
    train_size = int((1 - args.val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    
    logging.info(f"Creating model: {args.model_name}")
    model = create_model(args.model_name, version='v1', dropout=args.dropout)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        patience=args.patience,
        model_name=args.model_name,
        max_length=args.max_length
    )
    
    best_rmse = trainer.train(args.num_epochs)
    
    logging.info("\n" + "="*50)
    logging.info(f"Training completed!")
    logging.info(f"Best RMSE_VA: {best_rmse:.4f}")
    logging.info(f"Model saved to: {args.output_dir}/best_model.pt")
    logging.info("="*50)


if __name__ == "__main__":
    main()
