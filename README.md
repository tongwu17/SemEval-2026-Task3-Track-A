# SemEval-2026-Task3-Track-A

## Project Overview
This project provides a solution for SemEval 2026 Task - Dimensional ABSA, Track A Subtask 1 (DimASR).

## Methodology
Fine-tuning approach based on pretrained language model (XLM-RoBERTa) for sentiment dimension regression prediction.

### Model Architecture
```
Input: [CLS] Text [SEP] Aspect [SEP]
  ↓
XLM-RoBERTa Encoder
  ↓
[CLS] Representation
  ↓
├─ Valence Head (Linear + Sigmoid*8 + 1)
└─ Arousal Head (Linear + Sigmoid*8 + 1)
```

## Project Structure
```
src/
├── model.py                # Model definition
├── data_loader.py          # Data loading
├── train.py                # Training script
├── predict.py              # Prediction script
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

**Note**: After training, `checkpoints/`, `outputs/`, and `training_log_*.txt` will be generated (not tracked in git).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py \
    --data_dir ../task-dataset/track_a/subtask_1/<LANG> \
    --train_file <LANG>_<DOMAIN>_train_<task_type>.jsonl \
    --model_name xlm-roberta-base \
    --output_dir ./checkpoints/<LANG>_<DOMAIN> \
    --batch_size 16 \
    --num_epochs 10 \
    --val_split 0.1

# Examples:
# English restaurant: <LANG>=eng, <DOMAIN>=restaurant, <task_type>=alltasks
# Chinese finance:    <LANG>=zho, <DOMAIN>=finance, <task_type>=task1
```

### 3. Generate Predictions
```bash
# General command template
python predict.py \
    --checkpoint ./checkpoints/<LANG>_<DOMAIN>/best_model.pt \
    --test_file ../task-dataset/track_a/subtask_1/<LANG>/<LANG>_<DOMAIN>_dev_task1.jsonl \
    --output_file ./outputs/<LANG>/pred_<LANG>_<DOMAIN>.jsonl

# Examples:
# English restaurant: <LANG>=eng, <DOMAIN>=restaurant
# Chinese finance:    <LANG>=zho, <DOMAIN>=finance
```

## Experimental Settings
- **Pretrained Model**: xlm-roberta-base
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Training Epochs**: 10
- **Optimizer**: AdamW
- **Loss Function**: MSE Loss

**Note**: Val RMSE_VA is the result on validation set during training, Codabench RMSE is the official score after submission.

