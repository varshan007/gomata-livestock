#!/usr/bin/env python3
"""
train_sequence_model_v10.py

Trains BiLSTM Sequential Model (Phase 10) on 48h history sliding windows.
Features trajectory learning, cross-entropy + focal loss, and temporal consistency penalty.
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("TrainV10")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")
SEQ_LEN = 288  # 48 hours at 10-min ticks

# ── 1. DATASET ──

class CowSequenceDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.samples = []
        
        # Raw + minimal engineered features 
        self.features = [
            'temp', 'hr', 'resp', 'activity', 'rumination', 'lying', 
            'thi', 'ambient_temp', 'humidity', 
            'milk_yield', 'feed_intake', 'conductivity', 'body_weight', 
            'hours_since_vaccination', 'hours_since_antibiotic', 
            'parity', 'bcs', 'age'
        ]
        
        logger.info("Building sequence windows...")
        # We group by animal to ensure no cross-animal sequence leakage
        for aid, g in df.groupby('animal_id'):
            indices = g.index.values
            if len(indices) >= seq_len:
                for start_idx in range(len(indices) - seq_len + 1):
                    self.samples.append(indices[start_idx : start_idx + seq_len])
                    
        logger.info(f"Total sequences created: {len(self.samples)}")
        
        # Extract matrices
        self.X = df[self.features].values.astype(np.float32)
        
        # Standardize features
        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0) + 1e-8
        self.X = (self.X - mean) / std
        
        # Save feature scalers for inference later
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "v10_scalers.json"), "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist(), "features": self.features}, f)
            
        self.y_cls = df['target_disease_24h'].values.astype(np.float32)
        self.y_reg = df['target_severity_24h'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        seq_idx = self.samples[idx]
        x = self.X[seq_idx]
        # We take the targets for the entire sequence to compute sequence-to-sequence loss
        y_cls = self.y_cls[seq_idx]
        y_reg = self.y_reg[seq_idx]
        return torch.tensor(x), torch.tensor(y_cls), torch.tensor(y_reg)

# ── 2. MODEL ──

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=0.2)
        self.fc_cls = nn.Linear(hidden_dim, 1)
        self.fc_reg = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x) # out shape: (batch, seq_len, 2*hidden_dim)
        
        # Predict at every timestep
        logits_cls = self.fc_cls(out).squeeze(-1) # (batch, seq_len)
        pred_reg = self.fc_reg(out).squeeze(-1)   # (batch, seq_len)
        
        return logits_cls, pred_reg

# Focal Loss
def focal_loss_with_logits(logits, targets, alpha=0.90, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    # alpha weighting for positive class
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = alpha_t * (1 - pt)**gamma * bce_loss
    return loss.mean()

# ── 3. TRAINING LOOP ──

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def train_model():
    start_time = time.time()
    logger.info("="*60)
    logger.info("🧠 Phase 10 — Sequence Intelligence Training")
    logger.info("="*60)
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    data_path = os.path.join(DATA_DIR, "v10_flat_sequences.csv")
    if not os.path.exists(data_path):
        logger.error(f"{data_path} not found. Run prepare_sequences_v10.py first!")
        return
        
    logger.info("Loading tabular data into memory...")
    # Load and split train/test based on animal_id (80/20)
    df = pd.read_csv(data_path)
    animals = df['animal_id'].unique()
    np.random.seed(42)
    np.random.shuffle(animals)
    
    split_idx = int(len(animals) * 0.8)
    train_animals = animals[:split_idx]
    test_animals = animals[split_idx:]
    
    train_df = df[df['animal_id'].isin(train_animals)].reset_index(drop=True)
    test_df = df[df['animal_id'].isin(test_animals)].reset_index(drop=True)
    
    # We heavily drop early healthy rows in train to balance the dataset
    # We want hard negative mining (AR noise) + positive cases.
    # Keep all positives, and downsample negatives by keeping 20% of random negatives
    logger.info("Downsampling train negatives ...")
    pos_mask = train_df['target_disease_24h'] == 1
    neg_mask = train_df['target_disease_24h'] == 0
    sampled_negs = train_df[neg_mask].sample(frac=0.2, random_state=42)
    train_df_sampled = pd.concat([train_df[pos_mask], sampled_negs]).sort_values(["animal_id", "timestamp"]).reset_index(drop=True)
    
    logger.info("Building Datasets...")
    train_dataset = CowSequenceDataset(train_df_sampled)
    test_dataset = CowSequenceDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    model = BiLSTMModel(input_dim=len(train_dataset.features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 15
    best_val_auc = 0
    
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        logger.info(f"Epoch {epoch+1} starting iteration...")
        batch_idx = 0
        for x, y_cls, y_reg in train_loader:
            batch_idx += 1
            if batch_idx == 1:
                logger.info("Got first batch!")
            x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
            
            optimizer.zero_grad()
            logits_cls, pred_reg = model(x)
            
            # Loss 1: Focal Loss on end-of-sequence predictions
            cls_loss = focal_loss_with_logits(logits_cls[:, -1], y_cls[:, -1], alpha=0.9, gamma=2.0)
            
            # Loss 2: Severity Regression
            reg_loss = F.mse_loss(pred_reg[:, -1], y_reg[:, -1])
            
            # Loss 3: Temporal Regularization (smooth differences over sequence)
            # Only penalize positive jumps to be smooth
            diffs = logits_cls[:, 1:] - logits_cls[:, :-1]
            temporal_penalty = (diffs ** 2).mean()
            
            loss = cls_loss + 0.3 * reg_loss + 0.2 * temporal_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y_cls, y_reg in test_loader:
                x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
                logits_cls, pred_reg = model(x)
                
                cls_loss = focal_loss_with_logits(logits_cls[:, -1], y_cls[:, -1])
                reg_loss = F.mse_loss(pred_reg[:, -1], y_reg[:, -1])
                loss = cls_loss + 0.3 * reg_loss
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits_cls[:, -1]).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y_cls[:, -1].cpu().numpy())
                
        auc = roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0
        pr_auc = average_precision_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - "
                    f"Val Loss: {val_loss/len(test_loader):.4f} - Val AUC: {auc:.4f} - Val PR-AUC: {pr_auc:.4f}")
        
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "onset_sequence_model_v10.pth"))
            logger.info("  >> Saved best model")

    elapsed = time.time() - start_time
    logger.info(f"✅ Training completed in {elapsed:.1f}s. Best AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    train_model()
