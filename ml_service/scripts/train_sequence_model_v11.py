#!/usr/bin/env python3
"""
train_sequence_model_v11.py — Phase 12
Builds the Multi-Head Sequence Engine (BiLSTM).
Simultaneously predicts 5 unique diseases + 1 continuous severity regression.
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("TrainV11")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

class MultiHeadDataset(Dataset):
    def __init__(self, X_path, Y_path):
        # Memory map the massive numpy arrays so they aren't loaded into RAM all at once
        self.X = np.load(X_path, mmap_mode='r')
        self.Y = np.load(Y_path, mmap_mode='r')
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # Only load the individual 288x69 slice into RAM when requested by the DataLoader worker
        x_slice = torch.tensor(self.X[idx], dtype=torch.float32)
        y_slice = self.Y[idx]
        
        y_cls = torch.tensor(y_slice[0:5], dtype=torch.float32)
        y_reg = torch.tensor(y_slice[5], dtype=torch.float32)
        
        return x_slice, y_cls, y_reg

class MultiHeadBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        # Backbone Feature Extractor
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # We take the final state (or standard sequential)
        # For this model, we'll pool across time or take the last hidden state
        # given that Target Y is for the END of the 48h sequence window
        
        # 5 Independent Classification Heads (BCE)
        self.head_infection = nn.Linear(hidden_dim, 1)
        self.head_heat = nn.Linear(hidden_dim, 1)
        self.head_mastitis = nn.Linear(hidden_dim, 1)
        self.head_lameness = nn.Linear(hidden_dim, 1)
        self.head_calving = nn.Linear(hidden_dim, 1)
        
        # 1 Continuous Regression Head (MSE)
        self.head_severity = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)       # out: (batch, seq_len, hidden)
        
        # Context vector at the very end of the 48h window
        seq_end_context = out[:, -1, :] # (batch, hidden_dim)
        
        # Multi-head branching
        logits_infection = self.head_infection(seq_end_context).squeeze(-1)
        logits_heat = self.head_heat(seq_end_context).squeeze(-1)
        logits_mastitis = self.head_mastitis(seq_end_context).squeeze(-1)
        logits_lameness = self.head_lameness(seq_end_context).squeeze(-1)
        logits_calving = self.head_calving(seq_end_context).squeeze(-1)
        
        pred_severity = self.head_severity(seq_end_context).squeeze(-1)
        
        # Combine all 5 disease logits into a single tensor (batch, 5)
        logits_cls = torch.stack([
            logits_infection, logits_heat, logits_mastitis, 
            logits_lameness, logits_calving
        ], dim=1)
        
        return logits_cls, pred_severity


def get_device():
    # Force CPU. MPS (Apple Silicon GPU) crashes on the massive 5-head tensor allocation
    return torch.device('cpu')


def train_model():
    start_time = time.time()
    logger.info("="*60)
    logger.info("🧠 Phase 12 — Multi-Head Sequence Engine Training")
    logger.info("="*60)
    
    device = get_device()
    logger.info(f"Using compute: {device}")
    
    # ── 1. LOAD DATA ──
    x_path = os.path.join(DATA_DIR, "v11_X_sequences.npy")
    y_path = os.path.join(DATA_DIR, "v11_Y_targets.npy")
    
    if not os.path.exists(x_path):
        logger.error("Tensor files not found. Run prepare_sequences_v11.py first.")
        return
        
    # We probe the shape without loading it all to memory
    X_probe = np.load(x_path, mmap_mode='r')
    num_samples = len(X_probe)
    input_dim = X_probe.shape[2]
    
    logger.info(f"Detected Sequence Tensor X: {X_probe.shape}")
    
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    split = int(0.8 * num_samples)
    
    train_idx, test_idx = indices[:split], indices[split:]
    
    # We pass the PATHS to the dataset so it manages its own mmap handles safely per-worker
    train_dataset = MultiHeadDataset(x_path, y_path)
    test_dataset = MultiHeadDataset(x_path, y_path)
    
    # Assign specific subset indices to the datasets using Subset
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    # Dataloaders - keep batch size small for memory
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    # Init Model
    model = MultiHeadBiLSTM(input_dim=input_dim).to(device)
    
    # We use BCEWithLogits for the 5 independent binary target heads
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    epochs = 12
    best_loss = float('inf')
    
    # ── 2. TRAINING LOOP ──
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x, y_cls, y_reg in train_loader:
            x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
            
            optimizer.zero_grad()
            logits_cls, pred_severity = model(x)
            
            # Loss is composite: 5x Classification + 1x Regression
            loss_cls = criterion_cls(logits_cls, y_cls)
            loss_reg = criterion_reg(pred_severity, y_reg)
            
            total_loss = loss_cls + (0.2 * loss_reg) # Regression holds less weight
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
        # ── 3. EVALUATION AND METRICS ──
        model.eval()
        val_loss = 0.0
        
        # Accumulate metrics per head
        y_true_cls = []
        y_pred_cls = []
        
        with torch.no_grad():
            for x, y_cls, y_reg in test_loader:
                x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
                logits_cls, pred_severity = model(x)
                
                loss_c = criterion_cls(logits_cls, y_cls)
                loss_r = criterion_reg(pred_severity, y_reg)
                val_loss += (loss_c + 0.2 * loss_r).item()
                
                probs = torch.sigmoid(logits_cls)
                y_true_cls.append(y_cls.cpu().numpy())
                y_pred_cls.append(probs.cpu().numpy())
                
        y_true_all = np.vstack(y_true_cls)
        y_pred_all = np.vstack(y_pred_cls)
        
        # Compute AUC for each disease separately
        disease_names = ["Infect", "Heat", "Mastit", "Lame", "Calving"]
        aucs = []
        for d_idx in range(5):
            t = y_true_all[:, d_idx]
            p = y_pred_all[:, d_idx]
            d_auc = roc_auc_score(t, p) if len(np.unique(t)) > 1 else 0
            aucs.append(d_auc)
            
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(test_loader)
        scheduler.step(avg_val_loss)
        
        auc_str = " | ".join([f"{name}: {a:.3f}" for name, a in zip(disease_names, aucs)])
        logger.info(f"Epoch {epoch+1:02d} | Train L: {avg_train_loss:.4f} | Val L: {avg_val_loss:.4f} || AUCs => {auc_str}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "multi_head_model_v11.pth"))
            
    logger.info(f"\n✅ Phase 12 Training Complete! Best Val Loss: {best_loss:.4f}")
    
if __name__ == "__main__":
    train_model()
