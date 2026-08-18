#!/usr/bin/env python3
"""
train_sequence_model_v13_attn.py — Phase 13
Adversarial Realism + Shared Attention + Survival Hazard Engine.
"""

import os, sys, time, logging, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("TrainV13")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

class HazardDataset(Dataset):
    def __init__(self, X_path, Y_path):
        # Memory map arrays to prevent 15GB spike
        self.X = np.load(X_path, mmap_mode='r')
        self.Y = np.load(Y_path, mmap_mode='r')
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x_slice = torch.tensor(self.X[idx], dtype=torch.float32)
        y_slice = self.Y[idx]
        
        # Y is [5 diseases + 1 current severity + 24 hourly hazards]
        y_cls = torch.tensor(y_slice[0:5], dtype=torch.float32)
        y_sev = torch.tensor(y_slice[5], dtype=torch.float32)
        y_hazard = torch.tensor(y_slice[6:30], dtype=torch.float32)
        
        return x_slice, y_cls, y_sev, y_hazard


class TemporalMHA(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        # Using a simple mean pool approach over self-attention.
        
    def forward(self, x):
        # x is [B, seq_len, hidden_dim]
        # self attention
        attn_out, attn_weights = self.mha(x, x, x) # attn_weights: [B, seq_len, seq_len]
        
        # Take the mean across sequence length to create a Global Context Vector
        context = attn_out.mean(dim=1) # [B, hidden_dim]
        
        # Taking mean of structural weights for pure UI interpretability (where did it look on average?)
        temporal_focus = attn_weights.mean(dim=1) # [B, seq_len]
        
        return context, temporal_focus


class SharedAttentionHazardEngine(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        
        # Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_out_dim = hidden_dim * 2
        
        # Shared Attention Layer
        self.attention = TemporalMHA(lstm_out_dim, num_heads=4)
        
        # 5 Independent Disease Classification Heads
        self.head_inf = nn.Linear(lstm_out_dim, 1)
        self.head_heat = nn.Linear(lstm_out_dim, 1)
        self.head_mast = nn.Linear(lstm_out_dim, 1)
        self.head_lame = nn.Linear(lstm_out_dim, 1)
        self.head_calv = nn.Linear(lstm_out_dim, 1)
        
        # Modulated Severity & Hazard Heads
        self.head_sev = nn.Linear(lstm_out_dim, 1)
        self.head_hazard = nn.Linear(lstm_out_dim, 24) # 24 discrete hours
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Z = A V, C = GlobalWeightedPooling(Z)
        context, temporal_weights = self.attention(lstm_out)
        
        # Basic logits
        l_inf = self.head_inf(context)
        l_heat = self.head_heat(context)
        l_mast = self.head_mast(context)
        l_lame = self.head_lame(context)
        l_calv = self.head_calv(context)
        
        # Biological Gating (Modulate severity and hazards based on disease probability)
        # We need probabilities to modulate
        p_inf = torch.sigmoid(l_inf)
        p_mast = torch.sigmoid(l_mast)
        p_lame = torch.sigmoid(l_lame)
        p_heat = torch.sigmoid(l_heat)
        p_calv = torch.sigmoid(l_calv)
        
        # If any of these diseases rise, severity is scaled up linearly as an interaction
        # g(p) = 1 + α * p
        modulation = 1.0 + (0.5 * p_inf) + (0.5 * p_mast) + (0.3 * p_lame) + (0.2 * p_heat) + (0.2 * p_calv)
        
        sev_out = self.head_sev(context) * modulation
        hazard_logits = self.head_hazard(context) * modulation
        
        logits_cls = torch.cat([l_inf, l_heat, l_mast, l_lame, l_calv], dim=1)
        
        return logits_cls, sev_out.squeeze(1), hazard_logits, temporal_weights

def get_device():
    # Force CPU to avoid MPS memory allocation crashes on large arrays in mixed precision
    return torch.device("cpu")

def train_model():
    # Use CPU for training to avoid M1 unified memory fragmenting on 11k massive sequences
    device = get_device()
    logger.info(f"Using compute: {device} with Mixed Precision via Autocast")
    
    x_path = os.path.join(DATA_DIR, "v13_X_sequences.npy")
    y_path = os.path.join(DATA_DIR, "v13_Y_targets.npy")
    
    if not os.path.exists(x_path):
        logger.error("Tensor files not found. Run prepare_sequences_v13.py first.")
        return
        
    X_probe = np.load(x_path, mmap_mode='r')
    num_samples = len(X_probe)
    input_dim = X_probe.shape[2]
    
    logger.info(f"Detected Sequence Tensor X: {X_probe.shape}")
    
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    split = int(0.8 * num_samples)
    
    train_idx, test_idx = indices[:split], indices[split:]
    
    train_dataset = HazardDataset(x_path, y_path)
    test_dataset = HazardDataset(x_path, y_path)
    
    train_subset = Subset(train_dataset, train_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
    
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_sev = nn.MSELoss()
    criterion_hazard = nn.BCEWithLogitsLoss() # 24 independent binary risks

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 12
    best_val_loss = float('inf')
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        start_t = time.time()
        for x_batch, y_cls_batch, y_sev_batch, y_hazard_batch in train_loader:
            x_batch = x_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)
            y_sev_batch = y_sev_batch.to(device)
            y_hazard_batch = y_hazard_batch.to(device)
            
            optimizer.zero_grad()
            
            logits_cls, preds_sev, hazard_logits, _ = model(x_batch)
            
            loss_cls = criterion_cls(logits_cls, y_cls_batch)
            loss_sev = criterion_sev(preds_sev, y_sev_batch)
            loss_hazard = criterion_hazard(hazard_logits, y_hazard_batch) * 5.0 # Weight the survival curve heavily
            
            loss = loss_cls + (0.1 * loss_sev) + loss_hazard
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        y_true_cls = []
        y_pred_cls = []
        
        with torch.no_grad():
            for x_batch, y_cls_batch, y_sev_batch, y_hazard_batch in test_loader:
                x_batch = x_batch.to(device)
                y_cls_batch = y_cls_batch.to(device)
                y_sev_batch = y_sev_batch.to(device)
                y_hazard_batch = y_hazard_batch.to(device)
                
                logits_cls, preds_sev, hazard_logits, _ = model(x_batch)
                
                loss_cls = criterion_cls(logits_cls, y_cls_batch)
                loss_sev = criterion_sev(preds_sev, y_sev_batch)
                loss_hazard = criterion_hazard(hazard_logits, y_hazard_batch) * 5.0
                
                loss = loss_cls + (0.1 * loss_sev) + loss_hazard
                val_loss += loss.item()
                    
                probs = torch.sigmoid(logits_cls).float().cpu().numpy()
                y_true_cls.append(y_cls_batch.float().cpu().numpy())
                y_pred_cls.append(probs)
                
        val_loss /= len(test_loader)
        
        y_true = np.vstack(y_true_cls)
        y_pred = np.vstack(y_pred_cls)
        
        aucs = []
        for i in range(5):
            t = y_true[:, i]
            p = y_pred[:, i]
            if len(np.unique(t)) > 1:
                aucs.append(roc_auc_score(t, p))
            else:
                aucs.append(0.0)
                
        metrics_str = f"Infect: {aucs[0]:.3f} | Heat: {aucs[1]:.3f} | Mastit: {aucs[2]:.3f} | Lame: {aucs[3]:.3f} | Calving: {aucs[4]:.3f}"
        logger.info(f"Epoch {epoch+1:02d} | Train L: {train_loss:.4f} | Val L: {val_loss:.4f} || AUCs => {metrics_str}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
            torch.save(model.state_dict(), model_path)
            
    logger.info(f"\n✅ Phase 13 Training Complete! Best Val Loss: {best_val_loss:.4f}")
    
if __name__ == "__main__":
    train_model()
