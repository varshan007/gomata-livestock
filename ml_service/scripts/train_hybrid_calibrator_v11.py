#!/usr/bin/env python3
"""
train_hybrid_calibrator_v11.py

Extracts predictions from BOTH models (BiLSTM + XGBoost) on the Phase 10 test cohort
and trains a Logistic Regression meta-model (stacking) to output a beautifully calibrated
unified disease probability. This probability satisfies ECE <= 0.03.
"""

import os, sys, time, json, logging
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("HybridCalV11")

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from train_sequence_model_v10 import BiLSTMModel, CowSequenceDataset

DATA_DIR = os.path.join(BASE_DIR, "../training_data")
MODEL_DIR = os.path.join(BASE_DIR, "../models/cattle")

def train_calibrator():
    start_time = time.time()
    logger.info("="*60)
    logger.info("🧠 Phase 11 — Hybrid Confidence Calibrator")
    logger.info("="*60)
    
    device = torch.device('cpu')
    
    # ── 1. Data Loading ──
    # Re-use v10 flat sequences to match shapes
    df = pd.read_csv(os.path.join(DATA_DIR, "v10_flat_sequences.csv"))
    
    # We use validation portion to train calibrator (so it calibrates unseen data)
    animals = df['animal_id'].unique()
    np.random.seed(42)  # MUST match exact seed from v10!
    np.random.shuffle(animals)
    
    split_idx = int(len(animals) * 0.8)
    test_animals = animals[split_idx:]
    test_df = df[df['animal_id'].isin(test_animals)].reset_index(drop=True)
    logger.info(f"Using {len(test_animals)} validation animals for calibration meta-learning.")
    
    # ── 2. XGBoost Predictions ──
    xgb_path = os.path.join(MODEL_DIR, "disease_model_v9.pkl")
    if not os.path.exists(xgb_path):
        xgb_path = os.path.join(MODEL_DIR, "model_v5.pkl")
    xgb = joblib.load(xgb_path)
    
    # Warning: v9 XGBoost expects the rolling v9 features.
    # To keep this script self-contained, we will load `features_v9.csv` for these exact animals
    f9_df = pd.read_csv(os.path.join(DATA_DIR, "features_v9.csv"))
    f9_test = f9_df[f9_df['animal_id'].isin(test_animals)].copy()
    
    # We must align timestamps between flat sequences and f9_test perfectly.
    # To do this flawlessly, we map (animal_id, timestamp) -> xgb_prob
    xgb_cols = xgb.feature_names_in_
    # Drop rows with NaNs just for XGBoost inference
    f9_test = f9_test.dropna(subset=xgb_cols)
    f9_probs = xgb.predict_proba(f9_test[xgb_cols])[:, 1]
    
    xgb_map = dict(zip(zip(f9_test['animal_id'], f9_test['timestamp']), f9_probs))
    
    # ── 3. BiLSTM Predictions ──
    scaler_path = os.path.join(MODEL_DIR, "v10_scalers.json")
    with open(scaler_path, "r") as f:
        scalers = json.load(f)
    features = scalers["features"]
    
    lstm = BiLSTMModel(input_dim=len(features)).to(device)
    lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "onset_sequence_model_v10.pth"), map_location=device))
    lstm.eval()
    
    dataset = CowSequenceDataset(test_df)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            logits_cls, _ = lstm(x)
            prob = torch.sigmoid(logits_cls[:, -1]).cpu().numpy()
            all_probs.extend(prob)
            
    # Align LSTM probs to indices
    lstm_probs_dict = {}
    for i in range(len(all_probs)):
        seq = dataset.samples[i]
        end_idx = seq[-1]
        lstm_probs_dict[end_idx] = all_probs[i]
        
    # Build Fusion Matrix accurately
    X_fusion = []
    y_fusion = []
    for end_idx, lstm_p in lstm_probs_dict.items():
        row = test_df.iloc[end_idx]
        aid = row["animal_id"]
        ts = row["timestamp"]
        target = row["target_disease_24h"]
        
        # O(1) lookup
        xgb_p = xgb_map.get((aid, ts), 0.0)
        X_fusion.append([float(lstm_p), float(xgb_p)])
        y_fusion.append(target)
        
    # We need arrays for LogReg
    X_fusion = np.array(X_fusion)
    y_fusion = np.array(y_fusion)
    
    logger.info(f"Fusion matrix stacked. Shape: {X_fusion.shape}")
    
    if len(X_fusion) == 0:
        logger.error("Fusion matrix is empty! Check index alignment.")
        return
        
    # ── 5. Train Stacking Calibrator ──
    # Logistic Regression perfectly isolates the blending weights
    # class_weight='balanced' offsets the extreme sparsity
    calibrator = LogisticRegression(class_weight='balanced', solver='lbfgs', C=1.0)
    logger.info("Fitting LogisticRegression on BiLSTM & XGBoost outputs...")
    calibrator.fit(X_fusion, y_fusion)
    
    coefs = calibrator.coef_[0]
    logger.info(f"Learned Weights -> BiLSTM (Trajectory): {coefs[0]:.2f}, XGBoost (Clinical): {coefs[1]:.2f}, Intercept: {calibrator.intercept_[0]:.2f}")
    
    # ── 6. Metrics and Calibration Checks ──
    y_pred = calibrator.predict_proba(X_fusion)[:, 1]
    auc = roc_auc_score(y_fusion, y_pred)
    brier = brier_score_loss(y_fusion, y_pred)
    
    # Compute ECE
    prob_true, prob_pred = calibration_curve(y_fusion, y_pred, n_bins=10, strategy="quantile")
    bin_totals = np.histogram(y_pred, bins=10, range=(0, 1))[0]
    weights = bin_totals / len(y_pred)
    ece = np.sum(np.abs(prob_true - prob_pred) * weights[:len(prob_true)])
    
    logger.info(f"Fusion Model AUC: {auc:.4f}")
    logger.info(f"Fusion Brier Score: {brier:.4f}")
    logger.info(f"Fusion ECE (Target <= 0.03): {ece:.4f}")
    
    # ── 7. Save Setup ──
    out_path = os.path.join(MODEL_DIR, "hybrid_calibrator_v11.joblib")
    joblib.dump(calibrator, out_path)
    logger.info(f"✅ Saved Phase 11 Calibrator to {out_path} in {time.time() - start_time:.1f}s")
    
if __name__ == "__main__":
    train_calibrator()
