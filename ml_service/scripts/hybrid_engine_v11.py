#!/usr/bin/env python3
"""
hybrid_engine_v11.py

Fuses BiLSTM (Early Trajectory Engine) + XGBoost (Clinical Classifier).
Implements a two-stage biologically aligned decision system:
1. Early Watchlist (BiLSTM sustained anomaly gating)
2. Clinical Escalation (XGBoost validation OR fused escalation)
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import joblib
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("HybridV11")

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from train_sequence_model_v10 import BiLSTMModel, CowSequenceDataset

MODEL_DIR = os.path.join(BASE_DIR, "../models/cattle")
DATA_DIR = os.path.join(BASE_DIR, "../training_data")

class HybridLivestockEngine:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        logger.info(f"Loading Hybrid Models to {self.device}...")
        
        # 1. Load XGBoost Clinical Model
        xgb_path = os.path.join(MODEL_DIR, "disease_model_v9.pkl")
        if not os.path.exists(xgb_path):
            logger.warning(f"XGBoost v9 not found at {xgb_path}. Looking for v5...")
            xgb_path = os.path.join(MODEL_DIR, "model_v5.pkl")
            
        self.xgb_model = joblib.load(xgb_path)
        
        # 2. Load BiLSTM Trajectory Model
        scaler_path = os.path.join(MODEL_DIR, "v10_scalers.json")
        with open(scaler_path, "r") as f:
            scalers = json.load(f)
        self.seq_features = scalers["features"]
        
        self.lstm_model = BiLSTMModel(input_dim=len(self.seq_features)).to(self.device)
        self.lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "onset_sequence_model_v10.pth"), map_location=self.device))
        self.lstm_model.eval()
        
        # 3. Load Calibrator (if exists)
        cal_path = os.path.join(MODEL_DIR, "hybrid_calibrator_v11.joblib")
        self.calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
        
        # 4. State Management (Per Animal)
        self.state = {}
        
    def _init_animal(self, animal_id):
        if animal_id not in self.state:
            self.state[animal_id] = {
                "lstm_prob_hist": [],
                "xgb_prob_hist": [],
                "watchlist": False,
                "ticks_below_threshold": 0
            }
            
    def predict_tick(self, animal_id, lstm_input_seq, xgb_features, tick_timestamp=None):
        """
        Runs one tick of hybrid inference for an animal.
        lstm_input_seq: shape [288, features] array (48 hours at 10m ticks)
        xgb_features: DataFrame row with XGBoost features computed over last 24h
        """
        self._init_animal(animal_id)
        st = self.state[animal_id]
        
        # --- 1. BI-LSTM INFERENCE ---
        with torch.no_grad():
            x = torch.tensor(lstm_input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits_cls, pred_reg = self.lstm_model(x)
            lstm_prob = float(torch.sigmoid(logits_cls[:, -1]).cpu().numpy()[0])
            
        # --- 2. XGBOOST INFERENCE ---
        # Select correct feature order matching training
        xgb_cols = self.xgb_model.feature_names_in_
        x_xgb = xgb_features[xgb_cols]
        xgb_prob = float(self.xgb_model.predict_proba(x_xgb)[0, 1])
        
        # Update history
        st["lstm_prob_hist"].append(lstm_prob)
        st["xgb_prob_hist"].append(xgb_prob)
        if len(st["lstm_prob_hist"]) > 12:
            st["lstm_prob_hist"].pop(0)
            st["xgb_prob_hist"].pop(0)
            
        # --- 3. FUSED CALIBRATION ---
        fused_prob = 0.0
        if self.calibrator:
            fused_prob = float(self.calibrator.predict_proba([[lstm_prob, xgb_prob]])[0, 1])
        else:
            # Fallback heuristic blending
            fused_prob = (0.6 * lstm_prob) + (0.4 * xgb_prob)
            
        # --- 4. ENGINE LOGIC ---
        alert_level = "NORMAL"
        
        # Step 4a. Auto-Clear Logic
        if lstm_prob < 0.30:
            st["ticks_below_threshold"] += 1
            if st["ticks_below_threshold"] >= 12: # 2 hours of quiet
                st["watchlist"] = False
        else:
            st["ticks_below_threshold"] = 0
            
        # Step 4b. Early Watchlist Trigger (BiLSTM sustained gating)
        if len(st["lstm_prob_hist"]) >= 3:
            sustained_high = sum(1 for x in st["lstm_prob_hist"][-3:] if x > 0.55) >= 3
            if sustained_high:
                st["watchlist"] = True
                
        # Step 4c. Clinical Escalation (XGBoost validation OR blended rapid rise)
        lstm_rising = False
        if len(st["lstm_prob_hist"]) >= 3:
            lstm_rising = st["lstm_prob_hist"][-1] > st["lstm_prob_hist"][-3] + 0.05
            
        if xgb_prob > 0.70 or (lstm_rising and xgb_prob > 0.40) or fused_prob > 0.65:
            alert_level = "CLINICAL_ALERT"
        elif st["watchlist"]:
            alert_level = "EARLY_WATCHLIST"
            
        return {
            "animal_id": animal_id,
            "timestamp": tick_timestamp,
            "lstm_prob": round(lstm_prob, 4),
            "xgb_prob": round(xgb_prob, 4),
            "fused_prob": round(fused_prob, 4),
            "alert_level": alert_level,
            "watchlist_active": st["watchlist"]
        }

if __name__ == "__main__":
    logger.info("Hybrid Engine defined. Run evaluation scripts to test.")
