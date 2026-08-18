#!/usr/bin/env python3
"""
stress_test_v14_hardware.py — Phase 14
Hardware Corruption & Asynchronous Edge Streaming Reality Test

Tests the Multi-Head Shared Attention PyTorch model against:
- 30% random IoT sensor packet drops
- 2-hour physical gateway blackouts
- Stuck sensors (repeating values)
- Hardware calibration drift (+1.5C bias)
- Missing / asynchronous production data
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StressV14_Hardware")

# Import the PyTorch V13 model architecture
sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")


def apply_hardware_chaos(sensor_df, prod_df, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    df = pd.merge(sensor_df, prod_df, on=["animalId", "timestamp"], how="left")
    
    # Simulate asynchronous production data (Forward fill last known milk/feed up to 24h)
    for col in ["milkYield", "feedIntake", "conductivity", "antibioticActive"]:
        df[col] = df.groupby("animalId")[col].ffill(limit=144).fillna(0)
    
    n_total = len(df)
    logger.info("Injecting 30% random sensor packet loss...")
    # 30% drop on high-frequency IoT sensors
    drop_mask = rng.random(n_total) < 0.30
    for col in ["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index"]:
        df.loc[drop_mask, col] = np.nan
        
    logger.info("Injecting 2-hour physical blackouts (Gateway failures)...")
    animals = df["animalId"].unique()
    for aid in animals:
        a_mask = df["animalId"] == aid
        n_a = a_mask.sum()
        if n_a < 100: continue
        
        # 3 blackouts per animal
        for _ in range(3):
            st = rng.randint(0, n_a - 12)
            # 12 ticks = 2 hours
            idx_start = df[a_mask].index[st]
            idx_end = df[a_mask].index[st+12]
            for col in ["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index"]:
                df.loc[idx_start:idx_end, col] = np.nan
                
        # Stuck sensor logic: 15% chance HR gets stuck for 24 hours (144 ticks)
        if rng.random() < 0.15:
            st = rng.randint(0, n_a - 144)
            idx_start = df[a_mask].index[st]
            idx_end = df[a_mask].index[st+144]
            stuck_val = df.loc[idx_start, "heartRate_bpm"]
            if np.isnan(stuck_val): stuck_val = 60.0
            df.loc[idx_start:idx_end, "heartRate_bpm"] = stuck_val
            
        # Calibration drift logic: 20% chance Temp biased by +1.5C permanently midway
        if rng.random() < 0.20:
            st = rng.randint(int(n_a * 0.4), int(n_a * 0.8))
            idx_start = df[a_mask].index[st]
            df.loc[idx_start:, "temperature_C"] += 1.5

    # Forward fill up to 3 ticks (30 mins) then leave as NaN to be zeroed by scaler logic
    for col in ["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index"]:
        df[col] = df.groupby("animalId")[col].ffill(limit=3).fillna(0)
        
    return df


def evaluate_hardware_robustness(df, model_path, scalers_path):
    logger.info("Extracting streaming features under Hardware Chaos...")
    
    windows = {"6h": 36, "12h": 72, "24h": 144}
    for w_name, w_ticks in windows.items():
        for col in ["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index", "milkYield", "conductivity"]:
            df[f"{col}_mean_{w_name}"] = df.groupby("animalId")[col].transform(lambda x: x.rolling(w_ticks, min_periods=1).mean())
            df[f"{col}_std_{w_name}"] = df.groupby("animalId")[col].transform(lambda x: x.rolling(w_ticks, min_periods=1).std().fillna(0))
            df[f"{col}_delta_{w_name}"] = df[col] - df[f"{col}_mean_{w_name}"]
            
    df["thermal_strain_index"] = df["heartRate_bpm_delta_6h"] / (df["thi"] - 72 + 1e-5)
    df["lameness_suppression"] = (1.0 - df["activity_index"]) * (df["feedIntake"] / 22.0)
    df["mastitis_spike_index"] = df["conductivity_delta_12h"] * (df["milkYield"] / 30.0)
    df["fever_decoupled"] = df["temperature_C"] - (38.5 + (0.01 * np.maximum(df["thi"]-72, 0)))
    df.fillna(0, inplace=True)
    
    with open(scalers_path, "r") as f:
        scaler_data = json.load(f)
    features = scaler_data["features"]
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_data["means"])
    scaler.scale_ = np.array(scaler_data["scales"])
    df[features] = scaler.transform(df[features])
    
    SEQ_LEN = 288
    STRIDE = 12
    X_list, Y_cls_list = [], []
    
    grouped = df.groupby("animalId")
    for _, group in grouped:
        v_features = group[features].values
        v_targets = group[["infectionBinary", "heatStressBinary", "mastitisBinary", "lamenessBinary", "calvingBinary"]].values
        
        n_ticks = len(v_features)
        if n_ticks < SEQ_LEN + 144: continue
        
        for start_idx in range(0, n_ticks - (SEQ_LEN + 144), STRIDE):
            end_idx = start_idx + SEQ_LEN
            X_list.append(v_features[start_idx:end_idx])
            Y_cls_list.append(v_targets[end_idx - 1])
            
    X_mat = np.array(X_list, dtype=np.float32)
    Y_cls = np.array(Y_cls_list, dtype=np.float32)
    logger.info(f"Generated {len(X_mat)} sliding evaluation windows.")
    
    device = get_device()
    model = SharedAttentionHazardEngine(input_dim=len(features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    batch_size = 256
    y_pred_cls = []
    
    with torch.no_grad():
        for i in range(0, len(X_mat), batch_size):
            x_batch = torch.tensor(X_mat[i:i+batch_size], dtype=torch.float32).to(device)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits_cls, _, _, _ = model(x_batch)
            probs = torch.sigmoid(logits_cls).float().cpu().numpy()
            y_pred_cls.append(probs)
            
    Y_pred = np.vstack(y_pred_cls)
    DISEASES = ["Infection", "HeatStress", "Mastitis", "Lameness", "Calving"]
    
    m_dict = {}
    for i, disease in enumerate(DISEASES):
        t = Y_cls[:, i]
        p = Y_pred[:, i]
        auc = roc_auc_score(t, p) if len(np.unique(t)) > 1 else 0
        pred_labels = (p > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(t, pred_labels, average='binary', zero_division=0)
        m_dict[disease] = {"AUC": float(auc), "Recall": float(recall), "Precision": float(precision)}
        
    precisions = [m['Precision'] for k, m in m_dict.items() if m['Precision'] > 0.0]
    fp_rate = 1.0 - np.mean(precisions) if precisions else 0.0
    
    logger.info("--- HARDWARE CORRUPTION RESULTS ---")
    logger.info(f"Infect AUC: {m_dict['Infection']['AUC']:.4f} | Heat AUC: {m_dict['HeatStress']['AUC']:.4f}")
    logger.info(f"Mastit AUC: {m_dict['Mastitis']['AUC']:.4f}  | Lame AUC: {m_dict['Lameness']['AUC']:.4f}")
    logger.info(f"Target False Positives: {fp_rate * 7:.2f}/week")
    
    return m_dict

def main():
    logger.info("=" * 60)
    logger.info("🔥 Phase 14 — Hardware Chaos & Asynchronous Pipeline Test")
    logger.info("=" * 60)
    
    sensor_path = os.path.join(DATA_DIR, "v7_sensor_raw.csv")
    prod_path = os.path.join(DATA_DIR, "v7_production_raw.csv")
    model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
    scalers_path = os.path.join(MODEL_DIR, "v13_scalers.json")
    
    if not os.path.exists(sensor_path) or not os.path.exists(model_path):
        logger.error("Missing raw V7 data or trained V13 model. Please ensure pipeline is executed.")
        return
        
    logger.info("Loading clean Baseline V7 Sensor & Production data...")
    sensor_df = pd.read_csv(sensor_path)
    prod_df = pd.read_csv(prod_path)
    
    df_chaos = apply_hardware_chaos(sensor_df, prod_df)
    evaluate_hardware_robustness(df_chaos, model_path, scalers_path)
    logger.info("✅ HARDWARE TEST COMPLETE")

if __name__ == "__main__":
    main()
