#!/usr/bin/env python3
"""
prepare_sequences_v13.py — Phase 13
Sequence Extractor for Survival Hazard Modeling.

Merges adversarial v7 raw data and extracts 288-tick sequences.
Replaces the single future-regression with a 24-hour discrete 
Survival Hazard risk curve mapping.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("PrepareV13")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

SEQ_LEN = 288  # 48 hours at 10m ticks
STRIDE = 12    # shift window 2 hours
HAZARD_HOURS = 24
TICKS_PER_HOUR = 6

def build_flat_timeseries():
    logger.info("Loading V7 Adversarial CSV dumps...")
    df_sensor = pd.read_csv(os.path.join(DATA_DIR, "v7_sensor_raw.csv"), parse_dates=["timestamp"])
    df_prod = pd.read_csv(os.path.join(DATA_DIR, "v7_production_raw.csv"), parse_dates=["timestamp"])
    
    logger.info("Aligning asynchronous production and sensor streams...")
    df_sensor = df_sensor.sort_values(by="timestamp").reset_index(drop=True)
    df_prod = df_prod.sort_values(by="timestamp").reset_index(drop=True)
    
    df_merged = pd.merge_asof(
        df_sensor, 
        df_prod, 
        on="timestamp", 
        by="animalId", 
        direction="backward"
    )
    
    df_merged["milkYield"].fillna(30.0, inplace=True)
    df_merged["feedIntake"].fillna(22.0, inplace=True)
    df_merged["conductivity"].fillna(5.0, inplace=True)
    df_merged["antibioticActive"].fillna(0, inplace=True)
    
    return df_merged


def compute_multimodal_features(df):
    logger.info("Computing 110-dimensional cross-modal biological features...")
    df = df.sort_values(by=["animalId", "timestamp"]).reset_index(drop=True)
    
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
    
    feature_cols = [c for c in df.columns if c not in ["animalId", "timestamp", "simulationVersion"] and not c.endswith("Binary") and c != "severityLevel"]
    
    logger.info(f"Generated {len(feature_cols)} continuous features.")
    
    # Base targets logic (Hazard mapping happens during sequence extraction)
    target_cols = [
        "infectionBinary", "heatStressBinary", "mastitisBinary", "lamenessBinary", "calvingBinary",
        "severityLevel"
    ]
    
    return df, feature_cols, target_cols


def extract_pytorch_sequences(df, feature_cols, target_cols):
    logger.info("Extracting PyTorch Sequence Tensors with 24h Hazard Lookahead...")
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "v13_scalers.json"), "w") as f:
        json.dump({
            "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(),
            "features": feature_cols,
            "targets": target_cols,
            "hazards": 24
        }, f)
        
    total_seqs = 0
    grouped = df.groupby("animalId")
    for aid, group in grouped:
        n_ticks = len(group)
        if n_ticks >= SEQ_LEN + (HAZARD_HOURS * TICKS_PER_HOUR):
            total_seqs += (n_ticks - SEQ_LEN - (HAZARD_HOURS * TICKS_PER_HOUR)) // STRIDE + 1
            
    logger.info(f"Preallocating contiguous memory for {total_seqs} sequences...")
    
    X = np.zeros((total_seqs, SEQ_LEN, len(feature_cols)), dtype=np.float32)
    
    # Y is now [5 Diseases + 1 Current Severity + 24 Hourly Hazards] = 30 wide
    Y = np.zeros((total_seqs, len(target_cols) + HAZARD_HOURS), dtype=np.float32)
    
    processed = 0
    total = len(grouped)
    seq_idx = 0
    
    for aid, group in grouped:
        processed += 1
        if processed % 10 == 0:
            logger.info(f"  Sliced {processed}/{total} sub-trajectories...")
            
        group = group.sort_values("timestamp")
        
        features_arr = group[feature_cols].values
        targets_arr = group[target_cols].values
        
        n_ticks = len(features_arr)
        req_len = SEQ_LEN + (HAZARD_HOURS * TICKS_PER_HOUR)
        if n_ticks < req_len:
            continue
            
        for start_idx in range(0, n_ticks - req_len + 1, STRIDE):
            end_idx = start_idx + SEQ_LEN
            
            X[seq_idx] = features_arr[start_idx:end_idx]
            
            curr_y = targets_arr[end_idx - 1]
            
            # Severity is index 5 in target_cols
            future_severity = targets_arr[end_idx : end_idx + (HAZARD_HOURS * TICKS_PER_HOUR), 5]
            
            hazard_curve = np.zeros(HAZARD_HOURS, dtype=np.float32)
            for h in range(HAZARD_HOURS):
                hour_chunk = future_severity[h*TICKS_PER_HOUR : (h+1)*TICKS_PER_HOUR]
                if np.max(hour_chunk) >= 2.0:
                    hazard_curve[h] = 1.0
                    
            Y[seq_idx, :len(target_cols)] = curr_y
            Y[seq_idx, len(target_cols):] = hazard_curve
            seq_idx += 1
    
    logger.info(f"Final Tensor Shape: X={X.shape}, Y={Y.shape}")
    
    np.save(os.path.join(DATA_DIR, "v13_X_sequences.npy"), X)
    np.save(os.path.join(DATA_DIR, "v13_Y_targets.npy"), Y)
    
    logger.info("Sequence Context array mapped to disk successfully.")


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("🔄 Phase 13 — Survival Hazard Sequence Extractor v13")
    logger.info("=" * 60)
    
    df_flat = build_flat_timeseries()
    df_features, feature_cols, target_cols = compute_multimodal_features(df_flat)
    extract_pytorch_sequences(df_features, feature_cols, target_cols)
    
    logger.info(f"✅ V13 Pipeline Done in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
