#!/usr/bin/env python3
"""
profile_inference_v14.py — Phase 14
Production Resource Profiling

Profiles the latency and peak memory usage of the V14 engine
to guarantee edge-deployable scalability (< 100ms per cow, < 2GB RAM).
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import torch
import tracemalloc
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("ProfileV14")

sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")


def extract_features_from_df(df, scalers_path):
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
    # For Edge Inference, we only need the LATEST 48h sequence to predict current hazard!
    # So we only extract the final window.
    X_list = []
    
    grouped = df.groupby("animalId")
    for aid, group in grouped:
        v_features = group[features].values
        n_ticks = len(v_features)
        if n_ticks >= SEQ_LEN:
            X_list.append(v_features[-SEQ_LEN:]) # Just the most recent 288 ticks
            
    # X_mat is [N_COWS, 288, 110]
    return np.array(X_list, dtype=np.float32), len(features)


def main():
    logger.info("=" * 60)
    logger.info("⚙️ Phase 14 — Production Resource Profiling")
    logger.info("=" * 60)
    
    target_cows = 1000
    sensor_path = os.path.join(DATA_DIR, "v7_sensor_raw.csv")
    prod_path = os.path.join(DATA_DIR, "v7_production_raw.csv")
    model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
    scalers_path = os.path.join(MODEL_DIR, "v13_scalers.json")
    
    if not os.path.exists(sensor_path) or not os.path.exists(model_path):
        logger.error("V7 Data or V13 Model missing.")
        return
        
    logger.info(f"Loading Base Farm Data to scale up to {target_cows} Head Herd...")
    sensor_df = pd.read_csv(sensor_path)
    prod_df = pd.read_csv(prod_path)
    base_df = pd.merge(sensor_df, prod_df, on=["animalId", "timestamp"], how="left")
    for col in ["milkYield", "feedIntake", "conductivity", "antibioticActive"]:
        base_df[col] = base_df.groupby("animalId")[col].ffill(limit=144).fillna(0)
        
    # Isolate last 5 days of data for the benchmark speed
    base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])
    cutoff = base_df["timestamp"].max() - pd.Timedelta(days=5)
    base_df = base_df[base_df["timestamp"] >= cutoff].copy()
    
    # Scale dataset up to 1000 distinct animals by replicating
    original_animals = base_df["animalId"].unique()
    dfs = []
    for i in range(target_cows):
        source_aid = original_animals[i % len(original_animals)]
        df_copy = base_df[base_df["animalId"] == source_aid].copy()
        df_copy["animalId"] = f"scale_cow_{i:04d}"
        dfs.append(df_copy)
    
    scaled_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Scaled simulated stream built. ({len(scaled_df)} streaming ticks)")
    
    tracemalloc.start()
    
    # ── START END-TO-END LATENCY PROFILING ──
    t0_start = time.time()
    
    # 1. Feature Extraction Phase Latency
    X_mat, input_dim = extract_features_from_df(scaled_df, scalers_path)
    t1_features = time.time()
    feature_latency_ms = (t1_features - t0_start) * 1000.0 / target_cows
    
    logger.info(f"Feature Extraction / Scaler Time: {feature_latency_ms:.2f} ms per cow")
    
    # 2. PyTorch Inference Latency
    device = get_device()
    model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    x_tensor = torch.tensor(X_mat, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits_cls, sev_out, hazard_logits, w_attn = model(x_tensor)
            probs = torch.sigmoid(logits_cls).float().cpu().numpy()
            hazard = torch.sigmoid(hazard_logits).float().cpu().numpy()
            
    t2_infer = time.time()
    infer_latency_ms = (t2_infer - t1_features) * 1000.0 / target_cows
    
    logger.info(f"PyTorch Neural Inference Time: {infer_latency_ms:.2f} ms per cow")
    
    total_latency_ms = feature_latency_ms + infer_latency_ms
    logger.info(f"Total Deep-Stack Latency: {total_latency_ms:.2f} ms per cow (Target < 100ms)")
    
    # ── MEMORY PROFILING ──
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mem_gb = peak_mem / (1024 * 1024 * 1024)
    logger.info(f"Peak Edge Memory Footprint: {peak_mem_gb:.3f} GB (Target < 2.0GB)")
    
    if total_latency_ms <= 100.0 and peak_mem_gb <= 2.0:
        logger.info("\n✅ SUCCESS: The Phase 14 Architecture is certified for Farm-Grade Edge Deployment.")
    else:
        logger.warning("\n❌ WARNING: Resource overhead exceeded deployment budgets.")

if __name__ == "__main__":
    main()
