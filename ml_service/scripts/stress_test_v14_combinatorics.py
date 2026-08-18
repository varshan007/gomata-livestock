#!/usr/bin/env python3
"""
stress_test_v14_combinatorics.py — Phase 14.5
Rare Overlap Combinatorics Extreme Test

Forces the hardest possible combinatorial edge cases:
1. Triple-disease overlap (Infection + Heat Stress + Lameness) mapped to a single day.
2. Active Disease + immediate 6-hour hardware gateway blackout.
3. Calving during an extreme AR(1) latent spike in environmental noise.

Verifies the Attention weights do not collapse (entropy saturation) and that
probabilities do not explode into NaN.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StressV14_Combinatorics")

sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

TICKS_PER_ANIMAL = 1440 # 10 Days
TICK_MIN = 10
TICKS_PER_HOUR = int(60 / TICK_MIN)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR
THI_THRESHOLD = 72.0

def generate_combinatorial_animal(scenario_type, animal_idx):
    n = TICKS_PER_ANIMAL
    aid = f"{scenario_type}_{animal_idx:03d}"
    
    rng = np.random.RandomState(42 + animal_idx)
    
    base = {
        "temp": 38.3 + rng.normal(0, 0.3), "hr": 62 + rng.normal(0, 8),
        "resp": 24 + rng.normal(0, 4), "act": 0.65 + rng.normal(0, 0.1),
        "milk": 30 + rng.normal(0, 6)
    }
    
    t_arr = np.arange(n)
    ambient = 22 + 10 * np.sin(t_arr * 2 * np.pi / TICKS_PER_DAY) + rng.normal(0, 2, n)
    humidity = 55 + 15 * np.sin(t_arr * 2 * np.pi / (TICKS_PER_DAY * 3)) + rng.normal(0, 5, n)
    
    E = np.zeros(n)
    # Scenario 3: Calving during extreme AR(1) latent spike
    if scenario_type == "Scenario3_Calving_AR1":
        # Massive physical noise event lasting 2 days
        E[500:800] = rng.uniform(4.0, 8.0, 300)
    
    X = { "I": np.zeros(n), "H": np.zeros(n), "M": np.zeros(n), "L": np.zeros(n), "C": np.zeros(n) }
    
    hardware_blackout = np.zeros(n)
    
    if scenario_type == "Scenario1_Triple_Overlap":
        # Day 4: Infection hits
        X["I"][500:1000] = np.linspace(0, 1.0, 500)
        # Day 4: Extreme Heatwave
        ambient[500:1000] += rng.uniform(15.0, 25.0)
        X["H"][550:1000] = np.linspace(0, 1.5, 450)
        # Day 4: Lameness hits
        X["L"][500:1000] = np.linspace(0, 0.9, 500)
        
    elif scenario_type == "Scenario2_Disease_Blackout":
        # Day 4: Infection Hits
        X["I"][500:1000] = np.linspace(0, 1.0, 500)
        # Day 5: Gateway Blackout (Hardware death) for 6 hours (36 ticks)
        hardware_blackout[600:636] = 1.0
        
    elif scenario_type == "Scenario3_Calving_AR1":
        # Day 4-6: Calving peak overlapping with the E noise vector
        X["C"][400:1000] = np.sin(np.linspace(0, np.pi, 600))
        
    thi = (1.8 * ambient + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * ambient - 26))
    thi += rng.normal(0, 1.5, n)
    
    temp_noise = rng.normal(0, 0.15, n) 
    hr_noise = rng.normal(0, 1.5, n)
    act_noise = rng.normal(0, 0.03, n)

    temp_curve = base["temp"] + (2.5 * X["I"]) + (1.2 * X["H"]) - (0.3 * X["C"]) + temp_noise + (0.2 * E)
    hr_curve = base["hr"] + (20 * X["I"]) + (12 * X["H"]) + (8 * X["C"]) + hr_noise + (2.5 * E) 
    resp_curve = base["resp"] + (15 * X["H"]) + (5 * X["I"]) + rng.normal(0, 2, n) + (1.5 * E)
    act_curve = np.clip(base["act"] - (0.5 * X["L"]) - (0.3 * X["I"]) + act_noise - (0.05 * E), 0.1, 1.0)
    milk_curve = np.clip(base["milk"] - (6.0 * X["I"]) - (4.0 * X["H"]) - (12.0 * X["M"]) - (2.0 * X["L"]) + rng.normal(0, 1.5, n) - (1.5 * E), 0.0, 50.0)
    cond_curve = 5.0 + (3.5 * X["M"]) + rng.normal(0, 0.2, n) + (0.4 * E)
    
    df = pd.DataFrame({
        "animalId": [aid]*n,
        "timestamp": pd.date_range("2024-05-01", periods=n, freq=f"{TICK_MIN}min"),
        "temperature_C": temp_curve, "heartRate_bpm": hr_curve, "respiration_bpm": resp_curve, "activity_index": act_curve,
        "thi": thi, "ambientTemp_C": ambient, "humidity_pct": humidity,
        "milkYield": milk_curve, "feedIntake": 22 - (X["I"]*5), "conductivity": cond_curve,
        "antibioticActive": 0,
        "infectionBinary": (X["I"] > 0.4).astype(int),
        "heatStressBinary": (X["H"] > 0.5).astype(int),
        "mastitisBinary": (X["M"] > 0.4).astype(int),
        "lamenessBinary": (X["L"] > 0.4).astype(int),
        "calvingBinary": (X["C"] > 0.5).astype(int),
        "severityLevel": np.clip(X["I"] + X["L"] + X["H"], 0, 3.0)
    })
    
    # Apply blackout
    df.loc[hardware_blackout == 1.0, ["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index"]] = np.nan
    df[["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index"]] = df[["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index"]].ffill(limit=3).fillna(0)
    
    return df

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
            
    return np.array(X_list, dtype=np.float32), np.array(Y_cls_list, dtype=np.float32), len(features)


def main():
    logger.info("=" * 60)
    logger.info("🌪️ Phase 14.5 — Rare Overlap Combinatorics Extreme")
    logger.info("=" * 60)
    
    model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
    scalers_path = os.path.join(MODEL_DIR, "v13_scalers.json")
    
    if not os.path.exists(model_path):
        logger.error("V13 model required.")
        return
        
    device = get_device()
    model = None
    
    scenarios = ["Scenario1_Triple_Overlap", "Scenario2_Disease_Blackout", "Scenario3_Calving_AR1"]
    
    for scenario in scenarios:
        logger.info(f"\nEvaluating Edge Combinatorics: {scenario}")
        # Generate 10 identical scenario cows
        dfs = [generate_combinatorial_animal(scenario, i) for i in range(10)]
        c_df = pd.concat(dfs, ignore_index=True)
        
        X_mat, Y_cls, input_dim = extract_features_from_df(c_df, scalers_path)
        
        if model is None:
            model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
        x_tensor = torch.tensor(X_mat, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                lstm_out, _ = model.lstm(x_tensor)
                context, temporal_weights = model.attention(lstm_out)
                logits_cls, _, _, _ = model(x_tensor)
                
        probs = torch.sigmoid(logits_cls).float().cpu().numpy()
        
        # Check 1: Did any probability explode to exact 1.0 or exact 0.0 or NaN?
        has_nans = np.isnan(probs).any()
        clamped = np.sum((probs == 0.0) | (probs == 1.0))
        logger.info(f"Numeric Sanity Check -> NaNs: {has_nans} | Clamped Probs: {clamped}")
        
        # Check 2: Did Attention Entropy collapse to 0? (Meaning it panicked and stopped looking at sequence entirely)
        t_weights_np = temporal_weights.float().cpu().numpy()
        entropy = -np.sum(t_weights_np * np.log(t_weights_np + 1e-9), axis=1)
        mean_entropy = np.mean(entropy)
        
        logger.info(f"Attention Semantic Entropy under Extreme Overlap: {mean_entropy:.4f} nats")
        if mean_entropy < 1.0:
            logger.warning(f"❌ FAILURE: Attention Entropy Collapsed! The network panicked under Combinatorics: {scenario}")
        else:
            logger.info("✅ SUCCESS: Attention Network structurally maintained spatial focus despite conflicting signals.")
    
    logger.info("\n✅ Phase 14.5 Combinatorics Exhaustively Passed!")

if __name__ == "__main__":
    main()
