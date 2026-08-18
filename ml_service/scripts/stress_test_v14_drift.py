#!/usr/bin/env python3
"""
stress_test_v14_drift.py — Phase 14.5
Long-Horizon 90-Day Drift Assessment

Generates T=13000 ticks (~90 days) of continuous farm data, deeply injecting
macro-environmental seasonal shifts (temperature trending upwards +15C over months)
and biological baseline drift (aging heart rates). 
Evaluates if the Sequence Model explodes into False Positives on natural drift.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StressV14_Drift")

sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

N_ANIMALS_PER_FARM = 10
TICKS_PER_ANIMAL = 13000  # ~90 days at 10-minute intervals
TICK_MIN = 10
TICKS_PER_HOUR = int(60 / TICK_MIN)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR
THI_THRESHOLD = 72.0

class LongHorizonSimulationUniverse:
    def __init__(self, seed_offset):
        self.rng = np.random.RandomState(seed_offset)
        self.p_lambd = 0.05
        self.p_sigma = 0.8
        
    def generate_animal(self, animal_idx):
        n = TICKS_PER_ANIMAL
        aid = f"Drift_animal_{animal_idx:04d}"
        
        base = {
            "temp": 38.3 + self.rng.normal(0, 0.3),
            "hr": 62 + self.rng.normal(0, 8),
            "resp": 24 + self.rng.normal(0, 4),
            "act": 0.65 + self.rng.normal(0, 0.1),
            "milk": 30 + self.rng.normal(0, 6),
            "weight": 550 + self.rng.normal(0, 50),
            "heat_tolerance": self.rng.uniform(0.5, 1.5)
        }
        
        t_arr = np.arange(n)
        
        # ── SLOW MACRO DRIFT (SEASONAL CHANGE) ──
        # Temperature gradually rises by 15C across the 90 days (Spring -> Deep Summer)
        seasonal_temp_drift = np.linspace(0, 15.0, n)
        ambient = 15 + 10 * np.sin(t_arr * 2 * np.pi / TICKS_PER_DAY) + self.rng.normal(0, 2, n) + seasonal_temp_drift
        
        # Slowly rising humidity across the season
        seasonal_humid_drift = np.linspace(0, 20.0, n)
        humidity = 45 + 15 * np.sin(t_arr * 2 * np.pi / (TICKS_PER_DAY * 3)) + self.rng.normal(0, 5, n) + seasonal_humid_drift
        
        # Animal biology drifts with aging across the 3 months (Heart rate slows down slightly)
        aging_hr_drift = np.linspace(0, -4.0, n)
        
        thi = (1.8 * ambient + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * ambient - 26))
        thi += self.rng.normal(0, 1.5, n)
        
        E = np.zeros(n)
        for t in range(1, n):
            E[t] = E[t-1] - self.p_lambd * E[t-1] + self.rng.normal(0, self.p_sigma)
            
        X = { "I": np.zeros(n), "H": np.zeros(n), "M": np.zeros(n), "L": np.zeros(n), "C": np.zeros(n), "Imm": np.ones(n), "Comp": np.ones(n), "Fat": np.zeros(n) }
        
        # Sparse diseases (Mostly healthy over the 90 days to test False Positive resilience)
        has_infection = self.rng.random() < 0.15
        infection_start = self.rng.randint(int(0.1*n), int(0.9*n)) if has_infection else -1
        exposure = np.zeros(n)
        if has_infection: exposure[infection_start:infection_start + 144] = self.rng.uniform(0.01, 0.03)

        for t in range(1, n):
            thi_excess = max(thi[t] - THI_THRESHOLD, 0)
            dH = (0.01 * thi_excess) - (0.05 * base["heat_tolerance"]) + self.rng.normal(0, 0.01)
            X["H"][t] = np.clip(X["H"][t-1] + dH, 0, 1.5)

            dFat = (0.02 * X["H"][t]) + (0.05 * X["I"][t-1]) - 0.01
            X["Fat"][t] = np.clip(X["Fat"][t-1] + dFat, 0, 1.0)
            
            dImm = - (0.08 * X["Fat"][t]) - (0.05 * X["H"][t]) + 0.01
            X["Imm"][t] = np.clip(X["Imm"][t-1] + dImm, 0.1, 1.0)

            dI = exposure[t-1] + (0.05 * X["I"][t-1]) - (0.04 * X["Imm"][t]) + self.rng.normal(0, 0.005)
            X["I"][t] = np.clip(X["I"][t-1] + dI, 0, 1.0)

        temp_noise = np.zeros(n); hr_noise = np.zeros(n); resp_noise = np.zeros(n); act_noise = np.zeros(n)
        rho = 0.85
        for t in range(1, n):
            temp_noise[t] = rho * temp_noise[t-1] + self.rng.normal(0, 0.15)
            hr_noise[t] = rho * hr_noise[t-1] + self.rng.normal(0, 1.5)
            act_noise[t] = rho * act_noise[t-1] + self.rng.normal(0, 0.03)

        temp_curve = base["temp"] + (2.5 * X["I"]) + (1.2 * X["H"]) + temp_noise + (0.2 * E)
        hr_curve = base["hr"] + (20 * X["I"]) + (12 * X["H"]) + hr_noise + (2.5 * E) + aging_hr_drift
        resp_curve = base["resp"] + (15 * X["H"]) + (5 * X["I"]) + self.rng.normal(0, 2, n) + (1.5 * E)
        act_curve = np.clip(base["act"] - (0.3 * X["I"]) + act_noise - (0.05 * E), 0.1, 1.0)
        milk_curve = np.clip(base["milk"] - (6.0 * X["I"]) - (4.0 * X["H"]) + self.rng.normal(0, 1.5, n) - (1.5 * E), 0.0, 50.0)
        cond_curve = 5.0 + self.rng.normal(0, 0.2, n) + (0.4 * E)
        
        severity_label = np.clip(X["I"] + (X["H"]*0.5), 0, 3.0)

        df = pd.DataFrame({
            "animalId": [aid]*n,
            "timestamp": pd.date_range("2024-03-01", periods=n, freq=f"{TICK_MIN}min"),
            "temperature_C": temp_curve, "heartRate_bpm": hr_curve, "respiration_bpm": resp_curve, "activity_index": act_curve,
            "thi": thi, "ambientTemp_C": ambient, "humidity_pct": humidity,
            "milkYield": milk_curve, "feedIntake": 22 - (X["I"]*5), "conductivity": cond_curve,
            "antibioticActive": 0,
            "infectionBinary": (X["I"] > 0.4).astype(int),
            "heatStressBinary": (X["H"] > 0.5).astype(int),
            "mastitisBinary": 0, "lamenessBinary": 0, "calvingBinary": 0,
            "severityLevel": severity_label
        })
        
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
    STRIDE = 72 # evaluate every 12 hours for speed
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
    logger.info("🍂 Phase 14.5 — Long-Horizon 90-Day Drift Assessment")
    logger.info("=" * 60)
    
    model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
    scalers_path = os.path.join(MODEL_DIR, "v13_scalers.json")
    
    if not os.path.exists(model_path):
        logger.error("V13 model required.")
        return
        
    logger.info(f"Generating Massive 90-Day Seasonal Sequences ({TICKS_PER_ANIMAL} ticks per cow)...")
    sim = LongHorizonSimulationUniverse(seed_offset=800)
    farm_dfs = [sim.generate_animal(i) for i in range(N_ANIMALS_PER_FARM)]
    farm_df = pd.concat(farm_dfs, ignore_index=True)
    
    logger.info("Extracting rolling feature vectors (This takes a moment on 90-day sequences)...")
    X_mat, Y_cls, input_dim = extract_features_from_df(farm_df, scalers_path)
    
    device = get_device()
    model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    batch_size = 256
    y_pred_cls = []
    
    logger.info(f"Scanning through {len(X_mat)} continuous sliding evaluation windows...")
    with torch.no_grad():
        for i in range(0, len(X_mat), batch_size):
            x_batch = torch.tensor(X_mat[i:i+batch_size], dtype=torch.float32).to(device)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits_cls, _, _, _ = model(x_batch)
            probs = torch.sigmoid(logits_cls).float().cpu().numpy()
            y_pred_cls.append(probs)
            
    Y_pred = np.vstack(y_pred_cls)
    
    t_inf = Y_cls[:, 0]
    p_inf = Y_pred[:, 0]
    
    pred_labels = (p_inf > 0.65).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(t_inf, pred_labels, average='binary', zero_division=0)
    
    false_positives_total = np.sum((pred_labels == 1) & (t_inf == 0))
    # Calculate FP rate normalized over 90 straight days per ~100 cows
    # We evaluated 10 cows over 90 days = 900 cow-days. 
    # FP / 900 cow-days = FP / (128.5 weeks of cow exposure)
    fp_per_week_100cows = (false_positives_total / (N_ANIMALS_PER_FARM * 90 / 7)) * 100
    
    logger.info("\n--- 90-DAY SHIFT METRICS ---")
    logger.info(f"False Positives Triggered over 90 Continuous Days: {false_positives_total}")
    logger.info(f"Target FP Bounding: {fp_per_week_100cows:.2f} FP / 100-cows / week (Must be <= 5.0)")
    
    if fp_per_week_100cows <= 5.0:
        logger.info("✅ SUCCESS: The Model intelligently ignored massive ambient baseline creep across 3 months. False Positive bounds held.")
    else:
        logger.warning(f"❌ FAILURE: False Positives exponentiated due to slow seasonal drift ({fp_per_week_100cows:.2f}/wk)")
        
if __name__ == "__main__":
    main()
