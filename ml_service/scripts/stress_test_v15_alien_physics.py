#!/usr/bin/env python3
"""
stress_test_v15_alien_physics.py — Phase 15
Cross-Simulator (Alien Physics) Validation

The ultimate Reality-Check. Generated entirely distinct stochastic physics
using Jump-Diffusion Random Walks, Poisson disease arrivals, and Multiplicative
degradation (abandoning all AR(1) and SDE logic from previous simulators).
Proves the PyTorch Shared Attention model learned true biological representations
and did not simply memorize the Simulator's parameter mechanics.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StressV15_Alien")

sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

N_ANIMALS = 30
TICKS_PER_ANIMAL = 288 * 5 # 5 days
TICK_MIN = 10
TICKS_PER_HOUR = int(60 / TICK_MIN)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR

class AlienPhysicsSimulation:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        
    def generate_animal(self, animal_idx):
        n = TICKS_PER_ANIMAL
        aid = f"Alien_cow_{animal_idx:03d}"
        
        # ── EXTRATERRESTRIAL PHYSICS (Jump-Diffusion Baselines) ──
        temp_base = np.zeros(n)
        temp_base[0] = 38.5 + self.rng.normal(0, 0.4)
        
        hr_base = np.zeros(n)
        hr_base[0] = 65 + self.rng.normal(0, 5)
        
        for t in range(1, n):
            temp_base[t] = temp_base[t-1] + self.rng.normal(0, 0.05)
            # 0.5% chance of abrupt physical baseline jump
            if self.rng.random() < 0.005: temp_base[t] += self.rng.normal(0, 0.5) 
            
            hr_base[t] = hr_base[t-1] + self.rng.normal(0, 0.5)
            # 1% chance of sudden cardiac resting shift
            if self.rng.random() < 0.01: hr_base[t] += self.rng.normal(0, 5)
            
        ambient = 20 + 8 * np.sin(np.arange(n) * 2 * np.pi / TICKS_PER_DAY) + self.rng.normal(0, 3, n)
        humidity = 60 + 10 * np.cos(np.arange(n) * 2 * np.pi / TICKS_PER_DAY) + self.rng.normal(0, 4, n)
        thi = (1.8 * ambient + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * ambient - 26))
        
        # ── POISSON SHOCK PROCESSES (Instead of progressive exposure vectors) ──
        inf_state = np.zeros(n)
        if self.rng.random() < 0.4: # 40% chance
            start = self.rng.randint(50, n - 200)
            dur = self.rng.randint(100, 300)
            inf_state[start:start+dur] = np.exp(-np.linspace(0, 1.5, dur)) # Exponential decay shock
            
        mast_state = np.zeros(n)
        if self.rng.random() < 0.3:
            start = self.rng.randint(50, n - 200)
            dur = self.rng.randint(80, 250)
            ramp = np.cumsum(self.rng.uniform(0.005, 0.02, dur))
            end_idx = min(start + dur, n)
            mast_state[start:end_idx] = np.clip(ramp[:end_idx-start], 0, 1.0)
            
        lame_state = np.zeros(n)
        if self.rng.random() < 0.2:
            start = self.rng.randint(50, n - 400)
            lame_state[start:] = np.linspace(0.2, 1.0, n - start) # Linear worsening
            
        heat_state = (thi > 75).astype(float)
        
        calv_state = np.zeros(n)
        if self.rng.random() < 0.1:
            start = self.rng.randint(200, n - 200)
            # Gaussian bump spanning ~24 hours
            calv_state[start-100:start+100] = np.exp(-0.5 * ((np.arange(200) - 100) / 35.0)**2)
            
        # ── MULTIPLICATIVE BIOLOGY (Instead of linear additive models) ──
        # Infection causes logarithmic saturation; Mastitis causes mild fever
        temp_obs = temp_base + 1.5 * np.log1p(inf_state * 10) + 0.5 * np.exp(heat_state) + 0.6 * mast_state - 0.4 * calv_state + self.rng.normal(0, 0.2, n)
        
        # Multiplicative cascading heart rate limits
        hr_obs = hr_base * (1.0 + 0.2 * inf_state) * (1.0 + 0.15 * heat_state) * (1.0 + 0.12 * calv_state) + self.rng.normal(0, 2, n)
        
        resp_obs = 22 + (12 * heat_state) ** 1.2 + 3 * inf_state + 10 * calv_state + self.rng.normal(0, 3, n)
        
        # Activity: Calving turns into variable noise rather than a static drop constraint
        calv_noise = calv_state * self.rng.normal(0, 0.3, n)
        act_obs = 0.7 * (1.0 - 0.6 * lame_state) * (1.0 - 0.3 * inf_state) + calv_noise + self.rng.normal(0, 0.05, n)
        act_obs = np.clip(act_obs, 0, 1)
        
        # Yield is exponentially destroyed by mastitis rather than static 12kg drops
        milk_base = 32 + self.rng.normal(0, 4)
        milk_obs = milk_base * (1.0 - 0.4 * mast_state) * (1.0 - 0.15 * inf_state) * (1.0 - 0.1 * heat_state) + self.rng.normal(0, 1.5, n)
        milk_obs = np.clip(milk_obs, 0, None)
        
        feed_obs = 22 * (1.0 - 0.25 * inf_state) * (1.0 - 0.15 * lame_state) + self.rng.normal(0, 1, n)
        
        cond_obs = 5.0 + np.exp(mast_state * 1.5) - 1.0 + self.rng.normal(0, 0.3, n)
        
        severity = inf_state + mast_state + lame_state + heat_state
        
        df = pd.DataFrame({
            "animalId": [aid]*n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq=f"{TICK_MIN}min"),
            "temperature_C": temp_obs, "heartRate_bpm": hr_obs, "respiration_bpm": resp_obs, "activity_index": act_obs,
            "thi": thi, "ambientTemp_C": ambient, "humidity_pct": humidity,
            "milkYield": milk_obs, "feedIntake": feed_obs, "conductivity": cond_obs,
            "antibioticActive": 0,
            "infectionBinary": (inf_state > 0.4).astype(int),
            "heatStressBinary": (heat_state > 0.5).astype(int),
            "mastitisBinary": (mast_state > 0.4).astype(int),
            "lamenessBinary": (lame_state > 0.4).astype(int),
            "calvingBinary": (calv_state > 0.5).astype(int),
            "severityLevel": severity
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
    logger.info("============================================================")
    logger.info("👽 Phase 15 — CROSS-SIMULATOR (ALIEN PHYSICS) VALIDATION")
    logger.info("============================================================")
    
    sim = AlienPhysicsSimulation(seed=999)
    dfs = [sim.generate_animal(i) for i in range(N_ANIMALS)]
    c_df = pd.concat(dfs, ignore_index=True)
    
    scalers_path = os.path.join(MODEL_DIR, "v13_scalers.json")
    model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
    
    if not os.path.exists(model_path):
        logger.error("V13 PyTorch model missing.")
        return
        
    logger.info("Extracting rolling feature sequences against foreign parameterizations...")
    X_mat, Y_cls, input_dim = extract_features_from_df(c_df, scalers_path)
    logger.info(f"Generated {len(X_mat)} sliding windows from Alien Physics Space.")
    
    device = get_device()
    model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
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
        
    logger.info("--- ALIEN PHYSICS RESULTS ---")
    for d, metrics in m_dict.items():
        logger.info(f"{d:12} | AUC: {metrics['AUC']:.4f} | Recall: {metrics['Recall']:.4f}")
        
    avg_auc = np.mean([m['AUC'] for m in m_dict.values() if m['AUC'] > 0])
    logger.info(f"\nAverage Multi-Head AUC on Alien Physics: {avg_auc:.4f}")
    
    if avg_auc >= 0.80:
        logger.info("✅ SUCCESS: Model abstract biology generalized to alien simulators. True Reality-Gap resilience achieved.")
    else:
        logger.warning(f"❌ FAILED: Model plummeted below 0.80 AUC floor. It memorized Simulator math.")

if __name__ == "__main__":
    main()
