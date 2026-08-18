#!/usr/bin/env python3
"""
stress_test_v14_farms.py — Phase 14
Cross-Seed Stochastic & Farm Physics Shift Validation.

Generates 5 distinct "Farms" representing alternate realities
with massive shifts in AR(1) thermal regimes, disease prevalence,
and physical masking patterns. Tests the Pytorch V13 Engine.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StressV14_Farms")

# Import the PyTorch V13 model architecture
sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

N_ANIMALS_PER_FARM = 30
TICKS_PER_ANIMAL = 5000  
TICK_MIN = 10
TICKS_PER_HOUR = int(60 / TICK_MIN)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR
THI_THRESHOLD = 72.0


class FarmSimulationUniverse:
    def __init__(self, farm_name, seed_offset, params):
        self.farm_name = farm_name
        self.rng = np.random.RandomState(seed_offset)
        self.p_lambd = params.get("env_lambd", 0.05)
        self.p_sigma = params.get("env_sigma", 0.8)
        self.p_infection = params.get("prev_inf", 0.25)
        self.p_mastitis = params.get("prev_mast", 0.20)
        self.p_lameness = params.get("prev_lame", 0.15)
        self.p_ambiguous = params.get("pct_ambiguous", 0.25)
        self.p_heat_waves = params.get("heat_variance", 1.0)
        
    def generate_animal(self, animal_idx):
        n = TICKS_PER_ANIMAL
        aid = f"{self.farm_name}_animal_{animal_idx:04d}"
        
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
        ambient = 22 + 10 * np.sin(t_arr * 2 * np.pi / TICKS_PER_DAY) + self.rng.normal(0, 2, n)
        humidity = 55 + 15 * np.sin(t_arr * 2 * np.pi / (TICKS_PER_DAY * 3)) + self.rng.normal(0, 5, n)
        
        # Apply farm-specific variance to heatwaves
        hw_shift = self.rng.uniform(8, 15) * self.p_heat_waves
        ambient[int(0.4*n):int(0.6*n)] += hw_shift
        
        thi = (1.8 * ambient + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * ambient - 26))
        thi += self.rng.normal(0, 1.5, n)
        
        # Farm-specific Environmental Latent Chaos (dE = -λE dt + σdW)
        E = np.zeros(n)
        for t in range(1, n):
            E[t] = E[t-1] - self.p_lambd * E[t-1] + self.rng.normal(0, self.p_sigma)
            
        ambiguous_case = self.rng.random() < self.p_ambiguous
        transport_stress = np.zeros(n)
        if self.rng.random() < 0.2:
            ts_start = self.rng.randint(int(0.1*n), int(0.6*n))
            transport_stress[ts_start:ts_start+144] = self.rng.uniform(1.0, 3.0) 

        fake_hr_spike = np.zeros(n)
        fake_cond_spike = np.zeros(n)
        if ambiguous_case:
            for _ in range(3):
                st = self.rng.randint(0, n-20)
                fake_hr_spike[st:st+self.rng.randint(10, 30)] = self.rng.uniform(15, 35)
            for _ in range(2):
                st = self.rng.randint(0, n-100)
                fake_cond_spike[st:st+self.rng.randint(50, 100)] = self.rng.uniform(1.5, 3.5)
                
        X = { "I": np.zeros(n), "H": np.zeros(n), "M": np.zeros(n), "L": np.zeros(n), "C": np.zeros(n), "Imm": np.ones(n), "Comp": np.ones(n), "Fat": np.zeros(n) }

        has_infection = self.rng.random() < self.p_infection
        has_mastitis = self.rng.random() < self.p_mastitis
        has_lameness = self.rng.random() < self.p_lameness
        is_calving = self.rng.random() < 0.10

        infection_start = self.rng.randint(int(0.1*n), int(0.8*n)) if has_infection else -1
        primary_mastitis_start = self.rng.randint(int(0.1*n), int(0.8*n)) if has_mastitis else -1
        lameness_start = self.rng.randint(int(0.1*n), int(0.8*n)) if has_lameness else -1
        calving_midpoint = self.rng.randint(int(0.4*n), int(0.9*n)) if is_calving else -1

        abx_active = 0
        hoof_trm = 0
        exposure = np.zeros(n)
        if has_infection: exposure[infection_start:infection_start + 144] = self.rng.uniform(0.01, 0.03)

        for t in range(1, n):
            if is_calving:
                days_to_calving = (t - calving_midpoint) / TICKS_PER_DAY
                X["C"][t] = 1.0 / (1.0 + np.exp(-2.0 * days_to_calving))

            thi_excess = max(thi[t] - THI_THRESHOLD, 0)
            dH = (0.01 * thi_excess) - (0.05 * base["heat_tolerance"]) + self.rng.normal(0, 0.01)
            X["H"][t] = np.clip(X["H"][t-1] + dH, 0, 1.5)

            dFat = (0.02 * X["H"][t]) + (0.05 * X["I"][t-1]) + (0.03 * X["C"][t]) - 0.01 + (0.02 * transport_stress[t])
            X["Fat"][t] = np.clip(X["Fat"][t-1] + dFat, 0, 1.0)
            
            dImm = - (0.08 * X["Fat"][t]) - (0.05 * X["H"][t]) + 0.01
            X["Imm"][t] = np.clip(X["Imm"][t-1] + dImm, 0.1, 1.0)

            dI = exposure[t-1] + (0.05 * X["I"][t-1]) - (0.04 * X["Imm"][t]) - (0.2 * abx_active) + self.rng.normal(0, 0.005)
            X["I"][t] = np.clip(X["I"][t-1] + dI, 0, 1.0)

            primary_m = 0.05 if (primary_mastitis_start > 0 and t > primary_mastitis_start and t < primary_mastitis_start + 200) else 0
            secondary_m = 0.08 * max(X["I"][t] - 0.6, 0)
            dM = primary_m + secondary_m + (0.02 * X["M"][t-1]) - (0.05 * X["Imm"][t]) - (0.25 * abx_active) + self.rng.normal(0, 0.005)
            X["M"][t] = np.clip(X["M"][t-1] + dM, 0, 1.0)

            primary_l = 0.03 if (lameness_start > 0 and t > lameness_start and t < lameness_start + 500) else 0
            overlap_calving_lameness = 0.05 * X["C"][t]
            dL = primary_l + overlap_calving_lameness + (0.01 * X["L"][t-1]) - hoof_trm + self.rng.normal(0, 0.002)
            X["L"][t] = np.clip(X["L"][t-1] + dL, 0, 1.0)

            if (X["I"][t] > 0.8 or X["M"][t] > 0.7) and abx_active == 0 and self.rng.random() < 0.1: abx_active = 1.0
            if abx_active > 0: abx_active *= 0.95
            if X["L"][t] > 0.8 and hoof_trm == 0 and self.rng.random() < 0.05: hoof_trm = 0.5
            if hoof_trm > 0: hoof_trm *= 0.90
            
        temp_noise = np.zeros(n); hr_noise = np.zeros(n); resp_noise = np.zeros(n); act_noise = np.zeros(n)
        rho = 0.85
        for t in range(1, n):
            temp_noise[t] = rho * temp_noise[t-1] + self.rng.normal(0, 0.15)
            hr_noise[t] = rho * hr_noise[t-1] + self.rng.normal(0, 1.5)
            act_noise[t] = rho * act_noise[t-1] + self.rng.normal(0, 0.03)

        temp_curve = base["temp"] + (2.5 * X["I"]) + (1.2 * X["H"]) + (0.4 * X["M"]) - (0.3 * X["C"]) + temp_noise + (0.2 * E) + (0.6 * transport_stress)
        hr_curve = base["hr"] + (20 * X["I"]) + (12 * X["H"]) + (8 * X["C"]) + hr_noise + (2.5 * E) + (18 * transport_stress) + fake_hr_spike
        resp_curve = base["resp"] + (15 * X["H"]) + (5 * X["I"]) + self.rng.normal(0, 2, n) + (1.5 * E) + (8 * transport_stress)
        act_curve = np.clip(base["act"] - (0.5 * X["L"]) - (0.3 * X["I"]) + (X["C"] * self.rng.normal(0, 0.2, n)) + act_noise - (0.05 * E), 0.1, 1.0)
        milk_curve = np.clip(base["milk"] - (6.0 * X["I"]) - (4.0 * X["H"]) - (12.0 * X["M"]) - (2.0 * X["L"]) + self.rng.normal(0, 1.5, n) - (1.5 * E), 0.0, 50.0)
        cond_curve = 5.0 + (3.5 * X["M"]) + self.rng.normal(0, 0.2, n) + (0.4 * E) + fake_cond_spike
        
        severity_label = np.clip(X["I"] + (X["M"]*1.5) + (X["L"]*0.8) + (X["H"]*0.5), 0, 3.0)

        # Build feature DataFrame sequentially
        df = pd.DataFrame({
            "animalId": [aid]*n,
            "timestamp": pd.date_range("2024-07-01", periods=n, freq=f"{TICK_MIN}min"),
            "temperature_C": temp_curve, "heartRate_bpm": hr_curve, "respiration_bpm": resp_curve, "activity_index": act_curve,
            "thi": thi, "ambientTemp_C": ambient, "humidity_pct": humidity,
            "milkYield": milk_curve, "feedIntake": 22 - (X["I"]*5) - (transport_stress*2), "conductivity": cond_curve,
            "antibioticActive": 0, # simplified for testing
            "infectionBinary": (X["I"] > 0.4).astype(int),
            "heatStressBinary": (X["H"] > 0.5).astype(int),
            "mastitisBinary": (X["M"] > 0.4).astype(int),
            "lamenessBinary": (X["L"] > 0.4).astype(int),
            "calvingBinary": (X["C"] > 0.5).astype(int),
            "severityLevel": severity_label
        })
        
        return df


def evaluate_farm(df, farm_name, model_path, scalers_path):
    logger.info(f"Extracting streaming features for {farm_name}...")
    
    # ── COMPUTE FEATURES ──
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
    
    # ── EXTRACT SEQUENCES ──
    SEQ_LEN = 288
    STRIDE = 12
    X_list, Y_cls_list, Y_sev_list = [], [], []
    
    grouped = df.groupby("animalId")
    for _, group in grouped:
        v_features = group[features].values
        v_targets = group[["infectionBinary", "heatStressBinary", "mastitisBinary", "lamenessBinary", "calvingBinary"]].values
        v_sev = group["severityLevel"].values
        
        n_ticks = len(v_features)
        if n_ticks < SEQ_LEN + 144: continue
        
        for start_idx in range(0, n_ticks - (SEQ_LEN + 144), STRIDE):
            end_idx = start_idx + SEQ_LEN
            X_list.append(v_features[start_idx:end_idx])
            Y_cls_list.append(v_targets[end_idx - 1])
            Y_sev_list.append(v_sev[end_idx - 1])
            
    X_mat = np.array(X_list, dtype=np.float32)
    Y_cls = np.array(Y_cls_list, dtype=np.float32)
    logger.info(f"Farm {farm_name} generated {len(X_mat)} sliding evaluation windows.")
    
    # ── EVALUATE PyTorch MODEL ──
    device = get_device()
    model = SharedAttentionHazardEngine(input_dim=len(features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    batch_size = 256
    y_pred_cls = []
    
    with torch.no_grad():
        for i in range(0, len(X_mat), batch_size):
            x_batch = torch.tensor(X_mat[i:i+batch_size], dtype=torch.float32).to(device)
            # Mixed Precision
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
    
    logger.info(f"--- FARM: {farm_name} RESULTS ---")
    logger.info(f"Infect AUC: {m_dict['Infection']['AUC']:.4f} | Heat AUC: {m_dict['HeatStress']['AUC']:.4f}")
    logger.info(f"Mastit AUC: {m_dict['Mastitis']['AUC']:.4f}  | Lame AUC: {m_dict['Lameness']['AUC']:.4f}")
    logger.info(f"Target False Positives: {fp_rate * 7:.2f}/week")
    
    return m_dict

def main():
    logger.info("=" * 60)
    logger.info("🌍 Phase 14 — Cross-Seed & Cross-Farm Matrix")
    logger.info("=" * 60)
    
    # Defines the 5 alternate reality worlds
    FARMS = [
        {"name": "Farm_A_Baseline", "seed": 100, "params": {}},  # Expected highest matching Training Farm
        {"name": "Farm_B_HighChaos", "seed": 200, "params": {"env_lambd": 0.2, "env_sigma": 2.5}}, # Wild physics variations
        {"name": "Farm_C_HighOutbreak", "seed": 300, "params": {"prev_inf": 0.4, "prev_mast": 0.35}}, # Massive prevalence shift
        {"name": "Farm_D_Ambiguous", "seed": 400, "params": {"pct_ambiguous": 0.70}}, # 70% get fake exercise HR spikes
        {"name": "Farm_E_Heatwave", "seed": 500, "params": {"heat_variance": 4.0}} # Summer devastation Farm
    ]
    
    model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
    scalers_path = os.path.join(MODEL_DIR, "v13_scalers.json")
    
    if not os.path.exists(model_path):
        logger.error(f"Missing {model_path}. Please train V13 model first.")
        return

    results = {}
    for farm_config in FARMS:
        fname = farm_config["name"]
        logger.info(f"\n🚀 Generating Alternate Reality: {fname}")
        
        sim = FarmSimulationUniverse(fname, farm_config["seed"], farm_config["params"])
        farm_dfs = [sim.generate_animal(i) for i in range(N_ANIMALS_PER_FARM)]
        farm_df = pd.concat(farm_dfs, ignore_index=True)
        
        metrics = evaluate_farm(farm_df, fname, model_path, scalers_path)
        results[fname] = metrics
        
    logger.info("\n✅ CROSS-FARM MATRIX COMPLETE")
    for fname, metrics in results.items():
        avg_auc = np.mean([metrics[d]["AUC"] for d in metrics])
        logger.info(f"[{fname:20}] Avg Multi-Head AUC: {avg_auc:.4f}")

if __name__ == "__main__":
    main()
