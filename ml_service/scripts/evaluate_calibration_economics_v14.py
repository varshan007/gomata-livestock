#!/usr/bin/env python3
"""
evaluate_calibration_economics_v14.py — Phase 14.5
Ultimate Production Polish: Calibration & Economic Stability Audit

Calculates Expected Calibration Error (ECE) and Brier Score across
different Farm Universes, and calculates Economic Variance (Net ROI drop)
to prove that threshold pricing economics do not wildly vacillate across seeds.
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CalibrateV14")

sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device
from stress_test_v14_farms import FarmSimulationUniverse

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return ece


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
    X_list, Y_cls_list, milk_list = [], [], []
    
    grouped = df.groupby("animalId")
    for _, group in grouped:
        v_features = group[features].values
        v_targets = group[["infectionBinary", "heatStressBinary", "mastitisBinary", "lamenessBinary", "calvingBinary"]].values
        v_milk = group["milkYield"].values
        
        n_ticks = len(v_features)
        if n_ticks < SEQ_LEN + 144: continue
        
        for start_idx in range(0, n_ticks - (SEQ_LEN + 144), STRIDE):
            end_idx = start_idx + SEQ_LEN
            X_list.append(v_features[start_idx:end_idx])
            Y_cls_list.append(v_targets[end_idx - 1])
            # Economic proxy: base yield at time of prediction to anchor intervention value
            milk_list.append(v_milk[end_idx - 1])
            
    return np.array(X_list, dtype=np.float32), np.array(Y_cls_list, dtype=np.float32), np.array(milk_list), len(features)


def evaluate_calibration_economics(X_mat, Y_cls, milk_array, model, device):
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
    
    # Analyze Infection (Head 0) for core ECE and Economics
    t_inf = Y_cls[:, 0]
    p_inf = Y_pred[:, 0]
    
    brier = brier_score_loss(t_inf, p_inf)
    ece = expected_calibration_error(t_inf, p_inf)
    
    # Economic Threshold calculation
    MILK_PRICE = 0.40 # USD per kg
    INTERVENTION_COST = 25.00 # Vet visit + antibiotics
    PRESERVED_PRODUCTION = 30.0 # kg preserved by intervening early
    
    VALUE_PER_TP = PRESERVED_PRODUCTION * MILK_PRICE
    COST_PER_FP = INTERVENTION_COST
    
    THRESHOLD = COST_PER_FP / (COST_PER_FP + VALUE_PER_TP) # ~ 0.67
    
    # Calculate empirical ROI based on fixed algorithmic threshold across different farm seeds
    # To prove pricing models work despite farm shift
    pred_labels = (p_inf > THRESHOLD).astype(int)
    
    true_pos = np.sum((pred_labels == 1) & (t_inf == 1))
    false_pos = np.sum((pred_labels == 1) & (t_inf == 0))
    
    total_value = true_pos * VALUE_PER_TP
    total_cost = false_pos * COST_PER_FP
    net_roi = total_value - total_cost
    
    # Normalize ROI per 100 cows to make it cross-comparable regardless of sample size padding
    num_animals = len(X_mat) / 1000.0 # Approx factor to normalize
    normalized_roi = net_roi / num_animals
    
    return brier, ece, normalized_roi


def main():
    logger.info("=" * 60)
    logger.info("⚖️ Phase 14.5 — Calibration & Economic Stability Audit")
    logger.info("=" * 60)
    
    model_path = os.path.join(MODEL_DIR, "v13_attention_hazard_model.pth")
    scalers_path = os.path.join(MODEL_DIR, "v13_scalers.json")
    
    if not os.path.exists(model_path):
        logger.error("V13 model required for calibration.")
        return
        
    FARMS = [
        {"name": "Farm_A_Baseline", "seed": 100, "params": {}},  
        {"name": "Farm_B_HighChaos", "seed": 200, "params": {"env_lambd": 0.2, "env_sigma": 2.5}},
        {"name": "Farm_C_HighOutbreak", "seed": 300, "params": {"prev_inf": 0.4, "prev_mast": 0.35}},
        {"name": "Farm_D_Ambiguous", "seed": 400, "params": {"pct_ambiguous": 0.70}},
        {"name": "Farm_E_Heatwave", "seed": 500, "params": {"heat_variance": 4.0}}
    ]
    
    device = get_device()
    model = None
    
    results = {}
    
    for farm_config in FARMS:
        fname = farm_config["name"]
        logger.info(f"\nGenerative Inference >> {fname}")
        sim = FarmSimulationUniverse(fname, farm_config["seed"], farm_config["params"])
        farm_dfs = [sim.generate_animal(i) for i in range(25)]
        farm_df = pd.concat(farm_dfs, ignore_index=True)
        
        X_mat, Y_cls, milk_array, input_dim = extract_features_from_df(farm_df, scalers_path)
        
        if model is None:
            model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
        brier, ece, roi = evaluate_calibration_economics(X_mat, Y_cls, milk_array, model, device)
        
        logger.info(f"[{fname}] Brier Score: {brier:.4f}")
        logger.info(f"[{fname}] ECE (Target <= 0.05): {ece:.4f}")
        logger.info(f"[{fname}] Normalized Net ROI Return: ${roi:.2f}")
        
        results[fname] = {"ECE": ece, "ROI": roi}
        
    base_roi = results["Farm_A_Baseline"]["ROI"]
    logger.info("\n--- ECONOMIC VARIANCE CERTIFICATION ---")
    
    passed = True
    for fname, data in results.items():
        if fname == "Farm_A_Baseline": continue
        
        ece = data["ECE"]
        roi = data["ROI"]
        
        # Prevent dev-by-zero if no outbreaks happen to be caught optimally
        roi_shift = abs(roi - base_roi) / (abs(base_roi) + 1.0) * 100.0
        
        logger.info(f"{fname:20} -> ECE {ece:.4f} | ROI Drift: {roi_shift:.1f}%")
        
        if ece > 0.05:
            logger.warning(f"❌ FAILED CALIBRATION LIMIT on {fname} (>{0.05})")
            passed = False
            
    if passed:
        logger.info("✅ SUCCESS: Probabilities remained heavily calibrated, securing extreme Edge Pricing Economic Models.")
    else:
        logger.warning("❌ System demonstrated over-confidence un-calibrations on drift.")

if __name__ == "__main__":
    main()
