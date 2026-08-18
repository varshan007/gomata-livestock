#!/usr/bin/env python3
"""
evaluate_alien_recalibration_v15.py — Phase 15.1 Part 3
OPERATIONAL THRESHOLD DEFINITION & CROSS-DISTRIBUTION CALIBRATION

This script:
1. Generates fresh Alien Physics data (completely unseen by V14)
2. Runs V14 Domain-Randomized model inference
3. Splits into calibration set + evaluation set
4. Applies Isotonic Regression recalibration per-head
5. Solves for PRODUCTION METRIC: Max Recall under FP ≤ 5/week
6. Computes ECE, Brier Score, and full per-head operational report
"""

import os, sys, json, logging, time, gc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             brier_score_loss, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("AlienRecalV15")

sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

# ═══════════════════════════════════════════════════════════════
# ALIEN PHYSICS GENERATOR (Same as Phase 15 — totally foreign)
# ═══════════════════════════════════════════════════════════════

TICK_MIN = 10
TICKS_PER_DAY = 144
N_ANIMALS = 60
TICKS_PER_ANIMAL = 288 * 5

class AlienPhysicsGen:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def generate(self, idx):
        n = TICKS_PER_ANIMAL
        aid = f"AlienEval_{idx:03d}"
        tb = np.zeros(n); tb[0] = 38.5 + self.rng.normal(0, 0.4)
        hb = np.zeros(n); hb[0] = 65 + self.rng.normal(0, 5)
        for t in range(1, n):
            tb[t] = tb[t-1] + self.rng.normal(0, 0.05)
            if self.rng.random() < 0.005: tb[t] += self.rng.normal(0, 0.5)
            hb[t] = hb[t-1] + self.rng.normal(0, 0.5)
            if self.rng.random() < 0.01: hb[t] += self.rng.normal(0, 5)

        ambient = 20 + 8*np.sin(np.arange(n)*2*np.pi/TICKS_PER_DAY) + self.rng.normal(0, 3, n)
        humidity = 60 + 10*np.cos(np.arange(n)*2*np.pi/TICKS_PER_DAY) + self.rng.normal(0, 4, n)
        thi = (1.8*ambient+32) - ((0.55-0.0055*humidity)*(1.8*ambient-26))

        inf = np.zeros(n)
        if self.rng.random() < 0.35:
            s = self.rng.randint(50, n-200); d = self.rng.randint(100, 300); e = min(s+d, n)
            inf[s:e] = np.exp(-np.linspace(0, 1.5, d))[:e-s]
        mast = np.zeros(n)
        if self.rng.random() < 0.30:
            s = self.rng.randint(50, n-200); d = self.rng.randint(80, 250); e = min(s+d, n)
            ramp = np.cumsum(self.rng.uniform(0.005, 0.02, d))
            mast[s:e] = np.clip(ramp[:e-s], 0, 1)
        lame = np.zeros(n)
        if self.rng.random() < 0.20:
            s = self.rng.randint(50, n-400); lame[s:] = np.linspace(0.2, 1.0, n-s)
        heat = (thi > 75).astype(float)
        calv = np.zeros(n)
        if self.rng.random() < 0.12:
            s = self.rng.randint(200, n-200)
            calv[s-100:s+100] = np.exp(-0.5*((np.arange(200)-100)/35)**2)

        temp = tb + 1.5*np.log1p(inf*10) + 0.5*np.exp(heat) + 0.6*mast - 0.4*calv + self.rng.normal(0, 0.2, n)
        hr = hb*(1+0.2*inf)*(1+0.15*heat)*(1+0.12*calv) + self.rng.normal(0, 2, n)
        resp = 22 + (12*heat)**1.2 + 3*inf + 10*calv + self.rng.normal(0, 3, n)
        cn = calv*self.rng.normal(0, 0.3, n)
        act = np.clip(0.7*(1-0.6*lame)*(1-0.3*inf) + cn + self.rng.normal(0, 0.05, n), 0, 1)
        mb = 32 + self.rng.normal(0, 4)
        milk = np.clip(mb*(1-0.4*mast)*(1-0.15*inf)*(1-0.1*heat) + self.rng.normal(0, 1.5, n), 0, None)
        feed = 22*(1-0.25*inf)*(1-0.15*lame) + self.rng.normal(0, 1, n)
        cond = 5 + np.exp(mast*1.5) - 1 + self.rng.normal(0, 0.3, n)
        sev = inf + mast + lame + heat

        return pd.DataFrame({"animalId": [aid]*n,
            "timestamp": pd.date_range("2025-06-01", periods=n, freq=f"{TICK_MIN}min"),
            "temperature_C": temp, "heartRate_bpm": hr, "respiration_bpm": resp, "activity_index": act,
            "thi": thi, "ambientTemp_C": ambient, "humidity_pct": humidity,
            "milkYield": milk, "feedIntake": feed, "conductivity": cond,
            "antibioticActive": 0,
            "infectionBinary": (inf > 0.4).astype(int), "heatStressBinary": (heat > 0.5).astype(int),
            "mastitisBinary": (mast > 0.4).astype(int), "lamenessBinary": (lame > 0.4).astype(int),
            "calvingBinary": (calv > 0.5).astype(int), "severityLevel": sev})

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION + INFERENCE
# ═══════════════════════════════════════════════════════════════

def extract_and_infer(df, scalers_path, model_path):
    windows = {"6h": 36, "12h": 72, "24h": 144}
    for w, ticks in windows.items():
        for col in ["temperature_C","heartRate_bpm","respiration_bpm","activity_index","milkYield","conductivity"]:
            df[f"{col}_mean_{w}"] = df.groupby("animalId")[col].transform(lambda x: x.rolling(ticks, min_periods=1).mean())
            df[f"{col}_std_{w}"] = df.groupby("animalId")[col].transform(lambda x: x.rolling(ticks, min_periods=1).std().fillna(0))
            df[f"{col}_delta_{w}"] = df[col] - df[f"{col}_mean_{w}"]
    df["thermal_strain_index"] = df["heartRate_bpm_delta_6h"] / (df["thi"] - 72 + 1e-5)
    df["lameness_suppression"] = (1.0 - df["activity_index"]) * (df["feedIntake"] / 22.0)
    df["mastitis_spike_index"] = df["conductivity_delta_12h"] * (df["milkYield"] / 30.0)
    df["fever_decoupled"] = df["temperature_C"] - (38.5 + (0.01 * np.maximum(df["thi"].values - 72, 0)))
    df.fillna(0, inplace=True)

    with open(scalers_path, "r") as f:
        sd = json.load(f)
    features = sd["features"]
    scaler = StandardScaler()
    scaler.mean_ = np.array(sd["means"]); scaler.scale_ = np.array(sd["scales"])
    df[features] = scaler.transform(df[features])

    SEQ_LEN = 288; STRIDE = 12; HORIZON = 144
    X_list, Y_list = [], []
    for _, grp in df.groupby("animalId"):
        vf = grp[features].values.astype(np.float32)
        vc = grp[["infectionBinary","heatStressBinary","mastitisBinary","lamenessBinary","calvingBinary"]].values.astype(np.float32)
        nt = len(vf)
        if nt < SEQ_LEN + HORIZON: continue
        for s in range(0, nt - SEQ_LEN - HORIZON, STRIDE):
            e = s + SEQ_LEN
            X_list.append(vf[s:e])
            Y_list.append(vc[e:e+HORIZON].max(axis=0))

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    logger.info(f"Sliding windows: {len(X)}")

    device = get_device()
    model = SharedAttentionHazardEngine(input_dim=len(features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i:i+256], dtype=torch.float32).to(device)
            lc, sv, hz, aw = model(xb)
            preds.append(torch.sigmoid(lc).float().cpu().numpy())
    P = np.vstack(preds)
    return Y, P

# ═══════════════════════════════════════════════════════════════
# CALIBRATION & THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() == 0: continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
    return ece

def find_threshold_at_fp_budget(y_true, y_prob, max_fp_per_week, n_windows_per_week):
    """Find threshold that maximizes recall while keeping FP ≤ budget."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh = 0.5
    best_recall = 0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        fp = ((preds == 1) & (y_true == 0)).sum()
        fp_rate_per_week = fp / (len(y_true) / n_windows_per_week) if len(y_true) > 0 else 0
        if fp_rate_per_week <= max_fp_per_week:
            tp = ((preds == 1) & (y_true == 1)).sum()
            total_pos = y_true.sum()
            recall = tp / total_pos if total_pos > 0 else 0
            if recall > best_recall:
                best_recall = recall
                best_thresh = t
    return best_thresh, best_recall

def main():
    logger.info("=" * 70)
    logger.info("🔬 Phase 15.1 — ALIEN RECALIBRATION & OPERATIONAL THRESHOLD EVAL")
    logger.info("=" * 70)

    model_path = os.path.join(MODEL_DIR, "v14_domain_attention_model.pth")
    scalers_path = os.path.join(MODEL_DIR, "v14_domain_scalers.json")

    if not os.path.exists(model_path):
        logger.error("V14 model not found.")
        return

    # ── STEP 1: Generate Fresh Alien Physics ──
    logger.info("Step 1: Generating 60 Alien Physics animals (seed=7777)...")
    sim = AlienPhysicsGen(seed=7777)
    dfs = [sim.generate(i) for i in range(N_ANIMALS)]
    df = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()

    # ── STEP 2: Run V14 Inference ──
    logger.info("Step 2: Running V14 Domain-Randomized model inference...")
    Y_true, Y_prob_raw = extract_and_infer(df, scalers_path, model_path)
    del df; gc.collect()

    DISEASES = ["Infection", "HeatStress", "Mastitis", "Lameness", "Calving"]
    n_samples = len(Y_true)

    # ── STEP 3: Split Calibration / Evaluation (50/50) ──
    logger.info("Step 3: Splitting into Calibration (50%) / Evaluation (50%)...")
    idx = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(idx)
    cal_idx = idx[:n_samples // 2]
    eval_idx = idx[n_samples // 2:]

    Y_cal = Y_true[cal_idx]; P_cal_raw = Y_prob_raw[cal_idx]
    Y_eval = Y_true[eval_idx]; P_eval_raw = Y_prob_raw[eval_idx]

    # ── STEP 4: Isotonic Recalibration Per-Head ──
    logger.info("Step 4: Isotonic Regression recalibration per disease head...")
    isotonic_models = {}
    P_eval_calibrated = np.zeros_like(P_eval_raw)

    for i, disease in enumerate(DISEASES):
        if len(np.unique(Y_cal[:, i])) < 2:
            logger.warning(f"  {disease}: Skipped (single class in calibration set)")
            P_eval_calibrated[:, i] = P_eval_raw[:, i]
            continue
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(P_cal_raw[:, i], Y_cal[:, i])
        isotonic_models[disease] = iso
        P_eval_calibrated[:, i] = iso.predict(P_eval_raw[:, i])
        logger.info(f"  {disease}: Isotonic fitted (cal samples: {len(Y_cal)})")

    # ── STEP 5: Comprehensive Metrics ──
    logger.info("=" * 70)
    logger.info("📊 EVALUATION RESULTS — V14 Domain-Randomized on Alien Physics")
    logger.info("=" * 70)

    # Assume 1 week ≈ 1008 ticks (7 days * 144 ticks/day), stride 12 → ~84 windows/day → 588/week
    WINDOWS_PER_WEEK = 588
    FP_BUDGET = 5

    report = {}
    for i, disease in enumerate(DISEASES):
        t = Y_eval[:, i]
        p_raw = P_eval_raw[:, i]
        p_cal = P_eval_calibrated[:, i]

        # AUC
        auc_raw = roc_auc_score(t, p_raw) if len(np.unique(t)) > 1 else 0
        auc_cal = roc_auc_score(t, p_cal) if len(np.unique(t)) > 1 else 0

        # ECE
        ece_raw = compute_ece(t, p_raw)
        ece_cal = compute_ece(t, p_cal)

        # Brier Score
        brier_raw = brier_score_loss(t, p_raw)
        brier_cal = brier_score_loss(t, p_cal)

        # Fixed threshold recall (0.5)
        pred_fixed = (p_cal > 0.5).astype(int)
        prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(t, pred_fixed, average='binary', zero_division=0)

        # Operational threshold: Max Recall under FP ≤ 5/week
        opt_thresh, opt_recall = find_threshold_at_fp_budget(t, p_cal, FP_BUDGET, WINDOWS_PER_WEEK)
        pred_opt = (p_cal >= opt_thresh).astype(int)
        fp_opt = ((pred_opt == 1) & (t == 0)).sum()
        fp_per_week_opt = fp_opt / (len(t) / WINDOWS_PER_WEEK)

        report[disease] = {
            "AUC_raw": auc_raw, "AUC_calibrated": auc_cal,
            "ECE_raw": ece_raw, "ECE_calibrated": ece_cal,
            "Brier_raw": brier_raw, "Brier_calibrated": brier_cal,
            "Recall@0.5": rec_f, "F1@0.5": f1_f,
            "OptimalThreshold": opt_thresh, "Recall@FPbudget": opt_recall,
            "FP/week@OptThresh": fp_per_week_opt, "Positives": int(t.sum()),
            "Total": len(t)
        }

    # Print Report
    logger.info("")
    logger.info(f"{'Disease':12} | {'AUC_raw':>8} | {'AUC_cal':>8} | {'ECE_raw':>8} | {'ECE_cal':>8} | {'Brier_r':>8} | {'Brier_c':>8} | {'Rec@0.5':>8} | {'OptThresh':>9} | {'Rec@FP≤5':>9} | {'FP/wk':>6}")
    logger.info("-" * 135)
    for d, m in report.items():
        logger.info(f"{d:12} | {m['AUC_raw']:8.4f} | {m['AUC_calibrated']:8.4f} | {m['ECE_raw']:8.4f} | {m['ECE_calibrated']:8.4f} | {m['Brier_raw']:8.4f} | {m['Brier_calibrated']:8.4f} | {m['Recall@0.5']:8.4f} | {m['OptimalThreshold']:9.4f} | {m['Recall@FPbudget']:9.4f} | {m['FP/week@OptThresh']:6.2f}")

    # Summary
    avg_auc_cal = np.mean([m['AUC_calibrated'] for m in report.values() if m['AUC_calibrated'] > 0])
    avg_ece_cal = np.mean([m['ECE_calibrated'] for m in report.values()])
    avg_recall_op = np.mean([m['Recall@FPbudget'] for m in report.values()])

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"AVERAGE CALIBRATED AUC:       {avg_auc_cal:.4f}")
    logger.info(f"AVERAGE CALIBRATED ECE:       {avg_ece_cal:.4f}")
    logger.info(f"AVERAGE RECALL @ FP≤5/week:   {avg_recall_op:.4f}")
    logger.info("=" * 70)

    # ── Verdict ──
    if avg_auc_cal >= 0.80 and avg_recall_op >= 0.35:
        logger.info("✅ OPERATIONAL GENERALIZATION ACHIEVED — System is Commercial Production-Ready.")
    elif avg_auc_cal >= 0.80:
        logger.info("⚠️ Structural AUC holds but Operational Recall still below 35%. Threshold tuning needed.")
    else:
        logger.info("❌ AUC dropped below 0.80 — backbone needs retraining.")

    # Save JSON report
    report_path = os.path.join(MODEL_DIR, "alien_recalibration_report_v15.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved → {report_path}")

if __name__ == "__main__":
    main()
