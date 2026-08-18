#!/usr/bin/env python3
"""
hybrid_alert_engine_v8.py — Phase 8 Part 1
Gated ensemble: disease classifier + onset predictor → priority-tiered alerts.

Logic:
  - Clinical Alert (P1): disease_prob > threshold
  - Early Warning (P2): sustained onset + rising disease + instability
  - Monitor (P3): elevated onset only
  
FP reduction: requires sustained 3-tick onset AND rising disease momentum.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from v5_config import ALL_FEATURES as V5_FEATURES, DATA_DIR, MODEL_DIR
from split_strategy_v2 import animal_time_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("HybridV8")

TICK_HOURS = 5 / 60
TICKS_PER_WEEK = 2016
CLINICAL_THRESHOLD = 0.05  # From v5 threshold optimizer


def run_hybrid_engine(disease_prob, onset_prob, instability, severity, animal_ids):
    """Run hybrid gated alert engine on test data."""
    n = len(disease_prob)
    alert_level = np.zeros(n, dtype=int)  # 0=none, 1=clinical, 2=early_warning, 3=monitor

    # Per-animal buffers
    animal_buffers = {}

    for i in range(n):
        aid = animal_ids[i]
        dp = disease_prob[i]
        op = onset_prob[i]
        inst = instability[i]

        if aid not in animal_buffers:
            animal_buffers[aid] = {
                'disease_history': [], 'onset_history': [],
                'instability_history': []
            }
        buf = animal_buffers[aid]
        buf['disease_history'].append(dp)
        buf['onset_history'].append(op)
        buf['instability_history'].append(inst)

        # Keep last 6 ticks (30 min)
        for k in ['disease_history', 'onset_history', 'instability_history']:
            if len(buf[k]) > 6:
                buf[k] = buf[k][-6:]

        # Step 1: Rising disease momentum
        disease_rising = False
        if len(buf['disease_history']) >= 4:
            slope = dp - buf['disease_history'][-4]
            disease_rising = slope > 0.05

        # Step 2: Sustained onset (3+ of last 3 ticks > 0.7)
        recent_onset = buf['onset_history'][-3:] if len(buf['onset_history']) >= 3 else buf['onset_history']
        sustained_onset = sum(1 for x in recent_onset if x > 0.7) >= 3

        # Step 3: Instability rising
        instability_rising = False
        if len(buf['instability_history']) >= 3:
            inst_slope = inst - buf['instability_history'][-3]
            instability_rising = inst_slope > 0

        # Step 4: Clinical alert
        if dp > CLINICAL_THRESHOLD:
            alert_level[i] = 1  # P1 Clinical

        # Step 5: Early warning (gated)
        elif op > 0.7 and sustained_onset and disease_rising and instability_rising:
            alert_level[i] = 2  # P2 Early Warning

        # Monitor
        elif op > 0.5 and (disease_rising or sustained_onset):
            alert_level[i] = 3  # P3 Monitor

    return alert_level


def evaluate_alerts(alert_level, severity, animal_ids, label="Hybrid"):
    """Evaluate early detection and FP metrics for alert system."""
    # Map any alert (P1 or P2) as "detection alert"
    detection_alerts = (alert_level <= 2) & (alert_level > 0)  # P1 or P2
    early_alerts = alert_level == 2  # P2 only (early warning)
    any_alert = alert_level > 0  # Any alert

    d24 = 0; d12 = 0; d6 = 0; eps = 0
    fp_clinical = 0; fp_early = 0; fp_total = 0
    cw = 0; leads = []

    for aid in np.unique(animal_ids):
        m = animal_ids == aid
        sev_a = severity[m]
        det_a = detection_alerts[m]
        any_a = any_alert[m]
        n = m.sum()

        # Disease episodes
        in_ep = False
        for i in range(len(sev_a)):
            if sev_a[i] >= 2 and not in_ep:
                in_ep = True; eps += 1
                # Look back 48h for first detection alert
                look_back = min(i, 576)
                window = det_a[max(0, i - look_back):i]
                if window.any():
                    first = np.where(window)[0][0]
                    lead_h = (len(window) - first) * TICK_HOURS
                    leads.append(lead_h)
                    if lead_h >= 24: d24 += 1
                    if lead_h >= 12: d12 += 1
                    if lead_h >= 6: d6 += 1
            elif sev_a[i] < 2:
                in_ep = False

        # False positives: any alert when healthy
        fp_total += int((any_a & (sev_a < 0.5)).sum())
        cw += n / TICKS_PER_WEEK

    fp_wk = fp_total / max(cw, 1)
    p24 = d24 / max(eps, 1) * 100
    p12 = d12 / max(eps, 1) * 100
    p6 = d6 / max(eps, 1) * 100
    avg_lead = float(np.mean(leads)) if leads else 0

    results = {
        "episodes": int(eps), "detected": len(leads),
        "pct_24h": round(p24, 1), "pct_12h": round(p12, 1),
        "pct_6h": round(p6, 1), "avg_lead_h": round(avg_lead, 1),
        "fp_per_week": round(float(fp_wk), 2),
        "fp_total": int(fp_total),
        "alerts": {
            "clinical_p1": int((alert_level == 1).sum()),
            "early_warning_p2": int((alert_level == 2).sum()),
            "monitor_p3": int((alert_level == 3).sum()),
        },
        "pass_24h": p24 >= 35, "pass_fp": fp_wk <= 5,
    }
    return results


def economic_eval(alert_level, severity, animal_ids):
    """Economic impact using hybrid alerts."""
    alerts = (alert_level <= 2) & (alert_level > 0)
    total_no = 0; total_with = 0; treatments = 0; fa = 0
    decay = 0.5 ** (1 / (6 / TICK_HOURS))

    for aid in np.unique(animal_ids):
        m = animal_ids == aid
        sev_a = severity[m].astype(float)
        alert_a = alerts[m]
        in_ep = False; ep_start = None

        for i in range(len(sev_a)):
            if sev_a[i] > 0.5 and not in_ep:
                ep_start = i; in_ep = True
            elif sev_a[i] <= 0.1 and in_ep:
                if i - ep_start > 12:
                    ep_sev = sev_a[ep_start:i]
                    nl = float((ep_sev * 0.5 * TICK_HOURS).sum())
                    total_no += nl
                    ea = alert_a[ep_start:i]
                    if ea.any():
                        fi = np.where(ea)[0][0]
                        si = ep_sev.copy()
                        for k in range(fi, len(si)): si[k] *= decay ** (k - fi)
                        total_with += float((si * 0.5 * TICK_HOURS).sum())
                        treatments += 1
                    else:
                        total_with += nl
                in_ep = False
        fa += int((alert_a & (sev_a < 0.5)).sum())

    saved = total_no - total_with
    pct = saved / max(total_no, 0.001) * 100
    return {"pct_reduction": round(pct, 1), "milk_saved_L": round(saved, 2),
            "treatments": treatments, "pass": pct >= 45}


def main():
    logger.info("=" * 60)
    logger.info("🔀 Phase 8 Part 1 — Hybrid Gated Alert Engine")
    logger.info("=" * 60)

    # Load both models
    disease_model = joblib.load(os.path.join(MODEL_DIR, "model_v5.pkl"))
    onset_model = joblib.load(os.path.join(MODEL_DIR, "onset_model_v6.pkl"))
    iso = joblib.load(os.path.join(MODEL_DIR, "isotonic_cal_v6.pkl"))

    # Load v6 features (has both v5 + acceleration features)
    df = pd.read_csv(os.path.join(DATA_DIR, "features_v6.csv"))
    meta = {"animal_id", "disease_binary", "severity_level", "onset_binary",
            "temporal_weight", "timestamp"}
    all_fcols = [c for c in df.columns if c not in meta]
    v5_fcols = [c for c in V5_FEATURES if c in df.columns]
    logger.info(f"Loaded: {len(df)} rows, {len(all_fcols)} v6 features, {len(v5_fcols)} v5 features")

    # Split
    _, X_test_v6, _, y_test, _ = animal_time_split(
        df, all_fcols, "onset_binary",
        animal_train_ratio=0.8, time_train_ratio=0.7, seed=42)

    X_test_v5 = X_test_v6[v5_fcols] if all(c in X_test_v6.columns for c in v5_fcols) else X_test_v6[v5_fcols[:len(v5_fcols)]]

    # Predictions
    disease_prob = disease_model.predict_proba(X_test_v5)[:, 1]
    onset_prob_raw = onset_model.predict_proba(X_test_v6)[:, 1]
    onset_prob = iso.predict(onset_prob_raw)

    # Instability index
    instability = X_test_v6["temp_instability"].values if "temp_instability" in X_test_v6.columns else np.ones(len(X_test_v6))

    severity = df.loc[y_test.index, "severity_level"].values
    animal_ids = df.loc[y_test.index, "animal_id"].values

    logger.info(f"Disease prob: min={disease_prob.min():.4f} max={disease_prob.max():.4f}")
    logger.info(f"Onset prob: min={onset_prob.min():.4f} max={onset_prob.max():.4f}")

    # Run hybrid engine
    alert_level = run_hybrid_engine(disease_prob, onset_prob, instability, severity, animal_ids)

    # Evaluate
    results = evaluate_alerts(alert_level, severity, animal_ids)
    econ = economic_eval(alert_level, severity, animal_ids)

    logger.info(f"\n── Hybrid Alert Results ──")
    logger.info(f"  Episodes: {results['episodes']}")
    logger.info(f"  Detected: {results['detected']} ({results['pct_24h']}% ≥24h)")
    logger.info(f"  FP/week: {results['fp_per_week']}")
    logger.info(f"  Alerts: P1={results['alerts']['clinical_p1']}, "
                f"P2={results['alerts']['early_warning_p2']}, P3={results['alerts']['monitor_p3']}")
    logger.info(f"  Economic: {econ['pct_reduction']}% reduction")

    # Save
    report = {"hybrid_alerts": results, "economic": econ,
              "clinical_threshold": CLINICAL_THRESHOLD}
    with open(os.path.join(DATA_DIR, "hybrid_alert_log_v8.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  ✅ hybrid_alert_log_v8.json saved")


if __name__ == "__main__":
    main()
