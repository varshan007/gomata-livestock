#!/usr/bin/env python3
"""
robustness_eval_v2.py — GoMata Model Hardening v2, Phase 4
Automated Robustness Evaluation Loop

Re-runs all validation tests using the TRAINED model:
  1) Reality Gap (sensor drift, missing, temporal delay, misreporting)
  2) Counterfactual Herd Testing (vaccination scenarios)
  3) Intervention Validation (early vs late treatment)

Generates:
  - pilot_validation_v2_report.json
  - robustness_scorecard.csv

Usage:
  python robustness_eval_v2.py
"""

import os, sys, json, time, logging, subprocess
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from robust_augmentation_v2 import (sensor_drift, random_missing, block_missing,
                                     device_offline, timestamp_jitter, mgmt_misreport)

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "robustness_eval_v2.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("RobustnessEval")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")
SCRIPTS_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.join(SCRIPTS_DIR, "../../backend")


# ═══════════════════════════════════════════════════════════════════════════════
# Load model + features
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_config():
    """Load trained robust model and feature config."""
    model_path = os.path.join(MODEL_DIR, "robust_model_v2.pkl")
    config_path = os.path.join(MODEL_DIR, "feature_config_v2.json")

    if not os.path.exists(model_path):
        logger.error("robust_model_v2.pkl not found. Run train_classifier_v2.py first.")
        sys.exit(1)

    model = joblib.load(model_path)
    with open(config_path) as f:
        config = json.load(f)

    logger.info(f"Loaded model: {len(config['features'])} features")
    return model, config


def load_test_data():
    """Load fused test features."""
    v3_path = os.path.join(DATA_DIR, "features_v3_hardened.csv")
    v4_path = os.path.join(DATA_DIR, "features_v4_hardened.csv")

    df_v3 = pd.read_csv(v3_path)
    df_v4 = pd.read_csv(v4_path)

    min_len = min(len(df_v3), len(df_v4))
    df_v3, df_v4 = df_v3.iloc[:min_len], df_v4.iloc[:min_len]

    # Use last 20% as eval set
    split_idx = int(min_len * 0.8)
    v3_eval = df_v3.iloc[split_idx:].reset_index(drop=True)
    v4_eval = df_v4.iloc[split_idx:].reset_index(drop=True)

    return v3_eval, v4_eval


def fuse_features(df_v3, df_v4, features):
    """Fuse v3 + v4 features into single feature matrix."""
    fused = pd.DataFrame()
    for feat in features:
        if feat in df_v3.columns:
            fused[feat] = df_v3[feat].values
        elif feat in df_v4.columns:
            fused[feat] = df_v4[feat].values
        else:
            fused[feat] = 0
    return fused.fillna(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Model-based reality gap testing
# ═══════════════════════════════════════════════════════════════════════════════

def reality_gap_with_model(model, X_clean, y_true, features):
    """Run reality gap scenarios using the TRAINED model."""
    from sklearn.metrics import roc_auc_score

    # Clean baseline
    clean_prob = model.predict_proba(X_clean[features])[:, 1]
    clean_auc = roc_auc_score(y_true, clean_prob)
    logger.info(f"Clean baseline AUC: {clean_auc:.4f}")

    scenarios = [
        ("Sensor Drift", lambda df: sensor_drift(df)),
        ("Missing 5%", lambda df: random_missing(df, 0.05)),
        ("Missing 10%", lambda df: random_missing(df, 0.10)),
        ("Missing 15%", lambda df: random_missing(df, 0.15)),
        ("Missing 30%", lambda df: random_missing(df, 0.30)),
        ("Block Missing 4h", lambda df: block_missing(df, 4, 3)),
        ("Device Offline (temp)", lambda df: device_offline(df, 'temp')),
        ("Device Offline (hr)", lambda df: device_offline(df, 'hr')),
        ("Temporal Delay 30min", lambda df: timestamp_jitter(df, 6)),
        ("Temporal Delay 2h", lambda df: timestamp_jitter(df, 24)),
        ("Temporal Delay 6h", lambda df: timestamp_jitter(df, 72)),
        ("Mgmt Misreporting", lambda df: mgmt_misreport(df)),
    ]

    results = [{"Scenario": "Clean Baseline", "AUC": round(clean_auc, 4),
                "AUC_Drop_%": 0, "Verdict": "✅ PASS"}]

    for name, fn in scenarios:
        corrupted = fn(X_clean.copy())
        corrupted = corrupted.fillna(X_clean.median(numeric_only=True))
        noisy_prob = model.predict_proba(corrupted[features])[:, 1]

        try:
            noisy_auc = roc_auc_score(y_true, noisy_prob)
        except Exception:
            noisy_auc = 0.5

        drop_pct = (clean_auc - noisy_auc) / clean_auc * 100 if clean_auc > 0 else 0
        verdict = "✅ PASS" if drop_pct <= 15 else "❌ FRAGILE"
        results.append({
            "Scenario": name,
            "AUC": round(noisy_auc, 4),
            "AUC_Drop_%": round(drop_pct, 2),
            "Verdict": verdict
        })
        logger.info(f"  {name}: AUC={noisy_auc:.4f} (drop {drop_pct:.1f}%) {verdict}")

    return pd.DataFrame(results), clean_auc


def run_counterfactual():
    """Run counterfactual engine (reuses existing script)."""
    logger.info("\n── Running Counterfactual Herd Testing ──")
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "counterfactual_engine.py"),
         "--cows", "100", "--days", "30", "--runs", "5"],
        capture_output=True, text=True, timeout=300
    )

    cf_path = os.path.join(DATA_DIR, "counterfactual_results.csv")
    if os.path.exists(cf_path):
        df = pd.read_csv(cf_path)
        peaks = df['Peak Infected'].tolist()
        r0s = df['Estimated R₀'].tolist()
        peak_mono = all(peaks[i] >= peaks[i+1] for i in range(len(peaks)-1))
        r0_mono = all(r0s[i] >= r0s[i+1] for i in range(len(r0s)-1))
        return {'peak_monotonic': peak_mono, 'r0_monotonic': r0_mono,
                'passed': peak_mono and r0_mono, 'data': df.to_dict('records')}
    return {'passed': False, 'error': 'No results'}


def run_intervention():
    """Run intervention validation (reuses existing script)."""
    logger.info("\n── Running Intervention Validation ──")
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "intervention_validation.py"),
         "--cows", "50", "--runs", "5"],
        capture_output=True, text=True, timeout=300
    )

    iv_path = os.path.join(DATA_DIR, "intervention_results.csv")
    if os.path.exists(iv_path):
        df = pd.read_csv(iv_path)
        if len(df) >= 2:
            early = df[df['Trigger Severity'] == 1]
            late = df[df['Trigger Severity'] == 3]
            if len(early) > 0 and len(late) > 0:
                early_loss = early['Avg Milk Loss'].values[0]
                late_loss = late['Avg Milk Loss'].values[0]
                pct = ((late_loss - early_loss) / late_loss * 100) if late_loss > 0 else 0
                return {'diff_pct': round(pct, 1), 'passed': pct >= 20,
                        'early_loss': early_loss, 'late_loss': late_loss}
    return {'passed': False, 'error': 'No results'}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("🧬 GoMata Robustness Evaluation v2")
    logger.info("=" * 60)

    model, config = load_model_and_config()
    features = config['features']

    # ── Load eval data ─────────────────────────────────────────
    v3_eval, v4_eval = load_test_data()
    X_eval = fuse_features(v3_eval, v4_eval, features)
    y_eval = v3_eval['disease_binary']
    logger.info(f"Eval set: {len(X_eval)} rows, disease+ {y_eval.sum()}")

    # ── Part 1: Reality Gap ────────────────────────────────────
    logger.info("\n── Part 1: Reality Gap Testing (with trained model) ──")
    gap_results, clean_auc = reality_gap_with_model(model, X_eval, y_eval, features)

    # ── Part 2: Counterfactual ─────────────────────────────────
    cf_result = run_counterfactual()
    logger.info(f"Counterfactual: {'✅ PASS' if cf_result['passed'] else '❌ FAIL'}")

    # ── Part 3: Intervention ───────────────────────────────────
    iv_result = run_intervention()
    logger.info(f"Intervention: {'✅ PASS' if iv_result['passed'] else '❌ FAIL'} "
                f"({iv_result.get('diff_pct', 0)}% diff)")

    # ── Compute pilot readiness ────────────────────────────────
    criteria = {
        "reality_gap_15pct_missing": {
            "target": "AUC drop ≤12%",
            "passed": False,
            "detail": ""
        },
        "reality_gap_6h_delay": {
            "target": "AUC drop ≤10%",
            "passed": False,
            "detail": ""
        },
        "counterfactual": {
            "target": "Peak + R₀ monotonic",
            "passed": cf_result['passed'],
            "detail": "Peak + R₀ monotonic" if cf_result['passed'] else "Not monotonic"
        },
        "intervention": {
            "target": "Milk loss diff ≥20%",
            "passed": iv_result['passed'],
            "detail": f"{iv_result.get('diff_pct', 0)}% difference"
        }
    }

    # Check specific gap scenarios
    for _, row in gap_results.iterrows():
        if row['Scenario'] == 'Missing 15%':
            passed = row['AUC_Drop_%'] <= 12
            criteria['reality_gap_15pct_missing']['passed'] = passed
            criteria['reality_gap_15pct_missing']['detail'] = f"{row['AUC_Drop_%']}% drop"
        if row['Scenario'] == 'Temporal Delay 6h':
            passed = row['AUC_Drop_%'] <= 10
            criteria['reality_gap_6h_delay']['passed'] = passed
            criteria['reality_gap_6h_delay']['detail'] = f"{row['AUC_Drop_%']}% drop"

    total = len(criteria)
    passed = sum(1 for c in criteria.values() if c['passed'])
    score = (passed / total) * 100

    # ── Save scorecard ─────────────────────────────────────────
    scorecard = []
    for name, c in criteria.items():
        scorecard.append({
            'Criterion': name,
            'Target': c['target'],
            'Result': c['detail'],
            'Verdict': '✅ PASS' if c['passed'] else '❌ FAIL'
        })
    sc_df = pd.DataFrame(scorecard)
    sc_df.to_csv(os.path.join(DATA_DIR, "robustness_scorecard.csv"), index=False)

    # ── Save full report ───────────────────────────────────────
    report = {
        "version": "gomata_model_hardening_v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pilot_readiness_score": score,
        "criteria_passed": passed,
        "criteria_total": total,
        "clean_auc": clean_auc,
        "reality_gap": gap_results.to_dict('records'),
        "counterfactual": cf_result,
        "intervention": iv_result,
        "criteria": criteria
    }
    with open(os.path.join(DATA_DIR, "pilot_validation_v2_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start

    # ── Final verdict ──────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("📊 PILOT READINESS REPORT v2")
    logger.info(f"{'='*60}")
    logger.info(f"\n{sc_df.to_string(index=False)}")
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Pilot Readiness Score: {score:.0f}%")
    logger.info(f"  Criteria Passed: {passed}/{total}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info(f"{'─'*60}")

    if score >= 90:
        logger.info("\n🟢 SYSTEM QUALIFIED FOR PILOT DEPLOYMENT")
    elif score >= 75:
        logger.info("\n🟡 PARTIALLY READY — Close to deployment")
    elif score >= 50:
        logger.info("\n🟠 NEEDS IMPROVEMENT — Address failing criteria")
    else:
        logger.info("\n🔴 NOT READY — Significant issues remain")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
