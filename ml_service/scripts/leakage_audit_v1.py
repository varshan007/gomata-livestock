#!/usr/bin/env python3
"""
leakage_audit_v1.py — GoMata Leakage Audit
Phase 1: Feature Whitelist Enforcement

Scans all features used in training against forbidden and suspicious lists.
Computes label correlation, identifies hidden-state proxies.

Usage:
  python leakage_audit_v1.py
"""

import os, sys, json, logging
import numpy as np
import pandas as pd

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "leakage_audit.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("LeakageAudit")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")

# ═══════════════════════════════════════════════════════════════════════════════
# FORBIDDEN FEATURES — Must NEVER appear in training
# ═══════════════════════════════════════════════════════════════════════════════

FORBIDDEN_PATTERNS = [
    "hiddenState", "infectionLoad", "stressLoad", "immuneResponse",
    "compensationCollapse", "episodePhase", "diseaseBinary", "severityLevel",
    "infectionBinary", "stressBinary", "mixedStateBinary", "diseaseType",
    "infection_in_24h", "stress_in_24h", "interventionContext",
    "label", "disease_binary", "severity_level", "infection_binary",
    "stress_binary", "mixed_binary", "disease_type",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SUSPICIOUS FEATURES — Embedded features derived from simulator hidden state
# ═══════════════════════════════════════════════════════════════════════════════

SUSPICIOUS_EMBEDDED = {
    "rumination_drop":    "Derived from hidden infection/stress state → label proxy (r=0.69)",
    "composite_stress":   "composite_stress_index from hidden stressLoad → label proxy (r=0.56)",
    "hsi":                "heat_stress_index may encode hidden stress state",
    "temp_zscore":        "Embedded from features.temp_zscore — could encode hidden state",
    "autocorr_temp":      "Embedded from features.autocorrelation_temp — simulation artifact",
    "temp_slope_6h":      "Embedded from features.temp_6h_slope — may be computed WITH hidden state",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CLEAN WHITELIST — Only these features are allowed
# ═══════════════════════════════════════════════════════════════════════════════

V3_WHITELIST = [
    # Current signals — direct sensor readings
    "temp_current", "hr_current", "resp_current",
    "activity_current", "rumination_current", "lying_current",
    # Multi-scale windows — computed from raw signals only
    "temp_1h_avg", "temp_6h_avg", "temp_12h_median",
    "temp_24h_std", "temp_6h_std", "temp_1h_std",
    "hr_1h_avg", "hr_6h_avg", "hr_12h_median", "hr_6h_std",
    "activity_1h_avg", "activity_6h_avg", "activity_6h_std",
    "resp_6h_avg", "resp_6h_std",
    # Lag stacks — shifted raw signals only
    "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_12h",
    "hr_lag_1h", "hr_lag_3h", "hr_lag_6h", "hr_lag_12h",
    "activity_lag_1h", "activity_lag_3h", "activity_lag_6h", "activity_lag_12h",
    # Environment — external, non-hidden
    "thi", "ambient_temp", "humidity",
]  # 36 clean V3 features (removed 6 suspicious embedded)

V4_WHITELIST = [
    "milk_deviation", "conductivity_deviation", "feed_deviation", "weight_deviation",
    "hours_since_vaccination", "hours_since_antibiotic",
    "hours_since_transport", "hours_since_feed_change",
    "vacc_decay", "abx_decay", "transport_decay", "feed_decay",
    "total_antibiotic_days", "vaccination_count_12m", "feed_changes_30d",
    "parity", "bcs", "age",
]  # 18 clean V4 features

CLEAN_FEATURES = V3_WHITELIST + V4_WHITELIST  # 54 total


def audit_features(feature_list, label="model"):
    """Audit a feature list against forbidden and suspicious patterns."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 Auditing {label}: {len(feature_list)} features")
    logger.info(f"{'='*60}")

    forbidden_found = []
    suspicious_found = []
    clean = []

    for feat in feature_list:
        # Check forbidden
        is_forbidden = any(p.lower() in feat.lower() for p in FORBIDDEN_PATTERNS)
        if is_forbidden:
            forbidden_found.append(feat)
            logger.info(f"  ❌ FORBIDDEN: {feat}")
            continue

        # Check suspicious embedded
        if feat in SUSPICIOUS_EMBEDDED:
            suspicious_found.append({
                "feature": feat,
                "reason": SUSPICIOUS_EMBEDDED[feat]
            })
            logger.info(f"  ⚠️  SUSPICIOUS: {feat} — {SUSPICIOUS_EMBEDDED[feat]}")
            continue

        clean.append(feat)

    logger.info(f"\n  Summary:")
    logger.info(f"    Forbidden: {len(forbidden_found)}")
    logger.info(f"    Suspicious: {len(suspicious_found)}")
    logger.info(f"    Clean: {len(clean)}")

    passed = len(forbidden_found) == 0
    logger.info(f"    VERDICT: {'✅ NO FORBIDDEN' if passed else '❌ FORBIDDEN FEATURES DETECTED'}")

    return {
        "total": len(feature_list),
        "forbidden": forbidden_found,
        "suspicious": [s["feature"] for s in suspicious_found],
        "suspicious_details": suspicious_found,
        "clean": clean,
        "clean_count": len(clean),
        "passed_forbidden_check": passed,
        "has_suspicious": len(suspicious_found) > 0,
    }


def correlation_analysis(csv_path, label_col="disease_binary"):
    """Compute feature-label correlations to identify proxies."""
    df = pd.read_csv(csv_path)
    num = df.select_dtypes(include="number")
    label_cols = ["disease_binary", "severity_level", "infection_binary",
                  "stress_binary", "mixed_binary"]
    feat_cols = [c for c in num.columns if c not in label_cols]

    if label_col not in num.columns:
        return {}

    corrs = num[feat_cols].corrwith(num[label_col]).abs().sort_values(ascending=False)
    logger.info(f"\n  Top 10 correlations with {label_col}:")
    for feat, r in corrs.head(10).items():
        flag = " ⚠️ HIGH" if r > 0.5 else ""
        logger.info(f"    {feat}: {r:.4f}{flag}")

    high_corr = {feat: round(float(r), 4) for feat, r in corrs.items() if r > 0.5}
    return {"correlations": {k: round(float(v), 4) for k, v in corrs.head(20).items()},
            "high_correlation_features": high_corr}


def main():
    logger.info("=" * 60)
    logger.info("🔬 GoMata Leakage Audit v1 — Feature Whitelist Enforcement")
    logger.info("=" * 60)

    # Load v2 model config
    config_path = os.path.join(os.path.dirname(__file__),
                               "../models/cattle/feature_config_v2.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        v2_features = config.get("features", [])
    else:
        v2_features = []

    # Audit v2 model features
    v2_audit = audit_features(v2_features, "Model v2 (current)")

    # Correlation analysis
    logger.info("\n── Correlation Analysis ──")
    v3_corr = correlation_analysis(
        os.path.join(DATA_DIR, "features_v3_hardened.csv"))
    v4_corr = correlation_analysis(
        os.path.join(DATA_DIR, "features_v4_hardened.csv"))

    # Proposed clean whitelist
    logger.info("\n── Proposed Clean Feature Set ──")
    clean_audit = audit_features(CLEAN_FEATURES, "Proposed clean whitelist")

    # Save report
    report = {
        "audit_version": "gomata_leakage_audit_v1",
        "v2_model_audit": v2_audit,
        "proposed_clean_audit": clean_audit,
        "v3_correlations": v3_corr,
        "v4_correlations": v4_corr,
        "removed_features": list(SUSPICIOUS_EMBEDDED.keys()),
        "removed_reasons": SUSPICIOUS_EMBEDDED,
        "clean_feature_count": len(CLEAN_FEATURES),
        "recommendation": (
            "REMOVE 6 embedded features (rumination_drop, composite_stress, "
            "hsi, temp_zscore, autocorr_temp, temp_slope_6h). "
            "These are derived from simulator hiddenState and act as label proxies. "
            "Retrain with 54 clean features using animal+time split."
        )
    }

    report_path = os.path.join(DATA_DIR, "leakage_audit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"📋 Audit saved: {report_path}")
    logger.info(f"   Suspicious features to remove: {len(SUSPICIOUS_EMBEDDED)}")
    logger.info(f"   Clean features remaining: {len(CLEAN_FEATURES)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
