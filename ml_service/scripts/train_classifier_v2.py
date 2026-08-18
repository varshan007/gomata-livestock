#!/usr/bin/env python3
"""
train_classifier_v2.py — GoMata Model Hardening v2, Phase 3
Robust XGBoost Training with Augmentation Curriculum

- Fuses V3 (42 physiological) + V4 (18 management) features = 60 total
- 60/40 clean/noisy curriculum via robust_augmentation_v2
- XGBoost 500 trees, depth 6, robust hyperparams
- Multi-task: disease binary + severity level
- Robustness consistency verification (prediction stability check)

Outputs:
  - robust_model_v2.pkl (disease classifier)
  - robust_severity_model_v2.pkl (severity regressor)
  - feature_config_v2.json
  - noise_profile_stats.json

Usage:
  python train_classifier_v2.py
"""

import os, sys, json, time, logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, 
                             average_precision_score, mean_squared_error,
                             confusion_matrix, precision_recall_curve)

# Import augmentation engine
sys.path.insert(0, os.path.dirname(__file__))
from robust_augmentation_v2 import (augment_curriculum, apply_random_corruption,
                                     get_noise_stats)

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "train_v2.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("TrainerV2")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

V3_FEATURES = [
    "temp_current", "hr_current", "resp_current",
    "activity_current", "rumination_current", "lying_current",
    "temp_1h_avg", "temp_6h_avg", "temp_12h_median",
    "temp_24h_std", "temp_6h_std", "temp_1h_std",
    "hr_1h_avg", "hr_6h_avg", "hr_12h_median", "hr_6h_std",
    "activity_1h_avg", "activity_6h_avg", "activity_6h_std",
    "resp_6h_avg", "resp_6h_std",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_12h",
    "hr_lag_1h", "hr_lag_3h", "hr_lag_6h", "hr_lag_12h",
    "activity_lag_1h", "activity_lag_3h", "activity_lag_6h", "activity_lag_12h",
    "temp_zscore", "hsi", "composite_stress",
    "rumination_drop", "autocorr_temp", "temp_slope_6h",
    "thi", "ambient_temp", "humidity",
]

V4_FEATURES = [
    "milk_deviation", "conductivity_deviation", "feed_deviation", "weight_deviation",
    "hours_since_vaccination", "hours_since_antibiotic",
    "hours_since_transport", "hours_since_feed_change",
    "vacc_decay", "abx_decay", "transport_decay", "feed_decay",
    "total_antibiotic_days", "vaccination_count_12m", "feed_changes_30d",
    "parity", "bcs", "age",
]

ALL_FEATURES = V3_FEATURES + V4_FEATURES  # 60 total


def load_and_fuse():
    """Load V3 + V4 CSVs, fuse on animal_id index alignment."""
    v3_path = os.path.join(DATA_DIR, "features_v3_hardened.csv")
    v4_path = os.path.join(DATA_DIR, "features_v4_hardened.csv")

    if not os.path.exists(v3_path):
        logger.error(f"V3 features not found: {v3_path}. Run extract_features_v3.py first.")
        sys.exit(1)
    if not os.path.exists(v4_path):
        logger.error(f"V4 features not found: {v4_path}. Run extract_features_v4.py first.")
        sys.exit(1)

    df_v3 = pd.read_csv(v3_path)
    df_v4 = pd.read_csv(v4_path)
    logger.info(f"V3: {len(df_v3)} rows, {len(V3_FEATURES)} features")
    logger.info(f"V4: {len(df_v4)} rows, {len(V4_FEATURES)} features")

    # Align lengths — use shorter dataset
    min_len = min(len(df_v3), len(df_v4))
    df_v3 = df_v3.iloc[:min_len].reset_index(drop=True)
    df_v4 = df_v4.iloc[:min_len].reset_index(drop=True)

    # Fuse features
    fused = pd.DataFrame()
    for feat in V3_FEATURES:
        fused[feat] = df_v3[feat] if feat in df_v3.columns else 0
    for feat in V4_FEATURES:
        fused[feat] = df_v4[feat] if feat in df_v4.columns else 0

    # Labels from v3 (ground truth physiological)
    labels = pd.DataFrame({
        'disease_binary': df_v3['disease_binary'],
        'severity_level': df_v3['severity_level'],
        'infection_binary': df_v3.get('infection_binary', 0),
        'stress_binary': df_v3.get('stress_binary', 0),
    })

    logger.info(f"Fused dataset: {len(fused)} rows, {len(ALL_FEATURES)} features")
    logger.info(f"Disease: {labels['disease_binary'].sum()}/{len(labels)} "
                f"({labels['disease_binary'].mean()*100:.1f}%)")

    return fused, labels


def train_disease_model(X_train, y_train, X_test, y_test, scale_weight):
    """Train robust disease binary classifier."""
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,  # elevated for noise robustness
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    return model


def train_severity_model(X_train, y_train, X_test, y_test):
    """Train severity level regressor (0-3)."""
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    return model


def robustness_consistency_check(model, X_clean, y_clean, λ=0.2, n_trials=5):
    """
    Verify prediction stability: compare clean vs noisy predictions.
    L_robust = MSE(pred_clean, pred_noisy) — must be < threshold.
    """
    logger.info("\n── Robustness Consistency Check ──")
    pred_clean = model.predict_proba(X_clean)[:, 1]
    consistency_scores = []

    for trial in range(n_trials):
        rng = np.random.default_rng(trial + 100)
        X_noisy = apply_random_corruption(X_clean.copy(), rng)
        X_noisy = X_noisy.fillna(X_clean.median(numeric_only=True))
        pred_noisy = model.predict_proba(X_noisy)[:, 1]

        mse = np.mean((pred_clean - pred_noisy) ** 2)
        # Weighted loss
        L_robust = λ * mse
        consistency_scores.append({
            'trial': trial + 1,
            'mse': round(mse, 6),
            'L_robust': round(L_robust, 6),
            'max_diff': round(np.max(np.abs(pred_clean - pred_noisy)), 4),
            'mean_diff': round(np.mean(np.abs(pred_clean - pred_noisy)), 4)
        })
        logger.info(f"  Trial {trial+1}: MSE={mse:.6f}, L_robust={L_robust:.6f}, "
                    f"maxΔ={consistency_scores[-1]['max_diff']}")

    avg_mse = np.mean([s['mse'] for s in consistency_scores])
    avg_l_robust = np.mean([s['L_robust'] for s in consistency_scores])
    passed = avg_mse < 0.05  # threshold for acceptable noise sensitivity
    logger.info(f"  Avg MSE: {avg_mse:.6f}, Avg L_robust: {avg_l_robust:.6f}")
    logger.info(f"  Consistency: {'✅ PASS' if passed else '⚠️ FRAGILE'} (threshold < 0.05)")

    return consistency_scores, passed


def main():
    start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🧬 GoMata Robust XGBoost v2 — Model Hardening")
    logger.info("=" * 60)

    # ── Load and fuse ──────────────────────────────────────────
    X_all, labels = load_and_fuse()
    y_disease = labels['disease_binary']
    y_severity = labels['severity_level']

    # ── Augment with curriculum ────────────────────────────────
    logger.info("\n── Applying 60/40 Clean/Noisy Curriculum ──")
    X_aug, y_aug, noise_mask = augment_curriculum(X_all, y_disease, noisy_ratio=0.4, seed=42)

    # ── Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, stratify=y_aug, random_state=42
    )
    # Also split severity
    sev_train = y_severity.iloc[y_train.index]
    sev_test = y_severity.iloc[y_test.index]

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    logger.info(f"Disease+ train: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")

    # ── Scale weight ───────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / max(pos, 1)
    logger.info(f"scale_pos_weight: {scale_weight:.3f}")

    # ── Train Disease Model ────────────────────────────────────
    logger.info("\n── Training Disease Binary Classifier ──")
    disease_model = train_disease_model(X_train, y_train, X_test, y_test, scale_weight)

    y_pred = disease_model.predict(X_test)
    y_prob = disease_model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC: {roc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"Confusion:\n{cm}")

    # ── Threshold tuning ───────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    best_f1, optimal_threshold = 0, 0.5
    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= 0.70 and p >= 0.80:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, optimal_threshold = f1, t
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1={best_f1:.4f})")

    # ── Train Severity Model ───────────────────────────────────
    logger.info("\n── Training Severity Regressor ──")
    severity_model = train_severity_model(X_train, sev_train, X_test, sev_test)
    sev_pred = severity_model.predict(X_test)
    sev_rmse = np.sqrt(mean_squared_error(sev_test, sev_pred))
    logger.info(f"Severity RMSE: {sev_rmse:.4f}")

    # ── Robustness Consistency ─────────────────────────────────
    consistency, robust_pass = robustness_consistency_check(
        disease_model, X_test, y_test, λ=0.2
    )

    # ── Feature Importance ─────────────────────────────────────
    logger.info("\n── Feature Importance (Top 15) ──")
    importances = disease_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': ALL_FEATURES[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    logger.info(f"\n{feat_imp.head(15).to_string()}")

    # ── Save Artifacts ─────────────────────────────────────────
    logger.info("\n── Saving Artifacts ──")

    # Models
    joblib.dump(disease_model, os.path.join(MODEL_DIR, "robust_model_v2.pkl"))
    joblib.dump(severity_model, os.path.join(MODEL_DIR, "robust_severity_model_v2.pkl"))
    logger.info(f"  ✅ robust_model_v2.pkl")
    logger.info(f"  ✅ robust_severity_model_v2.pkl")

    # Feature config
    config = {
        "version": "gomata_model_hardening_v2",
        "features": ALL_FEATURES,
        "v3_features": V3_FEATURES,
        "v4_features": V4_FEATURES,
        "total_feature_count": len(ALL_FEATURES),
        "threshold_default": 0.5,
        "threshold_sensitive": round(float(optimal_threshold), 4),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "severity_rmse": round(float(sev_rmse), 4),
        "robustness_consistent": bool(robust_pass),
        "curriculum_noisy_ratio": 0.4,
        "model_params": {
            "n_estimators": 500, "max_depth": 6,
            "learning_rate": 0.05, "min_child_weight": 5,
        }
    }
    with open(os.path.join(MODEL_DIR, "feature_config_v2.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"  ✅ feature_config_v2.json")

    # Noise profile
    noise_stats = get_noise_stats(
        X_all.iloc[:len(noise_mask)], X_aug.iloc[:len(noise_mask)], noise_mask
    )
    noise_stats['_meta'] = {
        'noisy_ratio': 0.4,
        'corruptions': ['sensor_drift', 'missingness', 'timestamp_jitter', 'mgmt_misreport'],
        'consistency_scores': consistency,
        'consistency_passed': robust_pass
    }
    with open(os.path.join(DATA_DIR, "noise_profile_stats.json"), "w") as f:
        json.dump(noise_stats, f, indent=2, default=str)
    logger.info(f"  ✅ noise_profile_stats.json")

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"🟢 Model Hardening v2 Training Complete")
    logger.info(f"   Disease AUC: {roc:.4f}")
    logger.info(f"   Severity RMSE: {sev_rmse:.4f}")
    logger.info(f"   Robust: {'✅' if robust_pass else '⚠️'}")
    logger.info(f"   Duration: {elapsed:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
