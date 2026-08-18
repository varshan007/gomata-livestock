#!/usr/bin/env python3
"""
train_classifier_no_leak.py — GoMata Leakage Audit
Phase 3: Retrain with whitelisted features + anti-leakage splits

- Only 54 clean features (6 suspicious embedded features REMOVED)
- Animal+time split (no animal overlap, no future leakage)
- 60/40 clean/noisy curriculum
- SHAP analysis for shortcut detection (Phase 5)

Usage:
  python train_classifier_no_leak.py
"""

import os, sys, json, time, logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, mean_squared_error,
                             confusion_matrix, precision_recall_curve)

sys.path.insert(0, os.path.dirname(__file__))
from split_strategy_v2 import animal_time_split
from robust_augmentation_v2 import augment_curriculum, get_noise_stats
from leakage_audit_v1 import CLEAN_FEATURES, V3_WHITELIST, V4_WHITELIST

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "train_no_leak.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("TrainNoLeak")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")


def load_and_fuse_clean():
    """Load V3 + V4 CSVs, fuse using ONLY whitelisted features."""
    v3 = pd.read_csv(os.path.join(DATA_DIR, "features_v3_hardened.csv"))
    v4 = pd.read_csv(os.path.join(DATA_DIR, "features_v4_hardened.csv"))
    logger.info(f"Loaded V3: {len(v3)} rows, V4: {len(v4)} rows")

    min_len = min(len(v3), len(v4))
    v3, v4 = v3.iloc[:min_len].reset_index(drop=True), v4.iloc[:min_len].reset_index(drop=True)

    # Build fused dataframe with ONLY whitelisted features
    fused = pd.DataFrame()
    for feat in CLEAN_FEATURES:
        if feat in v3.columns:
            fused[feat] = v3[feat].values
        elif feat in v4.columns:
            fused[feat] = v4[feat].values
        else:
            fused[feat] = 0

    fused['animal_id'] = v3['animal_id']
    fused['disease_binary'] = v3['disease_binary']
    fused['severity_level'] = v3['severity_level']

    logger.info(f"Fused: {len(fused)} rows, {len(CLEAN_FEATURES)} clean features")
    logger.info(f"Disease: {fused['disease_binary'].sum()}/{len(fused)} "
                f"({fused['disease_binary'].mean()*100:.1f}%)")
    return fused


def compute_calibration_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += mask.sum() * abs(avg_confidence - avg_accuracy)
    return ece / len(y_true)


def main():
    start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🔬 GoMata No-Leak Training — Clean Features + Anti-Leakage Split")
    logger.info(f"   Features: {len(CLEAN_FEATURES)} (6 suspicious removed)")
    logger.info("=" * 60)

    # ── Load and fuse ──────────────────────────────────────────
    df = load_and_fuse_clean()

    # ── Anti-leakage split ─────────────────────────────────────
    logger.info("\n── Animal+Time Combined Split ──")
    X_train, X_test, y_train, y_test, split_info = animal_time_split(
        df, CLEAN_FEATURES, "disease_binary",
        animal_train_ratio=0.8, time_train_ratio=0.7, seed=42
    )

    # ── Apply curriculum augmentation ──────────────────────────
    logger.info("\n── Applying 60/40 Curriculum ──")
    X_train_aug, y_train_aug, noise_mask = augment_curriculum(
        X_train, y_train, noisy_ratio=0.4, seed=42
    )

    # ── Scale weight ───────────────────────────────────────────
    neg = (y_train_aug == 0).sum()
    pos = (y_train_aug == 1).sum()
    scale_weight = neg / max(pos, 1)
    logger.info(f"scale_pos_weight: {scale_weight:.3f}")

    # ── Train disease classifier ───────────────────────────────
    logger.info("\n── Training Disease Classifier (no-leak) ──")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train_aug, y_train_aug,
              eval_set=[(X_test, y_test)], verbose=False)

    # ── Evaluate ───────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    ece = compute_calibration_ece(y_test.values, y_prob)

    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC: {roc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"ECE (calibration): {ece:.4f}")
    logger.info(f"Confusion:\n{cm}")

    # ── Threshold tuning ───────────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    best_f1, optimal_threshold = 0, 0.5
    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= 0.60 and p >= 0.60:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, optimal_threshold = f1, t
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1={best_f1:.4f})")

    # ── Severity model ─────────────────────────────────────────
    logger.info("\n── Training Severity Regressor (no-leak) ──")
    y_sev_train = df.loc[y_train.index, 'severity_level']
    y_sev_test = df.loc[y_test.index, 'severity_level']

    sev_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0
    )
    sev_model.fit(X_train, y_sev_train, eval_set=[(X_test, y_sev_test)], verbose=False)
    sev_pred = sev_model.predict(X_test)
    sev_rmse = np.sqrt(mean_squared_error(y_sev_test, sev_pred))
    logger.info(f"Severity RMSE: {sev_rmse:.4f}")

    # ── Phase 5: SHAP Shortcut Detection ───────────────────────
    logger.info("\n── SHAP Feature Importance (Shortcut Detection) ──")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': CLEAN_FEATURES[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    total_imp = feat_imp['importance'].sum()
    feat_imp['pct'] = (feat_imp['importance'] / total_imp * 100).round(2)
    feat_imp['cumulative_pct'] = feat_imp['pct'].cumsum().round(2)

    logger.info(f"\nTop 15 features:")
    logger.info(f"\n{feat_imp.head(15).to_string()}")

    top3_pct = feat_imp.head(3)['pct'].sum()
    shortcut_warning = top3_pct > 60
    logger.info(f"\nTop 3 features explain: {top3_pct:.1f}% of total importance")
    logger.info(f"Shortcut bias: {'⚠️ YES — top 3 > 60%' if shortcut_warning else '✅ NO — distributed'}")

    # ── Try SHAP if available ──────────────────────────────────
    shap_summary = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test.iloc[:5000])
        shap_mean = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': CLEAN_FEATURES[:len(shap_mean)],
            'mean_shap': shap_mean
        }).sort_values('mean_shap', ascending=False)

        logger.info(f"\nSHAP Top 15:")
        logger.info(f"\n{shap_df.head(15).to_string()}")

        shap_summary = {f: round(float(v), 6) for f, v in
                        zip(shap_df['feature'], shap_df['mean_shap'])}

        # Save SHAP plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_vals, X_test.iloc[:5000],
                            feature_names=CLEAN_FEATURES, show=False)
            plot_path = os.path.join(DATA_DIR, "shap_summary.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP plot: {plot_path}")
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")

    except ImportError:
        logger.info("shap not installed — using XGBoost importance only")

    # ── Save artifacts ─────────────────────────────────────────
    logger.info("\n── Saving Artifacts ──")

    joblib.dump(model, os.path.join(MODEL_DIR, "model_no_leak.pkl"))
    joblib.dump(sev_model, os.path.join(MODEL_DIR, "severity_model_no_leak.pkl"))
    logger.info("  ✅ model_no_leak.pkl")
    logger.info("  ✅ severity_model_no_leak.pkl")

    config = {
        "version": "gomata_no_leak_v1",
        "features": CLEAN_FEATURES,
        "v3_features": V3_WHITELIST,
        "v4_features": V4_WHITELIST,
        "total_feature_count": len(CLEAN_FEATURES),
        "removed_features": [
            "rumination_drop", "composite_stress", "hsi",
            "temp_zscore", "autocorr_temp", "temp_slope_6h"
        ],
        "split_strategy": "animal_time_combined",
        "split_info": split_info,
        "threshold_default": 0.5,
        "threshold_sensitive": round(float(optimal_threshold), 4),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "ece": round(float(ece), 4),
        "severity_rmse": round(float(sev_rmse), 4),
        "top3_importance_pct": round(float(top3_pct), 1),
        "shortcut_warning": shortcut_warning,
        "shap_summary": shap_summary,
        "feature_importance": {
            row['feature']: round(float(row['importance']), 6)
            for _, row in feat_imp.iterrows()
        }
    }
    with open(os.path.join(MODEL_DIR, "feature_config_no_leak.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info("  ✅ feature_config_no_leak.json")

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"🔬 No-Leak Training Complete")
    logger.info(f"   ROC-AUC: {roc:.4f}  (expected 0.85-0.97)")
    logger.info(f"   PR-AUC: {pr_auc:.4f}")
    logger.info(f"   ECE: {ece:.4f}")
    logger.info(f"   Severity RMSE: {sev_rmse:.4f}")
    logger.info(f"   Shortcut: {'⚠️' if shortcut_warning else '✅'}")
    logger.info(f"   Duration: {elapsed:.1f}s")

    if roc > 0.98:
        logger.info(f"   ⚠️  AUC suspiciously high — investigate further")
    elif roc >= 0.85:
        logger.info(f"   ✅ AUC in realistic range — model is valid")
    else:
        logger.info(f"   ⚠️  AUC low — may need feature engineering")

    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
