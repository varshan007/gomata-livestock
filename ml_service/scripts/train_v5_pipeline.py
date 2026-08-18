#!/usr/bin/env python3
"""
train_v5_pipeline.py — GoMata v5 Full Training + Evaluation Pipeline

Steps 3-5 of the v5 pipeline:
  3. Animal+time split (anti-leakage)
  4. Train XGBoost with augmentation curriculum
  5. Full evaluation: leakage audit, reality gap, temporal shuffle, memorization

Usage:
  python train_v5_pipeline.py
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, mean_squared_error,
                             confusion_matrix, precision_recall_curve)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from split_strategy_v2 import animal_time_split
from robust_augmentation_v2 import (augment_curriculum, sensor_drift, random_missing,
                                     block_missing, timestamp_jitter, mgmt_misreport)

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "train_v5.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("TrainV5")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

# ═══════════════════════════════════════════════════════════════════════════════
# V5 CLEAN FEATURES (36 V3 + 18 V4 = 54 total)
# ═══════════════════════════════════════════════════════════════════════════════

V3_FEATURES = [
    "temp_current", "hr_current", "resp_current",
    "activity_current", "rumination_current", "lying_current",
    "temp_1h_avg", "temp_6h_avg", "temp_12h_median", "temp_24h_std",
    "temp_6h_std", "temp_1h_std",
    "hr_1h_avg", "hr_6h_avg", "hr_12h_median", "hr_6h_std",
    "activity_1h_avg", "activity_6h_avg", "activity_6h_std",
    "resp_6h_avg", "resp_6h_std",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_12h",
    "hr_lag_1h", "hr_lag_3h", "hr_lag_6h", "hr_lag_12h",
    "activity_lag_1h", "activity_lag_3h", "activity_lag_6h", "activity_lag_12h",
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

ALL_FEATURES = V3_FEATURES + V4_FEATURES


def load_and_fuse():
    """Load v5 features and fuse into training matrix."""
    v3_path = os.path.join(DATA_DIR, "features_v3_v5.csv")
    v4_path = os.path.join(DATA_DIR, "features_v4_v5.csv")

    v3 = pd.read_csv(v3_path)
    v4 = pd.read_csv(v4_path)
    logger.info(f"Loaded V3: {len(v3)} rows, V4: {len(v4)} rows")

    min_len = min(len(v3), len(v4))
    v3, v4 = v3.iloc[:min_len].reset_index(drop=True), v4.iloc[:min_len].reset_index(drop=True)

    fused = pd.DataFrame()
    for f in ALL_FEATURES:
        if f in v3.columns:
            fused[f] = v3[f].values
        elif f in v4.columns:
            fused[f] = v4[f].values
        else:
            fused[f] = 0

    fused['animal_id'] = v3['animal_id']
    fused['disease_binary'] = v3['disease_binary']
    fused['severity_level'] = v3['severity_level']

    logger.info(f"Fused: {len(fused)} × {len(ALL_FEATURES)} features")
    logger.info(f"Disease rate: {fused['disease_binary'].mean()*100:.1f}%")
    return fused


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece / len(y_true)


def run_reality_gap(model, X_clean, y_true, features):
    """Reality gap tests with v5 clean model."""
    clean_prob = model.predict_proba(X_clean[features])[:, 1]
    clean_auc = roc_auc_score(y_true, clean_prob)

    scenarios = [
        ("Sensor Drift", lambda df: sensor_drift(df)),
        ("Missing 5%", lambda df: random_missing(df, 0.05)),
        ("Missing 10%", lambda df: random_missing(df, 0.10)),
        ("Missing 15%", lambda df: random_missing(df, 0.15)),
        ("Missing 30%", lambda df: random_missing(df, 0.30)),
        ("Block Missing 4h", lambda df: block_missing(df, 4, 3)),
        ("Temporal Delay 30min", lambda df: timestamp_jitter(df, 6)),
        ("Temporal Delay 2h", lambda df: timestamp_jitter(df, 24)),
        ("Temporal Delay 6h", lambda df: timestamp_jitter(df, 72)),
        ("Mgmt Misreporting", lambda df: mgmt_misreport(df)),
    ]

    results = [{"Scenario": "Clean", "AUC": round(clean_auc, 4), "Drop_%": 0}]
    for name, fn in scenarios:
        corrupted = fn(X_clean.copy()).fillna(X_clean.median(numeric_only=True))
        try:
            noisy_prob = model.predict_proba(corrupted[features])[:, 1]
            noisy_auc = roc_auc_score(y_true, noisy_prob)
        except:
            noisy_auc = 0.5
        drop = (clean_auc - noisy_auc) / clean_auc * 100 if clean_auc > 0 else 0
        results.append({"Scenario": name, "AUC": round(noisy_auc, 4), "Drop_%": round(drop, 2)})
        logger.info(f"  {name}: AUC={noisy_auc:.4f} (drop {drop:.1f}%)")

    return pd.DataFrame(results), clean_auc


def run_temporal_shuffle_test(df, features, label_col):
    """Shuffle temporal features → AUC must drop."""
    logger.info("\n── Temporal Shuffle Test ──")
    rng = np.random.default_rng(42)

    # Original train/test
    X = df[features].fillna(0)
    y = df[label_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    sw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model_orig = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    scale_pos_weight=sw, random_state=42, n_jobs=-1, verbosity=0)
    model_orig.fit(X_tr, y_tr, verbose=False)
    orig_auc = roc_auc_score(y_te, model_orig.predict_proba(X_te)[:, 1])
    logger.info(f"  Original AUC: {orig_auc:.4f}")

    # Shuffle temporal columns
    lag_cols = [c for c in features if 'lag_' in c]
    window_cols = [c for c in features if any(w in c for w in ['_1h_', '_6h_', '_12h_', '_24h_'])]
    temporal_cols = list(set(lag_cols + window_cols))

    df_shuf = df.copy()
    for col in temporal_cols:
        if col in df_shuf.columns:
            df_shuf[col] = rng.permutation(df_shuf[col].values)

    X_s = df_shuf[features].fillna(0)
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y, test_size=0.2, stratify=y, random_state=42)

    model_shuf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    scale_pos_weight=sw, random_state=42, n_jobs=-1, verbosity=0)
    model_shuf.fit(X_tr_s, y_tr_s, verbose=False)
    shuf_auc = roc_auc_score(y_te_s, model_shuf.predict_proba(X_te_s)[:, 1])
    logger.info(f"  Shuffled AUC: {shuf_auc:.4f}")

    drop = orig_auc - shuf_auc
    passed = drop > 0.02  # Must show meaningful drop
    logger.info(f"  Drop: {drop:.4f} {'✅ Temporal features matter!' if passed else '❌ Still ignoring temporal!'}")
    return orig_auc, shuf_auc, passed


def run_memorization_test(df, features, label_col):
    """Random labels must give ~0.5 AUC."""
    logger.info("\n── Random Label Test ──")
    rng = np.random.default_rng(42)
    X = df[features].fillna(0)
    y_rand = pd.Series(rng.integers(0, 2, len(df)))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_rand, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, verbose=False)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    passed = auc < 0.55
    logger.info(f"  Random label AUC: {auc:.4f} {'✅' if passed else '❌'}")
    return auc, passed


def main():
    start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🧬 GoMata v5 Full Training + Evaluation Pipeline")
    logger.info(f"   Features: {len(ALL_FEATURES)} (36 V3 + 18 V4)")
    logger.info("=" * 60)

    # ── Step 2: Load fused features ────────────────────────────
    df = load_and_fuse()

    # ── Step 3: Anti-leakage split ─────────────────────────────
    logger.info("\n── Step 3: Animal+Time Split ──")
    X_train, X_test, y_train, y_test, split_info = animal_time_split(
        df, ALL_FEATURES, "disease_binary",
        animal_train_ratio=0.8, time_train_ratio=0.7, seed=42
    )

    # ── Step 4: Augmented training ─────────────────────────────
    logger.info("\n── Step 4: Training (60/40 curriculum) ──")
    X_train_aug, y_train_aug, noise_mask = augment_curriculum(
        X_train, y_train, noisy_ratio=0.4, seed=42
    )

    sw = (y_train_aug == 0).sum() / max((y_train_aug == 1).sum(), 1)
    logger.info(f"scale_pos_weight: {sw:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=sw,
        eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train_aug, y_train_aug, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    ece = compute_ece(y_test.values, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC: {roc:.4f}")
    logger.info(f"PR-AUC: {pr:.4f}")
    logger.info(f"ECE: {ece:.4f}")

    # Severity
    y_sev_train = df.loc[y_train.index, 'severity_level']
    y_sev_test = df.loc[y_test.index, 'severity_level']
    sev_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0
    )
    sev_model.fit(X_train, y_sev_train, eval_set=[(X_test, y_sev_test)], verbose=False)
    sev_rmse = np.sqrt(mean_squared_error(y_sev_test, sev_model.predict(X_test)))
    logger.info(f"Severity RMSE: {sev_rmse:.4f}")

    # Feature importance
    imp = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': ALL_FEATURES[:len(imp)], 'importance': imp})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    feat_imp['pct'] = (feat_imp['importance'] / feat_imp['importance'].sum() * 100).round(2)
    top3_pct = feat_imp.head(3)['pct'].sum()
    logger.info(f"\nTop 10 features:\n{feat_imp.head(10).to_string()}")
    logger.info(f"Top 3 explain: {top3_pct:.1f}%")

    # ── Step 5: Evaluation ─────────────────────────────────────
    logger.info("\n── Step 5a: Reality Gap ──")
    gap_df, clean_auc = run_reality_gap(model, X_test, y_test, ALL_FEATURES)

    logger.info("\n── Step 5b: Temporal Shuffle ──")
    orig_auc, shuf_auc, temporal_passed = run_temporal_shuffle_test(df, ALL_FEATURES, "disease_binary")

    logger.info("\n── Step 5c: Memorization ──")
    rand_auc, memo_passed = run_memorization_test(df, ALL_FEATURES, "disease_binary")

    # ── Save artifacts ─────────────────────────────────────────
    logger.info("\n── Saving ──")
    joblib.dump(model, os.path.join(MODEL_DIR, "model_v5.pkl"))
    joblib.dump(sev_model, os.path.join(MODEL_DIR, "severity_model_v5.pkl"))

    gap_df.to_csv(os.path.join(DATA_DIR, "reality_gap_v5.csv"), index=False)

    report = {
        "version": "gomata_v5_probabilistic",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": ALL_FEATURES,
        "total_features": len(ALL_FEATURES),
        "split": split_info,
        "metrics": {
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "ece": round(float(ece), 4),
            "severity_rmse": round(float(sev_rmse), 4),
            "top3_importance_pct": round(float(top3_pct), 1),
        },
        "temporal_test": {
            "original_auc": round(float(orig_auc), 4),
            "shuffled_auc": round(float(shuf_auc), 4),
            "drop": round(float(orig_auc - shuf_auc), 4),
            "passed": bool(temporal_passed),
        },
        "memorization_test": {
            "random_label_auc": round(float(rand_auc), 4),
            "passed": bool(memo_passed),
        },
        "reality_gap": gap_df.to_dict('records'),
        "feature_importance": {
            r['feature']: round(float(r['importance']), 6)
            for _, r in feat_imp.head(20).iterrows()
        },
        "auc_realistic": roc < 0.98,
    }

    with open(os.path.join(DATA_DIR, "v5_evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("  ✅ model_v5.pkl")
    logger.info("  ✅ v5_evaluation_report.json")

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info("📊 V5 PIPELINE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  ROC-AUC: {roc:.4f} (target 0.85-0.95)")
    logger.info(f"  PR-AUC: {pr:.4f}")
    logger.info(f"  ECE: {ece:.4f}")
    logger.info(f"  Severity RMSE: {sev_rmse:.4f}")
    logger.info(f"  Temporal shuffle: orig={orig_auc:.4f} → shuf={shuf_auc:.4f} "
                f"{'✅' if temporal_passed else '❌'}")
    logger.info(f"  Random labels: {rand_auc:.4f} {'✅' if memo_passed else '❌'}")
    logger.info(f"  AUC realistic (<0.98): {'✅' if roc < 0.98 else '❌'}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
