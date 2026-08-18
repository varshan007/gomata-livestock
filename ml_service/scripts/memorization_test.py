#!/usr/bin/env python3
"""
memorization_test.py — GoMata Leakage Audit
Phase 6: Memorization Detection

Tests if model has memorized animal-specific or temporal patterns:
  1. Shuffle animal IDs → AUC should stay similar (learns physiology, not identity)
  2. Shuffle timestamps → AUC should DROP (temporal features break)
  3. Random label test → AUC should be ~0.5 (no meaningful learning)

Usage:
  python memorization_test.py
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from leakage_audit_v1 import CLEAN_FEATURES

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "memorization_test.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("MemorizationTest")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")


def load_fused():
    v3 = pd.read_csv(os.path.join(DATA_DIR, "features_v3_hardened.csv"))
    v4 = pd.read_csv(os.path.join(DATA_DIR, "features_v4_hardened.csv"))
    m = min(len(v3), len(v4))
    v3, v4 = v3.iloc[:m].reset_index(drop=True), v4.iloc[:m].reset_index(drop=True)

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
    return fused.fillna(0)


def test_shuffle_animals(df):
    """Shuffle animal IDs — should NOT affect AUC much (model learns physiology)."""
    logger.info("\n── Test 1: Shuffle Animal IDs ──")
    shuffled = df.copy()
    rng = np.random.default_rng(42)
    shuffled['animal_id'] = rng.permutation(shuffled['animal_id'].values)

    X = shuffled[CLEAN_FEATURES]
    y = shuffled['disease_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1, verbosity=0,
                               scale_pos_weight=(y_train==0).sum()/max((y_train==1).sum(),1))
    model.fit(X_train, y_train, verbose=False)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    logger.info(f"  AUC with shuffled animals: {auc:.4f}")
    logger.info(f"  Expected: Similar to original (model learns physiology, not identity)")
    return auc


def test_shuffle_timestamps(df):
    """Shuffle timestamp ORDER — temporal features break, AUC should drop."""
    logger.info("\n── Test 2: Shuffle Timestamp Order ──")
    shuffled = df.copy()
    rng = np.random.default_rng(42)

    # Shuffle the lag features within each animal (breaks temporal coherence)
    lag_cols = [c for c in CLEAN_FEATURES if 'lag_' in c]
    window_cols = [c for c in CLEAN_FEATURES if any(w in c for w in ['_1h_', '_6h_', '_12h_', '_24h_'])]
    temporal_cols = lag_cols + window_cols

    for col in temporal_cols:
        if col in shuffled.columns:
            shuffled[col] = rng.permutation(shuffled[col].values)

    X = shuffled[CLEAN_FEATURES]
    y = shuffled['disease_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1, verbosity=0,
                               scale_pos_weight=(y_train==0).sum()/max((y_train==1).sum(),1))
    model.fit(X_train, y_train, verbose=False)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    logger.info(f"  AUC with shuffled timestamps: {auc:.4f}")
    logger.info(f"  Expected: LOWER than original (temporal features broken)")
    return auc


def test_random_labels(df):
    """Train on random labels — should get AUC ~0.5 (no signal to learn)."""
    logger.info("\n── Test 3: Random Labels ──")
    rng = np.random.default_rng(42)
    X = df[CLEAN_FEATURES]
    y = pd.Series(rng.integers(0, 2, len(df)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train, verbose=False)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    logger.info(f"  AUC with random labels: {auc:.4f}")
    logger.info(f"  Expected: ~0.50 (no signal = no learning)")
    return auc


def test_original_baseline(df):
    """Baseline: original clean model AUC."""
    logger.info("\n── Baseline: Original Clean Model ──")
    model = joblib.load(os.path.join(MODEL_DIR, "model_no_leak.pkl"))

    animals = df['animal_id'].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(animals)
    test_animals = animals[int(len(animals) * 0.8):]
    mask = df['animal_id'].isin(test_animals)

    X_test = df.loc[mask, CLEAN_FEATURES]
    y_test = df.loc[mask, 'disease_binary']
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    logger.info(f"  Baseline AUC (unseen animals): {auc:.4f}")
    return auc


def main():
    logger.info("=" * 60)
    logger.info("🧬 Memorization Detection Test")
    logger.info("=" * 60)

    df = load_fused()
    logger.info(f"Loaded: {len(df)} rows, {df['disease_binary'].sum()} positive")

    baseline_auc = test_original_baseline(df)
    shuffle_animal_auc = test_shuffle_animals(df)
    shuffle_time_auc = test_shuffle_timestamps(df)
    random_label_auc = test_random_labels(df)

    # Verdicts
    results = {
        "baseline_auc": round(float(baseline_auc), 4),
        "shuffle_animal_auc": round(float(shuffle_animal_auc), 4),
        "shuffle_time_auc": round(float(shuffle_time_auc), 4),
        "random_label_auc": round(float(random_label_auc), 4),
        "verdicts": {
            "animal_memorization": {
                "test": "Shuffle animal IDs",
                "result": round(float(shuffle_animal_auc), 4),
                "passed": abs(shuffle_animal_auc - baseline_auc) < 0.15,
                "reason": "Similar AUC = learns physiology not identity"
            },
            "temporal_dependency": {
                "test": "Shuffle timestamps",
                "result": round(float(shuffle_time_auc), 4),
                "passed": shuffle_time_auc < baseline_auc,
                "reason": "Lower AUC = genuinely uses temporal features"
            },
            "random_label_check": {
                "test": "Random labels",
                "result": round(float(random_label_auc), 4),
                "passed": random_label_auc < 0.55,
                "reason": "~0.5 AUC = no spurious patterns"
            }
        }
    }

    all_passed = all(v['passed'] for v in results['verdicts'].values())
    results['overall_passed'] = all_passed

    with open(os.path.join(DATA_DIR, "memorization_test.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("📊 MEMORIZATION TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Baseline AUC:          {baseline_auc:.4f}")
    logger.info(f"  Shuffle Animals:       {shuffle_animal_auc:.4f} "
                f"{'✅' if results['verdicts']['animal_memorization']['passed'] else '❌'}")
    logger.info(f"  Shuffle Timestamps:    {shuffle_time_auc:.4f} "
                f"{'✅' if results['verdicts']['temporal_dependency']['passed'] else '❌'}")
    logger.info(f"  Random Labels:         {random_label_auc:.4f} "
                f"{'✅' if results['verdicts']['random_label_check']['passed'] else '❌'}")
    logger.info(f"\n  Overall: {'✅ NO MEMORIZATION' if all_passed else '❌ MEMORIZATION DETECTED'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
