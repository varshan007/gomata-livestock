#!/usr/bin/env python3
"""
extract_features_v6.py — Phase 7: Acceleration + Pre-Onset Features
Vectorized version — fast enough for 185K rows.

Adds: slopes, acceleration, instability, variance ratios, cross-modal ratios.
Creates: onset_binary (severity≥2 within 24h), temporal_weight.
Output: features_v6.csv
"""

import os, sys, json, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from v5_config import ALL_FEATURES as V5_FEATURES, DATA_DIR

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "extract_v6.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V6Extractor")

TICKS_24H = 288


def vectorized_slope(series, window):
    """Fast slope: (last - first) / window for rolling window."""
    shifted = series.shift(window - 1)
    return ((series - shifted) / max(window, 1)).fillna(0)


def add_accel_features(df):
    """Add acceleration features per animal using fast vectorized ops."""
    logger.info("Computing acceleration features per animal...")
    results = []

    for animal_id, group in df.groupby("animal_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)

        for sig, prefix in [("temp_current", "temp"), ("hr_current", "hr"),
                            ("resp_current", "resp"), ("activity_current", "activity")]:
            s = g[sig].astype(float)

            # Slopes (simple: (current - past) / window)
            g[f"{prefix}_slope_1h"] = vectorized_slope(s, 12)
            g[f"{prefix}_slope_3h"] = vectorized_slope(s, 36)
            g[f"{prefix}_slope_6h"] = vectorized_slope(s, 72)

            # Acceleration (slope of slope_3h)
            g[f"{prefix}_accel_3h"] = vectorized_slope(g[f"{prefix}_slope_3h"], 36)

            # Instability = std_3h / std_24h
            std_3h = s.rolling(36, min_periods=1).std().fillna(0.001)
            std_24h = s.rolling(288, min_periods=1).std().fillna(0.001).clip(lower=0.001)
            g[f"{prefix}_instability"] = (std_3h / std_24h).round(4)

            # Variance ratio (6h vs 24h)
            var_6h = s.rolling(72, min_periods=1).var().fillna(0)
            var_24h = s.rolling(288, min_periods=1).var().fillna(0.001).clip(lower=0.001)
            g[f"{prefix}_var_ratio"] = (var_6h / var_24h).round(4)

            # 7-day baseline deviation
            baseline = s.rolling(288 * 7, min_periods=288).mean()
            baseline = baseline.fillna(s.expanding().mean())
            g[f"{prefix}_delta_7d"] = (s - baseline).round(4)

            # Volatility spike (std_1h > 2 × std_24h)
            std_1h = s.rolling(12, min_periods=1).std().fillna(0)
            g[f"{prefix}_volatility_spike"] = (std_1h > 2 * std_24h).astype(int)

        # Cross-modal ratios
        g["hr_temp_ratio"] = (g["hr_current"] / g["temp_current"].clip(lower=36)).round(4)
        g["resp_activity_ratio"] = (g["resp_current"] / g["activity_current"].clip(lower=0.01)).round(4)

        # Rumination velocity
        if "rumination_current" in g.columns:
            g["rumination_velocity"] = g["rumination_current"].diff().fillna(0).rolling(12, min_periods=1).mean()
        else:
            g["rumination_velocity"] = 0

        results.append(g)

    result = pd.concat(results, ignore_index=True)
    logger.info(f"Done: {len(result)} rows from {df['animal_id'].nunique()} animals")
    return result


def create_onset_label(df):
    """onset_binary: 1 if severity ≥2 within next 24h (288 ticks)."""
    logger.info("Creating onset_binary...")
    results = []
    for animal_id, group in df.groupby("animal_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        sev = g["severity_level"].values.astype(float)
        ng = len(g)
        onset = np.zeros(ng, dtype=int)
        # Vectorized: reverse cummax-style
        future_severe = np.zeros(ng, dtype=bool)
        for i in range(ng - 2, -1, -1):
            # Check if any severity ≥ 2 in next 288 ticks
            j_end = min(i + TICKS_24H, ng)
            if np.any(sev[i+1:j_end] >= 2):
                onset[i] = 1
        g["onset_binary"] = onset
        results.append(g)
    result = pd.concat(results, ignore_index=True)
    logger.info(f"Onset: {result['onset_binary'].sum()} positive ({result['onset_binary'].mean()*100:.2f}%)")
    return result


def create_temporal_weights(df):
    """w = 1 + 3 × exp(-hours_to_event / 24) for samples before severity events."""
    logger.info("Computing temporal weights...")
    results = []
    for animal_id, group in df.groupby("animal_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        sev = g["severity_level"].values.astype(float)
        ng = len(g)
        weights = np.ones(ng)
        sev_events = np.where(sev >= 2)[0]
        for evt in sev_events:
            for j in range(max(0, evt - 576), evt):
                hours_before = (evt - j) * (5 / 60)
                w = 1 + 3 * np.exp(-hours_before / 24)
                weights[j] = max(weights[j], w)
        g["temporal_weight"] = np.round(weights, 4)
        results.append(g)
    return pd.concat(results, ignore_index=True)


def main():
    logger.info("=" * 60)
    logger.info("🧠 Phase 7 — V6 Feature Extraction (vectorized)")
    logger.info("=" * 60)

    # Load v5 features
    v3 = pd.read_csv(os.path.join(DATA_DIR, "features_v3_v5.csv"))
    v4 = pd.read_csv(os.path.join(DATA_DIR, "features_v4_v5.csv"))
    logger.info(f"V3: {len(v3)}, V4: {len(v4)}")

    min_len = min(len(v3), len(v4))
    v3, v4 = v3.iloc[:min_len].reset_index(drop=True), v4.iloc[:min_len].reset_index(drop=True)

    df = v3.copy()
    for col in v4.columns:
        if col not in df.columns:
            df[col] = v4[col]

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="5min")

    logger.info(f"Fused: {len(df)} rows, {df['animal_id'].nunique()} animals")

    # Add acceleration features
    df = add_accel_features(df)

    # Create onset label
    df = create_onset_label(df)

    # Create temporal weights
    df = create_temporal_weights(df)

    # Save ALL columns (no filtering)
    meta_cols = {"animal_id", "disease_binary", "severity_level",
                 "onset_binary", "temporal_weight", "timestamp"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    save_cols = feature_cols + list(meta_cols & set(df.columns))

    out_path = os.path.join(DATA_DIR, "features_v6.csv")
    df[save_cols].to_csv(out_path, index=False)

    config = {
        "version": "v6_onset", "total_features": len(feature_cols),
        "rows": len(df), "animals": int(df["animal_id"].nunique()),
        "onset_positive": int(df["onset_binary"].sum()),
        "onset_rate": round(df["onset_binary"].mean() * 100, 2),
    }
    with open(os.path.join(DATA_DIR, "feature_config_v6.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ V6 Features: {out_path}")
    logger.info(f"   {len(df)} rows × {len(feature_cols)} features")
    logger.info(f"   Animals: {df['animal_id'].nunique()}")
    logger.info(f"   Onset rate: {config['onset_rate']:.2f}%")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
