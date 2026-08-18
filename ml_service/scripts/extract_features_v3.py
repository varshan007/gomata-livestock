#!/usr/bin/env python3
"""
extract_features_v3.py — GoMata Model Hardening v2, Phase 1A
V3 Feature Engine (Sensor Backbone) — Optimized Batch Version

Extracts 42 physiological features from validation_clean_v3.
Uses embedded features from DB + computes lag/window features per-animal.

Usage:
  python extract_features_v3.py [--source validation_clean_v3] [--limit 200000]
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
from pymongo import MongoClient

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "extract_v3.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V3Extractor")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"


def batch_load(db, source, limit):
    """Batch load all records and flatten."""
    logger.info(f"Loading from {source} (limit={limit})...")
    col = db[source]
    cursor = col.find({}).sort("timestamp", 1).limit(limit)

    rows = []
    for doc in cursor:
        sig = doc.get("signals", {})
        feat = doc.get("features", {})
        env = doc.get("environment", {})
        lbl = doc.get("labels", {})
        hid = doc.get("hiddenState", {})

        rows.append({
            "animal_id": str(doc.get("animalId")),
            "timestamp": doc.get("timestamp"),
            # Signals
            "temp_raw": sig.get("temperature_C", 38.5),
            "hr_raw": sig.get("heartRate_bpm", 70),
            "resp_raw": sig.get("respiration_bpm", 28),
            "activity_raw": sig.get("activity_index", 0.5),
            "rumination_raw": sig.get("rumination_min", 30),
            "lying_raw": sig.get("lying_min", 30),
            # Embedded features
            "temp_6h_avg_emb": feat.get("temp_6h_avg", 38.5),
            "temp_6h_std_emb": feat.get("temp_6h_std", 0.1),
            "temp_6h_slope_emb": feat.get("temp_6h_slope", 0),
            "temp_zscore_emb": feat.get("temp_zscore", 0),
            "hr_6h_avg_emb": feat.get("hr_6h_avg", 70),
            "hr_6h_std_emb": feat.get("hr_6h_std", 2),
            "activity_6h_avg_emb": feat.get("activity_6h_avg", 0.5),
            "activity_6h_std_emb": feat.get("activity_6h_std", 0.1),
            "hsi_emb": feat.get("heat_stress_index", 0),
            "composite_stress_emb": feat.get("composite_stress_index", 0),
            "rumination_drop_emb": feat.get("rumination_drop", 0),
            "autocorr_temp_emb": feat.get("autocorrelation_temp", 0),
            # Environment
            "thi": env.get("thi", 65),
            "ambient_temp": env.get("ambientTemp_C", 25),
            "humidity": env.get("humidity_pct", 60),
            # Labels
            "disease_binary": lbl.get("diseaseBinary", 0),
            "severity_level": lbl.get("severityLevel", 0),
            "infection_binary": lbl.get("infectionBinary", 0),
            "stress_binary": lbl.get("stressBinary", 0),
            "mixed_binary": lbl.get("mixedStateBinary", 0),
            "disease_type": lbl.get("diseaseType", "none"),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} records from {len(df['animal_id'].unique())} animals")
    return df


def compute_per_animal_features(df):
    """Compute lag stack and multi-scale window features per animal."""
    logger.info("Computing per-animal temporal features...")
    all_animals = []

    for animal_id, group in df.groupby("animal_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        n = len(g)

        # Multi-scale windows (vectorized pandas rolling)
        for sig, prefix in [("temp_raw", "temp"), ("hr_raw", "hr"),
                            ("activity_raw", "activity"), ("resp_raw", "resp")]:
            s = g[sig].astype(float)
            g[f"{prefix}_1h_avg"] = s.rolling(12, min_periods=1).mean()
            if prefix in ("temp", "hr", "activity"):
                g[f"{prefix}_6h_avg"] = s.rolling(72, min_periods=1).mean()
                g[f"{prefix}_6h_std"] = s.rolling(72, min_periods=1).std().fillna(0)
            if prefix in ("temp", "hr"):
                g[f"{prefix}_12h_median"] = s.rolling(144, min_periods=1).median()
            if prefix == "temp":
                g[f"{prefix}_24h_std"] = s.rolling(288, min_periods=1).std().fillna(0)
                g[f"{prefix}_1h_std"] = s.rolling(12, min_periods=1).std().fillna(0)
            if prefix == "resp":
                g[f"{prefix}_6h_avg"] = s.rolling(72, min_periods=1).mean()
                g[f"{prefix}_6h_std"] = s.rolling(72, min_periods=1).std().fillna(0)

        # Lag stack features
        for sig, prefix in [("temp_raw", "temp"), ("hr_raw", "hr"), ("activity_raw", "activity")]:
            s = g[sig].astype(float)
            g[f"{prefix}_lag_1h"] = s.shift(12).fillna(method='bfill')
            g[f"{prefix}_lag_3h"] = s.shift(36).fillna(method='bfill')
            g[f"{prefix}_lag_6h"] = s.shift(72).fillna(method='bfill')
            g[f"{prefix}_lag_12h"] = s.shift(144).fillna(method='bfill')

        all_animals.append(g)

    result = pd.concat(all_animals, ignore_index=True)
    logger.info(f"Computed features for {len(df['animal_id'].unique())} animals, {len(result)} rows")
    return result


def build_final_dataset(df):
    """Build final feature matrix with all 42 features."""
    out = pd.DataFrame()

    # Current signals (6)
    out["temp_current"] = df["temp_raw"]
    out["hr_current"] = df["hr_raw"]
    out["resp_current"] = df["resp_raw"]
    out["activity_current"] = df["activity_raw"]
    out["rumination_current"] = df["rumination_raw"]
    out["lying_current"] = df["lying_raw"]

    # Multi-scale temp (6)
    out["temp_1h_avg"] = df["temp_1h_avg"]
    out["temp_6h_avg"] = df.get("temp_6h_avg", df["temp_6h_avg_emb"])
    out["temp_12h_median"] = df.get("temp_12h_median", df["temp_raw"])
    out["temp_24h_std"] = df.get("temp_24h_std", 0)
    out["temp_6h_std"] = df.get("temp_6h_std", df["temp_6h_std_emb"])
    out["temp_1h_std"] = df.get("temp_1h_std", 0)

    # Multi-scale HR (4)
    out["hr_1h_avg"] = df["hr_1h_avg"]
    out["hr_6h_avg"] = df.get("hr_6h_avg", df["hr_6h_avg_emb"])
    out["hr_12h_median"] = df.get("hr_12h_median", df["hr_raw"])
    out["hr_6h_std"] = df.get("hr_6h_std", df["hr_6h_std_emb"])

    # Multi-scale activity (3)
    out["activity_1h_avg"] = df["activity_1h_avg"]
    out["activity_6h_avg"] = df.get("activity_6h_avg", df["activity_6h_avg_emb"])
    out["activity_6h_std"] = df.get("activity_6h_std", df["activity_6h_std_emb"])

    # Multi-scale resp (2)
    out["resp_6h_avg"] = df.get("resp_6h_avg", df["resp_raw"])
    out["resp_6h_std"] = df.get("resp_6h_std", 0)

    # Lag stack temp (4)
    out["temp_lag_1h"] = df["temp_lag_1h"]
    out["temp_lag_3h"] = df["temp_lag_3h"]
    out["temp_lag_6h"] = df["temp_lag_6h"]
    out["temp_lag_12h"] = df["temp_lag_12h"]

    # Lag stack HR (4)
    out["hr_lag_1h"] = df["hr_lag_1h"]
    out["hr_lag_3h"] = df["hr_lag_3h"]
    out["hr_lag_6h"] = df["hr_lag_6h"]
    out["hr_lag_12h"] = df["hr_lag_12h"]

    # Lag stack activity (4)
    out["activity_lag_1h"] = df["activity_lag_1h"]
    out["activity_lag_3h"] = df["activity_lag_3h"]
    out["activity_lag_6h"] = df["activity_lag_6h"]
    out["activity_lag_12h"] = df["activity_lag_12h"]

    # Stability (6)
    out["temp_zscore"] = df["temp_zscore_emb"]
    out["hsi"] = df["hsi_emb"]
    out["composite_stress"] = df["composite_stress_emb"]
    out["rumination_drop"] = df["rumination_drop_emb"]
    out["autocorr_temp"] = df["autocorr_temp_emb"]
    out["temp_slope_6h"] = df["temp_6h_slope_emb"]

    # Environment (3)
    out["thi"] = df["thi"]
    out["ambient_temp"] = df["ambient_temp"]
    out["humidity"] = df["humidity"]

    # Labels
    out["disease_binary"] = df["disease_binary"]
    out["severity_level"] = df["severity_level"]
    out["infection_binary"] = df["infection_binary"]
    out["stress_binary"] = df["stress_binary"]
    out["mixed_binary"] = df["mixed_binary"]
    out["disease_type"] = df["disease_type"]
    out["animal_id"] = df["animal_id"]

    return out.fillna(0)


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
]  # 42 features


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="validation_clean_v3")
    parser.add_argument("--limit", type=int, default=200000)
    args = parser.parse_args()

    start = time.time()
    logger.info("=" * 60)
    logger.info("🧬 V3 Feature Engine — Sensor Backbone (42 features)")
    logger.info("=" * 60)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    df_raw = batch_load(db, args.source, args.limit)
    df_feat = compute_per_animal_features(df_raw)
    df_final = build_final_dataset(df_feat)

    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "features_v3_hardened.csv")
    df_final.to_csv(csv_path, index=False)

    config = {"features": V3_FEATURES, "count": len(V3_FEATURES), "version": "v3_hardened"}
    with open(os.path.join(output_dir, "feature_config_v3.json"), "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ V3 Extraction Complete")
    logger.info(f"   Records: {len(df_final)}")
    logger.info(f"   Features: {len(V3_FEATURES)}")
    logger.info(f"   Disease: {df_final['disease_binary'].sum():.0f} / {len(df_final)}")
    logger.info(f"   CSV: {csv_path}")
    logger.info(f"   Duration: {elapsed:.1f}s")
    logger.info("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
