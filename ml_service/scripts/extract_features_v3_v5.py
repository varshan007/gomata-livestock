#!/usr/bin/env python3
"""
extract_features_v3_v5.py — v5 Sensor Feature Extraction
Reads from trainingevents_v5, computes rolling/lag features from raw signals.
No embedded features (v5 has raw signals only).

Output: features_v3_v5.csv (36 clean sensor features + labels)
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
from pymongo import MongoClient

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "extract_v3_v5.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V3V5Extractor")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"

# 36 clean V3 features (no rumination_drop, composite_stress, hsi, temp_zscore, autocorr_temp, temp_slope_6h)
V3_FEATURES = [
    # Current signals (6)
    "temp_current", "hr_current", "resp_current",
    "activity_current", "rumination_current", "lying_current",
    # Multi-scale windows (15)
    "temp_1h_avg", "temp_6h_avg", "temp_12h_median", "temp_24h_std",
    "temp_6h_std", "temp_1h_std",
    "hr_1h_avg", "hr_6h_avg", "hr_12h_median", "hr_6h_std",
    "activity_1h_avg", "activity_6h_avg", "activity_6h_std",
    "resp_6h_avg", "resp_6h_std",
    # Lag stacks (12)
    "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_12h",
    "hr_lag_1h", "hr_lag_3h", "hr_lag_6h", "hr_lag_12h",
    "activity_lag_1h", "activity_lag_3h", "activity_lag_6h", "activity_lag_12h",
    # Environment (3)
    "thi", "ambient_temp", "humidity",
]


def batch_load(db, source, limit):
    """Load raw signals from v5 collection."""
    logger.info(f"Loading from {source} (limit={limit})...")
    col = db[source]
    cursor = col.find({}).sort("timestamp", 1).limit(limit)
    rows = []
    for doc in cursor:
        sig = doc.get("signals", {})
        env = doc.get("environment", {})
        lbl = doc.get("labels", {})
        rows.append({
            "animal_id": str(doc.get("animalId")),
            "timestamp": doc.get("timestamp"),
            "temp_raw": sig.get("temperature_C", 38.5),
            "hr_raw": sig.get("heartRate_bpm", 65),
            "resp_raw": sig.get("respiration_bpm", 26),
            "activity_raw": sig.get("activity_index", 0.7),
            "rumination_raw": sig.get("rumination_min", 35),
            "lying_raw": sig.get("lying_min", 25),
            "thi": env.get("thi", 68),
            "ambient_temp": env.get("ambientTemp_C", 25),
            "humidity": env.get("humidity_pct", 60),
            "disease_binary": lbl.get("diseaseBinary", 0),
            "severity_level": lbl.get("severityLevel", 0),
            "infection_binary": lbl.get("infectionBinary", 0),
            "stress_binary": lbl.get("stressBinary", 0),
            "disease_type": lbl.get("diseaseType", "none"),
        })
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} records from {df['animal_id'].nunique()} animals")
    return df


def compute_per_animal_features(df):
    """Compute lag stack + rolling window features per animal."""
    logger.info("Computing per-animal temporal features...")
    results = []
    for animal_id, group in df.groupby("animal_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)

        # Rolling windows
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
                g["temp_24h_std"] = s.rolling(288, min_periods=1).std().fillna(0)
                g["temp_1h_std"] = s.rolling(12, min_periods=1).std().fillna(0)
            if prefix == "resp":
                g["resp_6h_avg"] = s.rolling(72, min_periods=1).mean()
                g["resp_6h_std"] = s.rolling(72, min_periods=1).std().fillna(0)

        # Lag stacks
        for sig, prefix in [("temp_raw", "temp"), ("hr_raw", "hr"), ("activity_raw", "activity")]:
            s = g[sig].astype(float)
            g[f"{prefix}_lag_1h"] = s.shift(12).bfill()
            g[f"{prefix}_lag_3h"] = s.shift(36).bfill()
            g[f"{prefix}_lag_6h"] = s.shift(72).bfill()
            g[f"{prefix}_lag_12h"] = s.shift(144).bfill()

        results.append(g)

    result = pd.concat(results, ignore_index=True)
    logger.info(f"Computed features for {df['animal_id'].nunique()} animals, {len(result)} rows")
    return result


def build_final(df):
    """Build final feature matrix."""
    out = pd.DataFrame()
    out["temp_current"] = df["temp_raw"]
    out["hr_current"] = df["hr_raw"]
    out["resp_current"] = df["resp_raw"]
    out["activity_current"] = df["activity_raw"]
    out["rumination_current"] = df["rumination_raw"]
    out["lying_current"] = df["lying_raw"]

    for feat in V3_FEATURES:
        if feat not in out.columns and feat in df.columns:
            out[feat] = df[feat]

    out["animal_id"] = df["animal_id"]
    out["disease_binary"] = df["disease_binary"]
    out["severity_level"] = df["severity_level"]

    logger.info(f"Final V3 matrix: {len(out)} rows × {len(V3_FEATURES)} features")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="trainingevents_v5")
    parser.add_argument("--limit", type=int, default=200000)
    args = parser.parse_args()

    start = time.time()
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    df = batch_load(db, args.source, args.limit)
    df = compute_per_animal_features(df)
    final = build_final(df)

    out_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "features_v3_v5.csv")
    final.to_csv(csv_path, index=False)

    config = {"version": "v5_sensor", "features": V3_FEATURES, "count": len(V3_FEATURES),
              "rows": len(final), "source": args.source}
    with open(os.path.join(out_dir, "feature_config_v3_v5.json"), "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ V3 v5 Features: {csv_path}")
    logger.info(f"   {len(final)} rows × {len(V3_FEATURES)} features")
    logger.info(f"   Disease rate: {final['disease_binary'].mean()*100:.1f}%")
    logger.info(f"   Duration: {elapsed:.1f}s")
    logger.info(f"{'='*60}")
    client.close()


if __name__ == "__main__":
    main()
