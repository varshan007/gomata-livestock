#!/usr/bin/env python3
"""
extract_features_v4_v5.py — v5 Production Feature Extraction
Reads from trainingevents_v5_production, computes production/management features.

Output: features_v4_v5.csv (18 clean production/management features + labels)
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
from pymongo import MongoClient

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "extract_v4_v5.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V4V5Extractor")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"

# Decay constants for time-since features
K_VACC = 0.005       # vaccination: slow decay (~6 day half-life)
K_ABX = 0.02         # antibiotic: fast decay (~1.5 day half-life)
K_TRANSPORT = 0.03   # transport: very fast decay (~1 day)
K_FEED = 0.01        # feed change: moderate decay (~3 day)

V4_FEATURES = [
    "milk_deviation", "conductivity_deviation", "feed_deviation", "weight_deviation",
    "hours_since_vaccination", "hours_since_antibiotic",
    "hours_since_transport", "hours_since_feed_change",
    "vacc_decay", "abx_decay", "transport_decay", "feed_decay",
    "total_antibiotic_days", "vaccination_count_12m", "feed_changes_30d",
    "parity", "bcs", "age",
]


def batch_load(db, source, limit):
    """Load from v5 production collection."""
    logger.info(f"Loading from {source} (limit={limit})...")
    col = db[source]
    cursor = col.find({}).sort("timestamp", 1).limit(limit)
    rows = []
    for doc in cursor:
        prod = doc.get("production", {})
        mgmt = doc.get("management", {})
        env = doc.get("environment", {})
        lbl = doc.get("labels", {})
        hs = doc.get("hiddenState", {})
        rows.append({
            "animal_id": str(doc.get("animalId")),
            "timestamp": doc.get("timestamp"),
            "milkYield": prod.get("milkYield", 28),
            "feedIntake": prod.get("feedIntake", 22),
            "conductivity": prod.get("conductivity", 5.0),
            "bodyWeight": prod.get("bodyWeight", 550),
            "vaccinationEffective": mgmt.get("vaccinationEffective", 0),
            "antibioticEffective": mgmt.get("antibioticEffective", 0),
            "thi": env.get("thi", 68),
            "infectionLoad": hs.get("infectionLoad", 0),
            "disease_binary": lbl.get("diseaseBinary", 0),
            "severity_level": lbl.get("severityLevel", 0),
            "disease_type": lbl.get("diseaseType", "none"),
        })
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} records from {df['animal_id'].nunique()} animals")
    return df


def compute_v4_features(df):
    """Compute production deviation + management context features per animal."""
    logger.info("Computing V4 features per animal...")
    results = []

    for animal_id, group in df.groupby("animal_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        ng = len(g)

        # Per-animal baselines
        base_milk = g["milkYield"].rolling(288, min_periods=1).mean()
        base_feed = g["feedIntake"].rolling(288, min_periods=1).mean()
        base_cond = g["conductivity"].rolling(288, min_periods=1).mean()
        base_weight = g["bodyWeight"].rolling(288, min_periods=1).mean()

        g["milk_deviation"] = g["milkYield"] - base_milk
        g["conductivity_deviation"] = g["conductivity"] - base_cond
        g["feed_deviation"] = g["feedIntake"] - base_feed
        g["weight_deviation"] = g["bodyWeight"] - base_weight

        # Management time-since features (simulated, realistic intervals)
        np.random.seed(hash(animal_id) % 2**31)
        vacc_interval = np.random.randint(90, 365)
        abx_interval = np.random.randint(30, 180)
        transport_interval = np.random.randint(60, 365)
        feed_change_interval = np.random.randint(14, 60)

        ticks = np.arange(ng)
        g["hours_since_vaccination"] = ((ticks % (vacc_interval * 288)) / 12).round(1)
        g["hours_since_antibiotic"] = ((ticks % (abx_interval * 288)) / 12).round(1)
        g["hours_since_transport"] = ((ticks % (transport_interval * 288)) / 12).round(1)
        g["hours_since_feed_change"] = ((ticks % (feed_change_interval * 288)) / 12).round(1)

        # Exponential decay embeddings
        g["vacc_decay"] = np.exp(-K_VACC * g["hours_since_vaccination"]).round(4)
        g["abx_decay"] = np.exp(-K_ABX * g["hours_since_antibiotic"]).round(4)
        g["transport_decay"] = np.exp(-K_TRANSPORT * g["hours_since_transport"]).round(4)
        g["feed_decay"] = np.exp(-K_FEED * g["hours_since_feed_change"]).round(4)

        # Cumulative exposure
        g["total_antibiotic_days"] = (g["antibioticEffective"].cumsum() / 288).round(1)
        g["vaccination_count_12m"] = np.random.randint(1, 4)
        g["feed_changes_30d"] = np.random.randint(0, 5)

        # Animal profile
        g["parity"] = np.random.randint(1, 6)
        g["bcs"] = round(2.5 + np.random.random() * 1.5, 1)
        g["age"] = round(2 + np.random.random() * 6, 1)

        results.append(g)

    result = pd.concat(results, ignore_index=True)
    logger.info(f"V4 features computed: {len(result)} rows from {df['animal_id'].nunique()} animals")
    return result


def build_final(df):
    """Build final V4 feature matrix."""
    out = pd.DataFrame()
    for feat in V4_FEATURES:
        out[feat] = df[feat] if feat in df.columns else 0.0

    out["animal_id"] = df["animal_id"]
    out["disease_binary"] = df["disease_binary"]
    out["severity_level"] = df["severity_level"]

    logger.info(f"Final V4 matrix: {len(out)} rows × {len(V4_FEATURES)} features")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="trainingevents_v5_production")
    parser.add_argument("--limit", type=int, default=200000)
    args = parser.parse_args()

    start = time.time()
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    df = batch_load(db, args.source, args.limit)
    df = compute_v4_features(df)
    final = build_final(df)

    out_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "features_v4_v5.csv")
    final.to_csv(csv_path, index=False)

    config = {"version": "v5_production", "features": V4_FEATURES, "count": len(V4_FEATURES),
              "rows": len(final), "source": args.source}
    with open(os.path.join(out_dir, "feature_config_v4_v5.json"), "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ V4 v5 Features: {csv_path}")
    logger.info(f"   {len(final)} rows × {len(V4_FEATURES)} features")
    logger.info(f"   Disease rate: {final['disease_binary'].mean()*100:.1f}%")
    logger.info(f"   Duration: {elapsed:.1f}s")
    logger.info(f"{'='*60}")
    client.close()


if __name__ == "__main__":
    main()
