#!/usr/bin/env python3
"""
extract_features_v4.py — GoMata Model Hardening v2, Phase 1B
V4 Feature Engine (Production + Management Context)

Extracts 18+ contextual features from validation_clean_v4:
  - Production deviations (milk, conductivity, feed, weight)
  - Time-since management events (vaccination, antibiotic, transport)
  - Event decay embeddings: exp(-k * hours_since)
  - Cumulative exposure (antibiotic days, vacc count, feed changes)
  - Animal profile features

Usage:
  python extract_features_v4.py [--source validation_clean_v4] [--limit 200000]
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
from pymongo import MongoClient

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "extract_v4.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V4Extractor")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"

# Decay constants (per-hour)
K_VACC = 0.005       # vaccination: slow decay (~6 day half-life)
K_ABX = 0.02         # antibiotic: fast decay (~1.5 day half-life)
K_TRANSPORT = 0.03   # transport: very fast decay (~1 day)
K_FEED = 0.01        # feed change: moderate decay (~3 day)


def extract_v4_features(db, source_collection="validation_clean_v4", limit=200000):
    """Extract 18+ production/management features per animal time series."""
    col = db[source_collection]
    total = col.count_documents({})
    logger.info(f"Source: {source_collection} ({total} docs)")

    animals = col.distinct("animalId")
    logger.info(f"Found {len(animals)} animals")

    all_rows = []
    for idx, animal_id in enumerate(animals):
        docs = list(col.find({"animalId": animal_id}).sort("timestamp", 1))
        if len(docs) < 12:
            continue

        # Track management event history per animal
        last_vacc_tick = -9999
        last_abx_tick = -9999
        last_transport_tick = -9999
        last_feed_change_tick = -9999
        cum_abx_ticks = 0
        cum_vacc_count = 0
        cum_feed_changes = 0
        prev_vacc = False
        prev_abx = False
        prev_feed = False

        for i, doc in enumerate(docs):
            prod = doc.get("production", {})
            mgmt = doc.get("managementEvents", {})
            profile = doc.get("animalProfile", {})
            labels = doc.get("labels", {})
            hidden = doc.get("hiddenState", {})

            # Track management transitions
            vacc_active = mgmt.get("vaccinationActive", False)
            abx_active = mgmt.get("antibioticActive", False)
            transport_active = mgmt.get("transportActive", False)
            feed_active = mgmt.get("feedChangeActive", False)

            if vacc_active and not prev_vacc:
                last_vacc_tick = i
                cum_vacc_count += 1
            if abx_active and not prev_abx:
                last_abx_tick = i
            if feed_active and not prev_feed:
                last_feed_change_tick = i
                cum_feed_changes += 1
            if transport_active:
                last_transport_tick = i
            if abx_active:
                cum_abx_ticks += 1

            prev_vacc = vacc_active
            prev_abx = abx_active
            prev_feed = feed_active

            # Time-since (convert ticks → hours, 5 min/tick)
            ticks_since_vacc = max(0, i - last_vacc_tick)
            ticks_since_abx = max(0, i - last_abx_tick)
            ticks_since_transport = max(0, i - last_transport_tick)
            ticks_since_feed = max(0, i - last_feed_change_tick)

            hours_since_vacc = ticks_since_vacc * 5 / 60
            hours_since_abx = ticks_since_abx * 5 / 60
            hours_since_transport = ticks_since_transport * 5 / 60
            hours_since_feed = ticks_since_feed * 5 / 60

            # Event decay embeddings
            vacc_decay = np.exp(-K_VACC * hours_since_vacc)
            abx_decay = np.exp(-K_ABX * hours_since_abx)
            transport_decay = np.exp(-K_TRANSPORT * hours_since_transport)
            feed_decay = np.exp(-K_FEED * hours_since_feed)

            # Cumulative exposure
            cum_abx_days = cum_abx_ticks * 5 / 1440  # ticks → days

            # Production deviations
            baseline_milk = profile.get("baselineMilkYield", 20)
            baseline_weight = profile.get("baselineWeight", 450)
            milk_yield = prod.get("milkYield", baseline_milk * 0.7)
            conductivity = prod.get("milkConductivity", 5.0)
            feed_intake = prod.get("feedIntake", 18)
            body_weight = prod.get("bodyWeight", baseline_weight)

            milk_dev = milk_yield / max(baseline_milk, 1)
            cond_dev = conductivity / 5.0
            feed_dev = feed_intake / 18.0
            weight_dev = body_weight / max(baseline_weight, 1)

            row = {
                # ── Production deviations (4) ──
                "milk_deviation": round(milk_dev, 4),
                "conductivity_deviation": round(cond_dev, 4),
                "feed_deviation": round(feed_dev, 4),
                "weight_deviation": round(weight_dev, 4),

                # ── Time-since features (4) ──
                "hours_since_vaccination": round(hours_since_vacc, 1),
                "hours_since_antibiotic": round(hours_since_abx, 1),
                "hours_since_transport": round(hours_since_transport, 1),
                "hours_since_feed_change": round(hours_since_feed, 1),

                # ── Event decay embeddings (4) ──
                "vacc_decay": round(vacc_decay, 4),
                "abx_decay": round(abx_decay, 4),
                "transport_decay": round(transport_decay, 4),
                "feed_decay": round(feed_decay, 4),

                # ── Cumulative exposure (3) ──
                "total_antibiotic_days": round(cum_abx_days, 2),
                "vaccination_count_12m": cum_vacc_count,
                "feed_changes_30d": cum_feed_changes,

                # ── Animal profile (3) ──
                "parity": profile.get("parity", 1),
                "bcs": profile.get("bodyConditionScore", 3.0),
                "age": profile.get("age", 4),

                # ── Labels ──
                "disease_binary": labels.get("diseaseBinary", 0),
                "severity_level": labels.get("severityLevel", 0),
                "infection_binary": labels.get("infectionBinary", 0),
                "stress_binary": labels.get("stressBinary", 0),
                "disease_type": labels.get("diseaseType", "none"),

                # ── Metadata ──
                "animal_id": str(doc.get("animalId")),
            }
            all_rows.append(row)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx+1}/{len(animals)} animals ({len(all_rows)} rows)")

    return pd.DataFrame(all_rows)


# Feature column list for model training
V4_FEATURES = [
    # Production (4)
    "milk_deviation", "conductivity_deviation", "feed_deviation", "weight_deviation",
    # Time-since (4)
    "hours_since_vaccination", "hours_since_antibiotic",
    "hours_since_transport", "hours_since_feed_change",
    # Event decay (4)
    "vacc_decay", "abx_decay", "transport_decay", "feed_decay",
    # Cumulative (3)
    "total_antibiotic_days", "vaccination_count_12m", "feed_changes_30d",
    # Profile (3)
    "parity", "bcs", "age",
]  # Total: 18 features


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="validation_clean_v4")
    parser.add_argument("--limit", type=int, default=200000)
    args = parser.parse_args()

    start = time.time()
    logger.info("=" * 60)
    logger.info("🧬 V4 Feature Engine — Production + Management (18 features)")
    logger.info("=" * 60)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    df = extract_v4_features(db, args.source, args.limit)

    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "features_v4_hardened.csv")
    df.to_csv(csv_path, index=False)

    config = {"features": V4_FEATURES, "count": len(V4_FEATURES), "version": "v4_hardened"}
    with open(os.path.join(output_dir, "feature_config_v4.json"), "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ V4 Extraction Complete")
    logger.info(f"   Records: {len(df)}")
    logger.info(f"   Features: {len(V4_FEATURES)}")
    logger.info(f"   Disease: {df['disease_binary'].sum():.0f} / {len(df)}")
    logger.info(f"   CSV: {csv_path}")
    logger.info(f"   Duration: {elapsed:.1f}s")
    logger.info("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
