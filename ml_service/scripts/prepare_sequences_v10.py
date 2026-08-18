#!/usr/bin/env python3
"""
prepare_sequences_v10.py

Extracts v5.2 data from MongoDB, downsamples from 5-min to 10-min ticks,
computes raw + minimal engineered features, and saves a flat time-series 
dataset. The PyTorch Dataset will dynamically slice this into 48h sequences.
"""

import os, sys, time, logging
import numpy as np
import pandas as pd
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("PrepV10")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
TICK_HOURS = 10 / 60  # Downsampled to 10 minutes

def extract_flat_time_series():
    start = time.time()
    logger.info("── Loading v5.2 from MongoDB ──")
    client = MongoClient(MONGO_URI)
    db = client["livestock_monitoring"]

    logger.info("Loading sensor docs (this might take a minute)...")
    sensor_cursor = db["trainingevents_v5_2"].find({}, {"_id": 0}).sort([("animalId", 1), ("timestamp", 1)])
    prod_cursor = db["trainingevents_v5_2_production"].find({}, {"_id": 0}).sort([("animalId", 1), ("timestamp", 1)])

    sensor_docs = list(sensor_cursor)
    prod_docs = list(prod_cursor)
    
    logger.info(f"Loaded {len(sensor_docs)} sensor and {len(prod_docs)} production docs.")
    client.close()

    rows = []
    # Assume sorting matches perfectly since they were generated in sync
    for i in range(len(sensor_docs)):
        s = sensor_docs[i]
        p = prod_docs[i]
        
        # Stride by 2 to get 10-minute bins
        if i % 2 != 0:
            continue
            
        sig = s.get("signals", {})
        env = s.get("environment", {})
        lab = s.get("labels", {})
        
        prod = p.get("production", {})
        mgmt = p.get("management", {})

        rows.append({
            "animal_id": s["animalId"],
            "timestamp": s["timestamp"],
            
            # Raw features
            "temp": sig.get("temperature_C", 38.5),
            "hr": sig.get("heartRate_bpm", 70),
            "resp": sig.get("respiration_bpm", 25),
            "activity": sig.get("activity_index", 0.5),
            "rumination": sig.get("rumination_min", 38),
            "lying": sig.get("lying_min", 25),
            
            "thi": env.get("thi", 65),
            "ambient_temp": env.get("ambientTemp_C", 22),
            "humidity": env.get("humidity_pct", 55),
            
            "milk_yield": prod.get("milkYield", 25),
            "feed_intake": prod.get("feedIntake", 20),
            "conductivity": prod.get("conductivity", 5.0),
            "body_weight": prod.get("bodyWeight", 550),
            
            "vaccination_event": mgmt.get("vaccinationEffective", 0),
            "antibiotic_event": mgmt.get("antibioticEffective", 0),
            
            # Target labels
            "disease": lab.get("diseaseBinary", 0),
            "severity": lab.get("severityLevel", 0),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Flattened DataFrame: {len(df)} rows (10-min ticks)")
    
    # Calculate minimal features per animal
    logger.info("── Computing minimal engineered features ──")
    results = []
    
    for aid, g in df.groupby("animal_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        # Random invariants for the animal
        rng = np.random.RandomState(hash(aid) % 2**31)
        g["parity"] = rng.randint(1, 5)
        g["bcs"] = 3.0 + rng.normal(0, 0.3)
        g["age"] = rng.randint(2, 8)
        
        # Calculate hours since events
        for ev_col in ["vaccination_event", "antibiotic_event"]:
            base_name = ev_col.split("_")[0]
            last_event = np.zeros(len(g))
            for i in range(len(g)):
                if g[ev_col].iloc[i]:
                    last_event[i:] = i
            hours_since = (np.arange(len(g)) - last_event) * TICK_HOURS
            g[f"hours_since_{base_name}"] = np.where(last_event > 0, hours_since, 999)
            
        # Target vectors: Disease within next 24h (144 ticks), max severity within next 24h
        # 144 ticks @ 10m = 24 hours. Let's make sure it's 144.
        ng = len(g)
        sev = g["severity"].values
        disease_24h = np.zeros(ng, dtype=int)
        severity_24h = np.zeros(ng, dtype=float)
        
        for i in range(ng):
            j_end = min(i + 144, ng)
            window_sev = sev[i+1:j_end]
            if len(window_sev) > 0:
                severity_24h[i] = np.max(window_sev)
                if np.any(window_sev >= 2.0):  # Using 2.0 as the clinical threshold
                    disease_24h[i] = 1
                    
        g["target_disease_24h"] = disease_24h
        g["target_severity_24h"] = np.round(severity_24h, 3)
        
        results.append(g)

    final_df = pd.concat(results, ignore_index=True)
    
    # Drop intermediate event columns
    final_df = final_df.drop(columns=["vaccination_event", "antibiotic_event"])
    
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "v10_flat_sequences.csv")
    final_df.to_csv(out_path, index=False)
    
    elapsed = time.time() - start
    logger.info(f"✅ Saved v10 tabular time-series to {out_path} in {elapsed:.1f}s")
    logger.info(f"   Shape: {final_df.shape}. Animals: {final_df['animal_id'].nunique()}")
    logger.info(f"   Positives (disease in next 24h): {final_df['target_disease_24h'].sum()} ({final_df['target_disease_24h'].mean()*100:.1f}%)")

if __name__ == "__main__":
    extract_flat_time_series()
