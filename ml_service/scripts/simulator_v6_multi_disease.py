#!/usr/bin/env python3
"""
simulator_v6_multi_disease.py — Phase 12
Multi-Disease Biological Intelligence Engine Simulator.

Extends the 1D infection state to an 8D fully coupled biological state vector:
X(t) = [Infection, Heat Stress, Mastitis, Lameness, Calving, Immunity, Compensation, Fatigue]
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("SimV6")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"

N_ANIMALS = 30
TICKS_PER_ANIMAL = 5000  # ~34 days at 10m ticks to capture calving and overlap events
TICK_MIN = 10
TICKS_PER_HOUR = int(60 / TICK_MIN)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR

# Biological Constants
THI_THRESHOLD = 72.0


def generate_animal_v6(animal_idx):
    rng = np.random.RandomState(animal_idx * 17 + 42)
    n = TICKS_PER_ANIMAL
    aid = f"v6_animal_{animal_idx:04d}"

    # ── BASE PHYSIOLOGICAL PROFILE ──
    base = {
        "temp": 38.3 + rng.normal(0, 0.3),
        "hr": 62 + rng.normal(0, 8),
        "resp": 24 + rng.normal(0, 4),
        "act": 0.65 + rng.normal(0, 0.1),
        "milk": 30 + rng.normal(0, 6),
        "weight": 550 + rng.normal(0, 50),
        "heat_tolerance": rng.uniform(0.5, 1.5) # Genetic tolerance modifier
    }

    # ── ENVIRONMENT (Diurnal + Weather Trends) ──
    t_arr = np.arange(n)
    
    # Base diurnal swing
    ambient = 22 + 10 * np.sin(t_arr * 2 * np.pi / TICKS_PER_DAY) + rng.normal(0, 2, n)
    humidity = 55 + 15 * np.sin(t_arr * 2 * np.pi / (TICKS_PER_DAY * 3)) + rng.normal(0, 5, n)
    
    # Introduce a 5-day heatwave in the middle of the month
    heatwave_start = int(0.4 * n)
    heatwave_end = int(0.6 * n)
    ambient[heatwave_start:heatwave_end] += rng.uniform(8, 15)
    
    thi = (1.8 * ambient + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * ambient - 26))
    thi += rng.normal(0, 1.5, n)

    # ── 8D BIOLOGICAL STATE VECTOR ──
    X = {
        "I": np.zeros(n),    # Systemic Infection Load
        "H": np.zeros(n),    # Heat Stress Load
        "M": np.zeros(n),    # Mastitis Load
        "L": np.zeros(n),    # Lameness Load
        "C": np.zeros(n),    # Calving Proximity
        "Imm": np.ones(n),   # Immunity (starts at 1.0)
        "Comp": np.ones(n),  # Biological Compensation
        "Fat": np.zeros(n)   # Fatigue
    }

    # ── EVENT TRIGGERS (Asynchronous Disease Seeds) ──
    has_infection = rng.random() < 0.25
    has_mastitis = rng.random() < 0.20
    has_lameness = rng.random() < 0.15
    is_calving = rng.random() < 0.10

    infection_start = rng.randint(int(0.1*n), int(0.8*n)) if has_infection else -1
    primary_mastitis_start = rng.randint(int(0.1*n), int(0.8*n)) if has_mastitis else -1
    lameness_start = rng.randint(int(0.1*n), int(0.8*n)) if has_lameness else -1
    
    # Calving is a dedicated fixed biological clock if active
    calving_midpoint = rng.randint(int(0.4*n), int(0.9*n)) if is_calving else -1

    # ── STOCHASTIC DIFFERENTIAL EVOLUTION ──
    abx_active = 0
    hoof_trm = 0
    exposure = np.zeros(n)
    if has_infection: exposure[infection_start:infection_start + 144] = rng.uniform(0.01, 0.03)

    for t in range(1, n):
        # 1. Calving Clock (Deterministic mapping independent of load)
        if is_calving:
            days_to_calving = (t - calving_midpoint) / TICKS_PER_DAY
            X["C"][t] = 1.0 / (1.0 + np.exp(-2.0 * days_to_calving)) # Sigmoid approach

        # 2. Environmental Heat Load
        thi_excess = max(thi[t] - THI_THRESHOLD, 0)
        dH = (0.01 * thi_excess) - (0.05 * base["heat_tolerance"]) + rng.normal(0, 0.01)
        X["H"][t] = np.clip(X["H"][t-1] + dH, 0, 1.5) # Heat load caps out

        # 3. Immunity and Fatigue (Cross-coupled)
        dFat = (0.02 * X["H"][t]) + (0.05 * X["I"][t-1]) + (0.03 * X["C"][t]) - 0.01
        X["Fat"][t] = np.clip(X["Fat"][t-1] + dFat, 0, 1.0)
        
        # Heat and Fatigue crash immunity
        dImm = - (0.08 * X["Fat"][t]) - (0.05 * X["H"][t]) + 0.01
        X["Imm"][t] = np.clip(X["Imm"][t-1] + dImm, 0.1, 1.0)

        # 4. Infection Dynamics
        # Infection grows via exposure, suppressed by Immunity, killed by ABX
        dI = exposure[t-1] + (0.05 * X["I"][t-1]) - (0.04 * X["Imm"][t]) - (0.2 * abx_active) + rng.normal(0, 0.005)
        X["I"][t] = np.clip(X["I"][t-1] + dI, 0, 1.0)

        # 5. Mastitis Dynamics
        # Can trigger independently, or secondary to heavy systemic infection
        primary_m = 0.05 if (primary_mastitis_start > 0 and t > primary_mastitis_start and t < primary_mastitis_start + 200) else 0
        secondary_m = 0.08 * max(X["I"][t] - 0.6, 0) # Only triggers if I > 0.6
        dM = primary_m + secondary_m + (0.02 * X["M"][t-1]) - (0.05 * X["Imm"][t]) - (0.25 * abx_active) + rng.normal(0, 0.005)
        X["M"][t] = np.clip(X["M"][t-1] + dM, 0, 1.0)

        # 6. Lameness Dynamics
        # Triggered by base probability or indirectly via excessive resting/fatigue compounding (simplified here)
        primary_l = 0.03 if (lameness_start > 0 and t > lameness_start and t < lameness_start + 500) else 0
        dL = primary_l + (0.01 * X["L"][t-1]) - hoof_trm + rng.normal(0, 0.002)
        X["L"][t] = np.clip(X["L"][t-1] + dL, 0, 1.0)

        # (Farmer Interventions)
        # If any load gets violently high, farmer notices and treats 24h later
        if (X["I"][t] > 0.8 or X["M"][t] > 0.7) and abx_active == 0 and rng.random() < 0.1:
            abx_active = 1.0 # STUB: In reality we'd delay this, but applying instantly for diff eq stability
        if abx_active > 0:
            abx_active *= 0.95 # decays out

        if X["L"][t] > 0.8 and hoof_trm == 0 and rng.random() < 0.05:
            hoof_trm = 0.5
        if hoof_trm > 0:
            hoof_trm *= 0.90


    # ── SIGNAL-LEVEL GENERATION (Physics Mapping) ──
    # AR(1) Hardware Noise
    temp_noise = np.zeros(n); hr_noise = np.zeros(n); resp_noise = np.zeros(n); act_noise = np.zeros(n)
    rho = 0.85
    for t in range(1, n):
        temp_noise[t] = rho * temp_noise[t-1] + rng.normal(0, 0.15)
        hr_noise[t] = rho * hr_noise[t-1] + rng.normal(0, 1.5)
        act_noise[t] = rho * act_noise[t-1] + rng.normal(0, 0.03)

    # The actual observable curves
    temp_curve = base["temp"] + (2.5 * X["I"]) + (1.2 * X["H"]) + (0.4 * X["M"]) - (0.3 * X["C"]) + temp_noise
    hr_curve = base["hr"] + (20 * X["I"]) + (12 * X["H"]) + (8 * X["C"]) + hr_noise
    resp_curve = base["resp"] + (15 * X["H"]) + (5 * X["I"]) + rng.normal(0, 2, n)
    
    # Activity crushed by lameness, but calving causes "restlessness" spikes
    act_curve = base["act"] - (0.5 * X["L"]) - (0.3 * X["I"]) + (X["C"] * rng.normal(0, 0.2, n)) + act_noise
    act_curve = np.clip(act_curve, 0.1, 1.0)

    # Production
    milk_curve = base["milk"] - (6.0 * X["I"]) - (4.0 * X["H"]) - (12.0 * X["M"]) - (2.0 * X["L"]) + rng.normal(0, 1.5, n)
    milk_curve = np.clip(milk_curve, 0.0, 50.0)
    cond_curve = 5.0 + (3.5 * X["M"]) + rng.normal(0, 0.2, n)
    
    # Global severity metric (used for evaluation, regression targets)
    severity_label = np.clip(X["I"] + (X["M"]*1.5) + (X["L"]*0.8) + (X["H"]*0.5), 0, 3.0)

    timestamps = pd.date_range("2024-05-01", periods=n, freq=f"{TICK_MIN}min")

    sensor_docs = []
    prod_docs = []
    
    for t in range(n):
        base_obj = {"animalId": aid, "timestamp": timestamps[t], "simulationVersion": "v6_multi_disease"}
        
        # Sensor Collection
        sensor_docs.append({
            **base_obj,
            "signals": {
                "temperature_C": round(float(temp_curve[t]), 2),
                "heartRate_bpm": round(float(hr_curve[t]), 1),
                "respiration_bpm": round(float(resp_curve[t]), 1),
                "activity_index": round(float(act_curve[t]), 3)
            },
            "environment": {
                "thi": round(float(thi[t]), 1),
                "ambientTemp_C": round(float(ambient[t]), 1),
                "humidity_pct": round(float(humidity[t]), 1)
            },
            "labels": {
                "infectionBinary": int(X["I"][t] > 0.4),
                "heatStressBinary": int(X["H"][t] > 0.5),
                "mastitisBinary": int(X["M"][t] > 0.4),
                "lamenessBinary": int(X["L"][t] > 0.4),
                "calvingBinary": int(X["C"][t] > 0.5),
                "severityLevel": round(float(severity_label[t]), 2)
            }
        })
        
        # Production Collection (Only save once every 12 hours ~ 2 milkings a day)
        if t % (TICKS_PER_HOUR * 12) == 0:
            prod_docs.append({
                **base_obj,
                "production": {
                    "milkYield": round(float(milk_curve[t]), 1),
                    "feedIntake": round(float(22 - (X["I"][t]*5)), 1),
                    "conductivity": round(float(cond_curve[t]), 2)
                },
                "management": {
                    "antibioticActive": int(abx_active > 0)
                }
            })

    return sensor_docs, prod_docs


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("🧬 Phase 12 — Multi-Disease Evolution Simulator v6.0")
    logger.info("=" * 60)

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        db = client[DB_NAME]
        use_mongo = True
        logger.info("Connected to MongoDB. Will write to databases AND dump to CSV.")
        for col in ["trainingevents_v6_sensor", "trainingevents_v6_production"]:
            db[col].drop()
    except Exception as e:
        use_mongo = False
        logger.warning(f"Failed to connect to MongoDB ({e}). Will dump directly to CSV files instead.")

    all_sensor = []
    all_prod = []

    for i in range(N_ANIMALS):
        s, p = generate_animal_v6(i)
        all_sensor.extend(s)
        all_prod.extend(p)
        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i+1}/{N_ANIMALS} multi-disease trajectories...")

    logger.info(f"Total: {len(all_sensor)} Sensor Ticks, {len(all_prod)} Production Ticks")

    if use_mongo:
        bs = 20000
        logger.info("Writing Sensor data to MangoDB...")
        for i in range(0, len(all_sensor), bs):
            db["trainingevents_v6_sensor"].insert_many(all_sensor[i:i+bs])
            
        logger.info("Writing Production data to MongoDB...")
        for i in range(0, len(all_prod), bs):
            db["trainingevents_v6_production"].insert_many(all_prod[i:i+bs])

        logger.info("Building V6 Indexes...")
        db["trainingevents_v6_sensor"].create_index([("animalId", 1), ("timestamp", -1)])
        db["trainingevents_v6_production"].create_index([("animalId", 1), ("timestamp", -1)])

    logger.info("Writing flattened CSV representations to /training_data...")
    DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    flat_sensor = []
    for s in all_sensor:
        flat_sensor.append({
            "animalId": s["animalId"], "timestamp": s["timestamp"],
            **s["signals"], **s["environment"], **s["labels"]
        })
    df_sensor = pd.DataFrame(flat_sensor)
    df_sensor.to_csv(os.path.join(DATA_DIR, "v6_sensor_raw.csv"), index=False)
    
    flat_prod = []
    for p in all_prod:
        flat_prod.append({
            "animalId": p["animalId"], "timestamp": p["timestamp"],
            **p["production"], **p["management"]
        })
    df_prod = pd.DataFrame(flat_prod)
    df_prod.to_csv(os.path.join(DATA_DIR, "v6_production_raw.csv"), index=False)

    logger.info(f"\n✅ V6 Generation Done in {time.time() - start:.1f}s")
    if use_mongo: client.close()

if __name__ == "__main__":
    main()
