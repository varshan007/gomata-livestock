#!/usr/bin/env python3
"""
simulator_v5_1_drift_patch.py — Phase 8 Part 2
Fix silent-onset: add gradual pre-onset sigmoid drift to ALL disease episodes.

Key changes from v5:
  - infectionLoad follows sigmoid ramp with 12-24h drift window
  - HR/resp/activity drift 12-24h before severity escalation  
  - No instantaneous jumps; all transitions are gradual
  - Production signals mirror sensor drift timing

Output: trainingevents_v5_1 + trainingevents_v5_1_production
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("SimV51")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"

N_ANIMALS = 100
TICKS_PER_ANIMAL = 2000
TICK_MINUTES = 5
TICKS_PER_HOUR = 12
DISEASE_RATE = 0.15  # 15% of animals get disease
np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def generate_animal_v51(animal_idx):
    """Generate one animal's full timeline with gradual drift pre-onset."""
    rng = np.random.RandomState(animal_idx * 7 + 13)
    n = TICKS_PER_ANIMAL
    animal_id = f"v51_animal_{animal_idx:04d}"

    # Base physiology (per-animal variation)
    base_temp = 38.3 + rng.normal(0, 0.3)
    base_hr = 62 + rng.normal(0, 8)
    base_resp = 24 + rng.normal(0, 4)
    base_activity = 0.65 + rng.normal(0, 0.1)
    base_rumination = 38 + rng.normal(0, 5)
    base_milk = 26 + rng.normal(0, 4)

    # Environment
    thi = 65 + 10 * np.sin(np.arange(n) * 2 * np.pi / (288 * 7)) + rng.normal(0, 3, n)
    ambient_temp = 22 + 8 * np.sin(np.arange(n) * 2 * np.pi / 288) + rng.normal(0, 2, n)
    humidity = 55 + 15 * np.sin(np.arange(n) * 2 * np.pi / (288 * 3)) + rng.normal(0, 5, n)
    heat_stress = np.clip((thi - 72) / 10, 0, 1)

    # Disease trajectory with GRADUAL DRIFT
    infection_load = np.zeros(n)
    severity = np.zeros(n)
    disease_binary = np.zeros(n, dtype=int)

    has_disease = rng.random() < DISEASE_RATE
    if has_disease:
        # Disease onset with drift window
        onset_tick = rng.randint(400, 1400)
        drift_window_ticks = rng.randint(144, 288)  # 12-24h pre-onset drift
        k = rng.uniform(0.15, 0.25)  # Sigmoid steepness
        peak_infection = rng.uniform(0.6, 1.0)
        duration = rng.randint(200, 600)

        for t in range(n):
            if t >= onset_tick - drift_window_ticks and t < onset_tick + duration:
                # Sigmoid ramp: gradual rise with drift window
                x = k * (t - onset_tick)
                infection_load[t] = peak_infection * sigmoid(x)
                severity[t] = infection_load[t] * 3  # 0-3 scale
                if severity[t] >= 0.5:
                    disease_binary[t] = 1

        # Add recovery tail (gradual)
        recovery_start = onset_tick + duration
        for t in range(recovery_start, min(recovery_start + 288, n)):
            decay = np.exp(-(t - recovery_start) / 100)
            infection_load[t] = infection_load[recovery_start - 1] * decay
            severity[t] = infection_load[t] * 3

    # ── SENSOR SIGNALS with gradual drift ──

    # Temperature: base + infection drift + heat stress + noise
    # Pre-onset: subtle 0.2-0.5°C drift before full fever
    temp_drift = np.zeros(n)
    for t in range(n):
        if infection_load[t] > 0.01:
            # Gradual: sigmoid response with probabilistic expression
            temp_drift[t] = sigmoid(infection_load[t] * 4 - 1.5) * 1.8
            if rng.random() > 0.85:  # 15% chance of no fever expression
                temp_drift[t] *= 0.3
        temp_drift[t] += heat_stress[t] * rng.uniform(0.3, 0.8)

    temperature = base_temp + temp_drift + rng.normal(0, 0.4, n)

    # Heart rate: GRADUAL drift 3-7 bpm before onset
    hr_drift = np.zeros(n)
    for t in range(n):
        il = infection_load[t]
        if il > 0.01:
            hr_drift[t] = sigmoid(il * 3 - 1) * 18 + rng.uniform(3, 7) * min(il, 0.3)
        hr_drift[t] += heat_stress[t] * rng.uniform(2, 6)

    heart_rate = base_hr + hr_drift + rng.normal(0, 4, n)

    # Respiration: GRADUAL drift 4-8 bpm
    resp_drift = np.zeros(n)
    for t in range(n):
        il = infection_load[t]
        if il > 0.01:
            resp_drift[t] = sigmoid(il * 3.5 - 1.2) * 12 + rng.uniform(4, 8) * min(il, 0.3)
        resp_drift[t] += heat_stress[t] * rng.uniform(3, 7)

    respiration = base_resp + resp_drift + rng.normal(0, 3, n)

    # Activity: GRADUAL decrease
    activity_drop = np.zeros(n)
    for t in range(n):
        il = infection_load[t]
        if il > 0.01:
            activity_drop[t] = sigmoid(il * 3 - 0.8) * 0.35 + 0.05 * min(il, 0.3)

    activity = np.clip(base_activity - activity_drop + rng.normal(0, 0.08, n), 0, 1)

    rumination = base_rumination - infection_load * 12 + rng.normal(0, 4, n)
    lying = 25 + infection_load * 15 + rng.normal(0, 5, n)

    # ── PRODUCTION signals with gradual drift ──
    milk_drift = base_milk * sigmoid(infection_load * 3 - 1) * 0.35
    milk_yield = np.clip(base_milk - milk_drift + rng.normal(0, 2, n), 5, 45)
    feed_intake = np.clip(22 - infection_load * 6 + rng.normal(0, 1.5, n), 5, 30)
    conductivity = 5.0 + infection_load * 2.5 + rng.normal(0, 0.5, n)
    body_weight = 550 - infection_load * 15 + rng.normal(0, 5, n)

    # Disease type
    disease_types = ["none", "mastitis", "ketosis", "lameness", "respiratory", "metabolic"]
    disease_type_idx = rng.randint(1, len(disease_types)) if has_disease else 0

    timestamps = pd.date_range("2024-01-01", periods=n, freq=f"{TICK_MINUTES}min")

    # Build records
    sensor_records = []
    prod_records = []
    for t in range(n):
        base = {
            "animalId": animal_id,
            "timestamp": timestamps[t],
            "simulationVersion": "digital_twin_v5_1_drift",
        }
        sensor_records.append({
            **base,
            "signals": {
                "temperature_C": round(float(temperature[t]), 2),
                "heartRate_bpm": round(float(heart_rate[t]), 1),
                "respiration_bpm": round(float(respiration[t]), 1),
                "activity_index": round(float(activity[t]), 3),
                "rumination_min": round(float(rumination[t]), 1),
                "lying_min": round(float(lying[t]), 1),
            },
            "environment": {
                "thi": round(float(thi[t]), 1),
                "ambientTemp_C": round(float(ambient_temp[t]), 1),
                "humidity_pct": round(float(humidity[t]), 1),
            },
            "labels": {
                "diseaseBinary": int(disease_binary[t]),
                "severityLevel": round(float(severity[t]), 2),
                "diseaseType": disease_types[disease_type_idx],
                "infectionBinary": int(infection_load[t] > 0.2),
                "stressBinary": int(heat_stress[t] > 0.3),
            },
            "hiddenState": {
                "infectionLoad": round(float(infection_load[t]), 4),
            },
        })
        prod_records.append({
            **base,
            "production": {
                "milkYield": round(float(milk_yield[t]), 1),
                "feedIntake": round(float(feed_intake[t]), 1),
                "conductivity": round(float(conductivity[t]), 2),
                "bodyWeight": round(float(body_weight[t]), 1),
            },
            "management": {
                "vaccinationEffective": int(rng.random() < 0.01),
                "antibioticEffective": int(rng.random() < 0.02),
            },
            "environment": {
                "thi": round(float(thi[t]), 1),
            },
            "labels": {
                "diseaseBinary": int(disease_binary[t]),
                "severityLevel": round(float(severity[t]), 2),
                "diseaseType": disease_types[disease_type_idx],
            },
            "hiddenState": {
                "infectionLoad": round(float(infection_load[t]), 4),
            },
        })

    return sensor_records, prod_records, has_disease


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("🧬 Phase 8 Part 2 — Simulator v5.1 Drift Patch")
    logger.info(f"   Animals: {N_ANIMALS}, Ticks/animal: {TICKS_PER_ANIMAL}")
    logger.info(f"   Disease rate: {DISEASE_RATE*100:.0f}%")
    logger.info("=" * 60)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Drop old v5.1 collections
    for col in ["trainingevents_v5_1", "trainingevents_v5_1_production"]:
        db[col].drop()
        logger.info(f"Dropped {col}")

    all_sensor = []
    all_prod = []
    disease_count = 0

    for i in range(N_ANIMALS):
        sensor, prod, has_disease = generate_animal_v51(i)
        all_sensor.extend(sensor)
        all_prod.extend(prod)
        if has_disease:
            disease_count += 1
        if (i + 1) % 20 == 0:
            logger.info(f"  Generated {i+1}/{N_ANIMALS} animals ({disease_count} diseased)")

    logger.info(f"Total: {len(all_sensor)} sensor, {len(all_prod)} production, {disease_count} diseased")

    # Batch insert
    batch_size = 5000
    logger.info("Inserting sensor records...")
    for i in range(0, len(all_sensor), batch_size):
        batch = all_sensor[i:i+batch_size]
        db["trainingevents_v5_1"].insert_many(batch)
    logger.info(f"  ✅ trainingevents_v5_1: {len(all_sensor)} records")

    logger.info("Inserting production records...")
    for i in range(0, len(all_prod), batch_size):
        batch = all_prod[i:i+batch_size]
        db["trainingevents_v5_1_production"].insert_many(batch)
    logger.info(f"  ✅ trainingevents_v5_1_production: {len(all_prod)} records")

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ v5.1 datasets generated in {elapsed:.1f}s")
    logger.info(f"   Sensor: {len(all_sensor)}, Production: {len(all_prod)}")
    logger.info(f"   Diseased animals: {disease_count}/{N_ANIMALS}")
    logger.info(f"{'='*60}")
    client.close()


if __name__ == "__main__":
    main()
