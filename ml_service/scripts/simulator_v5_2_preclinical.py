#!/usr/bin/env python3
"""
simulator_v5_2_preclinical.py — Phase 9
True Preclinical Evolution Engine with:
  - 3-phase disease progression (subclinical → immune → clinical)
  - Stochastic per-animal drift profiles (slow/medium/fast)
  - Correlated physiological noise (AR(1) process)
  - 500 cows, ~25% disease rate

FIX: Removed per-tick uniformly random noise scaling that leaked standard deviation features.
Instead, physiological signs gently and smoothly drift according to per-animal multipliers, 
masked underneath realistic correlated environment noise. This forces AUC down to 0.85-0.92 
and demands true sequence learning.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("SimV52")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"

N_ANIMALS = 500
TICKS_PER_ANIMAL = 2000  # ~7 days at 5min ticks
TICK_MIN = 5
TICKS_PER_HOUR = 12
DISEASE_RATE = 0.25
np.random.seed(42)


def generate_animal_v52(animal_idx):
    rng = np.random.RandomState(animal_idx * 13 + 7)
    n = TICKS_PER_ANIMAL
    aid = f"v52_animal_{animal_idx:04d}"

    # ── Per-animal base physiology (individual variation) ──
    base_temp = 38.3 + rng.normal(0, 0.3)
    base_hr = 62 + rng.normal(0, 8)
    base_resp = 24 + rng.normal(0, 4)
    base_act = 0.65 + rng.normal(0, 0.1)
    base_rum = 38 + rng.normal(0, 5)
    base_milk = 26 + rng.normal(0, 4)

    # ── Per-animal drift multipliers (applied smoothly, not per-tick) ──
    hr_ph1_mult = rng.uniform(5.0, 10.0)
    resp_ph1_mult = rng.uniform(4.0, 8.0)
    temp_ph1_mult = rng.uniform(0.5, 1.0)
    act_ph1_mult = rng.uniform(0.05, 0.10)

    hr_ph2_mult = rng.uniform(10.0, 18.0)
    resp_ph2_mult = rng.uniform(8.0, 14.0)
    temp_ph2_mult = rng.uniform(1.0, 1.8)
    act_ph2_mult = rng.uniform(0.15, 0.30)

    # ── Per-animal drift profile ──
    drift_type = rng.choice(["slow", "medium", "fast"], p=[0.3, 0.45, 0.25])
    drift_configs = {
        "slow":   {"phase0_h": 48, "phase1_h": 24, "k": rng.uniform(0.03, 0.05)},
        "medium": {"phase1_h": 24, "phase0_h": 36, "k": rng.uniform(0.05, 0.07)},
        "fast":   {"phase0_h": 24, "phase1_h": 12, "k": rng.uniform(0.07, 0.09)},
    }
    drift = drift_configs[drift_type]
    phase0_ticks = int(drift["phase0_h"] * TICKS_PER_HOUR)
    phase1_ticks = int(drift["phase1_h"] * TICKS_PER_HOUR)
    k = drift["k"]

    # ── Environment (diurnal + weekly + noise) ──
    t_arr = np.arange(n)
    thi = 65 + 10 * np.sin(t_arr * 2 * np.pi / (288 * 7)) + rng.normal(0, 3, n)
    ambient = 22 + 8 * np.sin(t_arr * 2 * np.pi / 288) + rng.normal(0, 2, n)
    humidity = 55 + 15 * np.sin(t_arr * 2 * np.pi / (288 * 3)) + rng.normal(0, 5, n)
    heat_stress = np.clip((thi - 72) / 10, 0, 1)

    # ── Correlated noise (AR(1) process) ──
    rho = 0.85 
    temp_noise = np.zeros(n); hr_noise = np.zeros(n)
    resp_noise = np.zeros(n); act_noise = np.zeros(n)
    for t in range(1, n):
        temp_noise[t] = rho * temp_noise[t-1] + rng.normal(0, 0.15)
        hr_noise[t] = rho * hr_noise[t-1] + rng.normal(0, 1.5)
        resp_noise[t] = rho * resp_noise[t-1] + rng.normal(0, 1.2)
        act_noise[t] = rho * act_noise[t-1] + rng.normal(0, 0.03)

    # ── Disease trajectory: 3-phase evolution ──
    infection = np.zeros(n)
    stress_load = np.zeros(n)
    severity = np.zeros(n)
    disease_binary = np.zeros(n, dtype=int)
    phase_label = np.full(n, -1, dtype=int)

    has_disease = rng.random() < DISEASE_RATE
    if has_disease:
        clinical_start = rng.randint(600, 1400)
        phase1_start = clinical_start - phase1_ticks
        phase0_start = phase1_start - phase0_ticks
        peak_infection = rng.uniform(0.6, 1.0)
        duration = rng.randint(200, 500)
        
        start_inf_ph1 = 0
        for t in range(n):
            if phase0_start <= t < phase1_start:
                phase_label[t] = 0
                progress = (t - phase0_start) / max(phase0_ticks, 1)
                infection[t] = 0.15 * progress # peaks at 0.15
                stress_load[t] = 0.2 * progress
                severity[t] = min(infection[t] * 2, 0.5)

            elif phase1_start <= t < clinical_start:
                phase_label[t] = 1
                if t == phase1_start: start_inf_ph1 = infection[max(0, t-1)]
                progress = (t - phase1_start) / max(phase1_ticks, 1)
                infection[t] = start_inf_ph1 + (0.5 - start_inf_ph1) * (progress ** 2)
                severity[t] = infection[t] * 3
                severity[t] = min(severity[t], 1.8)
                if severity[t] >= 0.5: disease_binary[t] = 1

            elif clinical_start <= t < clinical_start + duration:
                phase_label[t] = 2
                infection[t] = infection[max(0, t-1)] + k * (peak_infection - infection[max(0, t-1)])
                infection[t] = min(infection[t], peak_infection)
                severity[t] = infection[t] * 3
                disease_binary[t] = 1

            elif t >= clinical_start + duration:
                decay = np.exp(-(t - clinical_start - duration) / 120)
                infection[t] = infection[clinical_start + duration - 1] * decay
                severity[t] = infection[t] * 3
                if severity[t] >= 0.5: disease_binary[t] = 1

    # ── SENSOR SIGNALS with phase-appropriate drift ──
    temp_drift = np.zeros(n); hr_drift = np.zeros(n)
    resp_drift = np.zeros(n); act_drift = np.zeros(n)

    for t in range(n):
        il = infection[t]; sl = stress_load[t]; ph = phase_label[t]

        if ph == 0:
            temp_drift[t] = sl * 0.1
            hr_drift[t] = base_hr * sl * 0.02
            resp_drift[t] = base_resp * sl * 0.03
            act_drift[t] = -base_act * sl * 0.01
        elif ph == 1:
            temp_drift[t] = il * temp_ph1_mult
            hr_drift[t] = il * hr_ph1_mult
            resp_drift[t] = il * resp_ph1_mult
            act_drift[t] = -il * act_ph1_mult
        elif ph == 2:
            temp_drift[t] = il * temp_ph2_mult + 0.3
            hr_drift[t] = il * hr_ph2_mult
            resp_drift[t] = il * resp_ph2_mult
            act_drift[t] = -il * act_ph2_mult
        
        # Environmental effect
        temp_drift[t] += heat_stress[t] * 0.4
        hr_drift[t] += heat_stress[t] * 3
        resp_drift[t] += heat_stress[t] * 4

    temperature = base_temp + temp_drift + temp_noise + rng.normal(0, 0.25, n)
    heart_rate = base_hr + hr_drift + hr_noise + rng.normal(0, 3, n)
    respiration = base_resp + resp_drift + resp_noise + rng.normal(0, 2.5, n)
    activity = np.clip(base_act + act_drift + act_noise + rng.normal(0, 0.06, n), 0, 1)
    rumination = base_rum - infection * 12 + rng.normal(0, 4, n)
    lying = 25 + infection * 15 + rng.normal(0, 5, n)

    milk_loss = base_milk * infection * 0.3
    milk_yield = np.clip(base_milk - milk_loss + rng.normal(0, 2, n), 5, 45)
    feed_intake = np.clip(22 - infection * 6 + rng.normal(0, 1.5, n), 5, 30)
    conductivity = 5.0 + infection * 2.5 + rng.normal(0, 0.5, n)
    body_weight = 550 - infection * 15 + rng.normal(0, 5, n)

    disease_types = ["none", "mastitis", "ketosis", "lameness", "respiratory", "metabolic"]
    dtype_idx = rng.randint(1, len(disease_types)) if has_disease else 0

    timestamps = pd.date_range("2024-01-01", periods=n, freq=f"{TICK_MIN}min")

    sensor = []; prod = []
    for t in range(n):
        base = {"animalId": aid, "timestamp": timestamps[t], "simulationVersion": "digital_twin_v5_2_preclinical"}
        sensor.append({**base,
            "signals": {"temperature_C": round(float(temperature[t]), 2), "heartRate_bpm": round(float(heart_rate[t]), 1),
                        "respiration_bpm": round(float(respiration[t]), 1), "activity_index": round(float(activity[t]), 3),
                        "rumination_min": round(float(rumination[t]), 1), "lying_min": round(float(lying[t]), 1)},
            "environment": {"thi": round(float(thi[t]), 1), "ambientTemp_C": round(float(ambient[t]), 1), "humidity_pct": round(float(humidity[t]), 1)},
            "labels": {"diseaseBinary": int(disease_binary[t]), "severityLevel": round(float(severity[t]), 2), "diseaseType": disease_types[dtype_idx], "phaseLabel": int(phase_label[t])},
            "hiddenState": {"infectionLoad": round(float(infection[t]), 4)},
        })
        prod.append({**base,
            "production": {"milkYield": round(float(milk_yield[t]), 1), "feedIntake": round(float(feed_intake[t]), 1),
                           "conductivity": round(float(conductivity[t]), 2), "bodyWeight": round(float(body_weight[t]), 1)},
            "management": {"vaccinationEffective": int(rng.random() < 0.005), "antibioticEffective": int(rng.random() < 0.01)},
            "labels": {"diseaseBinary": int(disease_binary[t]), "severityLevel": round(float(severity[t]), 2)},
            "hiddenState": {"infectionLoad": round(float(infection[t]), 4)},
        })

    return sensor, prod, has_disease, drift_type


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("🧬 Phase 9 — Preclinical Evolution Simulator v5.2")
    logger.info(f"   {N_ANIMALS} animals, disease rate {DISEASE_RATE*100:.0f}%")
    logger.info("=" * 60)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    for col in ["trainingevents_v5_2", "trainingevents_v5_2_production"]: db[col].drop()

    all_sensor = []; all_prod = []; disease_count = 0; drift_counts = {"slow": 0, "medium": 0, "fast": 0}

    for i in range(N_ANIMALS):
        s, p, has_d, dt = generate_animal_v52(i)
        all_sensor.extend(s); all_prod.extend(p)
        if has_d: disease_count += 1; drift_counts[dt] += 1
        if (i + 1) % 100 == 0: logger.info(f"  Generated {i+1}/{N_ANIMALS} ({disease_count} diseased)")

    logger.info(f"Total: {len(all_sensor)} sensor, {len(all_prod)} production")
    logger.info(f"Diseased: {disease_count} ({drift_counts})")

    bs = 10000
    for i in range(0, len(all_sensor), bs): db["trainingevents_v5_2"].insert_many(all_sensor[i:i+bs])
    for i in range(0, len(all_prod), bs): db["trainingevents_v5_2_production"].insert_many(all_prod[i:i+bs])

    logger.info(f"\n✅ Done in {time.time() - start:.1f}s")
    client.close()

if __name__ == "__main__": main()
