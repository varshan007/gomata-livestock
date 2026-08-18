#!/usr/bin/env python3
"""
simulator_v8_domain_randomized.py — Phase 15.1
DOMAIN RANDOMIZATION GENERATOR

Fuses standard GoMata Stochastic Differential Equations (SDEs)
with completely foreign Alien Physics (Poisson/Jump-Diffusion).
During training data generation, each cow is randomly assigned a "physics engine"
(70% Native SDE, 30% Alien Physics).

This forces the PyTorch Shared Attention model to natively learn
mechanically-invariant biological representations during training,
rather than attempting to transfer post-hoc.
"""

import os, sys, json, logging, time
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("SimulatorV8_DomainRand")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
os.makedirs(DATA_DIR, exist_ok=True)

N_ANIMALS_TRAIN = 500
TICKS_PER_ANIMAL = 2000
TICK_MIN = 10
TICKS_PER_HOUR = int(60 / TICK_MIN)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR
THI_THRESHOLD = 72.0

class DomainRandomizedUniverse:
    def __init__(self, seed_offset):
        self.rng = np.random.RandomState(seed_offset)
        self.p_lambd = 0.05
        self.p_sigma = 0.8
        
    def generate_native_sde_animal(self, animal_idx):
        """Standard Phase 13 AR(1) limits and Coupled SDE generation"""
        n = TICKS_PER_ANIMAL
        aid = f"DomainRand_Native_{animal_idx:04d}"
        
        base = {
            "temp": 38.3 + self.rng.normal(0, 0.3),
            "hr": 62 + self.rng.normal(0, 8),
            "resp": 24 + self.rng.normal(0, 4),
            "act": 0.65 + self.rng.normal(0, 0.1),
            "milk": 30 + self.rng.normal(0, 6),
            "heat_tolerance": self.rng.uniform(0.5, 1.5)
        }
        
        t_arr = np.arange(n)
        ambient = 22 + 10 * np.sin(t_arr * 2 * np.pi / TICKS_PER_DAY) + self.rng.normal(0, 2, n)
        humidity = 55 + 15 * np.sin(t_arr * 2 * np.pi / (TICKS_PER_DAY * 3)) + self.rng.normal(0, 5, n)
        thi = (1.8 * ambient + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * ambient - 26))
        thi += self.rng.normal(0, 1.5, n)
        
        E = np.zeros(n)
        for t in range(1, n):
            E[t] = E[t-1] - self.p_lambd * E[t-1] + self.rng.normal(0, self.p_sigma)
            
        X = { "I": np.zeros(n), "H": np.zeros(n), "M": np.zeros(n), "L": np.zeros(n), "C": np.zeros(n), "Imm": np.ones(n), "Fat": np.zeros(n) }

        has_infection = self.rng.random() < 0.25
        has_mastitis = self.rng.random() < 0.20
        has_lameness = self.rng.random() < 0.15
        is_calving = self.rng.random() < 0.10

        infection_start = self.rng.randint(int(0.1*n), int(0.8*n)) if has_infection else -1
        primary_mastitis_start = self.rng.randint(int(0.1*n), int(0.8*n)) if has_mastitis else -1
        lameness_start = self.rng.randint(int(0.1*n), int(0.8*n)) if has_lameness else -1
        calving_midpoint = self.rng.randint(int(0.4*n), int(0.9*n)) if is_calving else -1

        exposure = np.zeros(n)
        if has_infection: exposure[infection_start:infection_start + 144] = self.rng.uniform(0.01, 0.03)

        for t in range(1, n):
            if is_calving:
                days_to_calving = (t - calving_midpoint) / TICKS_PER_DAY
                X["C"][t] = 1.0 / (1.0 + np.exp(-2.0 * days_to_calving))

            thi_excess = max(thi[t] - THI_THRESHOLD, 0)
            dH = (0.01 * thi_excess) - (0.05 * base["heat_tolerance"]) + self.rng.normal(0, 0.01)
            X["H"][t] = np.clip(X["H"][t-1] + dH, 0, 1.5)

            dFat = (0.02 * X["H"][t]) + (0.05 * X["I"][t-1]) + (0.03 * X["C"][t]) - 0.01
            X["Fat"][t] = np.clip(X["Fat"][t-1] + dFat, 0, 1.0)
            
            dImm = - (0.08 * X["Fat"][t]) - (0.05 * X["H"][t]) + 0.01
            X["Imm"][t] = np.clip(X["Imm"][t-1] + dImm, 0.1, 1.0)

            dI = exposure[t-1] + (0.05 * X["I"][t-1]) - (0.04 * X["Imm"][t]) + self.rng.normal(0, 0.005)
            X["I"][t] = np.clip(X["I"][t-1] + dI, 0, 1.0)

            primary_m = 0.05 if (primary_mastitis_start > 0 and t > primary_mastitis_start and t < primary_mastitis_start + 200) else 0
            secondary_m = 0.08 * max(X["I"][t] - 0.6, 0)
            dM = primary_m + secondary_m + (0.02 * X["M"][t-1]) - (0.05 * X["Imm"][t]) + self.rng.normal(0, 0.005)
            X["M"][t] = np.clip(X["M"][t-1] + dM, 0, 1.0)

            primary_l = 0.03 if (lameness_start > 0 and t > lameness_start and t < lameness_start + 500) else 0
            overlap_calving_lameness = 0.05 * X["C"][t]
            dL = primary_l + overlap_calving_lameness + (0.01 * X["L"][t-1]) + self.rng.normal(0, 0.002)
            X["L"][t] = np.clip(X["L"][t-1] + dL, 0, 1.0)
            
        temp_noise = np.zeros(n); hr_noise = np.zeros(n); resp_noise = np.zeros(n); act_noise = np.zeros(n)
        rho = 0.85
        for t in range(1, n):
            temp_noise[t] = rho * temp_noise[t-1] + self.rng.normal(0, 0.15)
            hr_noise[t] = rho * hr_noise[t-1] + self.rng.normal(0, 1.5)
            act_noise[t] = rho * act_noise[t-1] + self.rng.normal(0, 0.03)

        temp_curve = base["temp"] + (2.5 * X["I"]) + (1.2 * X["H"]) + (0.4 * X["M"]) - (0.3 * X["C"]) + temp_noise + (0.2 * E)
        hr_curve = base["hr"] + (20 * X["I"]) + (12 * X["H"]) + (8 * X["C"]) + hr_noise + (2.5 * E) 
        resp_curve = base["resp"] + (15 * X["H"]) + (5 * X["I"]) + self.rng.normal(0, 2, n) + (1.5 * E)
        act_curve = np.clip(base["act"] - (0.5 * X["L"]) - (0.3 * X["I"]) + (X["C"] * self.rng.normal(0, 0.2, n)) + act_noise - (0.05 * E), 0.1, 1.0)
        milk_curve = np.clip(base["milk"] - (6.0 * X["I"]) - (4.0 * X["H"]) - (12.0 * X["M"]) - (2.0 * X["L"]) + self.rng.normal(0, 1.5, n) - (1.5 * E), 0.0, 50.0)
        cond_curve = 5.0 + (3.5 * X["M"]) + self.rng.normal(0, 0.2, n) + (0.4 * E)
        
        severity_label = np.clip(X["I"] + (X["M"]*1.5) + (X["L"]*0.8) + (X["H"]*0.5), 0, 3.0)

        return pd.DataFrame({
            "animalId": [aid]*n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq=f"{TICK_MIN}min"),
            "temperature_C": temp_curve, "heartRate_bpm": hr_curve, "respiration_bpm": resp_curve, "activity_index": act_curve,
            "thi": thi, "ambientTemp_C": ambient, "humidity_pct": humidity,
            "milkYield": milk_curve, "feedIntake": np.clip(22 - (X["I"]*5), 0, 50), "conductivity": cond_curve,
            "antibioticActive": 0,
            "infectionBinary": (X["I"] > 0.4).astype(int),
            "heatStressBinary": (X["H"] > 0.5).astype(int),
            "mastitisBinary": (X["M"] > 0.4).astype(int),
            "lamenessBinary": (X["L"] > 0.4).astype(int),
            "calvingBinary": (X["C"] > 0.5).astype(int),
            "severityLevel": severity_label,
            "domainEngine": "Native_SDE"
        })
        
    def generate_alien_physics_animal(self, animal_idx):
        """Phase 15 Alien Physics: Multiplicative, Poisson, Jump Diffusion"""
        n = TICKS_PER_ANIMAL
        aid = f"DomainRand_Alien_{animal_idx:04d}"
        
        temp_base = np.zeros(n)
        temp_base[0] = 38.5 + self.rng.normal(0, 0.4)
        
        hr_base = np.zeros(n)
        hr_base[0] = 65 + self.rng.normal(0, 5)
        
        for t in range(1, n):
            temp_base[t] = temp_base[t-1] + self.rng.normal(0, 0.05)
            if self.rng.random() < 0.005: temp_base[t] += self.rng.normal(0, 0.5) 
            hr_base[t] = hr_base[t-1] + self.rng.normal(0, 0.5)
            if self.rng.random() < 0.01: hr_base[t] += self.rng.normal(0, 5)
            
        ambient = 20 + 8 * np.sin(np.arange(n) * 2 * np.pi / TICKS_PER_DAY) + self.rng.normal(0, 3, n)
        humidity = 60 + 10 * np.cos(np.arange(n) * 2 * np.pi / TICKS_PER_DAY) + self.rng.normal(0, 4, n)
        thi = (1.8 * ambient + 32) - ((0.55 - 0.0055 * humidity) * (1.8 * ambient - 26))
        
        # Poisson Events
        inf_state = np.zeros(n)
        if self.rng.random() < 0.25:
            start = self.rng.randint(50, n - 200)
            dur = self.rng.randint(100, 300)
            end_idx = min(start + dur, n)
            inf_state[start:end_idx] = np.exp(-np.linspace(0, 1.5, dur))[:end_idx-start]
            
        mast_state = np.zeros(n)
        if self.rng.random() < 0.20:
            start = self.rng.randint(50, n - 200)
            dur = self.rng.randint(80, 250)
            ramp = np.cumsum(self.rng.uniform(0.005, 0.02, dur))
            end_idx = min(start + dur, n)
            mast_state[start:end_idx] = np.clip(ramp[:end_idx-start], 0, 1.0)
            
        lame_state = np.zeros(n)
        if self.rng.random() < 0.15:
            start = self.rng.randint(50, n - 400)
            lame_state[start:] = np.linspace(0.2, 1.0, n - start)
            
        heat_state = (thi > 75).astype(float)
        
        calv_state = np.zeros(n)
        if self.rng.random() < 0.10:
            start = self.rng.randint(200, n - 200)
            calv_state[start-100:start+100] = np.exp(-0.5 * ((np.arange(200) - 100) / 35.0)**2)
            
        # Multiplicative
        temp_obs = temp_base + 1.5 * np.log1p(inf_state * 10) + 0.5 * np.exp(heat_state) + 0.6 * mast_state - 0.4 * calv_state + self.rng.normal(0, 0.2, n)
        hr_obs = hr_base * (1.0 + 0.2 * inf_state) * (1.0 + 0.15 * heat_state) * (1.0 + 0.12 * calv_state) + self.rng.normal(0, 2, n)
        resp_obs = 22 + (12 * heat_state) ** 1.2 + 3 * inf_state + 10 * calv_state + self.rng.normal(0, 3, n)
        
        calv_noise = calv_state * self.rng.normal(0, 0.3, n)
        act_obs = 0.7 * (1.0 - 0.6 * lame_state) * (1.0 - 0.3 * inf_state) + calv_noise + self.rng.normal(0, 0.05, n)
        act_obs = np.clip(act_obs, 0, 1)
        
        milk_base = 32 + self.rng.normal(0, 4)
        milk_obs = milk_base * (1.0 - 0.4 * mast_state) * (1.0 - 0.15 * inf_state) * (1.0 - 0.1 * heat_state) + self.rng.normal(0, 1.5, n)
        milk_obs = np.clip(milk_obs, 0, None)
        
        feed_obs = 22 * (1.0 - 0.25 * inf_state) * (1.0 - 0.15 * lame_state) + self.rng.normal(0, 1, n)
        cond_obs = 5.0 + np.exp(mast_state * 1.5) - 1.0 + self.rng.normal(0, 0.3, n)
        
        severity = inf_state + mast_state + lame_state + heat_state
        
        return pd.DataFrame({
            "animalId": [aid]*n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq=f"{TICK_MIN}min"),
            "temperature_C": temp_obs, "heartRate_bpm": hr_obs, "respiration_bpm": resp_obs, "activity_index": act_obs,
            "thi": thi, "ambientTemp_C": ambient, "humidity_pct": humidity,
            "milkYield": milk_obs, "feedIntake": feed_obs, "conductivity": cond_obs,
            "antibioticActive": 0,
            "infectionBinary": (inf_state > 0.4).astype(int),
            "heatStressBinary": (heat_state > 0.5).astype(int),
            "mastitisBinary": (mast_state > 0.4).astype(int),
            "lamenessBinary": (lame_state > 0.4).astype(int),
            "calvingBinary": (calv_state > 0.5).astype(int),
            "severityLevel": severity,
            "domainEngine": "Alien_Multiplicative"
        })


def main():
    logger.info("============================================================")
    logger.info("🧬 Phase 15.1 — DOMAIN RANDOMIZATION GENERATOR STARTING")
    logger.info("============================================================")
    
    sim = DomainRandomizedUniverse(seed_offset=1024)
    dfs_sensor = []
    dfs_production = []
    
    counts = {"Native": 0, "Alien": 0}
    
    for i in range(N_ANIMALS_TRAIN):
        # 30% Chance to inject completely foreign alien physics representations
        use_alien = sim.rng.random() < 0.30
        
        if use_alien:
            df = sim.generate_alien_physics_animal(i)
            counts["Alien"] += 1
        else:
            df = sim.generate_native_sde_animal(i)
            counts["Native"] += 1
            
        df_sensor = df[["animalId", "timestamp", "temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index", "thi", "ambientTemp_C", "humidity_pct"]].copy()
        dfs_sensor.append(df_sensor)
        
        daily_mask = df["timestamp"].dt.hour % 12 == 0
        df_prod = df.loc[daily_mask, ["animalId", "timestamp", "milkYield", "feedIntake", "conductivity", "antibioticActive", "infectionBinary", "heatStressBinary", "mastitisBinary", "lamenessBinary", "calvingBinary", "severityLevel", "domainEngine"]].copy()
        dfs_production.append(df_prod)
        
        if (i+1) % 50 == 0:
            logger.info(f"Generated {i+1}/{N_ANIMALS_TRAIN} Animals (N: {counts['Native']}, A: {counts['Alien']})...")
            
    final_sensor = pd.concat(dfs_sensor, ignore_index=True)
    final_prod = pd.concat(dfs_production, ignore_index=True)
    
    f_sensor = os.path.join(DATA_DIR, "trainingevents_v8_sensor.csv")
    f_prod = os.path.join(DATA_DIR, "trainingevents_v8_production.csv")
    
    final_sensor.to_csv(f_sensor, index=False)
    final_prod.to_csv(f_prod, index=False)
    
    logger.info(f"✅ Saved v8 Sensor Matrix: {f_sensor}")
    logger.info(f"✅ Saved v8 Production Labels: {f_prod}")
    logger.info(f"Physics Engine Distribution: {counts['Native']} SDE | {counts['Alien']} ALIEN")

if __name__ == "__main__":
    main()
