#!/usr/bin/env python3
"""
generate_v5_dataset.py — GoMata Digital Twin v5
Probabilistic Biological Realism Upgrade

Reads hidden states from validation_clean_v3 / validation_clean_v4,
regenerates signals with probabilistic physics:
  - Sigmoid saturation (no infinite increase)
  - Delayed physiological response (3h temp, 1h HR, 2h resp)
  - Partial symptom expression (probabilistic fever)
  - Environmental confounding (heat → high temp in healthy)
  - Overlapping healthy/diseased distributions
  - Non-linear milk loss with delay
  - Stochastic management event outcomes

Targets:
  Instantaneous temp AUC ≤ 0.85
  Instantaneous milk AUC ≤ 0.80
  Full temporal AUC: 0.90-0.95

Usage:
  python generate_v5_dataset.py [--limit 200000]
"""

import os, sys, time, logging, json
import numpy as np
import pandas as pd
from pymongo import MongoClient

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "generate_v5.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V5Generator")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"


# ═══════════════════════════════════════════════════════════════════════════════
# SIGMOID AND UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def delayed_value(series, delay_ticks, default=0):
    """Get delayed value from a series (with bfill for initial period)."""
    shifted = series.shift(delay_ticks)
    return shifted.fillna(series.iloc[0] if len(series) > 0 else default)


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: V3 SENSOR REALISM
# ═══════════════════════════════════════════════════════════════════════════════

def generate_v3_realistic(df, rng):
    """Regenerate sensor signals with probabilistic physics.
    
    Replaces deterministic: temp = base + 3.0*I + noise(0.12)
    With:                   temp = base + sigmoid(I_delayed*4-1.5)*1.8 + heat_env + noise(0.4)
    """
    logger.info("Generating V3 realistic sensor signals...")
    out = df.copy()
    n = len(out)

    # Extract hidden states
    I = out['infectionLoad'].values.astype(float)
    S = out['stressLoad'].values.astype(float)
    C = out.get('compensation', pd.Series(np.ones(n))).values.astype(float)
    F = out.get('fatigue', pd.Series(np.zeros(n))).values.astype(float)

    # Per-animal baselines and delays
    results = []
    for animal_id, group in out.groupby('animalId'):
        g = group.reset_index(drop=True)
        ng = len(g)

        # Individual baselines (slight variation per cow)
        base_temp = 38.3 + rng.normal(0, 0.3)
        base_hr = 65 + rng.normal(0, 5)
        base_resp = 26 + rng.normal(0, 2)
        base_activity = 0.7 + rng.normal(0, 0.05)
        base_rumination = 35 + rng.normal(0, 3)
        base_lying = 25 + rng.normal(0, 3)

        I_g = g['infectionLoad'].values.astype(float)
        S_g = g['stressLoad'].values.astype(float)
        C_g = g.get('compensation', pd.Series(np.ones(ng))).values.astype(float)
        F_g = g.get('fatigue', pd.Series(np.zeros(ng))).values.astype(float)

        # ── Delayed hidden states ──────────────────────────────────
        I_s = pd.Series(I_g)
        S_s = pd.Series(S_g)
        # 3h delay for temp (36 ticks at 5-min), 1h for HR (12), 2h for resp (24)
        I_delayed_3h = delayed_value(I_s, 36).values
        I_delayed_1h = delayed_value(I_s, 12).values
        S_delayed_2h = delayed_value(S_s, 24).values
        I_delayed_6h = delayed_value(I_s, 72).values

        # ── Probabilistic symptom expression ───────────────────────
        # Not every infected cow shows fever
        fever_prob = np.where(I_delayed_3h > 0.4, 0.70, 0.30)
        fever_expressed = rng.random(ng) < fever_prob
        # Smooth the expression over time (don't flip every tick)
        for i in range(1, ng):
            if abs(I_delayed_3h[i] - I_delayed_3h[i-1]) < 0.05:
                fever_expressed[i] = fever_expressed[i-1]

        # ── Environmental confounding ──────────────────────────────
        ambient = g.get('ambientTemp', pd.Series(np.full(ng, 25))).values.astype(float)
        humidity = g.get('humidity', pd.Series(np.full(ng, 60))).values.astype(float)
        thi = g.get('thi', pd.Series(np.full(ng, 68))).values.astype(float)

        # Heat stress adds to temp even in healthy cows
        heat_env_temp = np.maximum(0, sigmoid((thi - 72) * 0.3) * 0.8)
        heat_env_hr = np.maximum(0, sigmoid((thi - 70) * 0.2) * 8)

        # ── Circadian variation ────────────────────────────────────
        tick_phase = np.arange(ng) % 288  # 288 ticks = 24h
        circadian_temp = 0.3 * np.sin(2 * np.pi * tick_phase / 288 - np.pi/3)
        circadian_act = 0.8 + 0.2 * np.cos(2 * np.pi * tick_phase / 288)

        # ═══════════════════════════════════════════════════════════
        # TEMPERATURE: sigmoid response, delayed, probabilistic
        # ═══════════════════════════════════════════════════════════
        infection_temp = sigmoid(I_delayed_3h * 4 - 1.5) * 1.8  # max +1.8°C
        stress_temp = sigmoid(S_delayed_2h * 3 - 1.5) * 0.5     # max +0.5°C

        # Probabilistic expression: some infected cows don't show fever
        infection_temp_expressed = np.where(fever_expressed, infection_temp, infection_temp * 0.15)

        temp_raw = (base_temp
                   + circadian_temp
                   + infection_temp_expressed
                   + stress_temp * 0.3  # stress contributes less
                   + heat_env_temp      # environment confounding
                   + rng.normal(0, 0.4, ng))  # 3x more noise than v2

        # Exponential smoothing (autocorrelation)
        temp_smooth = np.zeros(ng)
        temp_smooth[0] = temp_raw[0]
        for i in range(1, ng):
            temp_smooth[i] = 0.92 * temp_smooth[i-1] + 0.08 * temp_raw[i]

        g['temperature_C'] = np.clip(temp_smooth, 36.5, 42.5).round(2)

        # ═══════════════════════════════════════════════════════════
        # HEART RATE: sigmoid, delayed 1h
        # ═══════════════════════════════════════════════════════════
        infection_hr = sigmoid(I_delayed_1h * 3 - 1.2) * 15  # max +15 bpm (was +50!)
        stress_hr = sigmoid(S_delayed_2h * 2.5 - 1) * 10     # max +10 bpm

        hr_raw = (base_hr
                 + infection_hr
                 + stress_hr
                 + heat_env_hr
                 + F_g * 5  # fatigue adds modest amount
                 + rng.normal(0, 6, ng))  # 2.5x more noise

        hr_smooth = np.zeros(ng)
        hr_smooth[0] = hr_raw[0]
        for i in range(1, ng):
            hr_smooth[i] = 0.90 * hr_smooth[i-1] + 0.10 * hr_raw[i]

        g['heartRate_bpm'] = np.clip(hr_smooth, 40, 140).round(0).astype(int)

        # ═══════════════════════════════════════════════════════════
        # RESPIRATION: delayed 2h
        # ═══════════════════════════════════════════════════════════
        stress_resp = sigmoid(S_delayed_2h * 3 - 1) * 12
        infection_resp = sigmoid(I_delayed_1h * 2 - 0.8) * 8
        heat_resp = np.maximum(0, sigmoid((thi - 68) * 0.25) * 10)

        resp_raw = (base_resp + stress_resp + infection_resp + heat_resp
                   + rng.normal(0, 3, ng))

        resp_smooth = np.zeros(ng)
        resp_smooth[0] = resp_raw[0]
        for i in range(1, ng):
            resp_smooth[i] = 0.88 * resp_smooth[i-1] + 0.12 * resp_raw[i]

        g['respiration_bpm'] = np.clip(resp_smooth, 10, 70).round(0).astype(int)

        # ═══════════════════════════════════════════════════════════
        # ACTIVITY: sigmoid suppression
        # ═══════════════════════════════════════════════════════════
        infection_act = sigmoid(I_g * 2 - 1) * 0.25  # suppression
        fatigue_act = F_g * 0.1
        stress_act = sigmoid(S_g * 2 - 1.5) * 0.1

        act_raw = (circadian_act * (base_activity - infection_act - fatigue_act - stress_act)
                  + rng.normal(0, 0.08, ng))

        act_smooth = np.zeros(ng)
        act_smooth[0] = max(0, act_raw[0])
        for i in range(1, ng):
            act_smooth[i] = 0.85 * act_smooth[i-1] + 0.15 * max(0, act_raw[i])

        g['activity_index'] = np.clip(act_smooth, 0, 1).round(3)

        # ═══════════════════════════════════════════════════════════
        # RUMINATION & LYING
        # ═══════════════════════════════════════════════════════════
        rum_raw = (base_rumination
                  - sigmoid(I_delayed_1h * 3 - 1) * 12   # max -12 (was -25)
                  - sigmoid(S_delayed_2h * 2 - 1) * 5
                  + rng.normal(0, 4, ng))
        g['rumination_min'] = np.clip(rum_raw, 0, 55).round(1)

        lie_raw = (base_lying
                  + sigmoid(I_g * 2 - 0.5) * 8  # sick cows lie more
                  + F_g * 4
                  + rng.normal(0, 3, ng))
        g['lying_min'] = np.clip(lie_raw, 0, 55).round(1)

        results.append(g)

    result = pd.concat(results, ignore_index=True)
    logger.info(f"V3 signals regenerated: {len(result)} rows")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: V4 PRODUCTION REALISM
# ═══════════════════════════════════════════════════════════════════════════════

def generate_v4_realistic(df, rng):
    """Regenerate production signals with probabilistic physics."""
    logger.info("Generating V4 realistic production signals...")
    out = df.copy()
    n = len(out)

    results = []
    for animal_id, group in out.groupby('animalId'):
        g = group.reset_index(drop=True)
        ng = len(g)

        I_g = g['infectionLoad'].values.astype(float)
        S_g = g['stressLoad'].values.astype(float)
        I_s = pd.Series(I_g)
        S_s = pd.Series(S_g)

        # Delayed infection for milk (6h = 72 ticks)
        I_delayed_6h = delayed_value(I_s, 72).values
        I_delayed_12h = delayed_value(I_s, 144).values

        # Individual baselines
        base_milk = 28 + rng.normal(0, 4)  # L/day
        base_feed = 22 + rng.normal(0, 2)  # kg/day
        base_conductivity = 5.0 + rng.normal(0, 0.3)
        base_weight = 550 + rng.normal(0, 30)

        thi = g.get('thi', pd.Series(np.full(ng, 68))).values.astype(float)

        # ═══════════════════════════════════════════════════════════
        # MILK: non-linear, delayed, with overlap
        # ═══════════════════════════════════════════════════════════
        # Healthy variation: ±3 L/day (heat, feed, random)
        healthy_variation = (rng.normal(0, 0.8, ng)  # random daily variation
                           - np.maximum(0, sigmoid((thi - 72) * 0.3) * 2))  # heat reduces milk

        # Disease effect: delayed sigmoid loss
        disease_milk_loss = sigmoid(I_delayed_6h * 3 - 1) * 8  # max -8 L (delayed)
        stress_milk_loss = sigmoid(S_g * 2 - 1.5) * 2

        milk_raw = base_milk + healthy_variation - disease_milk_loss - stress_milk_loss
        g['milkYield'] = np.clip(milk_raw, 5, 45).round(1)

        # ═══════════════════════════════════════════════════════════
        # CONDUCTIVITY: increases with mastitis, noisy
        # ═══════════════════════════════════════════════════════════
        mastitis_flag = (g.get('diseaseType', pd.Series(['none'] * ng)) == 'mastitis').values
        cond_raw = (base_conductivity
                   + sigmoid(I_g * 2) * 0.3 * mastitis_flag.astype(float)
                   + rng.normal(0, 0.15, ng)
                   + np.maximum(0, sigmoid((thi - 72) * 0.2) * 0.1))
        g['conductivity'] = np.clip(cond_raw, 3.5, 8.0).round(2)

        # ═══════════════════════════════════════════════════════════
        # FEED: stochastic, heat-affected
        # ═══════════════════════════════════════════════════════════
        heat_feed_reduction = np.maximum(0, sigmoid((thi - 72) * 0.3) * 3)
        disease_feed_loss = sigmoid(I_delayed_6h * 2 - 0.5) * 4

        feed_raw = (base_feed
                   - heat_feed_reduction
                   - disease_feed_loss
                   + rng.normal(0, 1.5, ng))  # large variation
        g['feedIntake'] = np.clip(feed_raw, 8, 30).round(1)

        # ═══════════════════════════════════════════════════════════
        # WEIGHT: slow response, small signal
        # ═══════════════════════════════════════════════════════════
        weight_trend = -sigmoid(I_delayed_12h * 2 - 1) * 15  # slow weight loss
        weight_raw = base_weight + weight_trend + rng.normal(0, 5, ng)
        g['bodyWeight'] = np.clip(weight_raw, 400, 750).round(1)

        # ═══════════════════════════════════════════════════════════
        # MANAGEMENT EVENTS: Probabilistic
        # ═══════════════════════════════════════════════════════════
        # Vaccination effectiveness: 70-90% (not 100%)
        if 'vaccinationActive' in g.columns:
            vacc_effectiveness = rng.uniform(0.70, 0.90)
            original_vacc = g['vaccinationActive'].values.astype(bool)
            vacc_works = rng.random(ng) < vacc_effectiveness
            g['vaccinationEffective'] = (original_vacc & vacc_works).astype(int)
        else:
            g['vaccinationEffective'] = 0

        # Antibiotic response: variable delay (6-24h) + success rate 80-95%
        if 'antibioticActive' in g.columns:
            abx_delay_ticks = int(rng.uniform(6, 24) * 12)  # 6-24h in ticks
            abx_success_rate = rng.uniform(0.80, 0.95)
            original_abx = g['antibioticActive'].astype(bool)
            abx_delayed = original_abx.shift(abx_delay_ticks).fillna(False)
            abx_works = rng.random(ng) < abx_success_rate
            g['antibioticEffective'] = (abx_delayed & abx_works).astype(int)
        else:
            g['antibioticEffective'] = 0

        results.append(g)

    result = pd.concat(results, ignore_index=True)
    logger.info(f"V4 signals regenerated: {len(result)} rows")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: Load, regenerate, save to MongoDB
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_data(db, collection, limit):
    """Load raw hidden-state data from MongoDB."""
    col = db[collection]
    count = col.count_documents({})
    logger.info(f"Loading from {collection}: {count} total, limit={limit}")

    cursor = col.find({}).sort("timestamp", 1).limit(limit)
    rows = []
    for doc in cursor:
        sig = doc.get("signals", {})
        env = doc.get("environment", {})
        hs = doc.get("hiddenState", {})
        lbl = doc.get("labels", {})
        profile = doc.get("animalProfile", {})

        rows.append({
            "animalId": str(doc.get("animalId")),
            "timestamp": doc.get("timestamp"),
            # Hidden states (for regeneration)
            "infectionLoad": hs.get("infectionLoad", 0),
            "stressLoad": hs.get("stressLoad", 0),
            "compensation": hs.get("compensation", 1),
            "fatigue": hs.get("fatigue", 0),
            # Environment
            "ambientTemp": env.get("ambientTemp_C", 25),
            "humidity": env.get("humidity_pct", 60),
            "thi": env.get("thi", 68),
            # Labels (preserved)
            "diseaseBinary": lbl.get("diseaseBinary", 0),
            "severityLevel": lbl.get("severityLevel", 0),
            "infectionBinary": lbl.get("infectionBinary", 0),
            "stressBinary": lbl.get("stressBinary", 0),
            "diseaseType": lbl.get("diseaseType", "none"),
            # Management (for v4)
            "vaccinationActive": hs.get("vaccinationActive", 0),
            "antibioticActive": hs.get("antibioticActive", 0),
            # Profile
            "parity": profile.get("parity", 0),
            "bcs": profile.get("bodyConditionScore", 3.0),
            "lactationStage": profile.get("lactationStage", "mid"),
        })

    return pd.DataFrame(rows)


def validate_instant_auc(df, signal_col, label_col):
    """Quick instantaneous AUC check."""
    from sklearn.metrics import roc_auc_score
    y = df[label_col].values
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    x = df[signal_col].values
    try:
        return roc_auc_score(y, x)
    except Exception:
        return 0.5


def save_v5_to_mongo(db, df_v3, df_v4, tag_sensor, tag_production):
    """Save v5 data to MongoDB collections — chunked for memory efficiency."""
    batch_size = 5000

    # V3 sensors
    col_v3 = db[tag_sensor]
    col_v3.drop()
    logger.info(f"Saving V3 to {tag_sensor} ({len(df_v3)} rows)...")

    for chunk_start in range(0, len(df_v3), batch_size):
        chunk = df_v3.iloc[chunk_start:chunk_start + batch_size]
        docs = []
        for idx, row in chunk.iterrows():
            docs.append({
                "animalId": row['animalId'],
                "timestamp": row['timestamp'],
                "simulationVersion": "digital_twin_v5_realistic_sensor",
                "signals": {
                    "temperature_C": float(row.get('temperature_C', 38.5)),
                    "heartRate_bpm": int(row.get('heartRate_bpm', 65)),
                    "respiration_bpm": int(row.get('respiration_bpm', 26)),
                    "activity_index": float(row.get('activity_index', 0.7)),
                    "rumination_min": float(row.get('rumination_min', 35)),
                    "lying_min": float(row.get('lying_min', 25)),
                },
                "environment": {
                    "ambientTemp_C": float(row.get('ambientTemp', 25)),
                    "humidity_pct": float(row.get('humidity', 60)),
                    "thi": float(row.get('thi', 68)),
                },
                "hiddenState": {
                    "infectionLoad": float(row['infectionLoad']),
                    "stressLoad": float(row['stressLoad']),
                    "compensation": float(row.get('compensation', 1)),
                    "fatigue": float(row.get('fatigue', 0)),
                },
                "labels": {
                    "diseaseBinary": int(row.get('diseaseBinary', 0)),
                    "severityLevel": int(row.get('severityLevel', 0)),
                    "infectionBinary": int(row.get('infectionBinary', 0)),
                    "stressBinary": int(row.get('stressBinary', 0)),
                    "diseaseType": str(row.get('diseaseType', 'none')),
                }
            })
        col_v3.insert_many(docs)
        if chunk_start % 20000 == 0:
            logger.info(f"  V3: {chunk_start + len(chunk)}/{len(df_v3)}")
    logger.info(f"Saved {len(df_v3)} docs to {tag_sensor}")

    # V4 production
    col_v4 = db[tag_production]
    col_v4.drop()
    logger.info(f"Saving V4 to {tag_production} ({len(df_v4)} rows)...")

    for chunk_start in range(0, len(df_v4), batch_size):
        chunk = df_v4.iloc[chunk_start:chunk_start + batch_size]
        docs = []
        for idx, row in chunk.iterrows():
            docs.append({
                "animalId": row['animalId'],
                "timestamp": row['timestamp'],
                "simulationVersion": "digital_twin_v5_realistic_production",
                "signals": {
                    "temperature_C": float(row.get('temperature_C', 38.5)),
                    "heartRate_bpm": int(row.get('heartRate_bpm', 65)),
                    "respiration_bpm": int(row.get('respiration_bpm', 26)),
                    "activity_index": float(row.get('activity_index', 0.7)),
                    "rumination_min": float(row.get('rumination_min', 35)),
                    "lying_min": float(row.get('lying_min', 25)),
                },
                "production": {
                    "milkYield": float(row.get('milkYield', 28)),
                    "feedIntake": float(row.get('feedIntake', 22)),
                    "conductivity": float(row.get('conductivity', 5.0)),
                    "bodyWeight": float(row.get('bodyWeight', 550)),
                },
                "management": {
                    "vaccinationEffective": int(row.get('vaccinationEffective', 0)),
                    "antibioticEffective": int(row.get('antibioticEffective', 0)),
                },
                "environment": {
                    "ambientTemp_C": float(row.get('ambientTemp', 25)),
                    "humidity_pct": float(row.get('humidity', 60)),
                    "thi": float(row.get('thi', 68)),
                },
                "hiddenState": {
                    "infectionLoad": float(row['infectionLoad']),
                    "stressLoad": float(row['stressLoad']),
                },
                "labels": {
                    "diseaseBinary": int(row.get('diseaseBinary', 0)),
                    "severityLevel": int(row.get('severityLevel', 0)),
                    "infectionBinary": int(row.get('infectionBinary', 0)),
                    "stressBinary": int(row.get('stressBinary', 0)),
                    "diseaseType": str(row.get('diseaseType', 'none')),
                }
            })
        col_v4.insert_many(docs)
        if chunk_start % 20000 == 0:
            logger.info(f"  V4: {chunk_start + len(chunk)}/{len(df_v4)}")
    logger.info(f"Saved {len(df_v4)} docs to {tag_production}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()
    rng = np.random.default_rng(args.seed)

    logger.info("=" * 60)
    logger.info("🧬 GoMata Digital Twin v5 — Probabilistic Realism Generator")
    logger.info("=" * 60)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # ── Load from source collections ───────────────────────────
    df_v3_raw = load_raw_data(db, "validation_clean_v3", args.limit)
    df_v4_raw = load_raw_data(db, "validation_clean_v4", args.limit)
    logger.info(f"V3 raw: {len(df_v3_raw)} rows, V4 raw: {len(df_v4_raw)} rows")

    # ── Generate realistic signals ─────────────────────────────
    df_v3 = generate_v3_realistic(df_v3_raw, rng)
    df_v4_merged = df_v3.copy()  # Start with v3 signals
    # Merge production from v4 raw
    for col in ['vaccinationActive', 'antibioticActive']:
        if col in df_v4_raw.columns:
            min_len = min(len(df_v4_merged), len(df_v4_raw))
            df_v4_merged = df_v4_merged.iloc[:min_len]
            df_v4_raw_trunc = df_v4_raw.iloc[:min_len]
            df_v4_merged[col] = df_v4_raw_trunc[col].values

    df_v4 = generate_v4_realistic(df_v4_merged, rng)

    # ── Validation: instantaneous AUC checks ───────────────────
    logger.info("\n── Instantaneous AUC Checks ──")
    temp_auc = validate_instant_auc(df_v3, 'temperature_C', 'diseaseBinary')
    hr_auc = validate_instant_auc(df_v3, 'heartRate_bpm', 'diseaseBinary')
    activity_auc = validate_instant_auc(df_v3, 'activity_index', 'diseaseBinary')
    milk_auc = validate_instant_auc(df_v4, 'milkYield', 'diseaseBinary')

    logger.info(f"  Temperature AUC: {temp_auc:.4f} (target ≤ 0.85)")
    logger.info(f"  Heart Rate AUC:  {hr_auc:.4f}")
    logger.info(f"  Activity AUC:    {activity_auc:.4f}")
    logger.info(f"  Milk Yield AUC:  {milk_auc:.4f} (target ≤ 0.80)")

    temp_ok = temp_auc <= 0.85
    milk_ok = milk_auc <= 0.80
    logger.info(f"  Temp check: {'✅' if temp_ok else '⚠️ Adjust noise/coefficients'}")
    logger.info(f"  Milk check: {'✅' if milk_ok else '⚠️ Adjust noise/coefficients'}")

    # ── Save to MongoDB ────────────────────────────────────────
    logger.info("\n── Saving to MongoDB ──")
    save_v5_to_mongo(db, df_v3, df_v4,
                     "trainingevents_v5", "trainingevents_v5_production")

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Digital Twin v5 Generation Complete")
    logger.info(f"   V3 sensor: {len(df_v3)} rows → trainingevents_v5")
    logger.info(f"   V4 production: {len(df_v4)} rows → trainingevents_v5_production")
    logger.info(f"   Instant temp AUC: {temp_auc:.4f}")
    logger.info(f"   Instant milk AUC: {milk_auc:.4f}")
    logger.info(f"   Duration: {elapsed:.1f}s")
    logger.info(f"{'='*60}")

    client.close()


if __name__ == "__main__":
    main()
