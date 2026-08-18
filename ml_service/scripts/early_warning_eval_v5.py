#!/usr/bin/env python3
"""
early_warning_eval_v5.py — GoMata Phase 6, Part 3
Measures how early the model detects disease before severity ≥ 2.
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from v5_config import ALL_FEATURES, load_and_fuse, DATA_DIR, MODEL_DIR
from split_strategy_v2 import animal_time_split

log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "early_warning_v5.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("EarlyWarningV5")

TICK_HOURS = 5 / 60
TICKS_PER_WEEK = 2016


def compute_early_detection(y_prob, severity, animal_ids, thresholds):
    results = {}
    for thresh in thresholds:
        alerts = y_prob >= thresh
        lead_times = []
        detected_24h = 0
        detected_12h = 0
        total_episodes = 0
        total_fp = 0
        total_cow_weeks = 0

        for animal_id in np.unique(animal_ids):
            mask = animal_ids == animal_id
            prob_a = y_prob[mask]
            sev_a = severity[mask]
            alert_a = alerts[mask]
            n = mask.sum()

            # Find severity episodes (severity ≥ 2)
            in_episode = False
            for i in range(len(sev_a)):
                if sev_a[i] >= 2 and not in_episode:
                    ep_start = i
                    in_episode = True
                    total_episodes += 1

                    # Search backwards for earliest alert
                    pre = alert_a[:i]
                    if pre.any():
                        first_alert = np.where(pre)[0][-1]
                        lead_ticks = i - first_alert
                        lead_hours = lead_ticks * TICK_HOURS
                        lead_times.append(lead_hours)
                        if lead_hours >= 24:
                            detected_24h += 1
                        if lead_hours >= 12:
                            detected_12h += 1
                elif sev_a[i] < 2:
                    in_episode = False

            # False positives: alerts when healthy
            healthy_mask = sev_a < 1
            fp_count = int((alert_a & healthy_mask).sum())
            total_fp += fp_count
            total_cow_weeks += n / TICKS_PER_WEEK

        fp_per_week = total_fp / max(total_cow_weeks, 1)
        pct_24 = (detected_24h / max(total_episodes, 1)) * 100
        pct_12 = (detected_12h / max(total_episodes, 1)) * 100
        avg_lead = float(np.mean(lead_times)) if lead_times else 0

        results[str(thresh)] = {
            "threshold": thresh, "total_episodes": total_episodes,
            "detected_episodes": len(lead_times),
            "pct_detected_24h_early": round(pct_24, 1),
            "pct_detected_12h_early": round(pct_12, 1),
            "avg_lead_time_hours": round(avg_lead, 1),
            "fp_per_cow_per_week": round(float(fp_per_week), 2),
            "total_false_positives": int(total_fp),
            "pass_24h": pct_24 >= 50,
            "pass_fp": fp_per_week <= 0.10 * TICKS_PER_WEEK,
        }
        logger.info(f"  θ={thresh}: {total_episodes} episodes, {pct_24:.0f}% ≥24h, "
                    f"{pct_12:.0f}% ≥12h, FP/wk={fp_per_week:.2f}")

    return results


def main():
    logger.info("=" * 60)
    logger.info("⏱ Phase 6 Part 3 — Early Detection Capability")
    logger.info("=" * 60)

    model = joblib.load(os.path.join(MODEL_DIR, "model_v5.pkl"))
    df = load_and_fuse()
    logger.info(f"Fused: {len(df)} rows")

    _, X_test, _, y_test, _ = animal_time_split(
        df, ALL_FEATURES, "disease_binary",
        animal_train_ratio=0.8, time_train_ratio=0.7, seed=42)

    y_prob = model.predict_proba(X_test)[:, 1]
    severity = df.loc[y_test.index, 'severity_level'].values
    animal_ids = df.loc[y_test.index, 'animal_id'].values

    logger.info(f"Test: {len(y_test)} rows, {y_test.sum()} pos, "
                f"{(severity >= 2).sum()} ticks sev≥2")

    results = compute_early_detection(y_prob, severity, animal_ids, [0.5, 0.7, 0.85])

    with open(os.path.join(DATA_DIR, "early_detection_report_v5.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  ✅ early_detection_report_v5.json saved")


if __name__ == "__main__":
    main()
