#!/usr/bin/env python3
"""
economic_impact_eval_v5.py — GoMata Phase 6, Part 4
Simulates intervention at model alerts, computes milk loss reduction + ROI.
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
    handlers=[logging.FileHandler(os.path.join(log_dir, "economic_v5.log"), mode='w'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("EconomicV5")

TICK_HOURS = 5 / 60
MILK_PRICE = 0.45
TREATMENT_COST = 25.0
FA_COST = 5.0


def simulate_milk_loss(sev_series):
    return float((sev_series * 0.5 * TICK_HOURS).sum())


def simulate_with_intervention(sev_series, alert_tick):
    sev = sev_series.copy()
    decay = 0.5 ** (1 / (6 / TICK_HOURS))
    for i in range(alert_tick, len(sev)):
        sev[i] = sev[i] * (decay ** (i - alert_tick))
    return float((sev * 0.5 * TICK_HOURS).sum())


def compute_economic_impact(y_prob, severity, animal_ids, threshold):
    alerts = y_prob >= threshold
    total_no_int = 0
    total_with_int = 0
    total_treatments = 0
    total_fa = 0
    episodes_found = 0

    for animal_id in np.unique(animal_ids):
        mask = animal_ids == animal_id
        sev_a = severity[mask].astype(float)
        alert_a = alerts[mask]

        # Find disease stretches
        in_ep = False
        ep_start = None
        for i in range(len(sev_a)):
            if sev_a[i] > 0.5 and not in_ep:
                ep_start = i
                in_ep = True
            elif sev_a[i] <= 0.1 and in_ep:
                if i - ep_start > 12:
                    ep_sev = sev_a[ep_start:i]
                    ep_alert = alert_a[ep_start:i]
                    no_int = simulate_milk_loss(ep_sev)
                    total_no_int += no_int
                    episodes_found += 1

                    if ep_alert.any():
                        first = np.where(ep_alert)[0][0]
                        with_int = simulate_with_intervention(ep_sev, first)
                        total_with_int += with_int
                        total_treatments += 1
                    else:
                        total_with_int += no_int
                in_ep = False

        # End-of-series
        if in_ep and ep_start is not None and len(sev_a) - ep_start > 12:
            ep_sev = sev_a[ep_start:]
            ep_alert = alert_a[ep_start:]
            no_int = simulate_milk_loss(ep_sev)
            total_no_int += no_int
            episodes_found += 1
            if ep_alert.any():
                with_int = simulate_with_intervention(ep_sev, np.where(ep_alert)[0][0])
                total_with_int += with_int
                total_treatments += 1
            else:
                total_with_int += no_int

        # False alarms
        total_fa += int((alert_a & (sev_a < 0.5)).sum())

    saved = total_no_int - total_with_int
    pct = (saved / max(total_no_int, 0.001)) * 100
    rev = saved * MILK_PRICE
    treat_cost = total_treatments * TREATMENT_COST
    fa_cost = total_fa * FA_COST
    net = rev - treat_cost - fa_cost
    roi = net / max(treat_cost + fa_cost, 1) * 100

    return {
        "threshold": threshold, "episodes_found": episodes_found,
        "total_no_intervention_loss_L": round(total_no_int, 2),
        "total_with_intervention_loss_L": round(total_with_int, 2),
        "milk_saved_L": round(saved, 2),
        "pct_milk_loss_reduction": round(pct, 1),
        "revenue_saved_usd": round(rev, 2),
        "treatments": total_treatments,
        "treatment_cost_usd": round(treat_cost, 2),
        "false_alarms": total_fa,
        "false_alarm_cost_usd": round(fa_cost, 2),
        "net_benefit_usd": round(net, 2),
        "roi_pct": round(roi, 1),
        "pass": pct >= 20,
    }


def main():
    logger.info("=" * 60)
    logger.info("💰 Phase 6 Part 4 — Economic Impact Evaluation")
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

    results = {}
    for thresh in [0.3, 0.5, 0.7]:
        r = compute_economic_impact(y_prob, severity, animal_ids, thresh)
        results[str(thresh)] = r
        logger.info(f"  θ={thresh}: {r['pct_milk_loss_reduction']:.0f}% reduction, "
                    f"ROI={r['roi_pct']:.0f}%, net=${r['net_benefit_usd']:.0f}")

    with open(os.path.join(DATA_DIR, "economic_impact_report_v5.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  ✅ economic_impact_report_v5.json saved")


if __name__ == "__main__":
    main()
