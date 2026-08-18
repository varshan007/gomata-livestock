#!/usr/bin/env python3
"""
reality_gap_v3.py — GoMata Pilot Validation Framework v1, Part 1
Reality Gap Testing Engine

Stress-tests model robustness against real-world imperfections:
  A) Sensor drift (temp, HR, humidity, THI)
  B) Missing data injection (random, block, device failure)
  C) Temporal delay simulation (30min, 2h, 6h)
  D) Event misreporting (v4: vaccination, antibiotic, feed)

Outputs a degradation table: | Scenario | Metric | Clean | Noisy | % Drop |

Usage:
  python reality_gap_v3.py [--mongo-uri mongodb://...] [--limit 100000]
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pymongo import MongoClient
from copy import deepcopy
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "reality_gap.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RealityGap")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DB_NAME = "livestock_monitoring"

# ═════════════════════════════════════════════════════════════════════════════════
# CORRUPTION PIPELINES
# ═════════════════════════════════════════════════════════════════════════════════

class SensorDrift:
    """A) Sensor drift corruption — adds realistic noise + progressive drift."""

    @staticmethod
    def apply_temperature_drift(df, mu=0.3, sigma=0.1, progressive_rate=0.002):
        """Add Gaussian offset + progressive time drift to temperature."""
        out = df.copy()
        n = len(out)
        noise = np.random.normal(mu, sigma, n)
        drift = progressive_rate * np.arange(n)
        out['temp_current'] = out['temp_current'] + noise + drift
        if 'temp_6h_avg' in out.columns:
            out['temp_6h_avg'] = out['temp_6h_avg'] + noise * 0.5 + drift * 0.5
        return out

    @staticmethod
    def apply_heart_rate_drift(df, mu=0, sigma=5):
        """Add Gaussian noise to heart rate."""
        out = df.copy()
        noise = np.random.normal(mu, sigma, len(out))
        out['hr_current'] = out['hr_current'] + noise
        if 'hr_6h_avg' in out.columns:
            out['hr_6h_avg'] = out['hr_6h_avg'] + noise * 0.3
        return out

    @staticmethod
    def apply_humidity_drift(df, pct_variation=0.10):
        """±10% humidity perturbation."""
        out = df.copy()
        if 'humidity_pct' in out.columns:
            factor = 1 + np.random.uniform(-pct_variation, pct_variation, len(out))
            out['humidity_pct'] = out['humidity_pct'] * factor
        return out

    @staticmethod
    def apply_thi_drift(df, units=3):
        """±3 unit THI perturbation."""
        out = df.copy()
        if 'thi' in out.columns:
            noise = np.random.uniform(-units, units, len(out))
            out['thi'] = out['thi'] + noise
        return out

    @classmethod
    def apply_all(cls, df):
        """Apply all sensor drift corruptions."""
        out = cls.apply_temperature_drift(df)
        out = cls.apply_heart_rate_drift(out)
        out = cls.apply_humidity_drift(out)
        out = cls.apply_thi_drift(out)
        return out


class MissingData:
    """B) Missing data injection — random, block, and device failure."""

    SENSOR_COLS = ['temp_current', 'temp_6h_avg', 'temp_6h_std', 'temp_6h_slope',
                   'hr_current', 'hr_6h_avg', 'hr_6h_std',
                   'activity_current', 'activity_6h_avg', 'activity_6h_std',
                   'rumination_drop', 'heat_stress_index', 'composite_stress_index']

    @classmethod
    def random_missing(cls, df, rate=0.05):
        """Randomly set rate% of sensor values to NaN."""
        out = df.copy()
        cols = [c for c in cls.SENSOR_COLS if c in out.columns]
        mask = np.random.random((len(out), len(cols))) < rate
        for i, col in enumerate(cols):
            out.loc[mask[:, i], col] = np.nan
        return out

    @classmethod
    def block_missing(cls, df, block_hours=6, ticks_per_hour=12, num_blocks=5):
        """Insert contiguous 6-hour gaps."""
        out = df.copy()
        block_size = block_hours * ticks_per_hour
        cols = [c for c in cls.SENSOR_COLS if c in out.columns]
        for _ in range(num_blocks):
            start = np.random.randint(0, max(1, len(out) - block_size))
            end = min(start + block_size, len(out))
            for col in cols:
                out.loc[start:end, col] = np.nan
        return out

    @classmethod
    def device_failure(cls, df, signal='temp'):
        """Entire signal column dropout (simulate broken sensor)."""
        out = df.copy()
        target_cols = [c for c in out.columns if c.startswith(signal)]
        for col in target_cols:
            out[col] = np.nan
        return out


class TemporalDelay:
    """C) Temporal delay simulation — shift features forward."""

    @classmethod
    def shift_features(cls, df, ticks=6):
        """Shift all feature columns forward by N ticks (stale data)."""
        out = df.copy()
        feature_cols = [c for c in out.columns if c.startswith(('temp_', 'hr_', 'activity_',
                                                                 'rumination_', 'heat_stress', 'composite_'))]
        for col in feature_cols:
            out[col] = out[col].shift(ticks)
        # Fill leading NaN with first valid value
        out = out.fillna(method='bfill')
        return out


class EventMisreporting:
    """D) Event misreporting — v4 management flag corruption."""

    @staticmethod
    def flip_vaccination(df, rate=0.10):
        """Randomly flip 10% of vaccination flags."""
        out = df.copy()
        if 'vaccinationActive' in out.columns:
            mask = np.random.random(len(out)) < rate
            out.loc[mask, 'vaccinationActive'] = ~out.loc[mask, 'vaccinationActive'].astype(bool)
        return out

    @staticmethod
    def delay_antibiotic(df, delay_ticks=144):
        """Delay antibiotic activation by 12h (144 ticks at 5min)."""
        out = df.copy()
        if 'antibioticActive' in out.columns:
            out['antibioticActive'] = out['antibioticActive'].shift(delay_ticks).fillna(False)
        return out

    @staticmethod
    def remove_feed_change(df, rate=0.10):
        """Remove 10% of feed change records."""
        out = df.copy()
        if 'feedChangeActive' in out.columns:
            mask = np.random.random(len(out)) < rate
            out.loc[mask, 'feedChangeActive'] = False
        return out


# ═════════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════════

def compute_metrics(df, label_col='diseaseBinary', severity_col='severityLevel',
                    milk_col='milkYield', forecast_col='forecastRisk24h'):
    """Compute key metrics from a dataset. Uses simple proxy models when no trained model."""
    metrics = {}

    # ── Disease AUC (proxy: composite_stress_index as predictor) ──────
    if 'composite_stress_index' in df.columns and label_col in df.columns:
        valid = df.dropna(subset=['composite_stress_index', label_col])
        if len(valid) > 100 and valid[label_col].nunique() > 1:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['disease_auc'] = roc_auc_score(
                    valid[label_col], valid['composite_stress_index']
                )
            except Exception:
                metrics['disease_auc'] = 0.5
        else:
            metrics['disease_auc'] = 0.5
    else:
        metrics['disease_auc'] = 0.5

    # ── Severity MAE (proxy: temp_current → severity) ────────────────
    if severity_col in df.columns and 'temp_current' in df.columns:
        valid = df.dropna(subset=['temp_current', severity_col])
        if len(valid) > 100:
            from sklearn.metrics import mean_absolute_error
            # Simple proxy: normalize temp to 0-3 range
            temp_pred = np.clip((valid['temp_current'] - 38.0) / 1.5, 0, 3)
            metrics['severity_mae'] = mean_absolute_error(valid[severity_col], temp_pred)
        else:
            metrics['severity_mae'] = 0
    else:
        metrics['severity_mae'] = 0

    # ── Forecast RMSE (proxy: feature consistency) ───────────────────
    if forecast_col in df.columns:
        valid = df.dropna(subset=[forecast_col, label_col])
        if len(valid) > 100:
            from sklearn.metrics import mean_squared_error
            metrics['forecast_rmse'] = np.sqrt(mean_squared_error(
                valid[label_col], valid[forecast_col]
            ))
        else:
            metrics['forecast_rmse'] = 0
    else:
        # Use temporal feature consistency as proxy
        if 'temp_6h_slope' in df.columns:
            valid = df.dropna(subset=['temp_6h_slope'])
            metrics['forecast_rmse'] = valid['temp_6h_slope'].std() if len(valid) > 0 else 0
        else:
            metrics['forecast_rmse'] = 0

    # ── Milk Loss Prediction Error (v4 only) ─────────────────────────
    if milk_col in df.columns and severity_col in df.columns:
        valid = df.dropna(subset=[milk_col, severity_col])
        if len(valid) > 100:
            # Expected: higher severity = lower milk
            sev_milk = valid.groupby(severity_col)[milk_col].agg(['mean', 'count'])
            # Only check monotonicity for severity levels with >50 samples
            sev_means = sev_milk[sev_milk['count'] > 50]['mean']
            if len(sev_means) >= 3:
                diffs = sev_means.diff().dropna()
                metrics['milk_monotonic'] = all(diffs <= 0)
                metrics['milk_loss_error'] = abs(
                    sev_means.iloc[0] - sev_means.iloc[-1]
                )
            else:
                # Insufficient data per severity — skip monotonicity
                metrics['milk_monotonic'] = True
                metrics['milk_loss_error'] = 0
        else:
            metrics['milk_monotonic'] = True
            metrics['milk_loss_error'] = 0
    else:
        metrics['milk_monotonic'] = True
        metrics['milk_loss_error'] = 0

    return metrics


# ═════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════════

def load_validation_v3(db, limit=100000):
    """Load validation_clean_v3 as flattened DataFrame."""
    col = db["validation_clean_v3"]
    cur = col.find({}).limit(limit)
    rows = []
    for doc in cur:
        row = {}
        # Flatten features
        for k, v in doc.get('features', {}).items():
            row[k] = v
        # Flatten signals
        for k, v in doc.get('signals', {}).items():
            if isinstance(v, dict):
                continue
            row[k] = v
        # Flatten environment
        for k, v in doc.get('environment', {}).items():
            row[k] = v
        # Flatten labels
        for k, v in doc.get('labels', {}).items():
            row[k] = v
        # Hidden state (for ground truth only)
        for k, v in doc.get('hiddenState', {}).items():
            row[f'hidden_{k}'] = v
        row['animalId'] = str(doc.get('animalId'))
        row['timestamp'] = doc.get('timestamp')
        rows.append(row)
    return pd.DataFrame(rows)


def load_validation_v4(db, limit=100000):
    """Load validation_clean_v4 as flattened DataFrame."""
    col = db["validation_clean_v4"]
    cur = col.find({}).limit(limit)
    rows = []
    for doc in cur:
        row = {}
        for k, v in doc.get('features', {}).items():
            row[k] = v
        for k, v in doc.get('signals', {}).items():
            if isinstance(v, dict):
                continue
            row[k] = v
        for k, v in doc.get('environment', {}).items():
            row[k] = v
        for k, v in doc.get('labels', {}).items():
            row[k] = v
        for k, v in doc.get('production', {}).items():
            row[k] = v
        for k, v in doc.get('managementEvents', {}).items():
            row[k] = v
        for k, v in doc.get('hiddenState', {}).items():
            row[f'hidden_{k}'] = v
        for k, v in doc.get('animalProfile', {}).items():
            if isinstance(v, (str, int, float, bool)):
                row[f'profile_{k}'] = v
        row['animalId'] = str(doc.get('animalId'))
        row['timestamp'] = doc.get('timestamp')
        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER
# ═════════════════════════════════════════════════════════════════════════════════

def run_v3_scenarios(df_clean):
    """Run all v3 corruption scenarios and measure degradation."""
    results = []
    clean_metrics = compute_metrics(df_clean)
    results.append(('Clean Baseline', clean_metrics))
    logger.info(f"Clean v3 metrics: {clean_metrics}")

    scenarios = [
        ('Sensor Drift (all)',       SensorDrift.apply_all),
        ('Temp Drift Only',          SensorDrift.apply_temperature_drift),
        ('HR Drift Only',            SensorDrift.apply_heart_rate_drift),
        ('Missing 5%',               lambda d: MissingData.random_missing(d, 0.05)),
        ('Missing 15%',              lambda d: MissingData.random_missing(d, 0.15)),
        ('Missing 30%',              lambda d: MissingData.random_missing(d, 0.30)),
        ('Block Missing 6h',         MissingData.block_missing),
        ('Device Failure (temp)',     lambda d: MissingData.device_failure(d, 'temp')),
        ('Device Failure (hr)',       lambda d: MissingData.device_failure(d, 'hr')),
        ('Temporal Delay 30min',      lambda d: TemporalDelay.shift_features(d, 6)),
        ('Temporal Delay 2h',         lambda d: TemporalDelay.shift_features(d, 24)),
        ('Temporal Delay 6h',         lambda d: TemporalDelay.shift_features(d, 72)),
    ]

    for name, fn in scenarios:
        logger.info(f"  Running scenario: {name}")
        corrupted = fn(df_clean.copy())
        # Fill NaN with column mean for metric computation
        corrupted_filled = corrupted.fillna(corrupted.mean(numeric_only=True))
        noisy_metrics = compute_metrics(corrupted_filled)
        results.append((name, noisy_metrics))

    return results, clean_metrics


def run_v4_scenarios(df_clean):
    """Run v4-specific corruption scenarios (includes management events)."""
    results = []
    clean_metrics = compute_metrics(df_clean, milk_col='milkYield')
    results.append(('Clean Baseline (v4)', clean_metrics))
    logger.info(f"Clean v4 metrics: {clean_metrics}")

    scenarios = [
        ('Sensor Drift (all)',        SensorDrift.apply_all),
        ('Missing 5%',                lambda d: MissingData.random_missing(d, 0.05)),
        ('Missing 15%',               lambda d: MissingData.random_missing(d, 0.15)),
        ('Missing 30%',               lambda d: MissingData.random_missing(d, 0.30)),
        ('Temporal Delay 2h',         lambda d: TemporalDelay.shift_features(d, 24)),
        ('Temporal Delay 6h',         lambda d: TemporalDelay.shift_features(d, 72)),
        ('Vacc Misreporting 10%',     EventMisreporting.flip_vaccination),
        ('Antibiotic Delay 12h',      EventMisreporting.delay_antibiotic),
        ('Feed Change Missing 10%',   EventMisreporting.remove_feed_change),
    ]

    for name, fn in scenarios:
        logger.info(f"  Running scenario: {name}")
        corrupted = fn(df_clean.copy())
        corrupted_filled = corrupted.fillna(corrupted.mean(numeric_only=True))
        noisy_metrics = compute_metrics(corrupted_filled, milk_col='milkYield')
        results.append((name, noisy_metrics))

    return results, clean_metrics


# ═════════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═════════════════════════════════════════════════════════════════════════════════

def generate_degradation_table(results, clean_metrics, version='v3'):
    """Generate formatted degradation table."""
    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for scenario, metrics in results:
        for metric_name in ['disease_auc', 'severity_mae', 'forecast_rmse', 'milk_loss_error']:
            clean_val = clean_metrics.get(metric_name, 0)
            noisy_val = metrics.get(metric_name, 0)
            if clean_val > 0:
                pct_drop = ((noisy_val - clean_val) / clean_val) * 100
            else:
                pct_drop = 0
            rows.append({
                'Scenario': scenario,
                'Metric': metric_name,
                'Clean': round(clean_val, 4),
                'Noisy': round(noisy_val, 4),
                '% Change': round(pct_drop, 2)
            })

    table = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, f"reality_gap_{version}.csv")
    table.to_csv(csv_path, index=False)
    logger.info(f"Saved degradation table: {csv_path}")

    return table


def evaluate_pass_criteria(results, clean_metrics):
    """Evaluate pass/fail against pilot criteria."""
    verdicts = []

    for scenario, metrics in results:
        if scenario.startswith('Clean'):
            continue

        # AUC degradation ≤15%
        clean_auc = clean_metrics.get('disease_auc', 0.5)
        noisy_auc = metrics.get('disease_auc', 0.5)
        auc_drop = (clean_auc - noisy_auc) / clean_auc * 100 if clean_auc > 0 else 0

        # Forecast RMSE increase ≤20%
        clean_rmse = clean_metrics.get('forecast_rmse', 0)
        noisy_rmse = metrics.get('forecast_rmse', 0)
        rmse_increase = (noisy_rmse - clean_rmse) / clean_rmse * 100 if clean_rmse > 0 else 0

        # Milk monotonicity preserved
        milk_ok = metrics.get('milk_monotonic', True)

        passed = auc_drop <= 15 and rmse_increase <= 20 and milk_ok
        verdicts.append({
            'Scenario': scenario,
            'AUC Drop %': round(auc_drop, 2),
            'RMSE Increase %': round(rmse_increase, 2),
            'Milk Monotonic': milk_ok,
            'VERDICT': '✅ PASS' if passed else '❌ FRAGILE'
        })

    return pd.DataFrame(verdicts)


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Reality Gap Testing Engine')
    parser.add_argument('--mongo-uri', default=MONGO_URI)
    parser.add_argument('--limit', type=int, default=100000)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🧬 GoMata Reality Gap Testing Engine — v1")
    logger.info("=" * 60)

    client = MongoClient(args.mongo_uri)
    db = client[DB_NAME]

    output_dir = os.path.join(os.path.dirname(__file__), "../training_data")
    os.makedirs(output_dir, exist_ok=True)

    # ── V3 Scenarios ─────────────────────────────────────────────────────
    logger.info("\n── Loading validation_clean_v3 ──")
    df_v3 = load_validation_v3(db, args.limit)
    logger.info(f"Loaded {len(df_v3)} v3 records ({df_v3.columns.tolist()[:10]}...)")

    if len(df_v3) > 0:
        logger.info("\n── Running V3 Reality Gap Scenarios ──")
        v3_results, v3_clean = run_v3_scenarios(df_v3)
        v3_table = generate_degradation_table(v3_results, v3_clean, 'v3')
        v3_verdicts = evaluate_pass_criteria(v3_results, v3_clean)

        logger.info("\n── V3 Degradation Table ──")
        logger.info(f"\n{v3_table.to_string(index=False)}")
        logger.info("\n── V3 Pass/Fail Verdicts ──")
        logger.info(f"\n{v3_verdicts.to_string(index=False)}")

        v3_verdicts.to_csv(os.path.join(output_dir, "reality_gap_v3_verdicts.csv"), index=False)
    else:
        logger.warning("No v3 data found. Run generate_validation_sets.js first.")

    # ── V4 Scenarios ─────────────────────────────────────────────────────
    logger.info("\n── Loading validation_clean_v4 ──")
    df_v4 = load_validation_v4(db, args.limit)
    logger.info(f"Loaded {len(df_v4)} v4 records")

    if len(df_v4) > 0:
        logger.info("\n── Running V4 Reality Gap Scenarios ──")
        v4_results, v4_clean = run_v4_scenarios(df_v4)
        v4_table = generate_degradation_table(v4_results, v4_clean, 'v4')
        v4_verdicts = evaluate_pass_criteria(v4_results, v4_clean)

        logger.info("\n── V4 Degradation Table ──")
        logger.info(f"\n{v4_table.to_string(index=False)}")
        logger.info("\n── V4 Pass/Fail Verdicts ──")
        logger.info(f"\n{v4_verdicts.to_string(index=False)}")

        v4_verdicts.to_csv(os.path.join(output_dir, "reality_gap_v4_verdicts.csv"), index=False)
    else:
        logger.warning("No v4 data found. Run generate_validation_sets.js first.")

    logger.info("\n" + "=" * 60)
    logger.info("🟢 Reality Gap Testing Complete")
    logger.info("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
