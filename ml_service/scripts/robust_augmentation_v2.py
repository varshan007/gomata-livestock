#!/usr/bin/env python3
"""
robust_augmentation_v2.py — GoMata Model Hardening v2, Phase 2
Post-Extraction Corruption Engine

Applies stochastic corruption to extracted features (CSV) — NEVER to raw DB.

Corruption modes:
  A) Sensor drift (temp, HR, humidity, THI + progressive)
  B) Missingness (random 10-20%, block 3-6h, device offline)
  C) Timestamp jitter (shift lag/window features)
  D) Management misreporting (vacc flip, abx delay, feed drop)
  E) 60/40 clean/noisy curriculum mixing

Usage:
  from robust_augmentation_v2 import augment_curriculum
  X_augmented, y_augmented = augment_curriculum(X_clean, y_clean, noisy_ratio=0.4)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("Augmentation")

# ═══════════════════════════════════════════════════════════════════════════════
# A) SENSOR DRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def sensor_drift(df, rng=None):
    """Inject realistic sensor drift into physiological features."""
    rng = rng or np.random.default_rng()
    out = df.copy()
    n = len(out)

    # Temperature drift: N(0, 0.3) + progressive
    temp_cols = [c for c in out.columns if c.startswith('temp_') and 'lag' not in c]
    for col in temp_cols:
        if col in out.columns:
            drift_rate = rng.uniform(0.0001, 0.002)
            noise = rng.normal(0, 0.3, n) + drift_rate * np.arange(n)
            out[col] = out[col] + noise

    # Heart rate drift: N(0, 4)
    hr_cols = [c for c in out.columns if c.startswith('hr_') and 'lag' not in c]
    for col in hr_cols:
        if col in out.columns:
            out[col] = out[col] + rng.normal(0, 4, n)

    # Humidity drift: ±8%
    if 'humidity' in out.columns:
        factor = 1 + rng.uniform(-0.08, 0.08, n)
        out['humidity'] = out['humidity'] * factor

    # THI drift: ±2 units
    if 'thi' in out.columns:
        out['thi'] = out['thi'] + rng.uniform(-2, 2, n)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# B) MISSINGNESS
# ═══════════════════════════════════════════════════════════════════════════════

SENSOR_COLS = [
    'temp_current', 'temp_1h_avg', 'temp_6h_avg', 'temp_12h_median',
    'temp_24h_std', 'temp_6h_std', 'temp_1h_std',
    'hr_current', 'hr_1h_avg', 'hr_6h_avg', 'hr_12h_median', 'hr_6h_std',
    'activity_current', 'activity_1h_avg', 'activity_6h_avg', 'activity_6h_std',
    'resp_current', 'resp_6h_avg', 'resp_6h_std',
    'rumination_current', 'lying_current',
]

def random_missing(df, rate=0.15, rng=None):
    """Random 10-20% point missing across sensor features."""
    rng = rng or np.random.default_rng()
    out = df.copy()
    cols = [c for c in SENSOR_COLS if c in out.columns]
    mask = rng.random((len(out), len(cols))) < rate
    for i, col in enumerate(cols):
        idx = np.where(mask[:, i])[0]
        out.iloc[idx, out.columns.get_loc(col)] = np.nan
    return out


def block_missing(df, block_hours=4, num_blocks=3, rng=None):
    """3-6 hour contiguous block dropouts."""
    rng = rng or np.random.default_rng()
    out = df.copy()
    block_size = block_hours * 12  # 12 ticks/hour
    cols = [c for c in SENSOR_COLS if c in out.columns]
    for _ in range(num_blocks):
        start = rng.integers(0, max(1, len(out) - block_size))
        end = min(start + block_size, len(out))
        for col in cols:
            col_idx = out.columns.get_loc(col)
            out.iloc[start:end, col_idx] = np.nan
    return out


def device_offline(df, signal='temp', rng=None):
    """Full signal blackout — simulates broken device."""
    out = df.copy()
    target = [c for c in out.columns if signal in c.lower()]
    for col in target:
        out[col] = np.nan
    return out


def inject_missingness(df, mode='mixed', rng=None):
    """Inject one of the missingness modes."""
    rng = rng or np.random.default_rng()
    if mode == 'random':
        rate = rng.uniform(0.10, 0.20)
        return random_missing(df, rate, rng)
    elif mode == 'block':
        hours = rng.integers(3, 7)
        blocks = rng.integers(2, 5)
        return block_missing(df, hours, blocks, rng)
    elif mode == 'offline':
        signal = rng.choice(['temp', 'hr', 'activity'])
        return device_offline(df, signal, rng)
    else:  # mixed
        choice = rng.choice(['random', 'block', 'offline'], p=[0.5, 0.3, 0.2])
        return inject_missingness(df, choice, rng)


# ═══════════════════════════════════════════════════════════════════════════════
# C) TIMESTAMP JITTER
# ═══════════════════════════════════════════════════════════════════════════════

LAG_COLS = [
    'temp_lag_1h', 'temp_lag_3h', 'temp_lag_6h', 'temp_lag_12h',
    'hr_lag_1h', 'hr_lag_3h', 'hr_lag_6h', 'hr_lag_12h',
    'activity_lag_1h', 'activity_lag_3h', 'activity_lag_6h', 'activity_lag_12h',
]

def timestamp_jitter(df, max_shift_ticks=72, rng=None):
    """Shift lag and window features to simulate timestamp misalignment."""
    rng = rng or np.random.default_rng()
    out = df.copy()
    shift = rng.integers(1, max_shift_ticks + 1)

    # Shift lag features
    for col in LAG_COLS:
        if col in out.columns:
            out[col] = out[col].shift(shift).bfill().ffill()

    # Shift rolling window features
    window_cols = [c for c in out.columns if any(w in c for w in ['_1h_', '_6h_', '_12h_', '_24h_'])]
    for col in window_cols:
        if col not in LAG_COLS and col in out.columns:
            jitter = rng.integers(-shift//2, shift//2 + 1)
            if jitter != 0:
                out[col] = out[col].shift(jitter).bfill().ffill()

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# D) MANAGEMENT MISREPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def mgmt_misreport(df, rng=None):
    """Corrupt management features: flip vacc 5%, delay abx 6-12h, drop 10% feed."""
    rng = rng or np.random.default_rng()
    out = df.copy()

    # Flip 5% vaccination decay values (simulate false reporting)
    if 'vacc_decay' in out.columns:
        mask = rng.random(len(out)) < 0.05
        out.loc[mask, 'vacc_decay'] = 1 - out.loc[mask, 'vacc_decay']
        if 'hours_since_vaccination' in out.columns:
            out.loc[mask, 'hours_since_vaccination'] = rng.uniform(0, 200, mask.sum())

    # Delay antibiotic by 6-12 hours (72-144 ticks)
    if 'abx_decay' in out.columns:
        delay_hours = rng.uniform(6, 12)
        delay_ticks = int(delay_hours * 12)
        out['abx_decay'] = out['abx_decay'].shift(delay_ticks).fillna(0)
        if 'hours_since_antibiotic' in out.columns:
            out['hours_since_antibiotic'] = out['hours_since_antibiotic'].shift(delay_ticks).fillna(
                out['hours_since_antibiotic'].max())

    # Drop 10% feed change signals
    if 'feed_decay' in out.columns:
        mask = rng.random(len(out)) < 0.10
        out.loc[mask, 'feed_decay'] = 0
        if 'hours_since_feed_change' in out.columns:
            out.loc[mask, 'hours_since_feed_change'] = 999

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# E) CURRICULUM MIXING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_random_corruption(df, rng=None):
    """Apply a random combination of corruptions to a batch."""
    rng = rng or np.random.default_rng()
    out = df.copy()

    # Always apply sensor drift (with varying intensity)
    out = sensor_drift(out, rng)

    # Random missingness (70% chance)
    if rng.random() < 0.7:
        out = inject_missingness(out, 'mixed', rng)

    # Timestamp jitter (50% chance, varying magnitude)
    if rng.random() < 0.5:
        jitter_ticks = rng.choice([6, 24, 72])  # 30min, 2h, 6h
        out = timestamp_jitter(out, jitter_ticks, rng)

    # Management misreporting (40% chance, only if v4 features present)
    if 'vacc_decay' in df.columns and rng.random() < 0.4:
        out = mgmt_misreport(out, rng)

    return out


def augment_curriculum(X_clean, y_clean, noisy_ratio=0.4, seed=42):
    """
    Create 60/40 clean/noisy training mix.
    
    Returns:
        X_augmented (pd.DataFrame): fused clean + noisy features
        y_augmented (pd.Series): labels
        noise_mask (np.array): True for noisy samples
    """
    rng = np.random.default_rng(seed)
    n = len(X_clean)
    n_noisy = int(n * noisy_ratio)

    # Select random subset to corrupt
    noisy_indices = rng.choice(n, n_noisy, replace=False)
    noisy_mask = np.zeros(n, dtype=bool)
    noisy_mask[noisy_indices] = True

    X_noisy = apply_random_corruption(X_clean.iloc[noisy_indices].copy(), rng)

    # Fill NaN from corruption with column median (learned imputation)
    col_medians = X_clean.median(numeric_only=True)
    X_noisy = X_noisy.fillna(col_medians)

    # Combine: clean samples stay clean, noisy samples get corrupted
    X_out = X_clean.copy()
    X_out.iloc[noisy_indices] = X_noisy.values

    logger.info(f"Curriculum: {n - n_noisy} clean + {n_noisy} noisy = {n} total")
    return X_out, y_clean.copy(), noisy_mask


def get_noise_stats(X_clean, X_augmented, noisy_mask):
    """Compute per-feature noise profile statistics."""
    stats = {}
    clean_slice = X_clean[~noisy_mask]
    noisy_slice_original = X_clean[noisy_mask]
    noisy_slice_corrupted = X_augmented[noisy_mask]

    for col in X_clean.columns:
        if not np.issubdtype(X_clean[col].dtype, np.number):
            continue
        clean_mean = clean_slice[col].mean()
        clean_std = clean_slice[col].std()
        noisy_mean = noisy_slice_corrupted[col].mean()
        noisy_std = noisy_slice_corrupted[col].std()
        shift = abs(noisy_mean - clean_mean) / max(clean_std, 1e-6)

        stats[col] = {
            "clean_mean": round(float(clean_mean), 4),
            "clean_std": round(float(clean_std), 4),
            "noisy_mean": round(float(noisy_mean), 4),
            "noisy_std": round(float(noisy_std), 4),
            "distribution_shift": round(float(shift), 4)
        }

    return stats
