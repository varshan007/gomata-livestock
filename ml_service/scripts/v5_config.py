#!/usr/bin/env python3
"""
v5_config.py — Shared constants and utilities for v5 pipeline.
Does NOT configure logging — safe to import from any script.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

V3_FEATURES = [
    "temp_current", "hr_current", "resp_current",
    "activity_current", "rumination_current", "lying_current",
    "temp_1h_avg", "temp_6h_avg", "temp_12h_median", "temp_24h_std",
    "temp_6h_std", "temp_1h_std",
    "hr_1h_avg", "hr_6h_avg", "hr_12h_median", "hr_6h_std",
    "activity_1h_avg", "activity_6h_avg", "activity_6h_std",
    "resp_6h_avg", "resp_6h_std",
    "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_12h",
    "hr_lag_1h", "hr_lag_3h", "hr_lag_6h", "hr_lag_12h",
    "activity_lag_1h", "activity_lag_3h", "activity_lag_6h", "activity_lag_12h",
    "thi", "ambient_temp", "humidity",
]

V4_FEATURES = [
    "milk_deviation", "conductivity_deviation", "feed_deviation", "weight_deviation",
    "hours_since_vaccination", "hours_since_antibiotic",
    "hours_since_transport", "hours_since_feed_change",
    "vacc_decay", "abx_decay", "transport_decay", "feed_decay",
    "total_antibiotic_days", "vaccination_count_12m", "feed_changes_30d",
    "parity", "bcs", "age",
]

ALL_FEATURES = V3_FEATURES + V4_FEATURES


def load_and_fuse():
    """Load v5 features and fuse into training matrix."""
    v3_path = os.path.join(DATA_DIR, "features_v3_v5.csv")
    v4_path = os.path.join(DATA_DIR, "features_v4_v5.csv")

    v3 = pd.read_csv(v3_path)
    v4 = pd.read_csv(v4_path)

    min_len = min(len(v3), len(v4))
    v3, v4 = v3.iloc[:min_len].reset_index(drop=True), v4.iloc[:min_len].reset_index(drop=True)

    fused = pd.DataFrame()
    for f in ALL_FEATURES:
        if f in v3.columns:
            fused[f] = v3[f].values
        elif f in v4.columns:
            fused[f] = v4[f].values
        else:
            fused[f] = 0

    fused['animal_id'] = v3['animal_id']
    fused['disease_binary'] = v3['disease_binary']
    fused['severity_level'] = v3['severity_level']
    return fused
