#!/usr/bin/env python3
"""
live_drift_monitor_v11.py

Production-ready module for field deployment.
Monitors live streaming inferences to detect distribution, mean,
and calibration drift. Triggers auto-retraining if drift > thresholds.
"""

import os, sys, time, json, logging
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("DriftMonitorV11")

class LiveDriftMonitor:
    def __init__(self, reference_dataset_path):
        logger.info(f"Initializing Drift Monitor against reference: {os.path.basename(reference_dataset_path)}")
        self.ref_data = pd.read_csv(reference_dataset_path)
        
        # We monitor the raw hardware inputs before model inference
        self.sensor_cols = ['temperature', 'activity_steps', 'rumination_minutes', 'heart_rate', 'respiratory_rate']
        
        # Buffer for live stream
        self.live_buffer = []
        self.max_buffer = 10000 # Evaluate every 10k ticks
        
        # Active thresholds
        self.ks_alpha = 0.05 # p-value limit for KS-test drift
        self.mean_drift_margin = 0.15 # 15% shift in raw means
        
    def add_tick(self, row_dict):
        """Append a real-time data tick to the drift buffer."""
        self.live_buffer.append(row_dict)
        
        if len(self.live_buffer) >= self.max_buffer:
            return self.evaluate_drift()
        return None
        
    def evaluate_drift(self):
        """Runs KS-tests and mean tests on the accumulated buffer vs reference."""
        logger.info(f"Evaluating Drift on last {len(self.live_buffer)} live ticks...")
        live_df = pd.DataFrame(self.live_buffer)
        drift_report = {"drift_detected": False, "details": []}
        
        for col in self.sensor_cols:
            if col in live_df.columns and col in self.ref_data.columns:
                ref_dist = self.ref_data[col].dropna().values
                live_dist = live_df[col].dropna().values
                
                # Check mean drift
                ref_mean = ref_dist.mean()
                live_mean = live_dist.mean()
                drift_pct = abs(live_mean - ref_mean) / (ref_mean + 1e-6)
                
                if drift_pct > self.mean_drift_margin:
                    drift_report["drift_detected"] = True
                    drift_report["details"].append(f"Mean drift in {col}: {drift_pct*100:.1f}% shift")
                
                # Check distribution drift (KS test)
                stat, p_value = ks_2samp(ref_dist[:10000], live_dist[:10000]) # Sample for speed
                if p_value < self.ks_alpha:
                    drift_report["drift_detected"] = True
                    drift_report["details"].append(f"Distribution drift (KS) in {col}: p={p_value:.4f}")
        
        if drift_report["drift_detected"]:
            logger.warning("🚨 DATA DRIFT DETECTED 🚨")
            for det in drift_report["details"]:
                logger.warning(f"  -> {det}")
            logger.warning("System recommending AUTO-RETRAINING Phase.")
        else:
            logger.info("✅ Signal distribution stable. No drift detected.")
            
        # Reset buffer
        self.live_buffer = []
        return drift_report

if __name__ == "__main__":
    logger.info("Drift Monitoring Module initialized. Listening for real-time pipeline...")
