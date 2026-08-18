#!/usr/bin/env python3
"""
stress_test_v11.py

Evaluates the Hybrid System under Phase 11 "Extreme Noise" Reality Gap parameters:
- AR(1) intensity doubled
- Missing 30% data (dropped blocks)
- 15% management misreporting (fuzzing timestamps or labels)
- 12h temporal delay (shift severity tags)
"""

import os, sys, time, json, logging
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StressTestV11")

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from hybrid_engine_v11 import HybridLivestockEngine
from train_sequence_model_v10 import CowSequenceDataset

DATA_DIR = os.path.join(BASE_DIR, "../training_data")
MODEL_DIR = os.path.join(BASE_DIR, "../models/cattle")

TICK_HOURS = 10 / 60
TICKS_PER_WEEK = 1008

def apply_reality_gap(df):
    """Corrupts the data according to pilot stress conditions."""
    np.random.seed(99)
    df_noisy = df.copy()
    
    # 1. 30% Missing Data (block drops)
    # 2. Doubled AR(1) intensity -> we just add gaussian noise to scaled features for now to mock sensor fail
    cols_to_add_noise = ['temperature', 'activity_steps', 'rumination_minutes', 'heart_rate', 'respiratory_rate']
    
    for aid, g in df_noisy.groupby("animal_id"):
        idxs = g.index.values
        
        # Add random sensor noise (Doubling variance effectively)
        for c in cols_to_add_noise:
            if c in df_noisy.columns:
                std = df_noisy[c].std()
                noise = np.random.normal(0, std * 0.5, size=len(idxs)) # Massive noise
                df_noisy.loc[idxs, c] += noise
                
        # Drop 30% blocks
        n_blocks = len(idxs) // 6 # hourly blocks
        drop_blocks = np.random.choice(n_blocks, size=int(n_blocks * 0.30), replace=False)
        for b in drop_blocks:
            drop_idxs = idxs[b*6:(b+1)*6]
            # Replace physical cols with NaNs or 0 (forward fill to mock bad data handling)
            for c in cols_to_add_noise:
                if c in df_noisy.columns:
                    df_noisy.loc[drop_idxs, c] = np.nan
    
    # Forward fill missing to mimic imputation
    for c in cols_to_add_noise:
        if c in df_noisy.columns:
            df_noisy[c] = df_noisy[c].fillna(method='ffill').fillna(0)
            
    # 3. 12h temporal delay -> shift severity arrays right by 72 ticks
    shifted_sev = []
    for aid, g in df_noisy.groupby("animal_id"):
        sev = g["severity"].values
        s_sev = np.roll(sev, 72)
        s_sev[:72] = 0
        shifted_sev.extend(s_sev)
    df_noisy["severity"] = shifted_sev
    
    # 4. 15% Misreporting -> Randomly drop severity signals completely for 15% of sick animals
    sick_aids = df_noisy[df_noisy['severity'] >= 1]['animal_id'].unique()
    num_fuzzed = int(len(sick_aids) * 0.15)
    fuzzed_aids = np.random.choice(sick_aids, size=num_fuzzed, replace=False)
    
    mask = df_noisy['animal_id'].isin(fuzzed_aids)
    df_noisy.loc[mask, 'severity'] = 0.0
    
    return df_noisy
    

def run_stress_test():
    start_time = time.time()
    logger.info("="*60)
    logger.info("🌪️ Phase 11 — Extreme Reality Gap Stress Test")
    logger.info("="*60)
    
    # 1. Load Data
    data_path = os.path.join(DATA_DIR, "v10_flat_sequences.csv")
    df = pd.read_csv(data_path)
    
    animals = df['animal_id'].unique()
    np.random.seed(42)
    np.random.shuffle(animals)
    
    split_idx = int(len(animals) * 0.8)
    test_animals = animals[split_idx:]
    test_df = df[df['animal_id'].isin(test_animals)].reset_index(drop=True)
    logger.info(f"Base cohort: {len(test_animals)} animals")
    
    # Apply Noise
    logger.info("Applying Reality Gap filter (Noise / Drops / Delays)...")
    test_df = apply_reality_gap(test_df)
    
    engine = HybridLivestockEngine(device='cpu')
    lstm_dataset = CowSequenceDataset(test_df)
    loader = torch.utils.data.DataLoader(lstm_dataset, batch_size=512, shuffle=False)
    
    lstm_probs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(engine.device)
            logits, _ = engine.lstm_model(x)
            prob = torch.sigmoid(logits[:, -1]).cpu().numpy()
            lstm_probs.extend(prob)
            
    xgb_cols = engine.xgb_model.feature_names_in_
    # Recreate rolling features for XGBoost from corrupted series (shortcut: load v9 and noise it)
    f9_df = pd.read_csv(os.path.join(DATA_DIR, "features_v9.csv"))
    f9_test = f9_df[f9_df['animal_id'].isin(test_animals)].copy()
    
    # Add random AR noise directly to XGB probabilities to simulate corrupted rolling windows
    np.random.seed(55)
    
    f9_test = f9_test.dropna(subset=xgb_cols)
    f9_probs = engine.xgb_model.predict_proba(f9_test[xgb_cols])[:, 1]
    f9_probs = np.clip(f9_probs + np.random.normal(0, 0.1, size=len(f9_probs)), 0, 1) # Corrupt
    xgb_map = dict(zip(zip(f9_test['animal_id'], f9_test['timestamp']), f9_probs))
    
    probs_dict = {}
    for i in range(len(lstm_probs)):
        seq = lstm_dataset.samples[i]
        end_idx = seq[-1]
        probs_dict[end_idx] = lstm_probs[i]
        
    # Read optimal thresholds
    conf_path = os.path.join(MODEL_DIR, "hybrid_alert_config_v11.json")
    if os.path.exists(conf_path):
        with open(conf_path, "r") as f:
            cfg = json.load(f)
        w_th = cfg["watchlist_th"]
        x_th = cfg["xgb_th"]
        f_th = cfg["fused_th"]
    else:
        # Fallback to manual
        w_th, x_th, f_th = 0.55, 0.70, 0.65
        
    logger.info(f"Using Thresholds: W={w_th} X={x_th} F={f_th}")
    
    test_df["lstm_prob"] = 0.0
    test_df["xgb_prob"] = 0.0
    test_df["fused_prob"] = 0.0
    
    fused_inputs = []
    fused_indices = []
    for idx in range(len(test_df)):
        if idx in probs_dict:
            row = test_df.iloc[idx]
            aid = row["animal_id"]
            ts = row["timestamp"]
            p_lstm = probs_dict[idx]
            p_xgb = xgb_map.get((aid, ts), 0.0)
            
            test_df.loc[idx, "lstm_prob"] = p_lstm
            test_df.loc[idx, "xgb_prob"] = p_xgb
            fused_inputs.append([float(p_lstm), float(p_xgb)])
            fused_indices.append(idx)
            
    if engine.calibrator and len(fused_inputs) > 0:
        fprobs = engine.calibrator.predict_proba(np.array(fused_inputs))[:, 1]
        for i, idx in enumerate(fused_indices):
            test_df.loc[idx, "fused_prob"] = fprobs[i]
            
    logger.info("Evaluating robust logic gates...")
    alerts = np.zeros(len(test_df), dtype=int)
    for aid, g in test_df.groupby("animal_id"):
        idxs = g.index.values
        lstm_p = g["lstm_prob"].values
        xgb_p = g["xgb_prob"].values
        fused_p = g["fused_prob"].values
        
        lstm_hist = []
        xgb_hist = []
        watchlist = False
        ticks_below = 0
        
        for i in range(len(idxs)):
            idx = idxs[i]
            lp = lstm_p[i]
            xp = xgb_p[i]
            fp = fused_p[i]
            
            lstm_hist.append(lp)
            xgb_hist.append(xp)
            if len(lstm_hist) > 12:
                lstm_hist.pop(0)
                xgb_hist.pop(0)
                
            if lp < 0.30:
                ticks_below += 1
                if ticks_below >= 12: watchlist = False
            else:
                ticks_below = 0
                
            if len(lstm_hist) >= 3:
                if sum(1 for x in lstm_hist[-3:] if x > w_th) >= 3: watchlist = True
                    
            lstm_rising = False
            if len(lstm_hist) >= 3:
                lstm_rising = lstm_hist[-1] > lstm_hist[-3] + 0.05
                
            if xp > x_th or (lstm_rising and xp > 0.40) or fp > f_th:
                alerts[idx] = 1

    det_arr = alerts
    sev_arr = test_df["severity"].values
    aids = test_df["animal_id"].values
    
    d24 = 0; eps = 0; fp = 0; cw = 0
    tno = 0; twi = 0
    decay = 0.5 ** (1 / (6 / TICK_HOURS))

    for aid in np.unique(aids):
        m = aids == aid
        sa = sev_arr[m]
        da = det_arr[m]
        n = m.sum()
        
        ie = False
        for i in range(len(sa)):
            if sa[i] >= 2 and not ie:
                ie = True; eps += 1
                lb = min(i, int(576/2))
                w = da[max(0, i - lb):i]
                if w.any():
                    fi = np.where(w)[0][0]
                    lh = (len(w) - fi) * TICK_HOURS
                    if lh >= 24: d24 += 1
            elif sa[i] < 2:
                ie = False
                
        fp += int((da & (sa < 0.5)).sum())
        cw += n / TICKS_PER_WEEK
        
        sf = sa.astype(float)
        ie2 = False
        es = None
        for i in range(len(sf)):
            if sf[i] > 0.5 and not ie2:
                es = i; ie2 = True
            elif sf[i] <= 0.1 and ie2:
                if i - es > int(12/2):
                    ep = sf[es:i]
                    nl = float((ep * 0.5 * TICK_HOURS).sum())
                    tno += nl
                    ea = da[es:i]
                    if ea.any():
                        fi2 = np.where(ea)[0][0]
                        si = ep.copy()
                        for k in range(fi2, len(si)):
                            si[k] *= decay**(k - fi2)
                        twi += float((si * 0.5 * TICK_HOURS).sum())
                    else:
                        twi += nl
                ie2 = False
                
    fpw = fp / max(cw, 1)
    p24 = d24 / max(eps, 1) * 100
    sv = tno - twi
    pct = sv / max(tno, 0.001) * 100
    
    logger.info(f"🚨 EXTREME NOISE STRESS TEST RESULTS 🚨")
    logger.info(f"Early >=24h Detection: {p24:.1f}%")
    logger.info(f"False Positives/Week : {fpw:.1f}")
    logger.info(f"Economic Reduction   : {pct:.1f}%")
    
    pilot = {
        "version": "v11_hybrid_stress",
        "fp_week": round(fpw, 1),
        "pct_24h": round(p24, 1),
        "econ": round(pct, 1)
    }
    with open(os.path.join(DATA_DIR, "pilot_readiness_v11_stress.json"), "w") as f:
        json.dump(pilot, f, indent=2)
        
    logger.info(f"Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    run_stress_test()
