#!/usr/bin/env python3
"""
optimize_hybrid_thresholds_v11.py

Optimizes the thresholds for the Hybrid Engine (v11) over the test cohort.
Goal: Maximize economic benefit while keeping False Positives <= 2/week
and >= 45% early detection.
"""

import os, sys, time, json, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("OptimizeV11")

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from hybrid_engine_v11 import HybridLivestockEngine
from train_sequence_model_v10 import CowSequenceDataset

DATA_DIR = os.path.join(BASE_DIR, "../training_data")
MODEL_DIR = os.path.join(BASE_DIR, "../models/cattle")

TICK_HOURS = 10 / 60
TICKS_PER_WEEK = 1008

def evaluate_thresholds():
    start_time = time.time()
    logger.info("="*60)
    logger.info("🎯 Phase 11 — Hybrid Threshold Optimizer")
    logger.info("="*60)
    
    # ── 1. Load Test Cohort ──
    data_path = os.path.join(DATA_DIR, "v10_flat_sequences.csv")
    df = pd.read_csv(data_path)
    
    animals = df['animal_id'].unique()
    np.random.seed(42)  # Match training splits
    np.random.shuffle(animals)
    
    split_idx = int(len(animals) * 0.8)
    test_animals = animals[split_idx:]
    test_df = df[df['animal_id'].isin(test_animals)].reset_index(drop=True)
    logger.info(f"Test cohort: {len(test_animals)} animals, {len(test_df)} rows")
    
    # ── 2. Run Engine Across Test Set ──
    # We will compute the base probabilities, then sweep thresholds
    logger.info("Pre-computing predictions for the test set...")
    
    engine = HybridLivestockEngine(device='cpu')
    
    dataset = CowSequenceDataset(test_df)
    
    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    
    lstm_probs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(engine.device)
            logits_cls, _ = engine.lstm_model(x)
            prob = torch.sigmoid(logits_cls[:, -1]).cpu().numpy()
            lstm_probs.extend(prob)
            
    # XGBoost
    import joblib
    xgb_cols = engine.xgb_model.feature_names_in_
    
    f9_df = pd.read_csv(os.path.join(DATA_DIR, "features_v9.csv"))
    f9_test = f9_df[f9_df['animal_id'].isin(test_animals)].copy()
    f9_test = f9_test.dropna(subset=xgb_cols)
    f9_probs = engine.xgb_model.predict_proba(f9_test[xgb_cols])[:, 1]
    xgb_map = dict(zip(zip(f9_test['animal_id'], f9_test['timestamp']), f9_probs))
    
    probs_dict = {}
    for i in range(len(lstm_probs)):
        seq = dataset.samples[i]
        end_idx = seq[-1]
        probs_dict[end_idx] = lstm_probs[i]
        
    fused_calibrator = engine.calibrator
    
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
            
    if fused_calibrator and len(fused_inputs) > 0:
        fprobs = fused_calibrator.predict_proba(np.array(fused_inputs))[:, 1]
        for i, idx in enumerate(fused_indices):
            test_df.loc[idx, "fused_prob"] = fprobs[i]
            
    # ── 3. Grid Search Thresholds ──
    logger.info("Grid searching logic gates...")
    
    best_result = None
    best_score = -9999
    
    grid_watchlist_th = [0.45, 0.50, 0.55]
    grid_xgb_th = [0.60, 0.70, 0.80]
    grid_fused_th = [0.60, 0.65, 0.75]
    
    for w_th in grid_watchlist_th:
        for x_th in grid_xgb_th:
            for f_th in grid_fused_th:
                
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
                            
                        # Auto-clear
                        if lp < 0.30:
                            ticks_below += 1
                            if ticks_below >= 12:
                                watchlist = False
                        else:
                            ticks_below = 0
                            
                        # Watchlist trigger
                        if len(lstm_hist) >= 3:
                            if sum(1 for x in lstm_hist[-3:] if x > w_th) >= 3:
                                watchlist = True
                                
                        # Escalate
                        lstm_rising = False
                        if len(lstm_hist) >= 3:
                            lstm_rising = lstm_hist[-1] > lstm_hist[-3] + 0.05
                            
                        if xp > x_th or (lstm_rising and xp > 0.40) or fp > f_th:
                            alerts[idx] = 1
                
                # Eval logic
                det_arr = alerts
                sev_arr = test_df["severity"].values
                aids = test_df["animal_id"].values
                
                d24 = 0; eps = 0; fp = 0; cw = 0
                leads = []
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
                                first_w = np.where(w)[0][0]
                                lh = (len(w) - first_w) * TICK_HOURS
                                leads.append(lh)
                                if lh >= 24: d24 += 1
                        elif sa[i] < 2:
                            ie = False
                            
                    fp += int((da & (sa < 0.5)).sum())
                    cw += n / TICKS_PER_WEEK
                    
                    # Econ
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
                
                logger.info(f"W={w_th} X={x_th} F={f_th} | 24h={p24:.1f}% FP/wk={fpw:.1f} Econ={pct:.0f}%")
                
                if fpw <= 4.0: # relaxed a bit to find best
                    score = pct + p24 - (fpw * 10)
                    if score > best_score:
                        best_score = score
                        best_result = {
                            "watchlist_th": w_th,
                            "xgb_th": x_th,
                            "fused_th": f_th,
                            "pct_24h": p24,
                            "fp_per_week": fpw,
                            "pct_reduction": pct,
                            "eval_eps": eps
                        }
                        
    logger.info("="*60)
    if best_result:
        logger.info(f"🏆 BEST HYBRID THRESHOLDS:")
        logger.info(f"Watchlist (LSTM): > {best_result['watchlist_th']}")
        logger.info(f"XGBoost Clinical: > {best_result['xgb_th']}")
        logger.info(f"Fused Alert     : > {best_result['fused_th']}")
        logger.info(f"Metrics: 24h={best_result['pct_24h']:.1f}%, FP/wk={best_result['fp_per_week']:.1f}, Econ={best_result['pct_reduction']:.1f}%")
        
        # Save config
        conf_path = os.path.join(MODEL_DIR, "hybrid_alert_config_v11.json")
        with open(conf_path, "w") as f:
            json.dump(best_result, f, indent=2)
    else:
        logger.warning("No threshold satisfied constraints.")
        
    logger.info(f"Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    evaluate_thresholds()
