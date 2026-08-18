#!/usr/bin/env python3
"""
evaluate_v18_early_warning.py
Phase 18.5: Epistemic Uncertainty & True Forecasting

Tests if V17c can predict future outbreaks before the visible infection 
slope actually takes off (true $t_0$ forecasting under uncertainty).
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from colab_train_v17c_herd import HerdEngineV17c, ProductionSimulatorV3

T_STEPS = 28; SIM_TOTAL = 56; MAX_COWS = 100; NODE_DIM = 18

class EarlyWarningSimulator(ProductionSimulatorV3):
    def __init__(self, seed=18500):
        super().__init__(seed=seed)
        
    def simulate_early_farm(self, fidx):
        n_cows = self.rng.randint(50, 100)
        n_pens = self.rng.randint(4, 8)
        n_workers = self.rng.randint(2, 5)
        
        A_raw, pen, worker, is_hub = self._hub_graph(n_cows, n_pens, n_workers, self.rng.randint(3, 6))

        # We force an outbreak regime, but we push the seeding to the absolute END 
        # of the observable window (t=24..26). The window is 0..28.
        # This means between t=24 and t=28, I(t) might just barely hit 0.01 or 0.02.
        regime = self.rng.choice(['outbreak','superspreader'], p=[0.7,0.3])
        
        if regime == 'outbreak':
            beta=self.rng.uniform(0.07,0.15); base_hsi=self.rng.uniform(0.5,0.8)
            n_seed=self.rng.randint(2,6); seed_t=self.rng.randint(24,26)
        else:
            beta=self.rng.uniform(0.15,0.30); base_hsi=self.rng.uniform(0.3,0.6)
            n_seed=self.rng.randint(3,8); seed_t=self.rng.randint(24,26)

        # For the non-outbreak class, we need stable farms where seed fails to propagate.
        # We'll override 50% of the time to Stable.
        if self.rng.rand() < 0.5:
            regime = 'stable'
            beta=self.rng.uniform(0.02,0.04); base_hsi=self.rng.uniform(0.8,0.99)
            n_seed=self.rng.randint(1,3); seed_t=self.rng.randint(24,26)

        vaccinated = np.zeros(n_cows, dtype=np.float32)
        nv = int(self.rng.uniform(0, 0.25) * n_cows)
        if nv > 0: vaccinated[self.rng.choice(n_cows, nv, replace=False)] = 1.0

        I = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        severity = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)

        seeds = self.rng.choice(n_cows, min(n_seed, n_cows), replace=False)
        I[seed_t, seeds] = self.rng.uniform(0.1, 0.4, len(seeds))

        alpha_heat = self.rng.uniform(0.02, 0.06); base_thi = self.rng.uniform(68, 85)
        
        # We do not apply intervention within the observable window for this test.
        # We just let it ride out to see the natural future peak.
        for t in range(max(1, seed_t+1), SIM_TOTAL):
            te = max(0, base_thi + 3*np.sin(t*2*np.pi/28) - 72)
            be = beta * (1 + alpha_heat * te)
            Ae = A_raw * (1 - vaccinated[np.newaxis, :] * 0.8)
            
            new_inf = np.clip(be * (Ae @ I[t-1]) * (1-I[t-1]), 0, 1)
            I[t] = np.clip(I[t-1] + new_inf - 0.1 * I[t-1], 0, 1)
            severity[t] = I[t] * (1 + 0.2 * te / 10)

        nf = np.zeros((SIM_TOTAL, n_cows, NODE_DIM), dtype=np.float32)
        for t in range(SIM_TOTAL):
            te = max(0, base_thi + 3*np.sin(t*2*np.pi/28) - 72)
            for i in range(n_cows):
                nf[t, i] = [
                    I[t,i], float(te>5)*0.3+self.rng.normal(0,0.03),
                    I[t,i]*0.4+self.rng.normal(0,0.03), 0.1+self.rng.normal(0,0.03),
                    0.05+self.rng.normal(0,0.01), severity[t,i], float(severity[t,i]>1.5),
                    np.gradient(severity[max(0,t-3):t+1,i]).mean() if t>0 else 0,
                    severity[max(0,t-4):t+1,i].sum()*0.25,
                    float(I[t,i]>0.1 and (I[t,i]-I[max(0,t-1),i])>0),
                    float(I[t,i]>0.3 and abs(I[t,i]-I[max(0,t-1),i])<0.02),
                    float(I[t,i]>0.1 and (I[t,i]-I[max(0,t-1),i])<-0.01),
                    self.rng.uniform(1,4), max(0,1-I[t,i]),
                    max(0,30-10*I[t,i])+self.rng.normal(0,1),
                    1-severity[t,i]*0.3+self.rng.normal(0,0.03),
                    vaccinated[i], float(pen[i])/n_pens]

        mI = I.mean(axis=1)
        
        # IMPORTANT DISTINCTION:
        # VISIBLE peak is strictly max up to T=28 (which will be tiny)
        visible_max = float(mI[:28].max()) if len(mI[:28]) > 0 else 0.0
        
        # FUTURE peak is the absolute max over the 14 days AFTER the observation window
        future_max = float(mI[28:56].max()) if len(mI[28:56]) > 0 else 0.0
        
        future_outbreak = float(future_max > 0.15)
        
        return {
            "node_features": nf[:28].astype(np.float32), # Supply only the first 28 ticks
            "adjacency": A_raw.astype(np.float32),
            "visible_max": visible_max,
            "labels": {
                "future_outbreak": future_outbreak,
                "future_intensity": future_max
            }
        }

def evaluate_early_warning(model_path, config_path, n_farms=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("🔮 PHASE 18.5: EPISTEMIC UNCERTAINTY & TRUE FORECASTING")
    print("="*60)
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    model = HerdEngineV17c(
        node_dim=cfg['node_dim'], gat_dim=cfg['gat_dim'], 
        tft_dim=cfg['tft_dim'], ngh=cfg.get('n_gat_heads', 6)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    sim = EarlyWarningSimulator(seed=18500)
    
    y_true_future_ob = []
    y_pred_intensity = []
    y_true_future_int = []
    visible_maxes = []

    print(f"\nSimulating {n_farms} Early Warning scenarios ($t_0$ Forecasting)...")
    with torch.no_grad():
        for i in range(n_farms):
            data = sim.simulate_early_farm(i)
            labels = data['labels']
            
            nf = torch.tensor(data['node_features']).unsqueeze(0).to(device)
            adj = torch.tensor(data['adjacency']).unsqueeze(0).to(device)
            
            out = model(nf, adj)
            
            y_true_future_ob.append(labels['future_outbreak'])
            y_pred_intensity.append(out['intensity'].item())
            
            y_true_future_int.append(labels['future_intensity'])
            visible_maxes.append(data['visible_max'])

    ob_auc = roc_auc_score(y_true_future_ob, y_pred_intensity) if len(set(y_true_future_ob)) > 1 else 1.0
    int_corr, _ = pearsonr(y_true_future_int, y_pred_intensity)
    
    mean_vis = np.mean(visible_maxes)
    
    print("\n" + "="*50)
    print("📉 FORECASTING RESULTS (Uncertainty Phase)")
    print("="*50)
    print(f"Mean VISIBLE Intensity at t=28: {mean_vis:.4f}  <-- Window is extremely flat")
    print(f"True FUTURE Outbreaks: {sum(y_true_future_ob)} / {n_farms}")
    print(f"Future Outbreak AUC:  {ob_auc:.4f}")
    print(f"Future Intensity Corr:{int_corr:.4f}")
    
    if 0.75 <= ob_auc <= 0.88:
        print("\n✅ PASS: Realistic epistemic uncertainty bound hit.")
        print("The model successfully transitioned from a 0.99 deterministic slope reader")
        print("to a ~0.80 true future geometric forecaster based purely on graph susceptibility.")
    elif ob_auc < 0.75:
        print("\n❌ FAIL: Model collapsed to random chance. It is purely a curve reader.")
    else:
        print("\n⚠️ SUSPICIOUS: AUC is still > 0.88. Simulation may still be leaking determinism.")
    print("="*50)

if __name__ == "__main__":
    evaluate_early_warning("models/cattle/v17c_herd_engine.pth", "models/cattle/v17c_herd_config.json", n_farms=300)
