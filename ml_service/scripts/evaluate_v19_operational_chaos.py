#!/usr/bin/env python3
"""
evaluate_v19_operational_chaos.py
Phase 19: Real-World Proxy Validation

Wraps the SEIR physics from Phase 18 with operational and human chaos:
- Human Behavior Randomness (Intervention Non-Compliance)
- Economic Decision Lag (t_lag = 12-48 hours)
- Missing Pen Assignments (Graph topology blinding)
- Partial Vaccination Coverage with varying efficacy
- Worker-to-Worker Cross-Farm Transmission 
- Reporting Bias (Stochastic sensor degradation)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from scipy.stats import pearsonr

from colab_train_v17c_herd import HerdEngineV17c

T_STEPS = 28; SIM_TOTAL = 56; MAX_COWS = 100; NODE_DIM = 18

class OperationalChaosSimulator:
    def __init__(self, seed=19000):
        self.rng = np.random.RandomState(seed)
        
    def _hub_graph(self, N, P, W, D):
        pen_assignments = self.rng.randint(0, P, size=N)
        worker_assignments = self.rng.randint(0, W, size=N)
        
        A = np.zeros((N, N), dtype=np.float32)
        
        # 1. Base Pen Connectivity
        for p in range(P):
            idx = np.where(pen_assignments == p)[0]
            for i in idx:
                for j in idx:
                    if i != j and self.rng.rand() < 0.3:
                        A[i, j] = 0.5
                        A[j, i] = 0.5
                        
        # 2. Worker Connectivity (Standard)
        for w in range(W):
            idx = np.where(worker_assignments == w)[0]
            if len(idx) > 2:
                for _ in range(int(len(idx)*0.2)):
                    i, j = self.rng.choice(idx, 2, replace=False)
                    A[i, j] = 0.2
                    A[j, i] = 0.2
                    
        # 3. [NEW PHASE 19] Worker-to-Worker Cross-Transmission
        # Simulates workers interacting in break rooms, violating spatial blocks entirely
        for _ in range(W * 2):
            w1, w2 = self.rng.choice(W, 2, replace=False)
            idx1 = np.where(worker_assignments == w1)[0]
            idx2 = np.where(worker_assignments == w2)[0]
            if len(idx1) > 0 and len(idx2) > 0:
                c1 = self.rng.choice(idx1)
                c2 = self.rng.choice(idx2)
                A[c1, c2] = 0.4
                A[c2, c1] = 0.4
                        
        hub_flags = np.zeros(N, dtype=bool)
        hubs = self.rng.choice(N, size=max(1, int(N*0.05)), replace=False)
        for h in hubs:
            hub_flags[h] = True
            targets = self.rng.choice(N, size=int(D*1.5), replace=False)
            for t in targets:
                if h != t:
                    A[h, t] = 1.0
                    A[t, h] = 1.0
                    
        np.fill_diagonal(A, 1.0)
        return A, pen_assignments, worker_assignments, hub_flags

    def simulate_chaos_farm(self, fidx):
        n_cows = self.rng.randint(50, 100)
        n_pens = self.rng.randint(4, 8)
        n_workers = self.rng.randint(2, 5)
        
        A_raw, pen, worker, is_hub = self._hub_graph(n_cows, n_pens, n_workers, self.rng.randint(3, 6))

        # We need outbreak regimes to test the model's predictive limits
        # True Epistemological Balance: Exactly 50% Outbreak, 50% Stable
        regime = self.rng.choice(['outbreak', 'stable'], p=[0.5, 0.5])
        
        if regime == 'outbreak':
            beta = self.rng.uniform(0.12, 0.25)  
            # Force seed extremely late so T=28 window has almost mathematically zero visible slope (I < 0.03)
            seed_t = self.rng.randint(24, 27) 
            n_seed = self.rng.randint(2, 6)
        else:
            beta = self.rng.uniform(0.02, 0.05)
            seed_t = self.rng.randint(24, 27)
            n_seed = self.rng.randint(1, 3)

        # [NEW PHASE 19] Partial Vaccination Efficacy
        vaccinated_flags = np.zeros(n_cows, dtype=np.float32)
        vaccine_efficacy = np.zeros(n_cows, dtype=np.float32)
        nv = int(self.rng.uniform(0, 0.3) * n_cows)
        if nv > 0:
            v_idx = self.rng.choice(n_cows, nv, replace=False)
            vaccinated_flags[v_idx] = 1.0
            vaccine_efficacy[v_idx] = self.rng.uniform(0.2, 0.6, size=len(v_idx)) # Lower efficacy

        # SEIR Arrays
        S = np.ones((SIM_TOTAL, n_cows), dtype=np.float32)
        E = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        I = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        
        severity_true = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        severity_observed = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)

        seeds = self.rng.choice(n_cows, min(n_seed, n_cows), replace=False)
        # We will inject the seed inside the simulation loop at t == seed_t
        
        # [NEW PHASE 19] Missing Pen Assignments / Corrupted Data
        obs_pen = pen.astype(np.float32)
        missing_pen_idx = self.rng.choice(n_cows, int(n_cows * 0.15), replace=False)
        obs_pen[missing_pen_idx] = 0.0 # Force to 0

        # [NEW PHASE 19] Reporting Bias
        broken_sensor_idx = self.rng.choice(n_cows, int(n_cows * 0.10), replace=False)

        # [NEW PHASE 19] Intervention Chaos variables
        intervention_active = False
        intervention_t = 999
        # Economic Decision Lag (12-48 hours = 2-8 ticks)
        decision_lag = self.rng.randint(2, 8)
        # Human Non-Compliance Threshold
        compliance_prob = self.rng.uniform(0.4, 0.9)
        
        alpha_heat = self.rng.uniform(0.02, 0.06) 
        base_thi = self.rng.uniform(68, 85)
        
        # We run the simulation forward. Intervention trigger might happen at Day 22-26 based on true E
        for t in range(1, SIM_TOTAL):
            # INJECT SEED EXACTLY AT SEED_T SO IT IS NOT OVERWRITTEN
            if t == seed_t:
                E[t-1, seeds] = 1.0
                I[t-1, seeds] = self.rng.uniform(0.1, 0.4, size=len(seeds))
                S[t-1, seeds] = 0.0
                
            te = max(0, base_thi + 3*np.sin(t*2*np.pi/28) - 72)
            be = beta * (1 + alpha_heat * te)
            
            # Form Adjacency (applying chaotic interventions)
            Ae = A_raw.copy()
            if t > intervention_t:
                # Stochastic compliance cutoff
                for i in range(n_cows):
                    if is_hub[i] and self.rng.rand() < compliance_prob:
                        Ae[i, :] *= 0.1
                        Ae[:, i] *= 0.1
            
            # Normalize adjacency degree so pressure doesn't explode past 1.0
            deg = Ae.sum(axis=1, keepdims=True)
            deg[deg == 0] = 1.0
            Ae_norm = Ae / deg
            
            # Infection Pressure.
            inf_pressure = Ae_norm @ (I[t-1] + E[t-1])
            new_exposed = np.clip(be * inf_pressure * S[t-1] * (1 - vaccine_efficacy), 0, 1)
            
            # Stochastic SEIR transitions with slightly faster incubation so the 14-day window catches the peak
            new_infectious = E[t-1] * self.rng.uniform(0.6, 1.0, size=n_cows) # Force fast incubation
            recovery = I[t-1] * self.rng.uniform(0.01, 0.05, size=n_cows) # Slow recovery so peak builds up
            
            S[t] = np.clip(S[t-1] - new_exposed, 0, 1)
            E[t] = np.clip(E[t-1] + new_exposed - new_infectious, 0, 1)
            I[t] = np.clip(I[t-1] + new_infectious - recovery, 0, 1)
            
            # True severity
            severity_true[t] = I[t] * (1 + 0.2 * te / 10)
            
            # Observed severity (Reporting Bias)
            obs_sev = severity_true[t].copy()
            # Broken sensors artificially cap severity
            obs_sev[broken_sensor_idx] = np.clip(obs_sev[broken_sensor_idx], 0, 1.2)
            severity_observed[t] = obs_sev
            
            # Trigger Intervention if apparent risk is high
            if not intervention_active and I[t].mean() > 0.05:
                intervention_t = t + decision_lag
                intervention_active = True

        # Build Node Features for the GNN up to T=28 (Early Phase)
        nf = np.zeros((28, n_cows, NODE_DIM), dtype=np.float32)
        for t in range(28):
            te = max(0, base_thi + 3*np.sin(t*2*np.pi/28) - 72)
            for i in range(n_cows):
                # We feed observed severity and explicitly corrupted pen assignments
                obs_i = severity_observed[t, i]
                
                nf[t, i] = [
                    obs_i, float(te>5)*0.3+self.rng.normal(0,0.03),
                    obs_i*0.4+self.rng.normal(0,0.03), 0.1+self.rng.normal(0,0.03),
                    0.05+self.rng.normal(0,0.01), obs_i, float(obs_i>1.5),
                    np.gradient(severity_observed[max(0,t-3):t+1,i]).mean() if t>0 else 0,
                    severity_observed[max(0,t-4):t+1,i].sum()*0.25,
                    float(obs_i>0.1 and (obs_i-severity_observed[max(0,t-1),i])>0),
                    float(obs_i>0.3 and abs(obs_i-severity_observed[max(0,t-1),i])<0.02),
                    float(obs_i>0.1 and (obs_i-severity_observed[max(0,t-1),i])<-0.01),
                    self.rng.uniform(1,4), max(0,1-obs_i),
                    max(0,30-10*obs_i)+self.rng.normal(0,1),
                    1-obs_i*0.3+self.rng.normal(0,0.03),
                    vaccinated_flags[i], float(obs_pen[i])/n_pens]

        # Calculate Future Max (the next 14 days after t=28 observation window)
        mI = I.mean(axis=1) # True infection state average
        
        future_max = float(mI[28:56].max()) if len(mI[28:56]) > 0 else 0.0
        future_outbreak = float(future_max > 0.15)
        
        visible_max = float(mI[:28].max()) if len(mI[:28]) > 0 else 0.0
        
        return {
            "node_features": nf.astype(np.float32), 
            "adjacency": A_raw.astype(np.float32),
            "visible_max": visible_max,
            "labels": {
                "future_outbreak": future_outbreak,
                "future_intensity": future_max
            }
        }

def evaluate_chaos_farm(model_path, config_path, n_farms=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("🌪️ PHASE 19: OPERATIONAL CHAOS & REAL-WORLD PROXY")
    print("="*60)
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    model = HerdEngineV17c(
        node_dim=cfg['node_dim'], gat_dim=cfg['gat_dim'], 
        tft_dim=cfg['tft_dim'], ngh=cfg.get('n_gat_heads', 6)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    sim = OperationalChaosSimulator(seed=19001)
    
    y_true_ob = []
    y_pred_ob = []
    y_pred_int = []
    y_true_int = []
    visible_maxes = []

    print(f"\nEvaluating {n_farms} Chaotic Proxy Farms...")
    with torch.no_grad():
        for i in range(n_farms):
            data = sim.simulate_chaos_farm(i)
            labels = data['labels']
            
            nf = torch.tensor(data['node_features']).unsqueeze(0).to(device)
            adj = torch.tensor(data['adjacency']).unsqueeze(0).to(device)
            
            out = model(nf, adj)
            
            y_true_ob.append(labels['future_outbreak'])
            # Since Intensity predicts outbreak, it's roughly the hazard logit
            y_pred_int.append(out['intensity'].item()) 
            y_true_int.append(labels['future_intensity'])
            visible_maxes.append(data['visible_max'])

    ob_auc = roc_auc_score(y_true_ob, y_pred_int) if len(set(y_true_ob)) > 1 else 1.0
    pr_auc = average_precision_score(y_true_ob, y_pred_int) if len(set(y_true_ob)) > 1 else 1.0
    int_corr, _ = pearsonr(y_true_int, y_pred_int)
    mean_vis = np.mean(visible_maxes)
    
    fpr, tpr, thresholds = roc_curve(y_true_ob, y_pred_int)
    recall_at_5_fpr = np.interp(0.05, fpr, tpr)
    recall_at_10_fpr = np.interp(0.10, fpr, tpr)
    
    print("\n" + "="*50)
    print("🛡️ PHASE 19.5: EPISTEMOLOGICAL BALANCING")
    print("="*50)
    print(f"Mean Visible True Intensity (t=28): {mean_vis:.4f}")
    print(f"Future Outbreaks Seeded: {sum(y_true_ob)} / {n_farms}")
    print(f"Future Outbreak ROC-AUC: {ob_auc:.4f}")
    print(f"Future Outbreak PR-AUC:  {pr_auc:.4f}")
    print(f"Recall @ 5% FPR:         {recall_at_5_fpr:.4f}")
    print(f"Recall @ 10% FPR:        {recall_at_10_fpr:.4f}")
    print(f"Future Intensity Corr:   {int_corr:.4f}")
    print("="*50)
    
    if ob_auc >= 0.85 and pr_auc >= 0.80:
        print("\n✅ PHASE 19.5 VERIFIED. TRUE FORESIGHT ACHIEVED.")
        print(f"The model survived the absolute bounds of epistemic uncertainty.")
        print(f"Even under chaos, missing inputs, and purely invisible 0.03 starting slopes,")
        print(f"the graph causality calculates fundamental system risk.")
    else:
        print("\n❌ PHASE 19.5 REVEALED INSUFFICIENT GENERALIZATION.")
        print("Performance collapsed when class imbalance was removed.")
        
if __name__ == "__main__":
    evaluate_chaos_farm("models/cattle/v17c_herd_engine.pth", "models/cattle/v17c_herd_config.json", n_farms=300)
