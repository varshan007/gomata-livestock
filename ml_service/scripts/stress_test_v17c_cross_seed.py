#!/usr/bin/env python3
"""
cross_seed_validation_v17c.py
Reality Check: Cross-Seed Validation for V17c Herd Epidemiology Engine
"""

import os
import json
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

# Import simulator and model class
from colab_train_v17c_herd import ProductionSimulatorV3, HerdEngineV17c

def eval_seed(model, config, seed, n_farms=100, device='cpu'):
    print(f"\n--- Simulating {n_farms} farms on SEED {seed} ---")
    sim = ProductionSimulatorV3(seed=seed)
    
    y_true_ob = []
    y_pred_ob = []
    y_true_bd = []
    y_pred_bd = []
    
    with torch.no_grad():
        for i in range(n_farms):
            data = sim.simulate_farm(i)
            labels = data['labels']
            
            nf = torch.tensor(data['node_features']).unsqueeze(0).to(device)
            adj = torch.tensor(data['adjacency']).unsqueeze(0).to(device)
            
            out = model(nf, adj)
            
            # Outbreak predicting
            y_true_ob.append(labels['outbreak'])
            y_pred_ob.append(out['intensity'].item())
            
            # Breakdown predicting
            y_true_bd.append(labels['breakdown'])
            y_pred_bd.append(torch.sigmoid(out['breakdown']).item())
            
    ob_auc = roc_auc_score(y_true_ob, y_pred_ob) if len(set(y_true_ob)) > 1 else 1.0
    bd_auc = roc_auc_score(y_true_bd, y_pred_bd) if len(set(y_true_bd)) > 1 else 1.0
    
    print(f"SEED {seed} -> Outbreak AUC: {ob_auc:.4f} | Breakdown AUC: {bd_auc:.4f} | " 
          f"Outbreaks: {sum(y_true_ob)}/{n_farms} | Breakdowns: {sum(y_true_bd)}/{n_farms}")
    return ob_auc, bd_auc

def main():
    model_path = "models/cattle/v17c_herd_engine.pth"
    config_path = "models/cattle/v17c_herd_config.json"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    model = HerdEngineV17c(
        node_dim=cfg['node_dim'], 
        gat_dim=cfg['gat_dim'], 
        tft_dim=cfg['tft_dim'],
        ngh=cfg.get('n_gat_heads', 6)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print("============================================================")
    print("🛡️  V17c REALITY CHECK: CROSS-SEED GENERALIZATION")
    print("============================================================")
    
    seeds = [42, 1024, 7777, 99999]
    res_ob, res_bd = [], []
    
    for seed in seeds:
        ob, bd = eval_seed(model, cfg, seed=seed, n_farms=100, device=device)
        res_ob.append(ob)
        res_bd.append(bd)
        
    print("\n============================================================")
    print("📊 CROSS-SEED SUMMARY")
    print("============================================================")
    ob_mean = np.mean(res_ob)
    bd_mean = np.mean(res_bd)
    print(f"Outbreak AUC Mean:  {ob_mean:.4f}  (Min: {np.min(res_ob):.4f}, Max: {np.max(res_ob):.4f})")
    print(f"Breakdown AUC Mean: {bd_mean:.4f}  (Min: {np.min(res_bd):.4f}, Max: {np.max(res_bd):.4f})")
    
    if ob_mean > 0.90 and bd_mean > 0.90:
        print("✅ PASS: The model genuinely learned herd dynamics without data leakage.")
    else:
        print("❌ FAIL: The model collapsed on unseen seeds. Probable data leakage.")
    print("============================================================")

if __name__ == "__main__":
    main()
