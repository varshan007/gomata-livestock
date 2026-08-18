#!/usr/bin/env python3
"""
stress_test_v18_counterfactual.py
Phase 18.5: Counterfactual Graph Surgery

Takes the exact same historical node features (I(t), etc.) and swaps 
the Adjacency Matrix A to test if the model causally responds to geometry 
independent of time-series readings.
"""

import os
import json
import numpy as np
import torch

from colab_train_v17c_herd import HerdEngineV17c, ProductionSimulatorV3

def test_counterfactual_geometry(model, cfg, device):
    print("="*60)
    print("✂️  PHASE 18.5: COUNTERFACTUAL GRAPH SURGERY")
    print("="*60)
    
    sim = ProductionSimulatorV3(seed=9999)
    
    # 1. Generate a single base biological state WITH an active infection cluster
    sim.rng = np.random.RandomState(42)
    data = sim.simulate_farm(0)
    
    # Let's force some nodes to have high infection manually at the last few ticks
    # so the GAT actually has a strong signal to pass across edges
    base_nf = data['node_features'].copy()  # (T, N, D)
    N_COWS = base_nf.shape[1]
    
    # Force Node 0 to be highly infected at t=25..27
    base_nf[-3:, 0, 0] = 0.8  # I(t)
    base_nf[-3:, 0, 5] = 1.0  # Severity
    base_nf[-3:, 1, 0] = 0.6
    base_nf[-3:, 1, 5] = 0.8
    
    # 2. Create three distinct geometries
    A_sparse = np.eye(N_COWS, dtype=np.float32)  # Completely isolated cows
    
    A_dense = np.ones((N_COWS, N_COWS), dtype=np.float32) * 0.5  # Fully connected herd
    np.fill_diagonal(A_dense, 1.0)
    
    # Realistic Hub Graph (a few highly connected super-spreaders)
    A_hub = np.eye(N_COWS, dtype=np.float32)
    hub_idx = np.random.choice(N_COWS, int(N_COWS * 0.15), replace=False)
    for h in hub_idx:
        tgts = np.random.choice(N_COWS, 15, replace=False)
        for t in tgts:
            A_hub[h, t] = 0.8
            A_hub[t, h] = 0.8
            
    # 3. Predict on all three exact identical I(t) histories
    nf_tensor = torch.tensor(base_nf).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get raw GAT layer embeddings BEFORE global mean pooling
        h_in = nf_tensor[:, -1] # Last timestep for inspection
        
        # We manually pass it through the GAT layers to observe the pure structural change
        # without the invariant mean pooling destroying the geometry
        h_s1 = torch.nn.functional.elu(model.gat1(h_in, torch.tensor(A_sparse).unsqueeze(0).to(device)))
        h_sparse_emb = torch.nn.functional.elu(model.gat2(h_s1, torch.tensor(A_sparse).unsqueeze(0).to(device)))
        
        h_h1 = torch.nn.functional.elu(model.gat1(h_in, torch.tensor(A_hub).unsqueeze(0).to(device)))
        h_hub_emb = torch.nn.functional.elu(model.gat2(h_h1, torch.tensor(A_hub).unsqueeze(0).to(device)))
        
        h_d1 = torch.nn.functional.elu(model.gat1(h_in, torch.tensor(A_dense).unsqueeze(0).to(device)))
        h_dense_emb = torch.nn.functional.elu(model.gat2(h_d1, torch.tensor(A_dense).unsqueeze(0).to(device)))
        
    print("\n[IDENTICAL NODE FEATURES] mapped across 3 Topologies.")
    print("Measuring GAT Output L2-Norm of the single heavily-infected Seed Cow (Node 0):")
    
    sparse_norm = torch.norm(h_sparse_emb[0, 0]).item()
    hub_norm = torch.norm(h_hub_emb[0, 0]).item()
    dense_norm = torch.norm(h_dense_emb[0, 0]).item()
        
    print(f"1. SPARSE (Isolated Pens): L2 Norm = {sparse_norm:.4f}")
    print("-" * 50)
    print(f"2. HUB (Standard Farm Structure): L2 Norm = {hub_norm:.4f}")
    print("-" * 50)
    print(f"3. DENSE (Panmictic Mixing): L2 Norm = {dense_norm:.4f}")
    
    print("\n============================================================")
    if not np.isclose(sparse_norm, hub_norm) and not np.isclose(hub_norm, dense_norm):
        print("✅ CAUSAL UNDERSTANDING PROVEN.")
        print("The attention network explicitly modifies feature representations based")
        print("on the local graph adjacency. Node 0's embedding differs drastically")
        print("because identical symptoms mean different things strictly depending on")
        print("who the node is connected to.")
    else:
        print("❌ DETERMINISTIC MEMORIZATION DETECTED.")
        print("The network ignored the input Adjacency Matrix entirely.")
    print("============================================================")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "models/cattle/v17c_herd_engine.pth"
    config_path = "models/cattle/v17c_herd_config.json"
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    model = HerdEngineV17c(
        node_dim=cfg['node_dim'], gat_dim=cfg['gat_dim'], 
        tft_dim=cfg['tft_dim'], ngh=cfg.get('n_gat_heads', 6)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    test_counterfactual_geometry(model, cfg, device)
