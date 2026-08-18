#!/usr/bin/env python3
"""
colab_train_v20_herd.py — Phase 20.6.3
BACKBONE-PRESERVING STRUCTURAL PEN INJECTION (GoMata T4)
============================================================
Architecture:
  ResGAT(96d, 6h) × 2 → Branch:
    ├─ Branch B (node): h_node + α·spectral → MLP → ΔR₀, vacc_rank
    ├─ Branch C (pen):  PenMesoEncoder + struct_mlp → PenGAT → MLP → pen_risk
    └─ Branch A (herd): global_mean_pool → TFT → 9 herd heads

Phase 20.6.3 — Backbone FROZEN forever:
  ✓ Backbone (GAT, TFT, Herd, Node) NEVER unfreezes
  ✓ Only struct_mlp + pen_gat + pen_head trainable in BOTH stages
  ✓ Stage A: 8ep pen-only Pearson loss (lr=3e-4, λ=6.0)
  ✓ Stage B: 6ep pen + all-loss supervision (lr=5e-5, λ=3.0)
  ✓ Safety abort if ΔR₀ drops below 0.95
  ✓ Best model saved by pen correlation
  ✓ 8 structural topology features per pen + struct_mlp(8→64→96)

Inherited:
  ✓ GAT 96d×6h (< 450k params, T4-safe)
  ✓ PenMesoEncoder: scatter_mean + struct_inject → pen_adj → PenGAT → MLP
  ✓ 2000 farms, 8 graph families, nodes 40-150
"""

import os, sys, gc, json, math, time, logging, warnings
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
NUM_FARMS   = 2000
T_STEPS     = 28; SIM_TOTAL = 28
MAX_COWS    = 150
MAX_PENS    = 10
NODE_DIM    = 22   # 18 original + 4 spectral priors
GAT_DIM     = 96
GAT_HEADS   = 6
GAT_LAYERS  = 2
STAGE_A_EPOCHS = 8     # Freeze backbone, train struct_mlp + pen head
STAGE_B_EPOCHS = 6     # Backbone STILL frozen, pen + all-loss supervision
TOTAL_EPOCHS = STAGE_A_EPOCHS + STAGE_B_EPOCHS  # 14 total
BATCH_SIZE  = 8
T_SUBSAMPLE = 2    # 28→14 GAT passes
PEN_GAT_HEADS = 4  # Pen-level GAT heads
OUT_DIR     = "models/cattle"
DRIVE_DIR   = "/content/drive/MyDrive/HerdV20"
os.makedirs(OUT_DIR, exist_ok=True)

LOSS_WEIGHTS = {
    "intensity": 1.0, "slope": 1.0, "trend": 1.0,
    "HSI": 1.0, "R0_reduction": 1.0,
    "peak_day": 1.0, "peak_size": 1.0,
    "breakdown": 1.2,
    "resources": 1.0,
    "delta_R0": 4.0,       # ← BOOSTED
    "pen_risk": 3.0,       # ← BOOSTED
    "vaccination": 3.0,    # ← BOOSTED
}

# ═══════════════════════════════════════════════════════════════
# SECTION 1: PRODUCTION SIMULATOR V5
# ═══════════════════════════════════════════════════════════════

class ProductionSimulatorV5:
    """Production simulator: 8 graph families, 40-150 nodes, spectral priors."""

    GRAPH_FAMILIES = [
        'hub', 'scale_free', 'small_world', 'erdos_renyi',
        'bipartite', 'clustered', 'community', 'multi_hub'
    ]

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    # ── Graph Generators (8 families) ─────────────────────────
    def _hub_graph(self, N, n_pens, n_workers):
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        n_hubs = max(2, int(0.12 * N))
        hub_idx = self.rng.choice(N, n_hubs, replace=False)
        is_hub = np.zeros(N, dtype=bool); is_hub[hub_idx] = True
        bd = self.rng.randint(3, 6)
        for p in range(n_pens):
            ip = np.where(pen == p)[0]
            if len(ip) < 2: continue
            for i in ip:
                nd = (bd * 3) if is_hub[i] else (bd - 1)
                nd = min(nd, len(ip) - 1)
                if nd <= 0: continue
                ot = [j for j in ip if j != i]
                ch = self.rng.choice(ot, min(nd, len(ot)), replace=False)
                for j in ch:
                    w = self.rng.uniform(0.3, 1.0) * 1.5
                    A[i, j] = max(A[i, j], w); A[j, i] = A[i, j]
        for h in hub_idx:
            nc = self.rng.randint(3, 8)
            tgts = self.rng.choice(N, nc, replace=False)
            for t in tgts:
                if t != h and pen[t] != pen[h]:
                    w = self.rng.uniform(0.3, 0.8) * 1.3
                    A[h, t] = max(A[h, t], w); A[t, h] = A[h, t]
        return A, pen, worker

    def _scale_free_graph(self, N, n_pens, n_workers):
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        m = self.rng.randint(2, 5)
        for i in range(min(m+1, N)):
            for j in range(i+1, min(m+1, N)):
                w = self.rng.uniform(0.3, 1.0); A[i,j]=w; A[j,i]=w
        for new in range(m+1, N):
            deg = A[:new].sum(axis=1); total = deg.sum()
            probs = deg / total if total > 0 else np.ones(new) / new
            tgts = self.rng.choice(new, min(m, new), replace=False, p=probs)
            for t in tgts:
                w = self.rng.uniform(0.3, 1.0); A[new,t]=w; A[t,new]=w
        return A, pen, worker

    def _small_world_graph(self, N, n_pens, n_workers):
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        k = self.rng.randint(4, 8); p_rw = self.rng.uniform(0.05, 0.3)
        for i in range(N):
            for j in range(1, k//2 + 1):
                nb = (i + j) % N; w = self.rng.uniform(0.3, 1.0)
                A[i, nb] = w; A[nb, i] = w
        for i in range(N):
            for j in range(1, k//2 + 1):
                if self.rng.random() < p_rw:
                    nb = (i + j) % N; A[i, nb] = 0; A[nb, i] = 0
                    nn_ = self.rng.randint(0, N)
                    while nn_ == i or A[i, nn_] > 0: nn_ = self.rng.randint(0, N)
                    w = self.rng.uniform(0.3, 1.0); A[i, nn_] = w; A[nn_, i] = w
        return A, pen, worker

    def _erdos_renyi_graph(self, N, n_pens, n_workers):
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        p = self.rng.uniform(0.05, 0.15)
        for i in range(N):
            for j in range(i+1, N):
                if self.rng.random() < p:
                    w = self.rng.uniform(0.3, 1.0); A[i,j]=w; A[j,i]=w
        return A, pen, worker

    def _bipartite_graph(self, N, n_pens, n_workers):
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        for wk in range(n_workers):
            cw = np.where(worker == wk)[0]
            if len(cw) < 2: continue
            ne = min(len(cw) * 2, len(cw) * (len(cw)-1) // 2)
            for _ in range(ne):
                i, j = self.rng.choice(cw, 2, replace=False)
                w = self.rng.uniform(0.3, 0.8)
                A[i, j] = max(A[i, j], w); A[j, i] = A[i, j]
        for _ in range(N // 5):
            i, j = self.rng.choice(N, 2, replace=False)
            if worker[i] != worker[j]:
                w = self.rng.uniform(0.1, 0.4)
                A[i, j] = max(A[i, j], w); A[j, i] = A[i, j]
        return A, pen, worker

    def _clustered_graph(self, N, n_pens, n_workers):
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        for p in range(n_pens):
            ip = np.where(pen == p)[0]
            if len(ip) < 2: continue
            pi = self.rng.uniform(0.3, 0.6)
            for a in range(len(ip)):
                for b in range(a+1, len(ip)):
                    if self.rng.random() < pi:
                        w = self.rng.uniform(0.5, 1.2)
                        A[ip[a], ip[b]] = w; A[ip[b], ip[a]] = w
        nb = self.rng.randint(n_pens, n_pens * 3)
        for _ in range(nb):
            i, j = self.rng.choice(N, 2, replace=False)
            if pen[i] != pen[j]:
                w = self.rng.uniform(0.1, 0.4)
                A[i, j] = max(A[i, j], w); A[j, i] = A[i, j]
        return A, pen, worker

    def _community_graph(self, N, n_pens, n_workers):
        """Multi-community with dense intra, sparse inter."""
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        n_comm = self.rng.randint(3, 6)
        comm = self.rng.randint(0, n_comm, N)
        for c in range(n_comm):
            members = np.where(comm == c)[0]
            if len(members) < 2: continue
            p_in = self.rng.uniform(0.3, 0.5)
            for a in range(len(members)):
                for b in range(a+1, len(members)):
                    if self.rng.random() < p_in:
                        w = self.rng.uniform(0.4, 1.0)
                        A[members[a], members[b]] = w
                        A[members[b], members[a]] = w
        # Inter-community bridges
        for _ in range(N // 4):
            i, j = self.rng.choice(N, 2, replace=False)
            if comm[i] != comm[j]:
                w = self.rng.uniform(0.1, 0.3)
                A[i, j] = max(A[i, j], w); A[j, i] = A[i, j]
        return A, pen, worker

    def _multi_hub_graph(self, N, n_pens, n_workers):
        """Multiple super-hubs with overlapping influence zones."""
        pen = self.rng.randint(0, n_pens, N)
        worker = self.rng.randint(0, n_workers, N)
        A = np.zeros((N, N), dtype=np.float32)
        n_hubs = self.rng.randint(3, 7)
        hubs = self.rng.choice(N, n_hubs, replace=False)
        # Each hub connects to a random subset
        for h in hubs:
            n_conn = self.rng.randint(N // 5, N // 2)
            targets = self.rng.choice(N, n_conn, replace=False)
            for t in targets:
                if t != h:
                    w = self.rng.uniform(0.3, 1.0)
                    A[h, t] = max(A[h, t], w); A[t, h] = A[h, t]
        # Sparse background edges
        for _ in range(N):
            i, j = self.rng.choice(N, 2, replace=False)
            if A[i, j] == 0:
                w = self.rng.uniform(0.1, 0.3)
                A[i, j] = w; A[j, i] = w
        return A, pen, worker

    def _generate_graph(self, N, n_pens, n_workers):
        family = self.rng.choice(self.GRAPH_FAMILIES)
        gen = {
            'hub': self._hub_graph, 'scale_free': self._scale_free_graph,
            'small_world': self._small_world_graph, 'erdos_renyi': self._erdos_renyi_graph,
            'bipartite': self._bipartite_graph, 'clustered': self._clustered_graph,
            'community': self._community_graph, 'multi_hub': self._multi_hub_graph,
        }
        A, pen, worker = gen[family](N, n_pens, n_workers)
        # Dynamic contact perturbation
        noise = self.rng.uniform(0.9, 1.1, A.shape).astype(np.float32)
        A = A * noise; A = (A + A.T) / 2; np.fill_diagonal(A, 0)
        return A, pen, worker, family

    # ── Spectral Priors (vectorized) ──────────────────────────
    def _compute_spectral_priors(self, A):
        N = A.shape[0]; adj_bin = (A > 0).astype(float)
        deg = adj_bin.sum(axis=1)
        deg_norm = deg / (N - 1 + 1e-8)
        # PageRank proxy for betweenness
        D_inv = np.diag(1.0 / (deg + 1e-8)); P = D_inv @ A
        pr = np.ones(N) / N
        for _ in range(10): pr = 0.85 * (P.T @ pr) + 0.15 / N
        betw = pr / (pr.max() + 1e-8)
        # Eigenvector centrality
        try:
            evals, evecs = np.linalg.eigh(A)
            eig_cent = np.abs(evecs[:, -1])
            eig_cent /= (eig_cent.max() + 1e-8)
        except: eig_cent = deg_norm.copy()
        # Clustering coefficient (vectorized A³ diagonal)
        tri = np.diag(adj_bin @ adj_bin @ adj_bin) / 2
        pairs = deg * (deg - 1) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            clust = np.where(pairs > 0, tri / pairs, 0).astype(np.float32)
        return deg_norm.astype(np.float32), betw.astype(np.float32), eig_cent.astype(np.float32), clust

    # ── Core Math ─────────────────────────────────────────────
    def _graph_entropy(self, A):
        d = (A > 0).sum(axis=1).astype(float); t = d.sum()
        if t == 0: return 0.0
        p = d / t; p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _spectral_R0(self, A, beta, gamma):
        if gamma <= 0: return 0.0
        try:
            ev = np.linalg.eigvalsh((beta / gamma) * A)
            return float(np.max(np.abs(ev)))
        except: return 0.0

    def _compute_HSI(self, I_obs, A, mean_I):
        n = len(mean_I)
        s2 = float(np.clip(np.var(mean_I)*10, 0, 1))
        dI = np.diff(mean_I) if n > 1 else np.array([0.0])
        g = float(np.clip(np.mean(np.abs(dI))*20, 0, 1))
        d2I = np.diff(dI) if len(dI) > 1 else np.array([0.0])
        a = float(np.clip(np.mean(np.abs(d2I))*40, 0, 1))
        c = float(np.clip(np.mean(1.0 - I_obs.mean(axis=1)), 0, 1))
        H = self._graph_entropy(A)
        Hm = np.log(A.shape[0]) if A.shape[0] > 1 else 1.0
        Hn = float(np.clip(H / Hm, 0, 1))
        base = float(np.clip(0.30*(1-s2)+0.30*(1-g)+0.20*(1-a)+0.10*c+0.10*Hn, 0, 1))
        return float(np.clip(base * (1.0 - float(np.max(mean_I))**2), 0, 1))

    # ── Phase 20 Targets (O(1) spectral approx) ──────────────
    def _delta_R0_per_node(self, A, beta, gamma):
        N = A.shape[0]
        if gamma <= 0: return np.zeros(N, dtype=np.float32)
        try:
            K = (beta / gamma) * A
            _, evecs = np.linalg.eigh(K)
            v = np.abs(evecs[:, -1]); d = A.sum(axis=1)
            delta = v ** 2 * d * (beta / gamma)
        except: delta = np.zeros(N, dtype=np.float32)
        mx = delta.max()
        if mx > 0: delta /= mx
        return delta.astype(np.float32)

    def _pen_intensity_zscore(self, I_obs, pen, n_pens):
        """Z-scored pen intensity (per-farm normalization)."""
        mi = I_obs.mean(axis=0)
        raw = np.zeros(n_pens, dtype=np.float32)
        for p in range(n_pens):
            idx = np.where(pen == p)[0]
            if len(idx) > 0: raw[p] = float(mi[idx].mean())
        mu = raw.mean(); std = raw.std() + 1e-8
        return ((raw - mu) / std).astype(np.float32)

    def _vaccination_gain(self, A, beta, gamma, N):
        if gamma <= 0: return np.zeros(N, dtype=np.float32)
        try:
            K = (beta / gamma) * A
            _, evecs = np.linalg.eigh(K)
            v = np.abs(evecs[:, -1]); d = A.sum(axis=1)
            scores = v ** 2 * d
        except: scores = np.zeros(N, dtype=np.float32)
        mx = scores.max()
        if mx > 0: scores /= mx
        return scores.astype(np.float32)

    # ── Main Simulation ───────────────────────────────────────
    def simulate_farm(self, fidx):
        n_cows = self.rng.randint(40, 150)
        n_pens = self.rng.randint(3, 10)
        n_workers = self.rng.randint(2, 6)
        A, pen, worker, graph_family = self._generate_graph(n_cows, n_pens, n_workers)

        regime = self.rng.choice(
            ['stable','borderline','outbreak','superspreader'],
            p=[0.30, 0.25, 0.35, 0.10]  # 45% outbreak+SS → ~40% outbreak prevalence
        )
        if regime == 'stable':
            beta=self.rng.uniform(0.01,0.03); gamma=self.rng.uniform(0.15,0.30)
            n_seed=self.rng.randint(1,3); seed_t=0
        elif regime == 'borderline':
            beta=self.rng.uniform(0.03,0.055); gamma=self.rng.uniform(0.08,0.15)
            n_seed=self.rng.randint(2,5); seed_t=self.rng.randint(5,15)
        elif regime == 'outbreak':
            beta=self.rng.uniform(0.055,0.12); gamma=self.rng.uniform(0.04,0.08)
            n_seed=self.rng.randint(2,6); seed_t=self.rng.randint(5,18)
        else:
            beta=self.rng.uniform(0.12,0.25); gamma=self.rng.uniform(0.03,0.06)
            n_seed=self.rng.randint(3,8); seed_t=self.rng.randint(5,15)

        vaccinated = np.zeros(n_cows, dtype=np.float32)
        nv = int(self.rng.uniform(0, 0.25) * n_cows)
        if nv > 0: vaccinated[self.rng.choice(n_cows, nv, replace=False)] = 1.0

        I = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        S = np.ones((SIM_TOTAL, n_cows), dtype=np.float32)
        severity = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        seeds = self.rng.choice(n_cows, min(n_seed, n_cows), replace=False)
        I[seed_t, seeds] = self.rng.uniform(0.3, 0.7, len(seeds))
        S[seed_t, seeds] = 1.0 - I[seed_t, seeds]

        ah = self.rng.uniform(0.02, 0.06); bt = self.rng.uniform(68, 85)
        for t in range(max(1, seed_t+1), SIM_TOTAL):
            te = max(0, bt + 3*np.sin(t*2*np.pi/28) - 72)
            be = beta * (1 + ah * te)
            Ae = A * (1 - vaccinated[np.newaxis, :] * 0.8)
            ni = np.clip(be * (Ae @ I[t-1]) * S[t-1], 0, S[t-1])
            nr = gamma * I[t-1]
            S[t] = np.clip(S[t-1] - ni, 0, 1)
            I[t] = np.clip(I[t-1] + ni - nr, 0, 1)
            severity[t] = I[t] * (1 + 0.2 * te / 10)

        # Spectral priors
        deg_n, betw, eig_c, clust = self._compute_spectral_priors(A)

        # Node features (22-dim)
        nf = np.zeros((SIM_TOTAL, n_cows, NODE_DIM), dtype=np.float32)
        for t in range(SIM_TOTAL):
            te = max(0, bt + 3*np.sin(t*2*np.pi/28) - 72)
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
                    vaccinated[i], float(pen[i])/n_pens,
                    deg_n[i], betw[i], eig_c[i], clust[i],
                ]

        # Herd labels
        obs = T_STEPS; mI = I.mean(axis=1)
        intensity = float(mI[:obs].max())
        x = np.arange(obs, dtype=np.float32); xc = x - x.mean()
        slope = float((xc * mI[:obs]).sum() / ((xc**2).sum() + 1e-8))
        trend = float(mI[obs-1] - mI[max(0, obs-7)])
        dI_max = float(np.max(np.abs(np.diff(mI[:obs])))) if obs > 1 else 0.0
        hsi = self._compute_HSI(I[:obs], A, mI[:obs])
        ba = beta * (1 + ah * max(0, bt - 72))
        r0b = self._spectral_R0(A, ba, gamma)
        wd = A.sum(axis=1); isc = I[:obs].mean(axis=0)
        comb = wd * (1 + 5 * isc)
        nr_n = max(3, int(0.10 * n_cows))
        tn = np.argsort(comb)[-nr_n:][::-1]
        A2 = A.copy(); A2[tn,:] = 0; A2[:,tn] = 0
        r0p = self._spectral_R0(A2, ba, gamma)
        r0r = float((r0b - r0p) / r0b) if r0b > 0 else 0
        outbreak = float(intensity > 0.15)
        bd = float((intensity > 0.65) and (hsi < 0.65) and (dI_max > 0.08))
        pk = int(np.argmax(mI[:obs])); pkd = float(pk/4.0); pks = float(mI[pk])
        ml = float((I[:obs]*10).sum()/n_cows)
        ab = float((I[:obs]>0.3).sum())/n_cows
        iso = float((severity[:obs]>1.0).any(axis=0).sum())/n_cows

        # Phase 20 labels
        delta_r0 = self._delta_R0_per_node(A, ba, gamma)
        pen_int = self._pen_intensity_zscore(I[:obs], pen, n_pens)
        vacc_gain = self._vaccination_gain(A, ba, gamma, n_cows)

        return {
            "node_features": nf[:obs].astype(np.float32),
            "adjacency": A.astype(np.float32),
            "n_cows": n_cows, "pen_mapping": pen.astype(np.int64),
            "n_pens": n_pens, "graph_family": graph_family,
            "labels": {
                "intensity": intensity, "slope": float(np.clip(slope,-1,1)),
                "trend": float(np.clip(trend,-1,1)), "HSI": hsi,
                "R0_reduction": float(np.clip(r0r, 0, 1)),
                "outbreak": outbreak, "peak_day": pkd,
                "peak_size": float(np.clip(pks,0,1)),
                "breakdown": bd,
                "milk_loss": ml, "antibiotic": ab, "isolation": iso,
                "delta_R0_per_node": delta_r0,
                "pen_intensity": pen_int,
                "vaccination_gain": vacc_gain,
            }
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 2: MODEL — HerdEngineV20 (Phase 20.4)
# ═══════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    def __init__(self, din, dout, drop=0.2):
        super().__init__()
        self.W = nn.Linear(din, dout, bias=False)
        self.a = nn.Linear(2*dout, 1, bias=False)
        self.lk = nn.LeakyReLU(0.2); self.dp = nn.Dropout(drop)
    def forward(self, h, adj):
        Wh = self.W(h); N = Wh.size(1)
        a_in = torch.cat([Wh.unsqueeze(2).expand(-1,-1,N,-1),
                          Wh.unsqueeze(1).expand(-1,N,-1,-1)], dim=-1)
        e = self.lk(self.a(a_in).squeeze(-1))
        m = (adj == 0); e = e.masked_fill(m, -6e4)
        a_w = self.dp(torch.softmax(e, dim=-1))
        return a_w @ Wh

class ResGAT(nn.Module):
    def __init__(self, din, dout, nh=6, drop=0.2):
        super().__init__()
        self.hd = dout // nh; self.rem = dout - self.hd * (nh - 1)
        self.heads = nn.ModuleList([
            GATLayer(din, self.hd if i < nh-1 else self.rem, drop) for i in range(nh)
        ])
        self.norm = nn.LayerNorm(dout)
        self.proj = nn.Linear(din, dout) if din != dout else nn.Identity()
    def forward(self, h, adj):
        out = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        return self.norm(out + self.proj(h))

class TFTBlock(nn.Module):
    def __init__(self, dm, nh=4, ff=256, drop=0.1):
        super().__init__()
        self.gru = nn.GRU(dm, dm, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(dm*2, dm)
        self.attn = nn.MultiheadAttention(dm, nh, dropout=drop, batch_first=True)
        self.n1 = nn.LayerNorm(dm); self.n2 = nn.LayerNorm(dm)
        self.ff = nn.Sequential(nn.Linear(dm,ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff,dm))
        self.gate = nn.Sequential(nn.Linear(dm*2,dm), nn.Sigmoid())
    def forward(self, x):
        g, _ = self.gru(x); g = self.proj(g)
        gc = self.gate(torch.cat([x, g], dim=-1)) * g
        r = self.n1(x + gc); a, _ = self.attn(r, r, r)
        r = self.n2(r + a); return (r + self.ff(r)).mean(dim=1)


class PenMesoEncoder(nn.Module):
    """Phase 20.6.2 — Mesoscopic Pen Graph Encoder with Structural Injection.

    Pipeline:
      1. Scatter mean-pool node embeddings → pen base embeddings (B, P, d)
      2. Compute 8 structural topology features per pen
      3. struct_mlp(8→64→d) → inject into pen embeddings
      4. Construct pen adjacency via bmm
      5. PenGAT message passing (1 layer, multi-head, residual)
      6. MLP risk head
    """

    def __init__(self, gat_dim=GAT_DIM, max_pens=MAX_PENS, pen_heads=PEN_GAT_HEADS, drop=0.1):
        super().__init__()
        self.max_pens = max_pens
        self.gat_dim = gat_dim

        # Structural feature MLP: 8 topology features → gat_dim
        self.struct_mlp = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, gat_dim)
        )

        # Pen-level GAT (1 layer, multi-head, residual)
        hd = gat_dim // pen_heads
        rem = gat_dim - hd * (pen_heads - 1)
        self.pen_gat_heads = nn.ModuleList([
            GATLayer(gat_dim, hd if i < pen_heads - 1 else rem, drop)
            for i in range(pen_heads)
        ])
        self.pen_norm = nn.LayerNorm(gat_dim)

        # Pen risk head
        self.pen_head = nn.Sequential(
            nn.Linear(gat_dim, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 1)
        )

    def _aggregate_pens(self, H_node, pen_map):
        """Scatter mean-pool: node embeddings → pen embeddings."""
        B, N, d = H_node.shape; P = self.max_pens
        pidx = pen_map.clamp(0, P - 1)
        idx_exp = pidx.unsqueeze(-1).expand(-1, -1, d)
        pen_sum = torch.zeros(B, P, d, device=H_node.device, dtype=H_node.dtype)
        pen_sum.scatter_add_(1, idx_exp, H_node)
        pen_cnt = torch.zeros(B, P, 1, device=H_node.device, dtype=H_node.dtype)
        pen_cnt.scatter_add_(1, pidx.unsqueeze(-1),
                             torch.ones(B, N, 1, device=H_node.device, dtype=H_node.dtype))
        return pen_sum / pen_cnt.clamp(min=1.0), pen_cnt.squeeze(-1)  # (B,P,d), (B,P)

    def _build_pen_adj(self, adj, pen_map):
        """Construct pen-level adjacency via bmm."""
        B, N, _ = adj.shape; P = self.max_pens
        pidx = pen_map.clamp(0, P - 1)
        pen_oh = torch.zeros(B, N, P, device=adj.device, dtype=adj.dtype)
        pen_oh.scatter_(2, pidx.unsqueeze(-1), 1.0)
        A_pen = torch.bmm(pen_oh.transpose(1, 2), torch.bmm(adj, pen_oh))
        deg = A_pen.sum(dim=-1)
        dis = (deg + 1e-8).pow(-0.5)
        A_norm = torch.nan_to_num(A_pen * dis.unsqueeze(-1) * dis.unsqueeze(-2), nan=0.0)
        return A_norm, A_pen  # return both normalized and raw

    def _compute_struct_features(self, adj, pen_map, pen_cnt):
        """Compute 8 structural topology features per pen.
        Features: num_nodes, internal_edges, mean_degree, density,
                  cut_ratio, conductance, clustering, spectral_gap.
        Loop is over max_pens (<=10) per batch item — fast."""
        B, N, _ = adj.shape; P = self.max_pens
        pidx = pen_map.clamp(0, P - 1)
        feats = torch.zeros(B, P, 8, device=adj.device, dtype=adj.dtype)

        adj_bin = (adj > 0).float()
        total_edges = adj_bin.sum(dim=(1, 2)) / 2  # (B,) total edges per graph

        for b in range(B):
            for p in range(P):
                nip = (pidx[b] == p).nonzero(as_tuple=True)[0]
                nc = len(nip)
                if nc < 1:
                    continue

                # 1. num_nodes (normalized)
                feats[b, p, 0] = nc / max(N, 1)

                if nc < 2:
                    continue

                # Internal sub-adjacency
                sa = adj_bin[b][nip][:, nip]
                deg_in = sa.sum(dim=1)  # internal degree per node
                e_int = sa.sum() / 2  # internal edges

                # 2. internal_edge_count (normalized)
                feats[b, p, 1] = e_int / max(total_edges[b].item(), 1)

                # 3. mean_internal_degree (normalized)
                feats[b, p, 2] = deg_in.mean() / max(nc - 1, 1)

                # 4. density = 2E / (N*(N-1))
                max_e = nc * (nc - 1) / 2
                feats[b, p, 3] = (e_int / max(max_e, 1)).clamp(0, 1)

                # Boundary edges: edges from pen nodes to outside
                full_deg = adj_bin[b][nip].sum(dim=1)  # total degree
                boundary = (full_deg - deg_in).sum() / 2
                total_possible_boundary = nc * (N - nc)

                # 5. cut_ratio
                feats[b, p, 4] = (boundary / max(total_possible_boundary, 1)).clamp(0, 1)

                # 6. conductance
                vol = 2 * e_int + boundary
                feats[b, p, 5] = (boundary / max(vol.item(), 1)).clamp(0, 1)

                # 7. clustering coefficient (mean local)
                if nc >= 3:
                    tri = torch.diag(sa @ sa @ sa) / 2
                    pairs = deg_in * (deg_in - 1) / 2
                    cc = torch.where(pairs > 0, tri / pairs, torch.zeros_like(tri))
                    feats[b, p, 6] = cc.mean().clamp(0, 1)

                # 8. spectral_gap (λ2 - λ1 of pen adjacency, approximate)
                if nc >= 3:
                    try:
                        L = torch.diag(deg_in) - sa
                        evals = torch.linalg.eigvalsh(L.float())
                        feats[b, p, 7] = (evals[1] - evals[0]).clamp(0, 5) / 5.0
                    except Exception:
                        feats[b, p, 7] = 0.0

        # Z-score normalize across batch (per feature)
        mu = feats.mean(dim=(0, 1), keepdim=True)
        std = feats.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        feats = ((feats - mu) / std).clamp(-3, 3)
        feats = torch.nan_to_num(feats, nan=0.0)
        return feats  # (B, P, 8)

    def forward(self, H_node, pen_map, adj):
        # Step 1: Aggregate node → pen
        pen_emb, pen_cnt = self._aggregate_pens(H_node, pen_map)  # (B, P, d), (B, P)

        # Step 2: Compute structural features + inject
        struct_feats = self._compute_struct_features(adj, pen_map, pen_cnt)  # (B, P, 8)
        struct_emb = self.struct_mlp(struct_feats)  # (B, P, d)
        pen_emb = pen_emb + struct_emb  # additive injection

        # Step 3: Build pen adjacency
        A_pen, _ = self._build_pen_adj(adj, pen_map)  # (B, P, P)

        # Step 4: PenGAT message passing (1 layer, residual)
        pen_out = torch.cat([h(pen_emb, A_pen) for h in self.pen_gat_heads], dim=-1)
        pen_out = self.pen_norm(pen_out + pen_emb)  # residual

        # Step 5: Risk head
        scores = self.pen_head(pen_out).squeeze(-1)  # (B, P)
        return scores


class HerdEngineV20(nn.Module):
    """Phase 20.6.2 — Herd/Node/Pen towers with Structural Mesoscopic Pen Encoder."""

    def __init__(self, node_dim=NODE_DIM, gat_dim=GAT_DIM, tft_dim=None,
                 ngh=GAT_HEADS, nth=4):
        super().__init__()
        if tft_dim is None: tft_dim = gat_dim

        # GAT backbone (2 layers, 96d, 6 heads)
        self.gat1 = ResGAT(node_dim, gat_dim, ngh)
        self.gat2 = ResGAT(gat_dim, gat_dim, ngh)

        # Residual spectral injection (learnable α)
        self.spectral_alpha = nn.Parameter(torch.tensor(0.1))
        self.spectral_proj = nn.Linear(4, gat_dim)

        # ── Branch A: Herd (pool → TFT → 9 heads) ──
        self.herd_pool = nn.Sequential(nn.Linear(gat_dim, gat_dim), nn.GELU())
        self.tft = TFTBlock(gat_dim, nth, ff=256)
        self.h_int   = nn.Linear(tft_dim, 1)
        self.h_slope = nn.Linear(tft_dim, 1)
        self.h_trend = nn.Linear(tft_dim, 1)
        self.h_hsi   = nn.Sequential(nn.Linear(tft_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.h_r0r   = nn.Sequential(nn.Linear(tft_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.h_pkd   = nn.Linear(tft_dim, 1)
        self.h_pks   = nn.Linear(tft_dim, 1)
        self.h_bd    = nn.Linear(tft_dim, 1)
        self.h_res   = nn.Linear(tft_dim, 3)

        # ── Branch B: Node heads (operate on h_node directly) ──
        self.node_ss = nn.Sequential(
            nn.Linear(gat_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.node_vacc = nn.Sequential(
            nn.Linear(gat_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # ── Branch C: Pen (Phase 20.6 — Mesoscopic Encoder) ──
        self.pen_meso = PenMesoEncoder(gat_dim=gat_dim)

    def forward(self, ns, adj, pen_map=None):
        B, T, N, F = ns.shape
        herd_seq = []

        for t in range(T):
            h = ns[:, t]
            h = tF.elu(self.gat1(h, adj))
            h = tF.elu(self.gat2(h, adj))

            # Residual spectral injection: h += α · proj(spectral_features)
            spectral_feats = ns[:, t, :, -4:]  # last 4 features = spectral priors
            h = h + self.spectral_alpha * self.spectral_proj(spectral_feats)

            herd_seq.append(self.herd_pool(h.mean(dim=1)))

        # Use last timestep's node embeddings for node-level heads
        H_node = h  # (B, N, gat_dim) — last timestep

        # ── Branch A: Herd ──
        ctx = self.tft(torch.stack(herd_seq, dim=1))
        result = {
            "intensity":    self.h_int(ctx).squeeze(-1),
            "slope":        self.h_slope(ctx).squeeze(-1),
            "trend":        self.h_trend(ctx).squeeze(-1),
            "HSI":          self.h_hsi(ctx).squeeze(-1),
            "R0_reduction": self.h_r0r(ctx).squeeze(-1),
            "peak_day":     self.h_pkd(ctx).squeeze(-1),
            "peak_size":    self.h_pks(ctx).squeeze(-1),
            "breakdown":    self.h_bd(ctx).squeeze(-1),
            "resources":    self.h_res(ctx),
        }

        # ── Branch B: Node (directly on H_node, no pooling) ──
        result["node_super_spreader_score"] = self.node_ss(H_node).squeeze(-1)
        result["optimal_vaccination_rank"]  = self.node_vacc(H_node).squeeze(-1)

        # ── Branch C: Pen (Phase 20.6 — Mesoscopic Graph Encoder) ──
        if pen_map is not None:
            result["pen_risk_scores"] = self.pen_meso(H_node, pen_map, adj)
        else:
            result["pen_risk_scores"] = torch.zeros(B, MAX_PENS, device=ns.device)

        return result


# ═══════════════════════════════════════════════════════════════
# SECTION 3: DATASET
# ═══════════════════════════════════════════════════════════════

class HerdDSV20(Dataset):
    def __init__(self, farms): self.farms = farms
    def __len__(self): return len(self.farms)
    def __getitem__(self, i):
        f = self.farms[i]
        nf = f["node_features"]; adj = f["adjacency"]
        Ts, N, Ft = nf.shape; nc = min(N, MAX_COWS)

        np_ = np.zeros((Ts, MAX_COWS, Ft), dtype=np.float32)
        ap_ = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
        np_[:, :nc, :] = nf[:, :nc, :]
        ap_[:nc, :nc] = adj[:nc, :nc]

        L = f["labels"]
        pm = np.zeros(MAX_COWS, dtype=np.int64); pm[:nc] = f["pen_mapping"][:nc]
        dr0 = np.zeros(MAX_COWS, dtype=np.float32); dr0[:nc] = L["delta_R0_per_node"][:nc]
        vg = np.zeros(MAX_COWS, dtype=np.float32); vg[:nc] = L["vaccination_gain"][:nc]
        pi = np.zeros(MAX_PENS, dtype=np.float32)
        nps = f["n_pens"]; pi[:nps] = L["pen_intensity"][:nps]

        return (
            torch.tensor(np_), torch.tensor(ap_),
            torch.tensor(L["intensity"], dtype=torch.float32),
            torch.tensor(L["slope"], dtype=torch.float32),
            torch.tensor(L["trend"], dtype=torch.float32),
            torch.tensor(L["HSI"], dtype=torch.float32),
            torch.tensor(L["R0_reduction"], dtype=torch.float32),
            torch.tensor(L["peak_day"], dtype=torch.float32),
            torch.tensor(L["peak_size"], dtype=torch.float32),
            torch.tensor(L["breakdown"], dtype=torch.float32),
            torch.tensor([L["milk_loss"], L["antibiotic"], L["isolation"]], dtype=torch.float32),
            torch.tensor(pm), torch.tensor(dr0), torch.tensor(vg), torch.tensor(pi),
        )


# ═══════════════════════════════════════════════════════════════
# SECTION 4: MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("="*60)
    logger.info("🧠 Phase 20.6.3 — Backbone-Preserving Structural Pen Injection")
    logger.info("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                     f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")

    # ── Step 1: Simulate ──
    logger.info(f"Step 1: Simulating {NUM_FARMS} farms (8 families, 40-150 nodes)...")
    sim = ProductionSimulatorV5(seed=2025)
    farms = []; fam_cnt = {}
    for i in range(NUM_FARMS):
        f = sim.simulate_farm(i)
        farms.append(f)
        fam_cnt[f['graph_family']] = fam_cnt.get(f['graph_family'], 0) + 1
        if (i+1) % 500 == 0:
            ob = sum(1 for f in farms if f['labels']['outbreak'] > 0.5)
            logger.info(f"  {i+1}/{NUM_FARMS} ob:{ob} ({ob/(i+1):.0%})")
    nob = sum(1 for f in farms if f['labels']['outbreak'] > 0.5)
    nbd = sum(1 for f in farms if f['labels']['breakdown'] > 0.5)
    logger.info(f"Done: {nob} outbreaks ({nob/NUM_FARMS:.0%}), {nbd} breakdowns ({nbd/NUM_FARMS:.0%})")
    logger.info(f"Families: {fam_cnt}")

    # ── Step 2: DataLoader ──
    ds = HerdDSV20(farms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)
    del farms; gc.collect()

    # ── Step 3: Model (NO log_vars) ──
    model = HerdEngineV20().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HerdEngineV20 Params: {n_params:,}")

    # ── Load best checkpoint (Phase 20.6.1 or latest) ──
    ckpt_best = None
    for cp in ["/content/drive/MyDrive/HerdV20/v20_herd_engine.pth",
               "models/cattle/v20_herd_engine.pth",
               "v20_herd_engine.pth"]:
        if os.path.exists(cp):
            ckpt_best = cp; break
    if ckpt_best:
        logger.info(f"Loading checkpoint from {ckpt_best}...")
        state = torch.load(ckpt_best, map_location=device, weights_only=True)
        if 'model_state_dict' in state: state = state['model_state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        logger.info(f"  Loaded: {loaded} keys, new: {len(missing)}, skipped: {len(unexpected)}")
    else:
        logger.info("⚠️ No checkpoint found — training from scratch. Backbone weights random.")
        logger.info("  WARNING: ΔR₀ will not be preserved without pre-trained backbone!")

    amp_sc = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    mse = nn.MSELoss(); hub = nn.HuberLoss(delta=0.1); bce = nn.BCEWithLogitsLoss()

    # Pearson correlation loss for pen (prevents constant-output collapse)
    def pearson_pen_loss(pred, target):
        """1 - Pearson(pred, target) + variance penalty. Per-sample, averaged."""
        B = pred.shape[0]
        losses = []
        for b in range(B):
            p, t = pred[b], target[b]
            # Only use pens that have nonzero target variance
            mask = torch.ones_like(t, dtype=torch.bool)
            if mask.sum() < 3:
                losses.append(torch.tensor(0.0, device=pred.device))
                continue
            p_c = p - p.mean(); t_c = t - t.mean()
            cov = (p_c * t_c).sum()
            p_std = (p_c ** 2).sum().sqrt().clamp(min=1e-6)
            t_std = (t_c ** 2).sum().sqrt().clamp(min=1e-6)
            corr = cov / (p_std * t_std)
            # Pearson loss + variance regularizer
            var_pen = torch.relu(0.5 - p.std()).clamp(max=2.0)
            losses.append(1.0 - corr + 0.05 * var_pen)
        return torch.stack(losses).mean()

    # ════════════════════════════════════════════════════════
    # STAGE A: Freeze backbone, train ONLY pen meso (8 epochs)
    # ════════════════════════════════════════════════════════
    W = dict(LOSS_WEIGHTS)
    LAMBDA_PEN_A = 6.0

    for name, p in model.named_parameters():
        p.requires_grad = ('pen_meso' in name)

    pen_params = [p for p in model.parameters() if p.requires_grad]
    n_pen_params = sum(p.numel() for p in pen_params)
    logger.info(f"Stage A: Pen-only ({n_pen_params:,} params, {STAGE_A_EPOCHS}ep, λ_pen={LAMBDA_PEN_A}, lr=3e-4)")

    opt_a = torch.optim.AdamW(pen_params, lr=3e-4, weight_decay=1e-4)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=STAGE_A_EPOCHS * len(loader))

    model.train()
    for ep in range(STAGE_A_EPOCHS):
        tl = 0; nb = 0; t0 = time.time()
        for bi, batch in enumerate(loader):
            (nf, adj, inten, sl, tr, hsi, r0r, pd, ps, bd, res,
             pen_map, delta_r0, vacc_gain, pen_int) = [b.to(device, non_blocking=True) for b in batch]
            nf_sub = nf[:, ::T_SUBSAMPLE, :, :]
            opt_a.zero_grad(set_to_none=True)

            if amp_sc:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = model(nf_sub, adj, pen_map)
                    loss = LAMBDA_PEN_A * pearson_pen_loss(o["pen_risk_scores"], pen_int)
                amp_sc.scale(loss).backward()
                amp_sc.unscale_(opt_a)
                torch.nn.utils.clip_grad_norm_(pen_params, 1.0)
                amp_sc.step(opt_a); amp_sc.update()
            else:
                o = model(nf_sub, adj, pen_map)
                loss = LAMBDA_PEN_A * pearson_pen_loss(o["pen_risk_scores"], pen_int)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pen_params, 1.0)
                opt_a.step()

            sched_a.step()
            tl += loss.item(); nb += 1
            if bi % 40 == 0:
                with torch.no_grad():
                    l_pen = pearson_pen_loss(o["pen_risk_scores"], pen_int).item()
                logger.info(f"A{ep+1}/{STAGE_A_EPOCHS} B{bi}/{len(loader)} pen_corr_L:{l_pen:.4f}")

        elapsed = time.time() - t0
        logger.info(f"== STAGE A EP {ep+1} | pen_corr_L:{tl/nb:.4f} | {elapsed:.1f}s ==")

    # ════════════════════════════════════════════════════════
    # STAGE B: Pen-focused with all-loss supervision (backbone STILL FROZEN)
    # ════════════════════════════════════════════════════════
    LAMBDA_PEN_B = 3.0
    W["pen_risk"] = LAMBDA_PEN_B
    W["delta_R0"] = 4.0
    W["vaccination"] = 3.0

    # BACKBONE STAYS FROZEN — do NOT unfreeze
    # Only pen params remain trainable
    logger.info(f"Stage B: Pen + all-loss, backbone FROZEN ({STAGE_B_EPOCHS}ep, λ_pen={LAMBDA_PEN_B}, lr=5e-5)")

    opt_b = torch.optim.AdamW(pen_params, lr=5e-5, weight_decay=1e-4)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=STAGE_B_EPOCHS * len(loader))

    best_pen_corr = -1.0
    best_state = None
    abort_training = False

    model.train()
    for ep in range(STAGE_B_EPOCHS):
        if abort_training:
            break
        tl = 0; nb = 0; t0 = time.time()
        ep_dr0_corrs = []
        ep_pen_corrs = []

        for bi, batch in enumerate(loader):
            (nf, adj, inten, sl, tr, hsi, r0r, pd, ps, bd, res,
             pen_map, delta_r0, vacc_gain, pen_int) = [b.to(device, non_blocking=True) for b in batch]
            nf_sub = nf[:, ::T_SUBSAMPLE, :, :]
            opt_b.zero_grad(set_to_none=True)

            if amp_sc:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = model(nf_sub, adj, pen_map)
                    pen_loss = LAMBDA_PEN_B * pearson_pen_loss(o["pen_risk_scores"], pen_int)
                    loss = (
                        W["intensity"]    * mse(o["intensity"], inten) +
                        W["slope"]        * hub(o["slope"], sl) +
                        W["trend"]        * hub(o["trend"], tr) +
                        W["HSI"]          * mse(o["HSI"], hsi) +
                        W["R0_reduction"] * mse(o["R0_reduction"], r0r) +
                        W["peak_day"]     * mse(o["peak_day"], pd) +
                        W["peak_size"]    * mse(o["peak_size"], ps) +
                        W["breakdown"]    * bce(o["breakdown"], bd) +
                        W["resources"]    * mse(o["resources"], res) +
                        W["delta_R0"]     * mse(o["node_super_spreader_score"], delta_r0) +
                        pen_loss +
                        W["vaccination"]  * mse(o["optimal_vaccination_rank"], vacc_gain)
                    )
                amp_sc.scale(loss).backward()
                amp_sc.unscale_(opt_b)
                torch.nn.utils.clip_grad_norm_(pen_params, 1.0)
                amp_sc.step(opt_b); amp_sc.update()
            else:
                o = model(nf_sub, adj, pen_map)
                pen_loss = LAMBDA_PEN_B * pearson_pen_loss(o["pen_risk_scores"], pen_int)
                loss = (
                    W["intensity"]    * mse(o["intensity"], inten) +
                    W["slope"]        * hub(o["slope"], sl) +
                    W["trend"]        * hub(o["trend"], tr) +
                    W["HSI"]          * mse(o["HSI"], hsi) +
                    W["R0_reduction"] * mse(o["R0_reduction"], r0r) +
                    W["peak_day"]     * mse(o["peak_day"], pd) +
                    W["peak_size"]    * mse(o["peak_size"], ps) +
                    W["breakdown"]    * bce(o["breakdown"], bd) +
                    W["resources"]    * mse(o["resources"], res) +
                    W["delta_R0"]     * mse(o["node_super_spreader_score"], delta_r0) +
                    pen_loss +
                    W["vaccination"]  * mse(o["optimal_vaccination_rank"], vacc_gain)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pen_params, 1.0)
                opt_b.step()

            sched_b.step()
            tl += loss.item(); nb += 1

            # Track batch-level correlations for safety monitoring
            with torch.no_grad():
                for b_idx in range(delta_r0.shape[0]):
                    pr = o["node_super_spreader_score"][b_idx].cpu().numpy()
                    tr_ = delta_r0[b_idx].cpu().numpy()
                    valid = tr_.std() > 0 and pr.std() > 0
                    if valid:
                        from scipy.stats import pearsonr as _pr
                        c, _ = _pr(tr_[:50], pr[:50])  # fast subset
                        if not np.isnan(c): ep_dr0_corrs.append(c)
                    pp = o["pen_risk_scores"][b_idx].cpu().numpy()
                    tp = pen_int[b_idx].cpu().numpy()
                    if tp.std() > 0 and pp.std() > 0:
                        c, _ = _pr(tp, pp)
                        if not np.isnan(c): ep_pen_corrs.append(c)

            if bi % 40 == 0:
                with torch.no_grad():
                    l_dr0 = mse(o["node_super_spreader_score"], delta_r0).item()
                    l_pen = pearson_pen_loss(o["pen_risk_scores"], pen_int).item()
                logger.info(
                    f"B{ep+1}/{STAGE_B_EPOCHS} B{bi}/{len(loader)} L:{loss.item():.4f} "
                    f"[ΔR0_mse:{l_dr0:.4f} pen_corr_L:{l_pen:.4f}]"
                )

        ep_global = STAGE_A_EPOCHS + ep
        elapsed = time.time() - t0
        mean_dr0 = np.mean(ep_dr0_corrs) if ep_dr0_corrs else 0.0
        mean_pen = np.mean(ep_pen_corrs) if ep_pen_corrs else 0.0
        logger.info(
            f"== STAGE B EP {ep+1} (global {ep_global+1}) | L:{tl/nb:.4f} | {elapsed:.1f}s "
            f"| ΔR₀_corr:{mean_dr0:.4f} | pen_corr:{mean_pen:.4f} | α:{model.spectral_alpha.item():.4f} =="
        )

        # ── SAFETY ABORT: Stop if node performance degrades ──
        if mean_dr0 < 0.95 and len(ep_dr0_corrs) > 50:
            logger.info(f"🚨 ABORT: ΔR₀ dropped to {mean_dr0:.4f} (< 0.95). Reverting to best model.")
            abort_training = True

        # ── Save best model by pen correlation ──
        if mean_pen > best_pen_corr:
            best_pen_corr = mean_pen
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"⭐ New best pen_corr: {mean_pen:.4f}")

        # ── Drive checkpoint ──
        if (ep_global + 1) % 5 == 0 or ep == STAGE_B_EPOCHS - 1:
            ckpt = {"epoch": ep_global, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt_b.state_dict()}
            ckpt_path = f"{DRIVE_DIR}/v20_ep{ep_global+1}.pth"
            os.makedirs(DRIVE_DIR, exist_ok=True)
            torch.save(ckpt, ckpt_path)
            logger.info(f"💾 Checkpoint → {ckpt_path}")

    # ── Restore best model (by pen corr) ──
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"✅ Restored best model (pen_corr={best_pen_corr:.4f})")
    else:
        logger.info("⚠️ No best model recorded, using final weights.")

    # ── Step 5: Save Production Artifacts ──
    mc = model.to("cpu")

    # Model weights
    mp = os.path.join(OUT_DIR, "v20_herd_engine.pth")
    torch.save(mc.state_dict(), mp)

    # Config
    cfg = {
        "version": "20.6.3",
        "architecture": "ResGAT×2 → Branch(Herd|Node|PenMesoEncoder+StructInject+PenGAT) [backbone_frozen]",
        "node_dim": NODE_DIM, "gat_dim": GAT_DIM, "tft_dim": GAT_DIM,
        "pen_gat_heads": PEN_GAT_HEADS, "pen_loss": "pearson+var_reg",
        "struct_features": 8, "backbone_frozen": True,
        "n_gat_heads": GAT_HEADS, "n_gat_layers": GAT_LAYERS, "n_tft_heads": 4,
        "max_cows": MAX_COWS, "max_pens": MAX_PENS, "t_steps": T_STEPS,
        "loss_weights": LOSS_WEIGHTS,
        "heads": [
            "intensity", "slope", "trend", "HSI", "R0_reduction",
            "peak_day", "peak_size", "breakdown", "resources",
            "node_super_spreader_score", "pen_risk_scores", "optimal_vaccination_rank",
        ],
        "output_schema": {
            "herd_level": {
                "intensity": "float [0,1] — peak infection intensity",
                "slope": "float [-1,1] — infection trend slope",
                "trend": "float [-1,1] — 7-day trend",
                "HSI": "float [0,1] — herd stability index",
                "R0_reduction": "float [0,1] — spectral R0 reduction potential",
                "peak_day": "float — estimated peak day / 4",
                "peak_size": "float [0,1] — peak infection size",
                "breakdown": "logit — system breakdown probability",
                "resources": "float[3] — [milk_loss, antibiotic, isolation]",
            },
            "node_level": {
                "node_super_spreader_score": "float[N] [0,1] — ΔR₀ per node",
                "optimal_vaccination_rank": "float[N] [0,1] — vaccination priority",
            },
            "pen_level": {
                "pen_risk_scores": "float[P] — z-scored pen risk",
            },
        },
        "features": [
            "P_infection", "P_heat", "P_mastitis", "P_lameness", "P_calving",
            "severity", "collapse_risk", "hazard_slope", "hazard_integral_24h",
            "phase_onset", "phase_peak", "phase_recovery", "attention_entropy",
            "recovery_hazard", "milk_loss_est", "health_score",
            "vaccination_status", "pen_encoding",
            "degree_centrality", "betweenness_centrality",
            "eigenvector_centrality", "clustering_coefficient",
        ],
        "graph_families": ProductionSimulatorV5.GRAPH_FAMILIES,
        "training": {
            "farms": NUM_FARMS, "epochs": TOTAL_EPOCHS, "batch_size": BATCH_SIZE,
            "optimizer": "AdamW(lr=3e-4, wd=1e-4)",
            "scheduler": "OneCycleLR(max_lr=8e-4, pct_start=0.25)",
            "grad_clip": 1.5, "amp": True, "t_subsample": T_SUBSAMPLE,
        },
    }
    cp = os.path.join(OUT_DIR, "v20_herd_config.json")
    with open(cp, "w") as f: json.dump(cfg, f, indent=2)

    # Also save to Drive
    os.makedirs(DRIVE_DIR, exist_ok=True)
    torch.save(mc.state_dict(), f"{DRIVE_DIR}/v20_herd_engine.pth")
    with open(f"{DRIVE_DIR}/v20_herd_config.json", "w") as f: json.dump(cfg, f, indent=2)

    logger.info(f"✅ V20.6.3 Model  → {mp}")
    logger.info(f"✅ Config      → {cp}")
    logger.info(f"✅ Drive       → {DRIVE_DIR}/")
    logger.info("="*60)
    logger.info("files.download('models/cattle/v20_herd_engine.pth')")
    logger.info("files.download('models/cattle/v20_herd_config.json')")


if __name__ == "__main__":
    main()
