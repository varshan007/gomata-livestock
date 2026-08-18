#!/usr/bin/env python3
"""
evaluate_v20_decision_intelligence.py  (STANDALONE)
Phase 20.6 Evaluation: Mesoscopic Spectral Decision Engine
===========================================================
Self-contained. Supports v20.2, v20.4, v20.5, and v20.6 models.
Only needs: v20_herd_engine.pth + v20_herd_config.json
"""

import os, json, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

MAX_COWS = 150; MAX_PENS = 10; NODE_DIM = 22; T_STEPS = 28


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (universal: v20.2 / v20.4 / v20.5 / v20.6)
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
            GATLayer(din, self.hd if i < nh-1 else self.rem, drop) for i in range(nh)])
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


# ── Phase 20.6: Mesoscopic Pen Graph Encoder ──

class PenMesoEncoder(nn.Module):
    """Phase 20.6.2: Mesoscopic Pen Encoder with Structural Injection."""
    def __init__(self, gat_dim=96, max_pens=10, pen_heads=4, drop=0.1):
        super().__init__()
        self.max_pens = max_pens; self.gat_dim = gat_dim
        self.struct_mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, gat_dim))
        hd = gat_dim // pen_heads; rem = gat_dim - hd * (pen_heads - 1)
        self.pen_gat_heads = nn.ModuleList([
            GATLayer(gat_dim, hd if i < pen_heads - 1 else rem, drop)
            for i in range(pen_heads)])
        self.pen_norm = nn.LayerNorm(gat_dim)
        self.pen_head = nn.Sequential(
            nn.Linear(gat_dim, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))

    def _compute_struct_features(self, adj, pen_map):
        B, N, _ = adj.shape; P = self.max_pens
        pidx = pen_map.clamp(0, P - 1)
        feats = torch.zeros(B, P, 8, device=adj.device, dtype=adj.dtype)
        adj_bin = (adj > 0).float()
        total_edges = adj_bin.sum(dim=(1, 2)) / 2
        for b in range(B):
            for p in range(P):
                nip = (pidx[b] == p).nonzero(as_tuple=True)[0]; nc = len(nip)
                if nc < 1: continue
                feats[b, p, 0] = nc / max(N, 1)
                if nc < 2: continue
                sa = adj_bin[b][nip][:, nip]; deg_in = sa.sum(dim=1); e_int = sa.sum() / 2
                feats[b, p, 1] = e_int / max(total_edges[b].item(), 1)
                feats[b, p, 2] = deg_in.mean() / max(nc - 1, 1)
                feats[b, p, 3] = (e_int / max(nc*(nc-1)/2, 1)).clamp(0, 1)
                full_deg = adj_bin[b][nip].sum(dim=1); boundary = (full_deg - deg_in).sum() / 2
                feats[b, p, 4] = (boundary / max(nc*(N-nc), 1)).clamp(0, 1)
                vol = 2*e_int + boundary
                feats[b, p, 5] = (boundary / max(vol.item(), 1)).clamp(0, 1)
                if nc >= 3:
                    tri = torch.diag(sa @ sa @ sa) / 2; pairs = deg_in*(deg_in-1)/2
                    cc = torch.where(pairs > 0, tri/pairs, torch.zeros_like(tri))
                    feats[b, p, 6] = cc.mean().clamp(0, 1)
                    try:
                        L = torch.diag(deg_in) - sa; ev = torch.linalg.eigvalsh(L.float())
                        feats[b, p, 7] = (ev[1] - ev[0]).clamp(0, 5) / 5.0
                    except: feats[b, p, 7] = 0.0
        mu = feats.mean(dim=(0,1), keepdim=True); std = feats.std(dim=(0,1), keepdim=True).clamp(min=1e-6)
        return torch.nan_to_num(((feats - mu) / std).clamp(-3, 3), nan=0.0)

    def forward(self, H_node, pen_map, adj):
        B, N, d = H_node.shape; P = self.max_pens
        pidx = pen_map.clamp(0, P - 1); idx_exp = pidx.unsqueeze(-1).expand(-1, -1, d)
        pen_sum = torch.zeros(B, P, d, device=H_node.device, dtype=H_node.dtype)
        pen_sum.scatter_add_(1, idx_exp, H_node)
        pen_cnt = torch.zeros(B, P, 1, device=H_node.device, dtype=H_node.dtype)
        pen_cnt.scatter_add_(1, pidx.unsqueeze(-1),
                             torch.ones(B, N, 1, device=H_node.device, dtype=H_node.dtype))
        pen_emb = pen_sum / pen_cnt.clamp(min=1.0)
        struct_feats = self._compute_struct_features(adj, pen_map)
        pen_emb = pen_emb + self.struct_mlp(struct_feats)
        pen_oh = torch.zeros(B, N, P, device=adj.device, dtype=adj.dtype)
        pen_oh.scatter_(2, pidx.unsqueeze(-1), 1.0)
        A_pen = torch.bmm(pen_oh.transpose(1, 2), torch.bmm(adj, pen_oh))
        deg = A_pen.sum(dim=-1); dis = (deg + 1e-8).pow(-0.5)
        A_norm = torch.nan_to_num(A_pen * dis.unsqueeze(-1) * dis.unsqueeze(-2), nan=0.0)
        pen_out = torch.cat([h(pen_emb, A_norm) for h in self.pen_gat_heads], dim=-1)
        pen_out = self.pen_norm(pen_out + pen_emb)
        return self.pen_head(pen_out).squeeze(-1)


# ── Legacy: scatter_mean for v20.2/v20.4 ──

def scatter_mean_by_pen(H_node, pen_map, max_pens):
    B, N, d = H_node.shape
    H_pen = torch.zeros(B, max_pens, d, device=H_node.device, dtype=H_node.dtype)
    count = torch.zeros(B, max_pens, 1, device=H_node.device, dtype=H_node.dtype)
    pidx = pen_map.clamp(0, max_pens - 1)
    H_pen.scatter_add_(1, pidx.unsqueeze(-1).expand(-1,-1,d), H_node)
    count.scatter_add_(1, pidx.unsqueeze(-1), torch.ones(B, N, 1, device=H_node.device, dtype=H_node.dtype))
    return H_pen / (count + 1e-8)


class HerdEngineV20(nn.Module):
    """Universal model — loads 20.2, 20.4, 20.5, and 20.6."""
    def __init__(self, node_dim=22, gat_dim=128, tft_dim=128, ngh=6, nth=4,
                 n_gat_layers=3, max_pens=10, pen_heads=4):
        super().__init__()
        self.max_pens = max_pens; self.n_gat_layers = n_gat_layers
        self.gat1 = ResGAT(node_dim, gat_dim, ngh)
        self.gat2 = ResGAT(gat_dim, gat_dim, ngh)
        if n_gat_layers >= 3: self.gat3 = ResGAT(gat_dim, gat_dim, ngh)
        self.spectral_alpha = nn.Parameter(torch.tensor(0.1))
        self.spectral_proj = nn.Linear(4, gat_dim)
        self.pool = nn.Sequential(nn.Linear(gat_dim, gat_dim), nn.GELU())
        self.herd_pool = nn.Sequential(nn.Linear(gat_dim, gat_dim), nn.GELU())
        self.tft = TFTBlock(gat_dim, nth, ff=256)
        self.h_int = nn.Linear(gat_dim, 1); self.h_slope = nn.Linear(gat_dim, 1)
        self.h_trend = nn.Linear(gat_dim, 1)
        self.h_hsi = nn.Sequential(nn.Linear(gat_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.h_r0r = nn.Sequential(nn.Linear(gat_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.h_pkd = nn.Linear(gat_dim, 1); self.h_pks = nn.Linear(gat_dim, 1)
        self.h_bd = nn.Linear(gat_dim, 1); self.h_res = nn.Linear(gat_dim, 3)
        # Node (v20.2)
        self.h_node_ss = nn.Sequential(nn.Linear(gat_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.h_node_vacc = nn.Sequential(nn.Linear(gat_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # Node (v20.4+)
        self.node_ss = nn.Sequential(nn.Linear(gat_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.node_vacc = nn.Sequential(nn.Linear(gat_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # Pen (v20.2/v20.4 legacy)
        self.h_pen_risk = nn.Sequential(nn.Linear(gat_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        # Pen (v20.6 mesoscopic)
        self.pen_meso = PenMesoEncoder(gat_dim=gat_dim, max_pens=max_pens, pen_heads=pen_heads)
        self.log_vars = nn.Parameter(torch.zeros(12))
        self._version = "unknown"

    def _detect_version(self, state_dict):
        has_pen_meso = any('pen_meso.' in k for k in state_dict)
        has_pen_risk_net = any('pen_risk.net.' in k for k in state_dict)
        has_gat3 = any('gat3' in k for k in state_dict)
        has_node_ss_v4 = any(k.startswith('node_ss.') for k in state_dict)
        has_h_node_ss = any('h_node_ss' in k for k in state_dict)
        if has_pen_meso: return "20.6"
        if has_pen_risk_net: return "20.5"
        if has_node_ss_v4 and not has_gat3: return "20.4"
        if has_gat3 and has_h_node_ss: return "20.2"
        return "20.2"

    def forward(self, ns, adj, pen_map=None):
        B, T, N, F = ns.shape; ht = []
        for t in range(T):
            h = ns[:, t]
            h = tF.elu(self.gat1(h, adj)); h = tF.elu(self.gat2(h, adj))
            if self.n_gat_layers >= 3 and hasattr(self, 'gat3'):
                h = tF.elu(self.gat3(h, adj))
            if self._version in ("20.4", "20.5", "20.6"):
                h = h + self.spectral_alpha * self.spectral_proj(ns[:, t, :, -4:])
                ht.append(self.herd_pool(h.mean(dim=1)))
            else:
                ht.append(self.pool(h.mean(dim=1)))
        H_node = h
        ctx = self.tft(torch.stack(ht, dim=1))
        result = {
            "intensity": self.h_int(ctx).squeeze(-1), "slope": self.h_slope(ctx).squeeze(-1),
            "trend": self.h_trend(ctx).squeeze(-1), "HSI": self.h_hsi(ctx).squeeze(-1),
            "R0_reduction": self.h_r0r(ctx).squeeze(-1), "peak_day": self.h_pkd(ctx).squeeze(-1),
            "peak_size": self.h_pks(ctx).squeeze(-1), "breakdown": self.h_bd(ctx).squeeze(-1),
            "resources": self.h_res(ctx),
        }
        if self._version in ("20.4", "20.5", "20.6"):
            result["node_super_spreader_score"] = self.node_ss(H_node).squeeze(-1)
            result["optimal_vaccination_rank"] = self.node_vacc(H_node).squeeze(-1)
        else:
            result["node_super_spreader_score"] = self.h_node_ss(H_node).squeeze(-1)
            result["optimal_vaccination_rank"] = self.h_node_vacc(H_node).squeeze(-1)
        if pen_map is not None:
            if self._version == "20.6":
                result["pen_risk_scores"] = self.pen_meso(H_node, pen_map, adj)
            elif self._version == "20.5":
                # v20.5 legacy — not used in 20.6
                H_pen = scatter_mean_by_pen(H_node, pen_map, self.max_pens)
                result["pen_risk_scores"] = self.h_pen_risk(H_pen).squeeze(-1)
            else:
                H_pen = scatter_mean_by_pen(H_node, pen_map, self.max_pens)
                result["pen_risk_scores"] = self.h_pen_risk(H_pen).squeeze(-1)
        else:
            result["pen_risk_scores"] = torch.zeros(B, self.max_pens, device=ns.device)
        return result


# ═══════════════════════════════════════════════════════════════
# SIMULATOR (z-scored pen targets)
# ═══════════════════════════════════════════════════════════════

class EvalSimulator:
    def __init__(self, seed=9999):
        self.rng = np.random.RandomState(seed)

    def _hub_graph(self, N, n_pens):
        pen = self.rng.randint(0, n_pens, N)
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
                    w = self.rng.uniform(0.3, 0.8)
                    A[h, t] = max(A[h, t], w); A[t, h] = A[h, t]
        return A, pen

    def _compute_spectral_priors(self, A):
        N = A.shape[0]; adj_bin = (A > 0).astype(float); deg = adj_bin.sum(axis=1)
        deg_norm = deg / (N - 1 + 1e-8)
        D_inv = np.diag(1.0 / (deg + 1e-8)); P = D_inv @ A
        pr = np.ones(N) / N
        for _ in range(10): pr = 0.85 * (P.T @ pr) + 0.15 / N
        betw = pr / (pr.max() + 1e-8)
        try:
            evals, evecs = np.linalg.eigh(A)
            eig_c = np.abs(evecs[:, -1]); eig_c /= (eig_c.max() + 1e-8)
        except: eig_c = deg_norm.copy()
        tri = np.diag(adj_bin @ adj_bin @ adj_bin) / 2
        pairs = deg * (deg - 1) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            clust = np.where(pairs > 0, tri / pairs, 0).astype(np.float32)
        return deg_norm.astype(np.float32), betw.astype(np.float32), eig_c.astype(np.float32), clust

    def _spectral_R0(self, A, beta, gamma):
        if gamma <= 0: return 0.0
        try: return float(np.max(np.abs(np.linalg.eigvalsh((beta / gamma) * A))))
        except: return 0.0

    def _graph_entropy(self, A):
        d = (A > 0).sum(axis=1).astype(float); t = d.sum()
        if t == 0: return 0.0
        p = d / t; p = p[p > 0]; return float(-np.sum(p * np.log(p + 1e-12)))

    def _compute_HSI(self, I_obs, A, mean_I):
        s2 = float(np.clip(np.var(mean_I)*10, 0, 1))
        dI = np.diff(mean_I) if len(mean_I) > 1 else np.array([0.0])
        g = float(np.clip(np.mean(np.abs(dI))*20, 0, 1))
        d2I = np.diff(dI) if len(dI) > 1 else np.array([0.0])
        a = float(np.clip(np.mean(np.abs(d2I))*40, 0, 1))
        c = float(np.clip(np.mean(1.0 - I_obs.mean(axis=1)), 0, 1))
        H = self._graph_entropy(A); Hm = np.log(A.shape[0]) if A.shape[0] > 1 else 1.0
        base = float(np.clip(0.30*(1-s2)+0.30*(1-g)+0.20*(1-a)+0.10*c+0.10*np.clip(H/Hm,0,1), 0, 1))
        return float(np.clip(base * (1.0 - float(np.max(mean_I))**2), 0, 1))

    def _delta_R0_per_node(self, A, beta, gamma):
        N = A.shape[0]
        if gamma <= 0: return np.zeros(N, dtype=np.float32)
        try:
            K = (beta / gamma) * A; _, evecs = np.linalg.eigh(K)
            v = np.abs(evecs[:, -1]); d = A.sum(axis=1)
            delta = v ** 2 * d * (beta / gamma)
        except: delta = np.zeros(N, dtype=np.float32)
        mx = delta.max()
        if mx > 0: delta /= mx
        return delta.astype(np.float32)

    def _pen_intensity_zscore(self, I_obs, pen, n_pens):
        mi = I_obs.mean(axis=0); raw = np.zeros(n_pens, dtype=np.float32)
        for p in range(n_pens):
            idx = np.where(pen == p)[0]
            if len(idx) > 0: raw[p] = float(mi[idx].mean())
        return ((raw - raw.mean()) / (raw.std() + 1e-8)).astype(np.float32)

    def _vaccination_gain(self, A, beta, gamma, N):
        if gamma <= 0: return np.zeros(N, dtype=np.float32)
        try:
            K = (beta / gamma) * A; _, evecs = np.linalg.eigh(K)
            v = np.abs(evecs[:, -1]); d = A.sum(axis=1); scores = v ** 2 * d
        except: scores = np.zeros(N, dtype=np.float32)
        mx = scores.max()
        if mx > 0: scores /= mx
        return scores.astype(np.float32)

    def simulate_farm(self, fidx):
        n_cows = self.rng.randint(50, 100); n_pens = self.rng.randint(4, 8)
        A, pen = self._hub_graph(n_cows, n_pens)
        regime = self.rng.choice(['stable','borderline','outbreak','superspreader'], p=[0.35,0.25,0.30,0.10])
        if regime == 'stable': beta=self.rng.uniform(0.01,0.03); gamma=self.rng.uniform(0.15,0.30); n_seed=self.rng.randint(1,3); seed_t=0
        elif regime == 'borderline': beta=self.rng.uniform(0.03,0.055); gamma=self.rng.uniform(0.08,0.15); n_seed=self.rng.randint(2,5); seed_t=self.rng.randint(5,15)
        elif regime == 'outbreak': beta=self.rng.uniform(0.055,0.12); gamma=self.rng.uniform(0.04,0.08); n_seed=self.rng.randint(2,6); seed_t=self.rng.randint(5,18)
        else: beta=self.rng.uniform(0.12,0.25); gamma=self.rng.uniform(0.03,0.06); n_seed=self.rng.randint(3,8); seed_t=self.rng.randint(5,15)
        vaccinated = np.zeros(n_cows, dtype=np.float32)
        nv = int(self.rng.uniform(0, 0.25) * n_cows)
        if nv > 0: vaccinated[self.rng.choice(n_cows, nv, replace=False)] = 1.0
        I = np.zeros((T_STEPS, n_cows), dtype=np.float32); S = np.ones((T_STEPS, n_cows), dtype=np.float32)
        severity = np.zeros((T_STEPS, n_cows), dtype=np.float32)
        seeds = self.rng.choice(n_cows, min(n_seed, n_cows), replace=False)
        I[seed_t, seeds] = self.rng.uniform(0.3, 0.7, len(seeds)); S[seed_t, seeds] = 1.0 - I[seed_t, seeds]
        ah = self.rng.uniform(0.02, 0.06); bt = self.rng.uniform(68, 85)
        for t in range(max(1, seed_t+1), T_STEPS):
            te = max(0, bt + 3*np.sin(t*2*np.pi/28) - 72); be = beta * (1 + ah * te)
            Ae = A * (1 - vaccinated[np.newaxis, :] * 0.8)
            ni = np.clip(be * (Ae @ I[t-1]) * S[t-1], 0, S[t-1]); nr = gamma * I[t-1]
            S[t] = np.clip(S[t-1] - ni, 0, 1); I[t] = np.clip(I[t-1] + ni - nr, 0, 1)
            severity[t] = I[t] * (1 + 0.2 * te / 10)
        deg_n, betw, eig_c, clust = self._compute_spectral_priors(A)
        nf = np.zeros((T_STEPS, n_cows, NODE_DIM), dtype=np.float32)
        for t in range(T_STEPS):
            te = max(0, bt + 3*np.sin(t*2*np.pi/28) - 72)
            for i in range(n_cows):
                nf[t, i] = [I[t,i], float(te>5)*0.3+self.rng.normal(0,0.03),
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
                    deg_n[i], betw[i], eig_c[i], clust[i]]
        mI = I.mean(axis=1); intensity = float(mI.max())
        x = np.arange(T_STEPS, dtype=np.float32); xc = x - x.mean()
        slope = float((xc * mI).sum() / ((xc**2).sum() + 1e-8))
        trend = float(mI[-1] - mI[max(0, T_STEPS-7)])
        dI_max = float(np.max(np.abs(np.diff(mI))))
        hsi = self._compute_HSI(I, A, mI)
        ba = beta * (1 + ah * max(0, bt - 72)); r0b = self._spectral_R0(A, ba, gamma)
        wd = A.sum(axis=1); isc = I.mean(axis=0); comb = wd * (1 + 5 * isc)
        nr_n = max(3, int(0.10 * n_cows)); tn = np.argsort(comb)[-nr_n:][::-1]
        A2 = A.copy(); A2[tn,:] = 0; A2[:,tn] = 0; r0p = self._spectral_R0(A2, ba, gamma)
        r0r = float((r0b - r0p) / r0b) if r0b > 0 else 0
        outbreak = float(intensity > 0.15)
        bd = float((intensity > 0.65) and (hsi < 0.65) and (dI_max > 0.08))
        pk = int(np.argmax(mI)); pkd = float(pk/4.0); pks = float(mI[pk])
        delta_r0 = self._delta_R0_per_node(A, ba, gamma)
        pen_int = self._pen_intensity_zscore(I, pen, n_pens)
        vacc_gain = self._vaccination_gain(A, ba, gamma, n_cows)
        ml = float((I*10).sum()/n_cows); ab = float((I>0.3).sum())/n_cows
        iso = float((severity>1.0).any(axis=0).sum())/n_cows
        return {"node_features": nf, "adjacency": A, "n_cows": n_cows,
                "pen_mapping": pen, "n_pens": n_pens,
                "labels": {"intensity": intensity, "slope": float(np.clip(slope,-1,1)),
                    "trend": float(np.clip(trend,-1,1)), "HSI": hsi,
                    "R0_reduction": float(np.clip(r0r, 0, 1)), "outbreak": outbreak,
                    "peak_day": pkd, "peak_size": float(np.clip(pks,0,1)), "breakdown": bd,
                    "milk_loss": ml, "antibiotic": ab, "isolation": iso,
                    "delta_R0_per_node": delta_r0, "pen_intensity": pen_int,
                    "vaccination_gain": vacc_gain}}


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_v20(model_path, config_path, n_farms=200, seed=9999):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("🧠 PHASE 20.6: MESOSCOPIC SPECTRAL ENGINE EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    with open(config_path, 'r') as f: cfg = json.load(f)
    version = cfg.get('version', '20.2')
    max_cows = cfg.get('max_cows', MAX_COWS); max_pens = cfg.get('max_pens', MAX_PENS)
    gat_dim = cfg.get('gat_dim', 128); n_gat_layers = cfg.get('n_gat_layers', 3)
    pen_heads = cfg.get('pen_gat_heads', 4)
    print(f"Config: v{version} | GAT {gat_dim}d×{n_gat_layers}L | PenGAT {pen_heads}h | max_cows {max_cows}")

    model = HerdEngineV20(node_dim=cfg.get('node_dim', NODE_DIM), gat_dim=gat_dim,
        tft_dim=cfg.get('tft_dim', gat_dim), ngh=cfg.get('n_gat_heads', 6),
        n_gat_layers=n_gat_layers, max_pens=max_pens, pen_heads=pen_heads).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    model._version = model._detect_version(state)
    print(f"Detected model version: {model._version}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    important = [k for k in missing if not any(x in k for x in
        ['node_ss.','node_vacc.','pen_risk','pen_meso','herd_pool.','pen_pooler.',
         'h_node_ss.','h_node_vacc.','h_pen_risk.','pool.','spectral_alpha',
         'spectral_proj','gat3','log_vars'])]
    if important: print(f"⚠️  Missing: {important[:5]}")
    if unexpected: print(f"ℹ️  Extra ignored: {len(unexpected)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    model.eval()

    sim = EvalSimulator(seed=seed)
    y_ob_t, y_ob_p, y_bd_t, y_bd_p = [], [], [], []
    y_int_t, y_int_p, y_hsi_t, y_hsi_p = [], [], [], []
    dr0_c, pen_c, vacc_s = [], [], []

    print(f"\nEvaluating {n_farms} farms (seed={seed})...")
    with torch.no_grad():
        for i in range(n_farms):
            data = sim.simulate_farm(i); L = data['labels']
            nc = min(data['n_cows'], max_cows); np_ = data['n_pens']
            nfp = np.zeros((T_STEPS, max_cows, NODE_DIM), dtype=np.float32)
            ap = np.zeros((max_cows, max_cows), dtype=np.float32)
            pm = np.zeros(max_cows, dtype=np.int64)
            nfp[:,:nc,:] = data['node_features'][:,:nc,:]; ap[:nc,:nc] = data['adjacency'][:nc,:nc]
            pm[:nc] = data['pen_mapping'][:nc]
            nft = torch.tensor(nfp).unsqueeze(0).to(device)
            at = torch.tensor(ap).unsqueeze(0).to(device)
            pt = torch.tensor(pm).unsqueeze(0).to(device)
            o = model(nft, at, pt)
            y_ob_t.append(L['outbreak']); y_ob_p.append(o['intensity'].item())
            y_bd_t.append(L['breakdown']); y_bd_p.append(torch.sigmoid(o['breakdown']).item())
            y_int_t.append(L['intensity']); y_int_p.append(o['intensity'].item())
            y_hsi_t.append(L['HSI']); y_hsi_p.append(o['HSI'].item())
            ps = o['node_super_spreader_score'][0,:nc].cpu().numpy(); td = L['delta_R0_per_node'][:nc]
            if td.std()>0 and ps.std()>0:
                c,_ = pearsonr(td, ps)
                if not np.isnan(c): dr0_c.append(c)
            pp = o['pen_risk_scores'][0,:np_].cpu().numpy(); tp = L['pen_intensity'][:np_]
            if tp.std()>0 and pp.std()>0:
                c,_ = pearsonr(tp, pp)
                if not np.isnan(c): pen_c.append(c)
            pv = o['optimal_vaccination_rank'][0,:nc].cpu().numpy(); tv = L['vaccination_gain'][:nc]
            if tv.std()>0 and pv.std()>0:
                s,_ = spearmanr(tv, pv)
                if not np.isnan(s): vacc_s.append(s)
            if (i+1)%50==0: print(f"  {i+1}/{n_farms}")

    oa = roc_auc_score(y_ob_t, y_ob_p) if len(set(y_ob_t))>1 else 1.0
    ba = roc_auc_score(y_bd_t, y_bd_p) if len(set(y_bd_t))>1 else 1.0
    ic = pearsonr(y_int_t, y_int_p)[0]; hc = pearsonr(y_hsi_t, y_hsi_p)[0]
    md = np.mean(dr0_c) if dr0_c else 0; mp = np.mean(pen_c) if pen_c else 0
    mv = np.mean(vacc_s) if vacc_s else 0
    sd = np.std(dr0_c) if dr0_c else 0; sp = np.std(pen_c) if pen_c else 0
    sv = np.std(vacc_s) if vacc_s else 0
    wp = min(pen_c) if pen_c else 0

    def tag(v,t): return "✅" if v>=t else "⚠️"
    print("\n" + "="*60); print("📊 EVALUATION RESULTS"); print("="*60)
    print("\n── HERD-LEVEL ──")
    print(f"  {tag(oa,0.97)} Outbreak AUC:      {oa:.4f}   (≥ 0.97)")
    print(f"  {tag(ba,0.95)} Breakdown AUC:     {ba:.4f}   (≥ 0.95)")
    print(f"  {tag(ic,0.90)} Intensity Corr:    {ic:.4f}   (≥ 0.90)")
    print(f"  {tag(hc,0.90)} HSI Corr:          {hc:.4f}   (≥ 0.90)")
    print("\n── NODE-LEVEL ──")
    print(f"  {tag(md,0.90)} ΔR₀ Pearson:       {md:.4f} ± {sd:.4f}  (≥ 0.90)")
    print(f"     Samples: {len(dr0_c)}/{n_farms}")
    print(f"  {tag(mv,0.85)} Vacc Spearman:     {mv:.4f} ± {sv:.4f}  (≥ 0.85)")
    print(f"     Samples: {len(vacc_s)}/{n_farms}")
    print("\n── PEN-LEVEL ──")
    print(f"  {tag(mp,0.80)} Pen Risk Corr:     {mp:.4f} ± {sp:.4f}  (≥ 0.80)")
    print(f"     Worst-case:         {wp:.4f}  (≥ 0.55)")
    print(f"     Samples: {len(pen_c)}/{n_farms}")
    print("\n" + "="*60)
    all_ok = oa>=0.97 and ba>=0.95 and ic>=0.90 and hc>=0.90 and md>=0.90 and mp>=0.80 and mv>=0.85
    if all_ok: print("🏆 PRODUCTION GRADE — All 7 targets met.")
    elif mp >= 0.80: print("✅ PEN UPGRADE SUCCESSFUL.")
    elif oa >= 0.90: print("✅ HERD+NODE OK — Pen needs more training.")
    else: print("⚠️  NEEDS TRAINING.")
    print("="*60)


if __name__ == "__main__":
    paths = [
        ("models/cattle/v20_herd_engine.pth", "models/cattle/v20_herd_config.json"),
        ("v20_herd_engine.pth", "v20_herd_config.json"),
        ("/content/models/cattle/v20_herd_engine.pth", "/content/models/cattle/v20_herd_config.json"),
        ("/content/drive/MyDrive/HerdV20/v20_herd_engine.pth",
         "/content/drive/MyDrive/HerdV20/v20_herd_config.json"),
    ]
    for mp, cp in paths:
        if os.path.exists(mp) and os.path.exists(cp):
            print(f"Found: {mp}"); evaluate_v20(mp, cp, n_farms=200); break
    else:
        print("❌ Model files not found.")
        for mp, cp in paths: print(f"   {mp}")
