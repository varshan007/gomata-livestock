#!/usr/bin/env python3
"""
colab_train_v21_backbone.py — Phase 21
SPECTRAL BACKBONE RECONSTITUTION (GoMata T4)
============================================================
Clean rebuild from first principles. No pen modules.

Architecture:
  NodeEncoder(22 → 96) → GAT×2(96d, 4h, residual) → AttentionPool
  → Independent MLP heads: ΔR₀, Vacc, Outbreak, Breakdown, Intensity, HSI

Training:
  35 epochs, AdamW lr=3e-4, CosineAnnealingLR + 2ep warmup
  Early stop when ΔR₀≥0.98 AND Vacc≥0.90 AND Outbreak AUC≥0.97
  Best model saved by ΔR₀ Pearson

Targets:
  ΔR₀ Pearson ≥ 0.98   | Vacc Spearman ≥ 0.90
  Outbreak AUC ≥ 0.97   | Breakdown AUC ≥ 0.95
"""

import os, sys, gc, json, math, time, logging, warnings
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# ── Deterministic ──
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Constants ──
NUM_FARMS   = 2000
MAX_COWS    = 150
NODE_DIM    = 22   # 18 original + 4 spectral priors
GAT_DIM     = 96
GAT_HEADS   = 4
GAT_LAYERS  = 2
EPOCHS      = 35
WARMUP_EP   = 2
BATCH_SIZE  = 8
T_STEPS     = 28
T_SUBSAMPLE = 2
OUT_DIR     = "models/cattle"
DRIVE_DIR   = "/content/drive/MyDrive/HerdV21"
os.makedirs(OUT_DIR, exist_ok=True)

LOSS_WEIGHTS = {
    "outbreak":  1.0,
    "breakdown": 1.0,
    "intensity": 1.0,
    "HSI":       1.0,
    "delta_R0":  3.0,
    "vaccination": 2.0,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V21")


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """Single-head graph attention layer."""
    def __init__(self, din, dout, drop=0.1):
        super().__init__()
        self.W = nn.Linear(din, dout, bias=False)
        self.a = nn.Linear(2 * dout, 1, bias=False)
        self.lk = nn.LeakyReLU(0.2)
        self.dp = nn.Dropout(drop)

    def forward(self, h, adj):
        Wh = self.W(h)
        N = Wh.size(1)
        a_in = torch.cat([
            Wh.unsqueeze(2).expand(-1, -1, N, -1),
            Wh.unsqueeze(1).expand(-1, N, -1, -1)
        ], dim=-1)
        e = self.lk(self.a(a_in).squeeze(-1))
        mask = (adj == 0)
        e = e.masked_fill(mask, -6e4)
        a_w = self.dp(torch.softmax(e, dim=-1))
        return a_w @ Wh


class ResGAT(nn.Module):
    """Multi-head GAT with residual connection."""
    def __init__(self, din, dout, nh=4, drop=0.1):
        super().__init__()
        self.hd = dout // nh
        self.rem = dout - self.hd * (nh - 1)
        self.heads = nn.ModuleList([
            GATLayer(din, self.hd if i < nh - 1 else self.rem, drop)
            for i in range(nh)
        ])
        self.norm = nn.LayerNorm(dout)
        self.proj = nn.Linear(din, dout) if din != dout else nn.Identity()

    def forward(self, h, adj):
        out = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        return self.norm(out + self.proj(h))


class AttentionPool(nn.Module):
    """Attention-weighted node pooling → single herd vector."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, h):
        # h: (B, N, d)
        w = torch.softmax(self.attn(h), dim=1)  # (B, N, 1)
        return (w * h).sum(dim=1)  # (B, d)


def make_head(din, dh, dout):
    """Independent MLP head."""
    return nn.Sequential(
        nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dout)
    )


class HerdEngineV21_Backbone(nn.Module):
    """Phase 21 — Clean spectral backbone. No pen modules."""

    def __init__(self, node_dim=NODE_DIM, gat_dim=GAT_DIM, ngh=GAT_HEADS):
        super().__init__()
        # Node encoder
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, gat_dim), nn.GELU(), nn.LayerNorm(gat_dim)
        )
        # 2-layer GAT backbone
        self.gat1 = ResGAT(gat_dim, gat_dim, ngh, drop=0.1)
        self.gat2 = ResGAT(gat_dim, gat_dim, ngh, drop=0.1)
        # Attention pooling
        self.pool = AttentionPool(gat_dim)

        # Node-level heads (operate on per-node embeddings)
        self.head_dr0 = make_head(gat_dim, 64, 1)
        self.head_vacc = make_head(gat_dim, 64, 1)

        # Herd-level heads (operate on pooled herd embedding)
        self.head_outbreak = make_head(gat_dim, 64, 1)
        self.head_breakdown = make_head(gat_dim, 64, 1)
        self.head_intensity = make_head(gat_dim, 64, 1)
        self.head_hsi = make_head(gat_dim, 64, 1)

    def forward(self, ns, adj):
        """
        ns: (B, T, N, F)  node features over time
        adj: (B, N, N)    adjacency
        Returns dict of predictions.
        """
        B, T, N, num_feat = ns.shape

        # Process each timestep through backbone, collect herd embeddings
        herd_embs = []
        for t in range(T):
            h = self.node_enc(ns[:, t])         # (B, N, d)
            h = F.elu(self.gat1(h, adj))        # (B, N, d)
            h = F.elu(self.gat2(h, adj))        # (B, N, d)
            herd_embs.append(self.pool(h))      # (B, d)

        # Node embeddings from last timestep
        H_node = h  # (B, N, d)

        # Herd embedding: mean of temporal pooled embeddings
        H_herd = torch.stack(herd_embs, dim=1).mean(dim=1)  # (B, d)

        return {
            "delta_R0": self.head_dr0(H_node).squeeze(-1),       # (B, N)
            "vacc_rank": self.head_vacc(H_node).squeeze(-1),     # (B, N)
            "outbreak": self.head_outbreak(H_herd).squeeze(-1),  # (B,)
            "breakdown": self.head_breakdown(H_herd).squeeze(-1),# (B,)
            "intensity": self.head_intensity(H_herd).squeeze(-1),# (B,)
            "HSI": self.head_hsi(H_herd).squeeze(-1),           # (B,)
        }


# ═══════════════════════════════════════════════════════════════
# PAIRWISE RANKING LOSS
# ═══════════════════════════════════════════════════════════════

def pairwise_ranking_loss(pred, target, n_pairs=50):
    """Differentiable pairwise ranking loss for vaccination priority."""
    B, N = pred.shape
    losses = []
    for b in range(B):
        p, t = pred[b], target[b]
        # Only consider nodes with nonzero target
        valid = (t > 0.01)
        if valid.sum() < 5:
            losses.append(torch.tensor(0.0, device=pred.device))
            continue
        vi = valid.nonzero(as_tuple=True)[0]
        n = min(n_pairs, len(vi) * (len(vi) - 1) // 2)
        if n < 1:
            losses.append(torch.tensor(0.0, device=pred.device))
            continue
        # Sample pairs
        idx = torch.randint(0, len(vi), (n, 2), device=pred.device)
        i, j = vi[idx[:, 0]], vi[idx[:, 1]]
        # Target ordering
        t_diff = t[i] - t[j]
        p_diff = p[i] - p[j]
        # Margin ranking: if t[i] > t[j], then p[i] should > p[j]
        sign = torch.sign(t_diff)
        loss = torch.relu(0.5 - sign * p_diff).mean()
        losses.append(loss)
    return torch.stack(losses).mean()


# ═══════════════════════════════════════════════════════════════
# SIMULATOR (same proven simulator from v20)
# ═══════════════════════════════════════════════════════════════

class FarmSimulator:
    """Generates synthetic farm graphs with epidemiological dynamics."""

    FAMILIES = ['hub', 'community', 'small_world', 'scale_free',
                'erdos_renyi', 'clustered', 'bipartite', 'multi_hub']

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def _make_graph(self, N, family):
        A = np.zeros((N, N), dtype=np.float32)
        if family == 'hub':
            n_hubs = max(2, int(0.08 * N))
            hubs = self.rng.choice(N, n_hubs, replace=False)
            for h in hubs:
                nn_ = self.rng.randint(int(0.3 * N), int(0.6 * N))
                tgts = self.rng.choice(N, min(nn_, N - 1), replace=False)
                for t in tgts:
                    if t != h:
                        w = self.rng.uniform(0.3, 1.0)
                        A[h, t] = w; A[t, h] = w
        elif family == 'community':
            n_c = self.rng.randint(3, 6)
            assign = self.rng.randint(0, n_c, N)
            for i in range(N):
                for j in range(i + 1, N):
                    p = 0.4 if assign[i] == assign[j] else 0.02
                    if self.rng.random() < p:
                        w = self.rng.uniform(0.2, 0.8)
                        A[i, j] = w; A[j, i] = w
        elif family == 'small_world':
            k = self.rng.randint(4, 8)
            for i in range(N):
                for d in range(1, k // 2 + 1):
                    j = (i + d) % N
                    w = self.rng.uniform(0.3, 0.8)
                    A[i, j] = w; A[j, i] = w
            for i in range(N):
                for j in range(i + 1, N):
                    if A[i, j] > 0 and self.rng.random() < 0.1:
                        A[i, j] = 0; A[j, i] = 0
                        r = self.rng.randint(0, N)
                        if r != i:
                            w = self.rng.uniform(0.3, 0.8)
                            A[i, r] = w; A[r, i] = w
        elif family == 'scale_free':
            m = self.rng.randint(2, 5)
            for i in range(m, N):
                deg = A[:i].sum(axis=1) + 1
                p = deg / deg.sum()
                tgts = self.rng.choice(i, min(m, i), replace=False, p=p)
                for t in tgts:
                    w = self.rng.uniform(0.3, 0.9)
                    A[i, t] = w; A[t, i] = w
        elif family == 'erdos_renyi':
            p = self.rng.uniform(0.05, 0.15)
            for i in range(N):
                for j in range(i + 1, N):
                    if self.rng.random() < p:
                        w = self.rng.uniform(0.2, 0.8)
                        A[i, j] = w; A[j, i] = w
        elif family == 'clustered':
            nc = self.rng.randint(3, 7)
            centers = self.rng.choice(N, nc, replace=False)
            assign = np.argmin(np.abs(np.arange(N)[:, None] - centers[None, :]), axis=1)
            for i in range(N):
                for j in range(i + 1, N):
                    p = 0.5 if assign[i] == assign[j] else 0.01
                    if self.rng.random() < p:
                        w = self.rng.uniform(0.3, 0.9)
                        A[i, j] = w; A[j, i] = w
        elif family == 'bipartite':
            g = self.rng.randint(0, 2, N)
            for i in range(N):
                for j in range(i + 1, N):
                    if g[i] != g[j] and self.rng.random() < 0.12:
                        w = self.rng.uniform(0.2, 0.7)
                        A[i, j] = w; A[j, i] = w
        elif family == 'multi_hub':
            nh = self.rng.randint(3, 7)
            hubs = self.rng.choice(N, nh, replace=False)
            for h in hubs:
                nc = self.rng.randint(int(0.1 * N), int(0.3 * N))
                tgts = self.rng.choice(N, nc, replace=False)
                for t in tgts:
                    if t != h:
                        w = self.rng.uniform(0.3, 0.9)
                        A[h, t] = w; A[t, h] = w
            for i, h1 in enumerate(hubs):
                for h2 in hubs[i + 1:]:
                    w = self.rng.uniform(0.5, 1.0)
                    A[h1, h2] = w; A[h2, h1] = w
        return A

    def _spectral_priors(self, A):
        N = A.shape[0]
        ab = (A > 0).astype(float)
        deg = ab.sum(axis=1)
        deg_n = deg / (N - 1 + 1e-8)
        D_inv = np.diag(1.0 / (deg + 1e-8))
        P = D_inv @ A
        pr = np.ones(N) / N
        for _ in range(10):
            pr = 0.85 * (P.T @ pr) + 0.15 / N
        betw = pr / (pr.max() + 1e-8)
        try:
            _, evecs = np.linalg.eigh(A)
            eig_c = np.abs(evecs[:, -1])
            eig_c /= (eig_c.max() + 1e-8)
        except:
            eig_c = deg_n.copy()
        tri = np.diag(ab @ ab @ ab) / 2
        pairs = deg * (deg - 1) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            clust = np.where(pairs > 0, tri / pairs, 0).astype(np.float32)
        return deg_n.astype(np.float32), betw.astype(np.float32), eig_c.astype(np.float32), clust

    def _spectral_R0(self, A, beta, gamma):
        if gamma <= 0: return 0.0
        try:
            return float(np.max(np.abs(np.linalg.eigvalsh((beta / gamma) * A))))
        except:
            return 0.0

    def _delta_R0(self, A, beta, gamma):
        N = A.shape[0]
        if gamma <= 0: return np.zeros(N, dtype=np.float32)
        try:
            K = (beta / gamma) * A
            _, evecs = np.linalg.eigh(K)
            v = np.abs(evecs[:, -1])
            d = A.sum(axis=1)
            delta = v ** 2 * d * (beta / gamma)
        except:
            delta = np.zeros(N, dtype=np.float32)
        mx = delta.max()
        if mx > 0: delta /= mx
        return delta.astype(np.float32)

    def _vacc_gain(self, A, beta, gamma, N):
        if gamma <= 0: return np.zeros(N, dtype=np.float32)
        try:
            K = (beta / gamma) * A
            _, evecs = np.linalg.eigh(K)
            v = np.abs(evecs[:, -1])
            d = A.sum(axis=1)
            scores = v ** 2 * d
        except:
            scores = np.zeros(N, dtype=np.float32)
        mx = scores.max()
        if mx > 0: scores /= mx
        return scores.astype(np.float32)

    def _graph_entropy(self, A):
        d = (A > 0).sum(axis=1).astype(float)
        t = d.sum()
        if t == 0: return 0.0
        p = d / t; p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _compute_HSI(self, I, A, mI):
        s2 = float(np.clip(np.var(mI) * 10, 0, 1))
        dI = np.diff(mI) if len(mI) > 1 else np.array([0.0])
        g = float(np.clip(np.mean(np.abs(dI)) * 20, 0, 1))
        d2I = np.diff(dI) if len(dI) > 1 else np.array([0.0])
        a = float(np.clip(np.mean(np.abs(d2I)) * 40, 0, 1))
        c = float(np.clip(np.mean(1.0 - I.mean(axis=1)), 0, 1))
        H = self._graph_entropy(A)
        Hm = np.log(A.shape[0]) if A.shape[0] > 1 else 1.0
        base = float(np.clip(
            0.30 * (1 - s2) + 0.30 * (1 - g) + 0.20 * (1 - a) +
            0.10 * c + 0.10 * np.clip(H / Hm, 0, 1), 0, 1))
        return float(np.clip(base * (1.0 - float(np.max(mI)) ** 2), 0, 1))

    def simulate(self, idx):
        n_cows = self.rng.randint(40, 150)
        family = self.FAMILIES[idx % len(self.FAMILIES)]
        A = self._make_graph(n_cows, family)

        regime = self.rng.choice(
            ['stable', 'borderline', 'outbreak', 'superspreader'],
            p=[0.35, 0.25, 0.30, 0.10])

        if regime == 'stable':
            beta = self.rng.uniform(0.01, 0.03); gamma = self.rng.uniform(0.15, 0.30)
            n_seed = self.rng.randint(1, 3); seed_t = 0
        elif regime == 'borderline':
            beta = self.rng.uniform(0.03, 0.055); gamma = self.rng.uniform(0.08, 0.15)
            n_seed = self.rng.randint(2, 5); seed_t = self.rng.randint(5, 15)
        elif regime == 'outbreak':
            beta = self.rng.uniform(0.055, 0.12); gamma = self.rng.uniform(0.04, 0.08)
            n_seed = self.rng.randint(2, 6); seed_t = self.rng.randint(5, 18)
        else:
            beta = self.rng.uniform(0.12, 0.25); gamma = self.rng.uniform(0.03, 0.06)
            n_seed = self.rng.randint(3, 8); seed_t = self.rng.randint(5, 15)

        vacc = np.zeros(n_cows, dtype=np.float32)
        nv = int(self.rng.uniform(0, 0.25) * n_cows)
        if nv > 0: vacc[self.rng.choice(n_cows, nv, replace=False)] = 1.0

        I = np.zeros((T_STEPS, n_cows), dtype=np.float32)
        S = np.ones((T_STEPS, n_cows), dtype=np.float32)
        sev = np.zeros((T_STEPS, n_cows), dtype=np.float32)

        seeds = self.rng.choice(n_cows, min(n_seed, n_cows), replace=False)
        I[seed_t, seeds] = self.rng.uniform(0.3, 0.7, len(seeds))
        S[seed_t, seeds] = 1.0 - I[seed_t, seeds]

        ah = self.rng.uniform(0.02, 0.06)
        bt = self.rng.uniform(68, 85)

        for t in range(max(1, seed_t + 1), T_STEPS):
            te = max(0, bt + 3 * np.sin(t * 2 * np.pi / 28) - 72)
            be = beta * (1 + ah * te)
            Ae = A * (1 - vacc[np.newaxis, :] * 0.8)
            ni = np.clip(be * (Ae @ I[t-1]) * S[t-1], 0, S[t-1])
            nr = gamma * I[t-1]
            S[t] = np.clip(S[t-1] - ni, 0, 1)
            I[t] = np.clip(I[t-1] + ni - nr, 0, 1)
            sev[t] = I[t] * (1 + 0.2 * te / 10)

        deg_n, betw, eig_c, clust = self._spectral_priors(A)

        # Build node features
        nf = np.zeros((T_STEPS, n_cows, NODE_DIM), dtype=np.float32)
        for t in range(T_STEPS):
            te = max(0, bt + 3 * np.sin(t * 2 * np.pi / 28) - 72)
            for i in range(n_cows):
                nf[t, i] = [
                    I[t, i],
                    float(te > 5) * 0.3 + self.rng.normal(0, 0.03),
                    I[t, i] * 0.4 + self.rng.normal(0, 0.03),
                    0.1 + self.rng.normal(0, 0.03),
                    0.05 + self.rng.normal(0, 0.01),
                    sev[t, i],
                    float(sev[t, i] > 1.5),
                    np.gradient(sev[max(0, t-3):t+1, i]).mean() if t > 0 else 0,
                    sev[max(0, t-4):t+1, i].sum() * 0.25,
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0, t-1), i]) > 0),
                    float(I[t, i] > 0.3 and abs(I[t, i] - I[max(0, t-1), i]) < 0.02),
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0, t-1), i]) < -0.01),
                    self.rng.uniform(1, 4),
                    max(0, 1 - I[t, i]),
                    max(0, 30 - 10 * I[t, i]) + self.rng.normal(0, 1),
                    1 - sev[t, i] * 0.3 + self.rng.normal(0, 0.03),
                    vacc[i],
                    0.0,  # pen placeholder (unused)
                    deg_n[i], betw[i], eig_c[i], clust[i]
                ]

        # Labels
        mI = I.mean(axis=1)
        intensity = float(mI.max())
        outbreak = float(intensity > 0.15)

        dI_max = float(np.max(np.abs(np.diff(mI))))
        hsi = self._compute_HSI(I, A, mI)
        breakdown = float((intensity > 0.65) and (hsi < 0.65) and (dI_max > 0.08))

        ba = beta * (1 + ah * max(0, bt - 72))
        delta_r0 = self._delta_R0(A, ba, gamma)
        vacc_gain = self._vacc_gain(A, ba, gamma, n_cows)

        return {
            "node_features": nf, "adjacency": A, "n_cows": n_cows,
            "family": family,
            "labels": {
                "intensity": intensity, "HSI": hsi,
                "outbreak": outbreak, "breakdown": breakdown,
                "delta_R0": delta_r0, "vacc_gain": vacc_gain,
            }
        }


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class BackboneDataset(Dataset):
    def __init__(self, farms):
        self.farms = farms

    def __len__(self):
        return len(self.farms)

    def __getitem__(self, idx):
        f = self.farms[idx]; L = f['labels']; nc = f['n_cows']
        nf = np.zeros((T_STEPS, MAX_COWS, NODE_DIM), dtype=np.float32)
        adj = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
        dr0 = np.zeros(MAX_COWS, dtype=np.float32)
        vg = np.zeros(MAX_COWS, dtype=np.float32)

        nf[:, :nc, :] = f['node_features'][:, :nc, :]
        adj[:nc, :nc] = f['adjacency'][:nc, :nc]
        dr0[:nc] = L['delta_R0'][:nc]
        vg[:nc] = L['vacc_gain'][:nc]

        return (
            torch.tensor(nf), torch.tensor(adj),
            torch.tensor(L['intensity'], dtype=torch.float32),
            torch.tensor(L['HSI'], dtype=torch.float32),
            torch.tensor(L['outbreak'], dtype=torch.float32),
            torch.tensor(L['breakdown'], dtype=torch.float32),
            torch.tensor(dr0), torch.tensor(vg),
            nc,
        )


# ═══════════════════════════════════════════════════════════════
# INLINE EVALUATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, device, n_farms=200, seed=9999):
    """Evaluate on fresh farms. Returns metrics dict."""
    model.eval()
    sim = FarmSimulator(seed=seed)

    y_ob_t, y_ob_p, y_bd_t, y_bd_p = [], [], [], []
    y_int_t, y_int_p, y_hsi_t, y_hsi_p = [], [], [], []
    dr0_c, vacc_s = [], []

    for i in range(n_farms):
        data = sim.simulate(i); L = data['labels']
        nc = min(data['n_cows'], MAX_COWS)
        nf = np.zeros((T_STEPS, MAX_COWS, NODE_DIM), dtype=np.float32)
        ap = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
        nf[:, :nc, :] = data['node_features'][:, :nc, :]
        ap[:nc, :nc] = data['adjacency'][:nc, :nc]

        nft = torch.tensor(nf).unsqueeze(0).to(device)
        at = torch.tensor(ap).unsqueeze(0).to(device)

        # Subsample time
        nft = nft[:, ::T_SUBSAMPLE, :, :]
        o = model(nft, at)

        y_ob_t.append(L['outbreak']); y_ob_p.append(o['intensity'].item())
        y_bd_t.append(L['breakdown']); y_bd_p.append(torch.sigmoid(o['breakdown']).item())
        y_int_t.append(L['intensity']); y_int_p.append(o['intensity'].item())
        y_hsi_t.append(L['HSI']); y_hsi_p.append(o['HSI'].item())

        ps = o['delta_R0'][0, :nc].cpu().numpy()
        td = L['delta_R0'][:nc]
        if td.std() > 0 and ps.std() > 0:
            c, _ = pearsonr(td, ps)
            if not np.isnan(c): dr0_c.append(c)

        pv = o['vacc_rank'][0, :nc].cpu().numpy()
        tv = L['vacc_gain'][:nc]
        if tv.std() > 0 and pv.std() > 0:
            s, _ = spearmanr(tv, pv)
            if not np.isnan(s): vacc_s.append(s)

    oa = roc_auc_score(y_ob_t, y_ob_p) if len(set(y_ob_t)) > 1 else 1.0
    ba = roc_auc_score(y_bd_t, y_bd_p) if len(set(y_bd_t)) > 1 else 1.0
    ic = pearsonr(y_int_t, y_int_p)[0]
    hc = pearsonr(y_hsi_t, y_hsi_p)[0]
    md = np.mean(dr0_c) if dr0_c else 0
    mv = np.mean(vacc_s) if vacc_s else 0
    sd = np.std(dr0_c) if dr0_c else 0
    sv = np.std(vacc_s) if vacc_s else 0

    model.train()
    return {
        'outbreak_auc': oa, 'breakdown_auc': ba,
        'intensity_corr': ic, 'hsi_corr': hc,
        'dr0_mean': md, 'dr0_std': sd, 'dr0_n': len(dr0_c),
        'vacc_mean': mv, 'vacc_std': sv, 'vacc_n': len(vacc_s),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("🧠 Phase 21 — Spectral Backbone Reconstitution")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
    else:
        logger.info("⚠️  No GPU — training will be slow.")

    # ── Step 1: Simulate farms ──
    logger.info(f"Step 1: Simulating {NUM_FARMS} farms (8 families, 40-150 nodes)...")
    sim = FarmSimulator(seed=SEED)
    farms = []
    fam_cnt = {}
    for i in range(NUM_FARMS):
        f = sim.simulate(i)
        farms.append(f)
        fam = f['family']
        fam_cnt[fam] = fam_cnt.get(fam, 0) + 1
        if (i + 1) % 500 == 0:
            nob = sum(1 for x in farms if x['labels']['outbreak'] > 0.5)
            logger.info(f"  {i+1}/{NUM_FARMS} ob:{nob} ({nob/(i+1):.0%})")

    nob = sum(1 for f in farms if f['labels']['outbreak'] > 0.5)
    nbd = sum(1 for f in farms if f['labels']['breakdown'] > 0.5)
    logger.info(f"Done: {nob} outbreaks ({nob/NUM_FARMS:.0%}), {nbd} breakdowns ({nbd/NUM_FARMS:.0%})")
    logger.info(f"Families: {fam_cnt}")

    # ── Step 2: DataLoader ──
    ds = BackboneDataset(farms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    del farms; gc.collect()

    # ── Step 3: Model ──
    model = HerdEngineV21_Backbone().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HerdEngineV21_Backbone Params: {n_params:,}")

    amp_sc = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    W = LOSS_WEIGHTS
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Cosine with linear warmup
    total_steps = EPOCHS * len(loader)
    warmup_steps = WARMUP_EP * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_dr0 = -1.0
    best_state = None
    early_stopped = False

    # ── Step 4: Train ──
    logger.info(f"\nTraining {EPOCHS} epochs (warmup={WARMUP_EP}, early stop targets: ΔR₀≥0.98, Vacc≥0.90, AUC≥0.97)...")
    model.train()

    for ep in range(EPOCHS):
        if early_stopped:
            break
        tl = 0; nb = 0; t0 = time.time()

        for bi, batch in enumerate(loader):
            nf, adj, inten, hsi, outbreak, breakdown, dr0, vg, nc = batch
            nf = nf.to(device, non_blocking=True)
            adj = adj.to(device, non_blocking=True)
            inten = inten.to(device, non_blocking=True)
            hsi = hsi.to(device, non_blocking=True)
            outbreak = outbreak.to(device, non_blocking=True)
            breakdown = breakdown.to(device, non_blocking=True)
            dr0 = dr0.to(device, non_blocking=True)
            vg = vg.to(device, non_blocking=True)

            nf_sub = nf[:, ::T_SUBSAMPLE, :, :]
            optimizer.zero_grad(set_to_none=True)

            if amp_sc:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    o = model(nf_sub, adj)
                    loss = (
                        W["outbreak"]    * bce(o["outbreak"], outbreak) +
                        W["breakdown"]   * bce(o["breakdown"], breakdown) +
                        W["intensity"]   * mse(o["intensity"], inten) +
                        W["HSI"]         * mse(o["HSI"], hsi) +
                        W["delta_R0"]    * mse(o["delta_R0"], dr0) +
                        W["vaccination"] * pairwise_ranking_loss(o["vacc_rank"], vg)
                    )
                amp_sc.scale(loss).backward()
                amp_sc.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_sc.step(optimizer)
                amp_sc.update()
            else:
                o = model(nf_sub, adj)
                loss = (
                    W["outbreak"]    * bce(o["outbreak"], outbreak) +
                    W["breakdown"]   * bce(o["breakdown"], breakdown) +
                    W["intensity"]   * mse(o["intensity"], inten) +
                    W["HSI"]         * mse(o["HSI"], hsi) +
                    W["delta_R0"]    * mse(o["delta_R0"], dr0) +
                    W["vaccination"] * pairwise_ranking_loss(o["vacc_rank"], vg)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            tl += loss.item(); nb += 1

            if bi % 40 == 0:
                with torch.no_grad():
                    l_dr0 = mse(o["delta_R0"], dr0).item()
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    f"EP{ep+1}/{EPOCHS} B{bi}/{len(loader)} "
                    f"L:{loss.item():.4f} [ΔR0:{l_dr0:.4f}] lr:{lr_now:.2e}"
                )

        elapsed = time.time() - t0
        logger.info(f"== EP {ep+1} | L:{tl/nb:.4f} | {elapsed:.1f}s ==")

        # ── Evaluate every epoch ──
        m = evaluate_model(model, device, n_farms=100, seed=7777)
        logger.info(
            f"   EVAL: ΔR₀={m['dr0_mean']:.4f}±{m['dr0_std']:.4f} "
            f"Vacc={m['vacc_mean']:.4f} AUC={m['outbreak_auc']:.4f}"
        )

        # Save best by ΔR₀
        if m['dr0_mean'] > best_dr0:
            best_dr0 = m['dr0_mean']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"   ⭐ New best ΔR₀: {best_dr0:.4f}")

        # Early stopping
        if (m['dr0_mean'] >= 0.98 and m['vacc_mean'] >= 0.90
                and m['outbreak_auc'] >= 0.97):
            logger.info(f"   🏆 Early stop targets met at epoch {ep+1}!")
            early_stopped = True

        # Drive checkpoint
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1 or early_stopped:
            os.makedirs(DRIVE_DIR, exist_ok=True)
            ckpt = {"epoch": ep, "model_state_dict": model.state_dict()}
            torch.save(ckpt, f"{DRIVE_DIR}/v21_ep{ep+1}.pth")
            logger.info(f"   💾 → {DRIVE_DIR}/v21_ep{ep+1}.pth")

    # ── Restore best ──
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"✅ Restored best model (ΔR₀={best_dr0:.4f})")

    # ── Step 5: Final Evaluation (200 farms, seed=9999) ──
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL EVALUATION — 200 fresh farms (seed=9999)")
    logger.info("=" * 60)

    m = evaluate_model(model, device, n_farms=200, seed=9999)

    def tag(v, t): return "✅" if v >= t else "⚠️"

    print("\n── HERD-LEVEL ──")
    print(f"  {tag(m['outbreak_auc'],0.97)} Outbreak AUC:      {m['outbreak_auc']:.4f}   (≥ 0.97)")
    print(f"  {tag(m['breakdown_auc'],0.95)} Breakdown AUC:     {m['breakdown_auc']:.4f}   (≥ 0.95)")
    print(f"  {tag(m['intensity_corr'],0.90)} Intensity Corr:    {m['intensity_corr']:.4f}   (≥ 0.90)")
    print(f"  {tag(m['hsi_corr'],0.90)} HSI Corr:          {m['hsi_corr']:.4f}   (≥ 0.90)")

    print("\n── NODE-LEVEL ──")
    print(f"  {tag(m['dr0_mean'],0.98)} ΔR₀ Pearson:       {m['dr0_mean']:.4f} ± {m['dr0_std']:.4f}  (≥ 0.98)")
    print(f"     Samples: {m['dr0_n']}/200")
    print(f"  {tag(m['vacc_mean'],0.90)} Vacc Spearman:     {m['vacc_mean']:.4f} ± {m['vacc_std']:.4f}  (≥ 0.90)")
    print(f"     Samples: {m['vacc_n']}/200")

    # Save or reject
    accept = m['dr0_mean'] >= 0.95 and m['vacc_mean'] >= 0.85
    if not accept:
        logger.info("❌ Model REJECTED — below minimum thresholds.")
        logger.info(f"   ΔR₀={m['dr0_mean']:.4f} (min 0.95), Vacc={m['vacc_mean']:.4f} (min 0.85)")
        return

    # ── Step 6: Save Production Artifacts ──
    mc = model.to("cpu")
    mp = f"{OUT_DIR}/v21_backbone.pth"
    cp = f"{OUT_DIR}/v21_config.json"
    torch.save(mc.state_dict(), mp)

    cfg = {
        "version": "21.0",
        "architecture": "NodeEncoder → ResGAT×2(96d, 4h) → AttentionPool → MLP Heads",
        "node_dim": NODE_DIM, "gat_dim": GAT_DIM,
        "n_gat_heads": GAT_HEADS, "n_gat_layers": GAT_LAYERS,
        "max_cows": MAX_COWS, "t_steps": T_STEPS,
        "loss_weights": LOSS_WEIGHTS,
        "training": {
            "epochs_used": ep + 1 if early_stopped else EPOCHS,
            "early_stopped": early_stopped,
            "lr": 3e-4, "warmup": WARMUP_EP,
            "batch_size": BATCH_SIZE, "num_farms": NUM_FARMS,
            "seed": SEED,
        },
        "metrics": {
            "dr0_pearson": float(m['dr0_mean']),
            "vacc_spearman": float(m['vacc_mean']),
            "outbreak_auc": float(m['outbreak_auc']),
            "breakdown_auc": float(m['breakdown_auc']),
            "intensity_corr": float(m['intensity_corr']),
            "hsi_corr": float(m['hsi_corr']),
        },
    }
    with open(cp, "w") as f:
        json.dump(cfg, f, indent=2)

    # Download
    try:
        from google.colab import files
        files.download(mp)
        files.download(cp)
    except:
        pass

    # Drive copy
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        torch.save(mc.state_dict(), f"{DRIVE_DIR}/v21_backbone.pth")
        with open(f"{DRIVE_DIR}/v21_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    logger.info(f"\n✅ V21 Backbone → {mp}")
    logger.info(f"✅ Config      → {cp}")
    logger.info(f"✅ Drive       → {DRIVE_DIR}/")
    logger.info("=" * 60)
    logger.info("✅ Phase 21 Backbone Stabilized — Ready for Residual Pen Integration")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
