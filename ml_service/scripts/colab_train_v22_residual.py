#!/usr/bin/env python3
"""
colab_train_v22_residual.py — Phase 22
RESIDUAL MESOSCOPIC PEN CORRECTION (GoMata T4)
============================================================
Architecture Principle:
  Backbone (Phase 21) = epidemiological manifold (FROZEN forever)
  Pen module = bounded small corrective residual field

Workflow:
  1. Load v21_backbone.pth, freeze all parameters.
  2. For each pen: compute 6 structural features + mean node embedding.
  3. Pen input = 96d + 6d = 102d.
  4. PenResidualHead: MLP(102 → 64 → 32)
  5. Gated correction: gate = clamp(sigmoid(W_g * pen_embed), 0, 0.25)
  6. delta_R0_final = delta_R0_backbone + gate * pen_correction

Training:
  - Stage A: 6ep, lr=3e-4, λ_pen=4.0
  - Stage B: 8ep, lr=5e-5, λ_pen=2.5
  - Loss = delta_R0 (MSE) + vacc (Rank) + herd (BCE) + 2.5*pen_corr + 3.0*stability
  - Safety abort: if ΔR₀ < 0.95 -> restore checkpt, lr *= 0.5, λ_pen *= 0.70
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
MAX_PENS    = 10
NODE_DIM    = 22
GAT_DIM     = 96
GAT_HEADS   = 4
BATCH_SIZE  = 8
T_STEPS     = 28
T_SUBSAMPLE = 2
OUT_DIR     = "models/cattle"
DRIVE_DIR   = "/content/drive/MyDrive/HerdV22"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V22")


# ═══════════════════════════════════════════════════════════════
# PHASE 21 BACKBONE (FROZEN MULTI-HEAD)
# ═══════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    def __init__(self, din, dout, drop=0.1):
        super().__init__()
        self.W = nn.Linear(din, dout, bias=False)
        self.a = nn.Linear(2 * dout, 1, bias=False)
        self.lk = nn.LeakyReLU(0.2); self.dp = nn.Dropout(drop)
    def forward(self, h, adj):
        Wh = self.W(h); N = Wh.size(1)
        a_in = torch.cat([Wh.unsqueeze(2).expand(-1, -1, N, -1),
                          Wh.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1)
        e = self.lk(self.a(a_in).squeeze(-1))
        m = (adj == 0); e = e.masked_fill(m, -6e4)
        return self.dp(torch.softmax(e, dim=-1)) @ Wh

class ResGAT(nn.Module):
    def __init__(self, din, dout, nh=4, drop=0.1):
        super().__init__()
        self.hd = dout // nh; self.rem = dout - self.hd * (nh - 1)
        self.heads = nn.ModuleList([
            GATLayer(din, self.hd if i < nh - 1 else self.rem, drop) for i in range(nh)])
        self.norm = nn.LayerNorm(dout)
        self.proj = nn.Linear(din, dout) if din != dout else nn.Identity()
    def forward(self, h, adj):
        out = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        return self.norm(out + self.proj(h))

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(dim=1)

def make_head(din, dh, dout):
    return nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dout))

class HerdEngineV21_Backbone(nn.Module):
    """FROZEN backbone."""
    def __init__(self, node_dim=NODE_DIM, gat_dim=GAT_DIM, ngh=GAT_HEADS):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, gat_dim), nn.GELU(), nn.LayerNorm(gat_dim))
        self.gat1 = ResGAT(gat_dim, gat_dim, ngh, drop=0.1)
        self.gat2 = ResGAT(gat_dim, gat_dim, ngh, drop=0.1)
        self.pool = AttentionPool(gat_dim)
        self.head_dr0 = make_head(gat_dim, 64, 1)
        self.head_vacc = make_head(gat_dim, 64, 1)
        self.head_outbreak = make_head(gat_dim, 64, 1)
        self.head_breakdown = make_head(gat_dim, 64, 1)
        self.head_intensity = make_head(gat_dim, 64, 1)
        self.head_hsi = make_head(gat_dim, 64, 1)

    def forward(self, ns, adj):
        B, T, N, _ = ns.shape; herd_embs = []
        for t in range(T):
            h = self.node_enc(ns[:, t])
            h = F.elu(self.gat1(h, adj)); h = F.elu(self.gat2(h, adj))
            herd_embs.append(self.pool(h))
        H_node = h; H_herd = torch.stack(herd_embs, dim=1).mean(dim=1)
        return {
            "H_node": H_node, "H_herd": H_herd,
            "delta_R0": self.head_dr0(H_node).squeeze(-1),
            "vacc_rank": self.head_vacc(H_node).squeeze(-1),
            "outbreak": self.head_outbreak(H_herd).squeeze(-1),
            "breakdown": self.head_breakdown(H_herd).squeeze(-1),
            "intensity": self.head_intensity(H_herd).squeeze(-1),
        }


# ═══════════════════════════════════════════════════════════════
# PHASE 22 RESIDUAL PEN ENGINE
# ═══════════════════════════════════════════════════════════════

class PenResidualEngine(nn.Module):
    """
    Phase 22 Bounded Residual Corrector.
    Wraps frozen V21 backbone, computes bounded structural pen corrections.
    """
    def __init__(self, backbone, max_pens=MAX_PENS, gat_dim=GAT_DIM):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False  # Strict freeze

        self.max_pens = max_pens
        self.gat_dim = gat_dim

        # 96 (mean node emb) + 6 (structural feats) = 102
        PEN_INPUT_DIM = gat_dim + 6

        # MLP < 30k params
        self.pen_mlp = nn.Sequential(
            nn.Linear(PEN_INPUT_DIM, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.pen_corr_head = nn.Linear(32, 1)
        self.pen_risk_head = nn.Linear(32, 1)  # To predict actual pen risk
        
        # Bounded Gate (sigmoid -> clamp 0-0.25)
        self.gate_w = nn.Linear(32, 1)

    def _compute_pen_features(self, H_node, adj, pen_map):
        """Build independent structural + embedding vectors per pen."""
        B, N, _ = H_node.shape; P = self.max_pens
        pidx = pen_map.clamp(0, P - 1)

        # 1. Mean node embedding (96d)
        idx_exp = pidx.unsqueeze(-1).expand(-1, -1, self.gat_dim)
        pen_sum = torch.zeros(B, P, self.gat_dim, device=H_node.device, dtype=H_node.dtype)
        pen_sum.scatter_add_(1, idx_exp, H_node)
        pen_cnt = torch.zeros(B, P, 1, device=H_node.device, dtype=H_node.dtype)
        pen_cnt.scatter_add_(1, pidx.unsqueeze(-1),
                             torch.ones(B, N, 1, device=H_node.device, dtype=H_node.dtype))
        pen_emb = pen_sum / pen_cnt.clamp(min=1.0)  # (B, P, 96)

        # 2. Structural Features (6d)
        feats = torch.zeros(B, P, 6, device=adj.device, dtype=adj.dtype)
        adj_bin = (adj > 0).float()
        
        for b in range(B):
            for p in range(P):
                nip = (pidx[b] == p).nonzero(as_tuple=True)[0]
                nc = len(nip)
                if nc < 1: continue

                feats[b, p, 0] = nc / max(N, 1)  # pen size
                if nc < 2: continue

                sa = adj_bin[b][nip][:, nip]
                deg_in = sa.sum(dim=1)
                e_int = sa.sum() / 2
                
                feats[b, p, 1] = deg_in.mean() / max(nc - 1, 1)  # mean degree
                feats[b, p, 2] = deg_in.std() / max(nc - 1, 1) if nc > 1 else 0  # degree std
                feats[b, p, 3] = (e_int / max(nc*(nc-1)/2, 1)).clamp(0, 1)  # infection density (density)
                
                
                if nc >= 3:
                    tri = torch.diag(sa @ sa @ sa) / 2
                    pairs = deg_in * (deg_in - 1) / 2
                    cc = torch.zeros_like(tri)
                    valid = pairs > 0
                    cc[valid] = (tri[valid] / pairs[valid]).to(cc.dtype)
                    feats[b, p, 4] = cc.mean().clamp(0, 1)  # clustering
                    
                    try:
                        L = torch.diag(deg_in) - sa
                        ev = torch.linalg.eigvalsh(L.float())
                        # Normalize spectral radius approx
                        v_max = (ev[-1]).clamp(0, 5) / 5.0
                        feats[b, p, 5] = v_max
                    except: 
                        feats[b, p, 5] = 0.0

        if not torch.isfinite(feats).all():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        mu = feats.mean(dim=(0, 1), keepdim=True)
        std = feats.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        feats = torch.nan_to_num(((feats - mu) / std).clamp(-3, 3), nan=0.0)

        # Concatenate: 96 + 6 = 102
        return torch.cat([pen_emb, feats], dim=-1), pen_cnt.squeeze(-1)

    def forward(self, ns, adj, pen_map):
        # 1. run frozen backbone
        with torch.no_grad():
            bb_out = self.backbone(ns, adj)
            H_node = bb_out["H_node"]
            dr0_bb = bb_out["delta_R0"]
            vacc = bb_out["vacc_rank"]
            ob = bb_out["outbreak"]
            bd = bb_out["breakdown"]
            inten = bb_out["intensity"]

        # 2. Pen features (102d)
        pen_feats, pen_cnt = self._compute_pen_features(H_node, adj, pen_map)  # (B, P, 102)

        # 3. Pen MLP + Gate
        pen_h = self.pen_mlp(pen_feats)  # (B, P, 32)
        gate_raw = torch.sigmoid(self.gate_w(pen_h))  # (B, P, 1)
        gate = torch.clamp(gate_raw, 0.0, 0.25)       # Bounded 0.0 -> 0.25
        
        p_corr = self.pen_corr_head(pen_h)            # (B, P, 1)
        corr_gated = (gate * p_corr).squeeze(-1)      # (B, P)
        
        # Pen risk prediction for loss
        pen_risk = self.pen_risk_head(pen_h).squeeze(-1) # (B, P)

        # 4. Final Node Residual Addition
        B, N = dr0_bb.shape
        pidx = pen_map.clamp(0, self.max_pens - 1)
        node_corr = torch.gather(corr_gated, 1, pidx) # Broadcast pen corr to nodes
        
        dr0_final = dr0_bb + node_corr

        return {
            "delta_R0": dr0_final, "vacc_rank": vacc,
            "outbreak": ob, "breakdown": bd, "intensity": inten,
            "pen_risk": pen_risk, "gate": gate.mean(), "gate_max": gate.max()
        }


# ═══════════════════════════════════════════════════════════════
# LOSSES
# ═══════════════════════════════════════════════════════════════

def pairwise_ranking_loss(pred, target, n_pairs=50):
    B, N = pred.shape
    losses = []
    for b in range(B):
        p, t = pred[b], target[b]
        valid = (t > 0.01)
        if valid.sum() < 5: losses.append(torch.tensor(0.0, device=pred.device)); continue
        vi = valid.nonzero(as_tuple=True)[0]
        n = min(n_pairs, len(vi) * (len(vi) - 1) // 2)
        if n < 1: losses.append(torch.tensor(0.0, device=pred.device)); continue
        idx = torch.randint(0, len(vi), (n, 2), device=pred.device)
        i, j = vi[idx[:, 0]], vi[idx[:, 1]]
        t_diff = t[i] - t[j]; p_diff = p[i] - p[j]
        sign = torch.sign(t_diff)
        loss = torch.relu(0.5 - sign * p_diff).mean()
        losses.append(loss)
    return torch.stack(losses).mean()

def pearson_pen_loss(pred_risk, target_risk):
    B, P = pred_risk.shape
    c_loss = []
    for b in range(B):
        p, t = pred_risk[b], target_risk[b]
        if t.std() > 1e-4 and p.std() > 1e-4:
            pm = p - p.mean(); tm = t - t.mean()
            c = (pm * tm).sum() / (torch.sqrt((pm**2).sum()) * torch.sqrt((tm**2).sum()) + 1e-8)
            c_loss.append(1.0 - c)
    if not c_loss: return torch.tensor(1.0, device=pred_risk.device, requires_grad=True)
    return sum(c_loss) / len(c_loss)


# ═══════════════════════════════════════════════════════════════
# SIMULATOR (with Pen Integration)
# ═══════════════════════════════════════════════════════════════

class PenSimulator:
    FAMILIES = ['hub','community','small_world','scale_free',
                'erdos_renyi','clustered','bipartite','multi_hub']

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def _make_graph(self, N, family):
        A = np.zeros((N, N), dtype=np.float32)
        if family == 'hub':
            hubs = self.rng.choice(N, max(2, int(0.08*N)), replace=False)
            for h in hubs:
                tgts = self.rng.choice(N, min(self.rng.randint(int(0.3*N),int(0.6*N)), N-1), replace=False)
                for t in tgts:
                    if t != h: w = self.rng.uniform(0.3,1.0); A[h,t]=w; A[t,h]=w
        elif family == 'community':
            nc=self.rng.randint(3,6); assign=self.rng.randint(0,nc,N)
            for i in range(N):
                for j in range(i+1,N):
                    p=0.4 if assign[i]==assign[j] else 0.02
                    if self.rng.random()<p: w=self.rng.uniform(0.2,0.8); A[i,j]=w; A[j,i]=w
        elif family == 'small_world':
            k=self.rng.randint(4,8)
            for i in range(N):
                for d in range(1,k//2+1):
                    j=(i+d)%N; w=self.rng.uniform(0.3,0.8); A[i,j]=w; A[j,i]=w
        elif family == 'scale_free':
            m=self.rng.randint(2,5)
            for i in range(m,N):
                deg=A[:i].sum(axis=1)+1; p=deg/deg.sum()
                tgts=self.rng.choice(i,min(m,i),replace=False,p=p)
                for t in tgts: w=self.rng.uniform(0.3,0.9); A[i,t]=w; A[t,i]=w
        elif family == 'erdos_renyi':
            p=self.rng.uniform(0.05,0.15)
            for i in range(N):
                for j in range(i+1,N):
                    if self.rng.random()<p: w=self.rng.uniform(0.2,0.8); A[i,j]=w; A[j,i]=w
        elif family == 'clustered':
            nc=self.rng.randint(3,7); centers=self.rng.choice(N,nc,replace=False)
            assign=np.argmin(np.abs(np.arange(N)[:,None]-centers[None,:]),axis=1)
            for i in range(N):
                for j in range(i+1,N):
                    p=0.5 if assign[i]==assign[j] else 0.01
                    if self.rng.random()<p: w=self.rng.uniform(0.3,0.9); A[i,j]=w; A[j,i]=w
        elif family == 'bipartite':
            g=self.rng.randint(0,2,N)
            for i in range(N):
                for j in range(i+1,N):
                    if g[i]!=g[j] and self.rng.random()<0.12: w=self.rng.uniform(0.2,0.7); A[i,j]=w; A[j,i]=w
        elif family == 'multi_hub':
            nh=self.rng.randint(3,7); hubs=self.rng.choice(N,nh,replace=False)
            for h in hubs:
                nc=self.rng.randint(int(0.1*N),int(0.3*N)); tgts=self.rng.choice(N,nc,replace=False)
                for t in tgts:
                    if t!=h: w=self.rng.uniform(0.3,0.9); A[h,t]=w; A[t,h]=w
            for i,h1 in enumerate(hubs):
                for h2 in hubs[i+1:]: w=self.rng.uniform(0.5,1.0); A[h1,h2]=w; A[h2,h1]=w
        return A

    def _spectral_priors(self, A):
        N=A.shape[0]; ab=(A>0).astype(float); deg_n=ab.sum(axis=1)/(N-1+1e-8)
        P=np.diag(1.0/(ab.sum(axis=1)+1e-8))@A; pr=np.ones(N)/N
        for _ in range(10): pr=0.85*(P.T@pr)+0.15/N
        betw=pr/(pr.max()+1e-8)
        try: _,ev=np.linalg.eigh(A); eig=np.abs(ev[:,-1]); eig/=(eig.max()+1e-8)
        except: eig=deg_n.copy()
        
        tri = np.diag(ab@ab@ab)/2
        pairs = ab.sum(axis=1)*(ab.sum(axis=1)-1)/2
        clust = np.zeros_like(tri, dtype=np.float32)
        valid = pairs > 0
        clust[valid] = (tri[valid] / pairs[valid]).astype(np.float32)
        
        return deg_n, betw, eig, clust

    def simulate(self, idx):
        n_cows=self.rng.randint(40,150)
        family = self.FAMILIES[idx % len(self.FAMILIES)]
        A = self._make_graph(n_cows, family)

        # Pen logic
        num_pens = self.rng.randint(3, MAX_PENS+1)
        pen_map = np.zeros(n_cows, dtype=np.int64)
        cws = np.arange(n_cows); self.rng.shuffle(cws)
        sp = np.array_split(cws, num_pens)
        for p, arr in enumerate(sp): pen_map[arr] = p

        regime = self.rng.choice(['stable','borderline','outbreak','superspreader'],p=[0.35,0.25,0.30,0.10])
        if regime=='stable': beta=self.rng.uniform(0.01,0.03); gamma=self.rng.uniform(0.15,0.30); ns=self.rng.randint(1,3); st=0
        elif regime=='borderline': beta=self.rng.uniform(0.03,0.055); gamma=self.rng.uniform(0.08,0.15); ns=self.rng.randint(2,5); st=self.rng.randint(5,15)
        elif regime=='outbreak': beta=self.rng.uniform(0.055,0.12); gamma=self.rng.uniform(0.04,0.08); ns=self.rng.randint(2,6); st=self.rng.randint(5,18)
        else: beta=self.rng.uniform(0.12,0.25); gamma=self.rng.uniform(0.03,0.06); ns=self.rng.randint(3,8); st=self.rng.randint(5,15)

        vacc=np.zeros(n_cows,dtype=np.float32); nv=int(self.rng.uniform(0,0.25)*n_cows)
        if nv>0: vacc[self.rng.choice(n_cows,nv,replace=False)]=1.0
        I=np.zeros((T_STEPS,n_cows),dtype=np.float32); S=np.ones((T_STEPS,n_cows),dtype=np.float32)
        sev=np.zeros((T_STEPS,n_cows),dtype=np.float32)
        seeds=self.rng.choice(n_cows,min(ns,n_cows),replace=False)
        I[st,seeds]=self.rng.uniform(0.3,0.7,len(seeds)); S[st,seeds]=1.0-I[st,seeds]
        ah=self.rng.uniform(0.02,0.06); bt=self.rng.uniform(68,85)

        for t in range(max(1,st+1),T_STEPS):
            te=max(0,bt+3*np.sin(t*2*np.pi/28)-72); be=beta*(1+ah*te)
            ni=np.clip(be*(A*(1-vacc*0.8)@I[t-1])*S[t-1],0,S[t-1]); nr=gamma*I[t-1]
            S[t]=np.clip(S[t-1]-ni,0,1); I[t]=np.clip(I[t-1]+ni-nr,0,1)
            sev[t]=I[t]*(1+0.2*te/10)

        deg_n,betw,eig,clust=self._spectral_priors(A)
        nf=np.zeros((T_STEPS,n_cows,NODE_DIM),dtype=np.float32)
        for t in range(T_STEPS):
            te=max(0,bt+3*np.sin(t*2*np.pi/28)-72)
            for i in range(n_cows):
                nf[t,i]=[I[t,i],float(te>5)*0.3+self.rng.normal(0,0.03),
                    I[t,i]*0.4+self.rng.normal(0,0.03),0.1+self.rng.normal(0,0.03),
                    0.05+self.rng.normal(0,0.01),sev[t,i],float(sev[t,i]>1.5),
                    np.gradient(sev[max(0,t-3):t+1,i]).mean() if t>0 else 0,
                    sev[max(0,t-4):t+1,i].sum()*0.25,
                    float(I[t,i]>0.1 and (I[t,i]-I[max(0,t-1),i])>0),
                    float(I[t,i]>0.3 and abs(I[t,i]-I[max(0,t-1),i])<0.02),
                    float(I[t,i]>0.1 and (I[t,i]-I[max(0,t-1),i])<-0.01),
                    self.rng.uniform(1,4),max(0,1-I[t,i]),
                    max(0,30-10*I[t,i])+self.rng.normal(0,1),
                    1-sev[t,i]*0.3+self.rng.normal(0,0.03),
                    vacc[i],float(pen_map[i]),deg_n[i],betw[i],eig[i],clust[i]]

        mI=I.mean(axis=1); intensity=float(mI.max())
        outbreak=float(intensity>0.15); breakdown=float((intensity>0.65) and np.max(np.abs(np.diff(mI)))>0.08)
        ba=beta*(1+ah*max(0,bt-72))
        try:
            K=(ba/gamma)*A; _,ev=np.linalg.eigh(K); v=np.abs(ev[:,-1]); d=A.sum(axis=1)
            dr0=(v**2*d*(ba/gamma)).astype(np.float32); dr0/=(dr0.max()+1e-8)
            vg=(v**2*d).astype(np.float32); vg/=(vg.max()+1e-8)
        except: dr0=np.zeros(n_cows,dtype=np.float32); vg=np.zeros(n_cows,dtype=np.float32)

        pen_int = np.zeros(MAX_PENS, dtype=np.float32)
        node_int = I.max(axis=0)
        for p in range(num_pens):
            msk = (pen_map == p)
            if msk.sum() > 0: pen_int[p] = node_int[msk].mean()
        pen_int = pen_int / (pen_int.max() + 1e-8)

        return {"node_features":nf,"adjacency":A,"n_cows":n_cows,"family":family,"pen_map":pen_map,
                "labels":{"intensity":intensity,"outbreak":outbreak,"breakdown":breakdown,
                          "delta_R0":dr0,"vacc_gain":vg,"pen_int":pen_int}}

class PenDataset(Dataset):
    def __init__(self, farms): self.farms = farms
    def __len__(self): return len(self.farms)
    def __getitem__(self, idx):
        f = self.farms[idx]; L = f['labels']; nc = f['n_cows']
        nf=np.zeros((T_STEPS,MAX_COWS,NODE_DIM),dtype=np.float32)
        adj=np.zeros((MAX_COWS,MAX_COWS),dtype=np.float32)
        dr0=np.zeros(MAX_COWS,dtype=np.float32); vg=np.zeros(MAX_COWS,dtype=np.float32)
        pm=np.zeros(MAX_COWS,dtype=np.int64)
        nf[:,:nc,:]=f['node_features'][:,:nc,:]; adj[:nc,:nc]=f['adjacency'][:nc,:nc]
        dr0[:nc]=L['delta_R0'][:nc]; vg[:nc]=L['vacc_gain'][:nc]; pm[:nc]=f['pen_map']
        return (torch.tensor(nf), torch.tensor(adj), torch.tensor(L['intensity'],dtype=torch.float32),
                torch.tensor(L['outbreak'],dtype=torch.float32), torch.tensor(L['breakdown'],dtype=torch.float32),
                torch.tensor(pm), torch.tensor(dr0), torch.tensor(vg), torch.tensor(L['pen_int']))


# ═══════════════════════════════════════════════════════════════
# INLINE EVALUATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, device, n_farms=100, seed=7777):
    model.eval()
    sim = PenSimulator(seed=seed)
    y_ob_t, y_ob_p = [], []
    dr0_c, vacc_s, pen_c = [], [], []

    for i in range(n_farms):
        data = sim.simulate(i); L = data['labels']; nc = data['n_cows']
        nf = np.zeros((T_STEPS, MAX_COWS, NODE_DIM), dtype=np.float32)
        ap = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
        pm = np.zeros(MAX_COWS, dtype=np.int64)
        nf[:,:nc,:] = data['node_features'][:,:nc,:]; ap[:nc,:nc] = data['adjacency'][:nc,:nc]; pm[:nc] = data['pen_map']
        nft=torch.tensor(nf).unsqueeze(0).to(device)[:,::T_SUBSAMPLE,:,:]
        at=torch.tensor(ap).unsqueeze(0).to(device)
        pmt=torch.tensor(pm).unsqueeze(0).to(device)
        
        o = model(nft, at, pmt)
        y_ob_t.append(L['outbreak']); y_ob_p.append(o['intensity'].item() if 'intensity' in o else 0.0)

        ps=o['delta_R0'][0,:nc].cpu().numpy(); td=L['delta_R0'][:nc]
        if td.std()>0 and ps.std()>0:
            c,_=pearsonr(td,ps); dr0_c.append(c)

        pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
        if tv.std()>0 and pv.std()>0:
            s,_=spearmanr(tv,pv); vacc_s.append(s)

        pp=o['pen_risk'][0].cpu().numpy(); tp=L['pen_int']
        if tp.std()>0 and pp.std()>0:
            c,_=pearsonr(tp,pp); pen_c.append(c)

    oa = roc_auc_score(y_ob_t, y_ob_p) if len(set(y_ob_t))>1 else 1.0
    model.train()
    return {
        'dr0': np.mean(dr0_c) if dr0_c else 0, 'vacc': np.mean(vacc_s) if vacc_s else 0,
        'ob_auc': oa, 'pen_mean': np.mean(pen_c) if pen_c else 0, 'pen_min': np.min(pen_c) if pen_c else 0
    }


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("="*60)
    logger.info("🧠 Phase 22 — Residual Mesoscopic Pen Correction")
    logger.info("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
    
    # Sim farms
    sim = PenSimulator(seed=SEED)
    farms = []
    logger.info(f"Simulating {NUM_FARMS} farms... This takes a minute.")
    for i in range(NUM_FARMS):
        farms.append(sim.simulate(i))
        if (i+1) % 500 == 0:
            logger.info(f"  ... {i+1}/{NUM_FARMS} farms generated")
    nob = sum(1 for f in farms if f['labels']['outbreak']>0.5)
    logger.info(f"Simulated {NUM_FARMS} farms. Outbreaks: {nob} ({nob/NUM_FARMS:.0%})")
    loader = DataLoader(PenDataset(farms), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    del farms; gc.collect()

    # Load Backbone
    backbone = HerdEngineV21_Backbone(node_dim=NODE_DIM, gat_dim=GAT_DIM)
    ckpt_paths = ["models/cattle/v21_backbone.pth", "/content/drive/MyDrive/HerdV21/v21_backbone.pth", "v21_backbone.pth"]
    bb_path = next((p for p in ckpt_paths if os.path.exists(p)), None)
    if not bb_path:
        logger.error("❌ Phase 21 Backbone not found. ABORTING.")
        return
    logger.info(f"Loading backbone from {bb_path}")
    state = torch.load(bb_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    backbone.load_state_dict(state)
    logger.info(f"Backbone Params: {sum(p.numel() for p in backbone.parameters()):,}")

    model = PenResidualEngine(backbone).to(device)
    pen_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Residual Pen Head Params: {sum(p.numel() for p in pen_params):,}")

    mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
    amp_sc = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    STAGE_A_EP = 6; STAGE_B_EP = 8
    best_pen = -1.0; best_state = None; aborts = 0; LAMBDA_PEN = 4.0; LR_NOW = 3e-4
    logger.info(f"Stage A: {STAGE_A_EP}ep | LR={LR_NOW} | λ_pen={LAMBDA_PEN}")

    opt = torch.optim.AdamW(pen_params, lr=LR_NOW, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(STAGE_A_EP+STAGE_B_EP)*len(loader))

    model.train()
    for ep in range(STAGE_A_EP + STAGE_B_EP):
        if ep == STAGE_A_EP:
            LAMBDA_PEN = 2.5; LR_NOW = 5e-5
            for param_group in opt.param_groups: param_group['lr'] = LR_NOW
            logger.info("="*40)
            logger.info(f"Stage B: {STAGE_B_EP}ep | LR={LR_NOW} | λ_pen={LAMBDA_PEN}")
            logger.info("="*40)

        t0 = time.time(); tr_loss = 0; batch_dr0 = []
        for bi, batch in enumerate(loader):
            nf, adj, inten, outbreak, breakdown, pm, dr0, vg, pen_int = [b.to(device) for b in batch]
            nf_sub = nf[:, ::T_SUBSAMPLE, :, :]
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16) if amp_sc else torch.autocast("cpu", enabled=False):
                o = model(nf_sub, adj, pm)
                l_dr0 = mse(o["delta_R0"], dr0)
                l_vac = pairwise_ranking_loss(o['vacc_rank'], vg)
                l_out = bce(o['outbreak'], outbreak)
                l_pen = pearson_pen_loss(o['pen_risk'], pen_int)
                
                # Stability penalty if batch delta_R0 drops
                with torch.no_grad():
                    bb = pearson_pen_loss(o['delta_R0'], dr0).item()
                    batch_dr0.append(1 - bb)
                
                l_stab = torch.clamp(torch.tensor(0.97 - (1 - bb)).to(device), min=0.0)
                loss = l_dr0 + 1.0*l_vac + 1.0*l_out + LAMBDA_PEN*l_pen + 3.0*l_stab

            if amp_sc:
                amp_sc.scale(loss).backward()
                amp_sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(pen_params, 1.0)
                amp_sc.step(opt); amp_sc.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(pen_params, 1.0); opt.step()
            
            sched.step(); tr_loss += loss.item()

        # ── Epoch Eval & Monitoring ──
        m = evaluate_model(model, device, n_farms=100)
        logger.info(f"EP{ep+1} | L:{tr_loss/len(loader):.3f} | ΔR₀:{m['dr0']:.4f} Vacc:{m['vacc']:.4f} Pen:{m['pen_mean']:.3f} Gate:{o['gate']:.3f}")
        
        # Save best
        if m['pen_mean'] > best_pen and m['dr0'] >= 0.95:
            best_pen = m['pen_mean']; best_state = {k: v.clone() for k,v in model.state_dict().items()}
            logger.info(f"   ⭐ New best Pen Corr: {best_pen:.4f}")

        # Safety Abort Mechanism
        if m['dr0'] < 0.95:
            aborts += 1
            if best_state: model.load_state_dict(best_state)
            LAMBDA_PEN *= 0.70; LR_NOW *= 0.50
            for param_group in opt.param_groups: param_group['lr'] = LR_NOW
            logger.info(f"   🚨 ABORT! ΔR₀ < 0.95. Rollback to best. LR={LR_NOW:.1e}, λ_pen={LAMBDA_PEN:.2f}")
            if aborts > 5: logger.error("Fatal instability limit reached"); break
        elif o['gate'].item() > 0.20:
            LR_NOW *= 0.70
            for param_group in opt.param_groups: param_group['lr'] = LR_NOW
            logger.info(f"   ⚠️ High gate output ({o['gate']:.3f}). Reducing LR to {LR_NOW:.1e}")

    # Restore best
    if best_state: model.load_state_dict(best_state)
    logger.info("="*60); logger.info("📊 FINAL EVALUATION"); logger.info("="*60)
    m = evaluate_model(model, device, n_farms=200, seed=9999)
    print(f"  ΔR₀ Pearson:       {m['dr0']:.4f}   (≥ 0.97)")
    print(f"  Vacc Spearman:     {m['vacc']:.4f}   (≥ 0.92)")
    print(f"  Herd Outbreak AUC: {m['ob_auc']:.4f}   (≥ 0.98)")
    print(f"  Pen Mean Corr:     {m['pen_mean']:.4f}   (≥ 0.80)")
    print(f"  Pen Min Corr:      {m['pen_min']:.4f}")
    
    mp = f"{OUT_DIR}/v22_residual_engine.pth"; cp = f"{OUT_DIR}/v22_config.json"
    torch.save(model.to("cpu").state_dict(), mp)
    cfg = {"version": "22.0", "architecture": "Frozen V21 Backbone + Bounded PenResidualHead(102d->64d->32d->1d)", "dr0": float(m['dr0']), "pen": float(m['pen_mean'])}
    with open(cp, "w") as f: json.dump(cfg, f, indent=2)
    os.makedirs(DRIVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{DRIVE_DIR}/v22_residual_engine.pth")
    logger.info(f"✅ V22 Residual Engine saved to {mp} and Drive.")

if __name__ == "__main__": main()
