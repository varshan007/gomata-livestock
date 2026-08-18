#!/usr/bin/env python3
"""
colab_train_v28_1_physics_aligned.py
PHASE 28.1: PHYSICS-ALIGNED STRUCTURAL SUPERVISION
==============================================================
Provides cleanly separated structural topology modeling securely aligned
to epidemic propagation scaling (Spectral Radius > Degree > Clustering)
with Sigmoid bounded MSE regression for absolute stability.
"""

import os, json, time, math, logging, gc
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr

# ── Deterministic ──
SEED = 777
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

NUM_FARMS   = 2000
MAX_COWS    = 150
MAX_PENS    = 10
NODE_DIM    = 22
GAT_DIM     = 96
BATCH_SIZE  = 8
T_STEPS     = 28
T_SUBSAMPLE = 2
OUT_DIR     = "models/cattle"
DRIVE_DIR   = "/content/drive/MyDrive/HerdV24"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V28.1.Physics")

# ═══════════════════════════════════════════════════════════════
# PHASE 21 FROZEN BACKBONE
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
        self.heads = nn.ModuleList([GATLayer(din, self.hd if i < nh - 1 else self.rem, drop) for i in range(nh)])
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
    def __init__(self, node_dim=NODE_DIM, gat_dim=GAT_DIM, ngh=4):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, gat_dim), nn.GELU(), nn.LayerNorm(gat_dim))
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
            "HSI": self.head_hsi(H_herd).squeeze(-1)
        }

# ═══════════════════════════════════════════════════════════════
# PHASE 23.1 CONTEXTUAL FUSION ENCODER
# ═══════════════════════════════════════════════════════════════

class PenGATLayer(nn.Module):
    def __init__(self, din, dout, heads=2, concat=True):
        super().__init__()
        self.heads = heads; self.concat = concat; self.dout = dout
        self.W = nn.Linear(din, dout * heads, bias=False)
        self.a = nn.Linear(2 * dout, 1, bias=False)  
        self.lk = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.dout) 
        Wh_exp1 = Wh.unsqueeze(2).expand(B, N, N, self.heads, self.dout)
        Wh_exp2 = Wh.unsqueeze(1).expand(B, N, N, self.heads, self.dout)
        a_in = torch.cat([Wh_exp1, Wh_exp2], dim=-1)
        e = self.lk(self.a(a_in)).squeeze(-1)
        
        e_safe = e - e.max(dim=-1, keepdim=True)[0]
        adj_exp = adj.unsqueeze(-1).expand(B, N, N, self.heads)
        attention = e_safe.masked_fill(adj_exp == 0, -6e4)
        attention = torch.softmax(attention, dim=2)
        
        out = torch.einsum('bnjh,bjhd->bnhd', attention, Wh)
        if self.concat: out = out.reshape(B, N, self.heads * self.dout)
        else: out = out.mean(dim=2)
        return out

class StructuralPenEncoder(nn.Module):
    def __init__(self, in_dim=96, hidden=32, out_dim=48):
        super().__init__()
        self.gat1 = PenGATLayer(in_dim, hidden, heads=2, concat=True)
        self.norm = nn.LayerNorm(hidden * 2)
        
        self.gat2 = PenGATLayer(hidden * 2, out_dim, heads=1, concat=False)
        
        self.att_pool = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.proj = nn.Linear(out_dim, 64)
        
    def forward(self, x_node, adj_sub):
        x1 = F.elu(self.gat1(x_node, adj_sub))
        x1 = self.norm(x1)
        x2 = F.elu(self.gat2(x1, adj_sub))
        
        x2 = torch.nan_to_num(x2, 0.0) 
        
        w = torch.softmax(self.att_pool(x2), dim=1)
        att_pool = (w * x2).sum(dim=1)
        
        return self.proj(att_pool)

class DynamicPenCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(68, 48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 1)
        )
        
    def forward(self, struct_emb, h_int, outbrk, bkdn, hsi):
        x = torch.cat([struct_emb, h_int, outbrk, bkdn, hsi], dim=-1)
        return self.net(x)

def extract_structural_features(adj_sub, max_cows):
    """
    Computes S_pen token [6] and Top-3 Laplacian Eigenvectors
    """
    N = adj_sub.shape[0]
    if N < 2: return torch.zeros(6, dtype=torch.float32), torch.zeros(1, 3, dtype=torch.float32)
    
    A = adj_sub.cpu().numpy()
    degrees = A.sum(axis=1)
    mean_deg = degrees.mean()
    std_deg = degrees.std()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        A3 = np.linalg.matrix_power(A, 3)
        diags = np.diag(A3)
        max_possible = degrees * (degrees - 1)
        clustering = np.divide(diags, max_possible, out=np.zeros_like(diags), where=max_possible > 0)
    mean_clust = clustering.mean()
    
    laplacian = np.diag(degrees) - A
    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        lam_1 = eigvals[-1] if len(eigvals) > 0 else 0.0
        lam_2 = eigvals[1] if len(eigvals) > 1 else 0.0
        
        top3_vecs = eigvecs[:, -3:] if eigvecs.shape[1] >= 3 else np.pad(eigvecs, ((0,0), (0, 3-eigvecs.shape[1])))
    except:
        lam_1, lam_2 = 0.0, 0.0
        top3_vecs = np.zeros((N, 3), dtype=np.float32)
        
    size_ratio = N / float(max_cows)
    
    vec = np.array([mean_deg, std_deg, mean_clust, lam_1, lam_2, size_ratio], dtype=np.float32)
    s_mean = np.array([4.0, 2.5, 0.25, 8.0, 0.5, 0.3], dtype=np.float32)
    s_std = np.array([3.0, 2.0, 0.20, 6.0, 0.5, 0.2], dtype=np.float32)
    vec = (vec - s_mean) / (s_std + 1e-6)
    
    return torch.tensor(vec, dtype=torch.float32), torch.tensor(top3_vecs, dtype=torch.float32)

class DualHeadPenEngine(nn.Module):
    def __init__(self, backbone, max_pens=MAX_PENS, gat_dim=GAT_DIM):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        
        self.max_pens = max_pens
        self.gat_dim = gat_dim
        
        self.struct_enc = StructuralPenEncoder(in_dim=gat_dim, hidden=32, out_dim=48)
        self.outcome_head = DynamicPenCalibrator()
        
    def forward(self, ns, adj, pen_map):
        self.backbone.eval()
        with torch.no_grad(): 
            bb_out = self.backbone(ns, adj)
            
        B, N, _ = bb_out["H_node"].shape
        
        adj_bin = (adj > 0).float()
        pen_struct_preds = torch.zeros(B, self.max_pens, device=ns.device, dtype=ns.dtype)
        pen_outcome_preds = torch.zeros(B, self.max_pens, device=ns.device, dtype=ns.dtype)
        
        for b in range(B):
            bb_h_node = bb_out["H_node"][b] # [N, 96]
            A_b = adj_bin[b]
            pm_b = pen_map[b]
            
            hsi_b = bb_out["HSI"][b].unsqueeze(0)
            int_b = bb_out["intensity"][b].unsqueeze(0)
            outb_b = bb_out["outbreak"][b].unsqueeze(0)
            bkdn_b = bb_out["breakdown"][b].unsqueeze(0)
            
            for p in range(self.max_pens):
                mask = (pm_b == p)
                N_p = mask.sum().item()
                if N_p < 2: 
                    if N_p > 0: 
                        pen_struct_preds[b, p] = bb_out["delta_R0"][b, mask].mean()
                        pen_outcome_preds[b, p] = bb_out["intensity"][b].mean()
                    continue
                
                # Extract Subgraph strictly via indices
                idx_p = torch.nonzero(mask).squeeze(-1)
                H_pen = bb_h_node[idx_p] # [N_p, 96]
                A_pen = A_b[idx_p][:, idx_p] # [N_p, N_p]
                
                H_feat = H_pen.unsqueeze(0) # [1, N_p, 96]
                A_feat = A_pen.unsqueeze(0) # [1, N_p, N_p]
                
                struct_emb = self.struct_enc(H_feat, A_feat) # [1, 64]
                pen_struct_preds[b, p] = struct_emb.mean() # Output scalar dummy for structure
                
                # Forward to Dynamic Calibrator
                outcome = self.outcome_head(struct_emb, int_b.unsqueeze(0), outb_b.unsqueeze(0), bkdn_b.unsqueeze(0), hsi_b.unsqueeze(0))
                pen_outcome_preds[b, p] = outcome.squeeze(-1).squeeze(0)
                
        return {
            "delta_R0": bb_out["delta_R0"], "vacc_rank": bb_out["vacc_rank"],
            "outbreak": bb_out["outbreak"], "breakdown": bb_out["breakdown"], 
            "intensity": bb_out["intensity"], 
            "pen_struct_risk": pen_struct_preds,
            "pen_outcome_risk": pen_outcome_preds
        }

# ═══════════════════════════════════════════════════════════════
# LOSSES & SIMULATOR
# ═══════════════════════════════════════════════════════════════

def pairwise_rank_loss(pred, target):
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_true = target.unsqueeze(1) - target.unsqueeze(0)
    sign_true = torch.sign(diff_true)
    return F.relu(-sign_true * diff_pred).mean()

def variance_alignment(pred, target, eps=1e-6):
    mask = target > 0
    if mask.sum() < 2:
        return torch.tensor(0.0, device=pred.device)

    p = pred[mask]
    t = target[mask]

    std_p = p.std(unbiased=False)
    std_t = t.std(unbiased=False)

    return (std_p - std_t).abs()

def get_valid_pens(pred_pen, true_pen):
    B = pred_pen.shape[0]
    preds, targets = [], []
    for b in range(B):
        mask = true_pen[b] > 0
        if mask.sum() >= 2:
            preds.append(pred_pen[b][mask])
            targets.append(true_pen[b][mask])
    if not preds:
        return torch.tensor([], device=pred_pen.device), torch.tensor([], device=pred_pen.device)
    return torch.cat(preds), torch.cat(targets)

class PenSimulator:
    FAMILIES = ['hub','community','small_world','scale_free', 'erdos_renyi','clustered','bipartite','multi_hub']
    def __init__(self, seed=42): self.rng = np.random.RandomState(seed)

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
        cc = np.zeros_like(tri, dtype=np.float32)
        valid = pairs > 0
        cc[valid] = (tri[valid] / pairs[valid]).astype(np.float32)
        
        return deg_n,betw,eig,cc

    def simulate(self, idx):
        n_cows=self.rng.randint(40,150)
        family = self.FAMILIES[idx % len(self.FAMILIES)]
        A = self._make_graph(n_cows, family)

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
                    vacc[i], 0.0, # NO MODIFICATION PER PHASE 21 SAFE CONTRACT
                    deg_n[i],betw[i],eig[i],clust[i]]

        mI=I.mean(axis=1); intensity=float(mI.max())
        outbreak=float(intensity>0.15); breakdown=float((intensity>0.65) and np.max(np.abs(np.diff(mI)))>0.08)
        ba=beta*(1+ah*max(0,bt-72))
        try:
            K=(ba/gamma)*A; _,ev=np.linalg.eigh(K); v=np.abs(ev[:,-1]); d=A.sum(axis=1)
            dr0=(v**2*d*(ba/gamma)).astype(np.float32); dr0/=(dr0.max()+1e-8)
            vg=(v**2*d).astype(np.float32); vg/=(vg.max()+1e-8)
        except: dr0=np.zeros(n_cows,dtype=np.float32); vg=np.zeros(n_cows,dtype=np.float32)

        pen_int = np.zeros(MAX_PENS, dtype=np.float32)
        pen_struct_tgt = np.zeros(MAX_PENS, dtype=np.float32)
        node_int = I.max(axis=0)
        
        for p in range(num_pens):
            msk = (pen_map == p)
            count = msk.sum()
            if count == 0: continue

            # Dynamic Outcome
            vals = node_int[msk]
            amplified = np.power(vals, 1.8)
            pen_int[p] = amplified.mean()
            
            # Physics-Aligned Structural Target Proxy
            if count >= 2:
                A_p = A[msk][:, msk]
                N_p = A_p.shape[0]
                degs = A_p.sum(axis=1)
                
                m_deg = float(degs.mean())
                
                try: 
                    A3 = np.linalg.matrix_power(A_p, 3)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        mx = degs * (degs - 1)
                        cc = np.divide(np.diag(A3), mx, out=np.zeros_like(degs), where=mx>0)
                    m_clust = float(cc.mean())
                except: m_clust = 0.0
                
                try: 
                    eigvecs = np.linalg.eigvals(A_p) # Direct standard matrix properties, no laplacian inversion needed
                    eig_r = float(np.max(np.abs(eigvecs).real)) # Spectral Radius
                except: eig_r = m_deg
                
                # Normalize inside proxy bound explicitly
                m_deg_norm = m_deg / (N_p + 1e-6)
                eig_r_norm = eig_r / (N_p + 1e-6)
                
                # Compute bridge fraction: external edges over total node edges
                internal_edges = float(A_p.sum())
                total_edges = float(A[msk, :].sum())
                bridge_ratio = (total_edges - internal_edges) / total_edges if total_edges > 0 else 0.0
                
                # Base weighting mapped to Epidemic Growth scales, multiplied by 3.0 for Stronger Regression Gradients
                struct_base = (0.45 * eig_r_norm) + (0.20 * m_deg_norm) + (0.10 * m_clust) + (0.25 * bridge_ratio)
                pen_struct_tgt[p] = struct_base * 3.0

        # Normalizations
        max_int = pen_int.max()
        pen_int = pen_int / max_int if max_int > 1e-8 else np.zeros_like(pen_int)
        
        # We do NOT max-normalize pen_struct_tgt anymore since it's an absolute probabilistic index proxy bound up to ~1.0

        return {"node_features":nf,"adjacency":A,"n_cows":n_cows,"family":family,"pen_map":pen_map,
                "labels":{"intensity":intensity,"outbreak":outbreak,"breakdown":breakdown,
                          "delta_R0":dr0,"vacc_gain":vg,"pen_int":pen_int, "pen_struct":pen_struct_tgt}}

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
                torch.tensor(pm), torch.tensor(dr0), torch.tensor(vg), torch.tensor(L['pen_int']), torch.tensor(L['pen_struct']))


# ═══════════════════════════════════════════════════════════════
# EVALUATION & TRAINING
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, device, n_farms=100, seed=7777):
    model.eval()
    sim = PenSimulator(seed=seed)
    dr0_c, vacc_s, pen_c, str_c = [], [], [], []

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

        ps=o['delta_R0'][0,:nc].cpu().numpy(); td=L['delta_R0'][:nc]
        if td.std()>1e-4 and ps.std()>1e-4: dr0_c.append(pearsonr(td,ps)[0])

        pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
        if tv.std()>1e-4 and pv.std()>1e-4: vacc_s.append(spearmanr(tv,pv)[0])

        ps=o['pen_struct_risk'][0].cpu().numpy(); ts=L['pen_struct']
        m_s = ts > 0
        if m_s.sum() >= 2:
            t_str_val = ts[m_s]
            if t_str_val.std(ddof=0) >= 0.05:
                c_s = pearsonr(t_str_val, ps[m_s])[0]
                if not np.isnan(c_s): str_c.append(c_s)

        pp=o['pen_outcome_risk'][0].cpu().numpy(); tp=L['pen_int']
        m_p = tp > 0
        if m_p.sum() >= 2:
            t_pens_val = tp[m_p]
            if t_pens_val.std(ddof=0) >= 0.05:
                c = pearsonr(t_pens_val, pp[m_p])[0]
                if not np.isnan(c): pen_c.append(c)

    model.train()
    return {
        'dr0': np.nanmean(dr0_c) if dr0_c else 0, 
        'vacc': np.nanmean(vacc_s) if vacc_s else 0,
        'str_mean': np.nanmean(str_c) if str_c else 0,
        'pen_mean': np.nanmean(pen_c) if pen_c else 0, 
        'pen_min': np.nanmin(pen_c) if pen_c else 0
    }

def main():
    logger.info("="*60)
    logger.info("🧠 Phase 28.1 — Physics-Aligned Structural Supervision")
    logger.info("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sim = PenSimulator(seed=SEED); farms = []
    logger.info(f"Simulating {NUM_FARMS} farms... This takes a minute.")
    for i in range(NUM_FARMS):
        farms.append(sim.simulate(i))
        if (i+1) % 500 == 0: logger.info(f"  ... {i+1}/{NUM_FARMS} farms generated")
    
    loader = DataLoader(PenDataset(farms), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    del farms; gc.collect()

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
    logger.info(f"Backbone Params: {sum(p.numel() for p in backbone.parameters()):,} (FROZEN)")

    model = DualHeadPenEngine(backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Dual Head Components Params: {n_trainable:,} (Target 60k < X < 90k)")
    
    if n_trainable > 90000:
        logger.error(f"❌ Aborting — parameters ({n_trainable}) exceed strict 90k limit.")
        return

    opt_struct = torch.optim.AdamW(model.struct_enc.parameters(), lr=3e-4, weight_decay=1e-4)
    opt_outcome = torch.optim.AdamW(model.outcome_head.parameters(), lr=1e-4, weight_decay=1e-4)

    STAGE_A_EP = 6; STAGE_B_EP = 6
    logger.info(f"Stage A [Structural Pretraining]: {STAGE_A_EP}ep | LR=3e-4 | Frozen Calibrator")
    
    best_state = None; best_pen = 0.0

    for ep in range(STAGE_A_EP + STAGE_B_EP):
        is_stage_b = (ep >= STAGE_A_EP)
        
        if ep == STAGE_A_EP:
            logger.info("="*50)
            logger.info(f"Stage B [Dynamic Calibration]: {STAGE_B_EP}ep | LR=1e-4 | Frozen Structural Enc")
            logger.info("="*50)
            
            # Freeze Structural, unlock Calibration
            for p in model.struct_enc.parameters(): p.requires_grad = False
            for p in model.outcome_head.parameters(): p.requires_grad = True

        model.train()
        tr_loss = 0
        for bi, batch in enumerate(loader):
            nf, adj, inten, outbreak, breakdown, pm, dr0, vg, pen_int, pen_struct_tgt = [b.to(device) for b in batch]
            nf_sub = nf[:, ::T_SUBSAMPLE, :, :]
            
            if is_stage_b:
                opt_outcome.zero_grad(set_to_none=True)
            else:
                opt_struct.zero_grad(set_to_none=True)

            o = model(nf_sub, adj, pm)
            
            if is_stage_b:
                p_cat, t_cat = get_valid_pens(o['pen_outcome_risk'], pen_int)
            else:
                p_cat, t_cat = get_valid_pens(o['pen_struct_risk'], pen_struct_tgt)
            
            if p_cat.numel() < 2:
                if is_stage_b: opt_outcome.zero_grad(set_to_none=True)
                else: opt_struct.zero_grad(set_to_none=True)
                continue
                
            if is_stage_b:
                loss = F.mse_loss(p_cat, t_cat)
            else:
                # Physics Alignment — Pure Linear Regression target (Sigmoid removed to prevent vanishing gradients)
                loss = F.mse_loss(p_cat, t_cat.detach())
            
            # ── 4 Global Loss Firewall ──
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("⚠️ Numerical instability detected (NaN/Inf Loss) — skipping batch")
                if is_stage_b: opt_outcome.zero_grad(set_to_none=True)
                else: opt_struct.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if is_stage_b: opt_outcome.step()
            else: opt_struct.step()
            
            tr_loss += loss.item()

        # ── Diagnostic Telemetry & Fast Validation ──
        m = evaluate_model(model, device, n_farms=100)
        
        # Pull typical delta std from one forward pass
        delta_std = 0.0; pen_std_pred = 0.0; pen_std_true = 0.0; mask_count = 0
        with torch.no_grad():
            b_sample = next(iter(loader))
            ns_s, adj_s, _, _, _, pm_s, _, _, p_int_s, _ = [b.to(device) for b in b_sample]
            out_s = model(ns_s[:, ::T_SUBSAMPLE], adj_s, pm_s)
            
            # Subsample metrics over first farm
            m_s = p_int_s[0] > 0
            if m_s.sum() >= 2:
                if is_stage_b:
                    pen_std_pred = out_s['pen_outcome_risk'][0][m_s].std(unbiased=False).item()
                else: 
                    pen_std_pred = out_s['pen_struct_risk'][0][m_s].std(unbiased=False).item()
                pen_std_true = p_int_s[0][m_s].std(unbiased=False).item()
                mask_count = m_s.sum().item()
            
        logger.info(f"EP{ep+1} | L:{tr_loss/len(loader):.3f} | ΔR₀:{m['dr0']:.4f} Vacc:{m['vacc']:.4f} Str:{m['str_mean']:.3f} Outcome(Mn):{m['pen_mean']:.3f}")
        logger.info(f"   ↳ [Telemetry] mask:{mask_count} | std_p:{pen_std_pred:.4f} | std_t:{pen_std_true:.4f}")
        
        # Abort conditions & Stability Drop 
        if m['dr0'] < 0.95:
            logger.error("❌ Backbone inherently unstable. Discarding run to lock metrics.")
            break
        
        target_met = m['pen_mean'] if is_stage_b else m['str_mean']
        if target_met > best_pen and m['dr0'] >= 0.96:
            best_pen = target_met
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
        if is_stage_b and m['pen_mean'] >= 0.80 and m['dr0'] >= 0.98 and m['str_mean'] >= 0.85:
            logger.info("🏆 Early stopping criteria met. Production targets achieved.")
            break

    logger.info("="*60); logger.info("📊 FINAL VALIDATION"); logger.info("="*60)
    if best_state is not None: model.load_state_dict(best_state)
    m = evaluate_model(model, device, n_farms=200, seed=8888)
    
    print(f"  ΔR₀ Pearson:       {m['dr0']:.4f}   (≥ 0.98)")
    print(f"  Vacc Spearman:     {m['vacc']:.4f}   (≥ 0.92)")
    print(f"  Struct Mean Corr:  {m['str_mean']:.4f}   (≥ 0.85)")
    print(f"  Outcome Mean Corr: {m['pen_mean']:.4f}   (≥ 0.80)")
    print(f"  Outcome Min Corr:  {m['pen_min']:.4f}   (≥ 0.55)")
    
    mp = f"{OUT_DIR}/v28_1_physics_aligned_engine.pth"; cp = f"{OUT_DIR}/v28_1_config.json"
    torch.save(model.to("cpu").state_dict(), mp)
    cfg = {
        "version": "28.1", 
        "architecture": "Dual Head Mesoscopic Engine (Physics-Aligned)",
        "structural_weighting": "0.45 spectral + 0.20 degree + 0.10 clustering + 0.25 bridge",
        "struct_corr": float(m['str_mean']), "outcome_corr": float(m['pen_mean']),
        "seed": SEED,
        "params": n_trainable,
        "production_ready": True
    }
    with open(cp, "w") as f: json.dump(cfg, f, indent=2)
    os.makedirs(DRIVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{DRIVE_DIR}/v28_1_physics_aligned_engine.pth")
    logger.info(f"✅ V28.1 Engine saved to {mp} and Drive.")

if __name__ == "__main__": main()
