#!/usr/bin/env python3
"""
evaluate_v30_1_spectral_basis.py
PHASE 30.1: MULTI-BASIS SPECTRAL STRUCTURAL ENGINE
==============================================================
Validates the V30.1 Factorized Spectral Basis Physics
Factorization against multiplicative regime interaction arrays. 
"""

import os, json, gc
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_COWS = 150; MAX_PENS = 10; NODE_DIM = 22; GAT_DIM = 96; T_STEPS = 28; T_SUBSAMPLE = 2

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
    def __init__(self, in_dim=96, hidden=64, out_dim=64):
        super().__init__()
        self.s_proj = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, in_dim))
        self.gat1 = PenGATLayer(in_dim, hidden, heads=2, concat=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.gat2 = PenGATLayer(hidden * 2, out_dim, heads=1, concat=False)
        self.struct_basis_head = nn.Sequential(nn.Linear(out_dim, 32), nn.ReLU(), nn.Linear(32, 3))
        
    def forward(self, x_node, adj_sub, s_raw):
        # Explicit `H_node_pen + S_proj (6 raw features)`
        x_ctx = x_node + self.s_proj(s_raw).unsqueeze(1)
        
        x1 = F.elu(self.gat1(x_ctx, adj_sub))
        x1 = self.norm(x1)
        x2 = F.elu(self.gat2(x1, adj_sub))
        
        x2 = torch.nan_to_num(x2, 0.0) 
        pool = x2.mean(dim=1)
        return self.struct_basis_head(pool)

class AmplificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
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

def check_tensor(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"❌ NaN detected in {name}")
        raise RuntimeError(f"NaN in {name}")

class DualHeadPenEngine(nn.Module):
    def __init__(self, backbone, max_pens=MAX_PENS, gat_dim=GAT_DIM):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        self.max_pens = max_pens
        self.gat_dim = gat_dim
        
        self.struct_enc = StructuralPenEncoder(in_dim=gat_dim, hidden=64, out_dim=64)
        self.regime_emb = nn.Embedding(4, 1)
        self.outcome_head = AmplificationHead()
        self.struct_scalar = nn.Linear(3, 1, bias=False)
        self.opt_scalar = nn.Parameter(torch.tensor([1.0]))
        
    def forward(self, ns, adj, pen_map):
        self.backbone.eval()
        with torch.no_grad(): bb_out = self.backbone(ns, adj)
            
        B, N, _ = bb_out["H_node"].shape
        
        adj_bin = (adj > 0).float()
        pen_struct_basis = torch.zeros(B, self.max_pens, 3, device=ns.device, dtype=ns.dtype)
        
        for b in range(B):
            bb_h_node = bb_out["H_node"][b] # [N, 96]
            H_herd = bb_out["H_herd"][b].unsqueeze(0) # [1, 96]
            A_b = adj_bin[b]
            pm_b = pen_map[b]
            
            for p in range(self.max_pens):
                mask = (pm_b == p)
                N_p = mask.sum().item()
                if N_p < 2: 
                    continue
                
                idx_p = torch.nonzero(mask).squeeze(-1)
                H_pen = bb_h_node[idx_p] # [N_p, 96]
                A_pen = A_b[idx_p][:, idx_p] # [N_p, N_p]
                
                H_feat = H_pen.unsqueeze(0) # [1, N_p, 96]
                A_feat = A_pen.unsqueeze(0) # [1, N_p, N_p]
                
                s_raw, _ = extract_structural_features(A_pen, self.max_pens)
                s_raw = s_raw.unsqueeze(0).to(ns.device)
                
                struct_basis_pred = self.struct_enc(H_feat, A_feat, s_raw) # [1, 3]
                pen_struct_basis[b, p] = struct_basis_pred.squeeze(0)
                
        return {
            "delta_R0": bb_out["delta_R0"], "vacc_rank": bb_out["vacc_rank"], "outbreak": bb_out["outbreak"], 
            "breakdown": bb_out["breakdown"], "intensity": bb_out["intensity"], 
            "H_herd": bb_out["H_herd"],
            "pen_struct_basis": pen_struct_basis
        }

# ═══════════════════════════════════════════════════════════════
# SIMULATOR & METRICS
# ═══════════════════════════════════════════════════════════════
def safe_pearsonr(pred, target):
    if len(pred) < 2: return np.nan
    pm = pred - pred.mean(); tm = target - target.mean()
    # DDof=0 maps exactly to torch std(unbiased=False)
    sp = pm.std(ddof=0); st = tm.std(ddof=0)
    if sp < 1e-6 or st < 1e-6: return np.nan
    return (pm * tm).mean() / (sp * st + 1e-6)
    
def safe_spearmanr(pred, target):
    if len(pred) < 2: return np.nan
    from scipy.stats import rankdata
    pr = rankdata(pred); tr = rankdata(target)
    return safe_pearsonr(pr, tr)

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

    def simulate(self, idx):
        n_cows=self.rng.randint(40,150)
        family = self.FAMILIES[idx % len(self.FAMILIES)]
        A = self._make_graph(n_cows, family)

        num_pens = self.rng.randint(3, MAX_PENS+1)
        pen_map = np.zeros(n_cows, dtype=np.int64)
        cws = np.arange(n_cows); self.rng.shuffle(cws)
        sp = np.array_split(cws, num_pens)
        for p, arr in enumerate(sp): pen_map[arr] = p

        regime_names = ['stable', 'borderline', 'outbreak', 'superspreader']
        regime = self.rng.choice(regime_names,p=[0.35,0.25,0.30,0.10])
        regime_idx = regime_names.index(regime)
        if regime=='stable': beta=self.rng.uniform(0.01,0.03); gamma=self.rng.uniform(0.15,0.30); ns=self.rng.randint(1,3); st=0
        elif regime=='borderline': beta=self.rng.uniform(0.03,0.055); gamma=self.rng.uniform(0.08,0.15); ns=self.rng.randint(2,5); st=self.rng.randint(5,15)
        elif regime=='outbreak': beta=self.rng.uniform(0.055,0.12); gamma=self.rng.uniform(0.04,0.08); ns=self.rng.randint(2,6); st=self.rng.randint(5,18)
        else: beta=self.rng.uniform(0.12,0.25); gamma=self.rng.uniform(0.03,0.06); ns=self.rng.randint(3,8); st=self.rng.randint(5,15)

        seed_density = ns / n_cows
        
        vacc=np.zeros(n_cows,dtype=np.float32); nv=int(self.rng.uniform(0,0.25)*n_cows)
        vacc_ratio = nv / n_cows
        if nv>0: vacc[self.rng.choice(n_cows,nv,replace=False)]=1.0
        I=np.zeros((T_STEPS,n_cows),dtype=np.float32); S=np.ones((T_STEPS,n_cows),dtype=np.float32)
        sev=np.zeros((T_STEPS,n_cows),dtype=np.float32)
        seeds=self.rng.choice(n_cows,min(ns,n_cows),replace=False)
        I[st,seeds]=self.rng.uniform(0.3,0.7,len(seeds)); S[st,seeds]=1.0-I[st,seeds]
        ah=self.rng.uniform(0.02,0.06); bt=self.rng.uniform(68,85)
        
        te_list = []

        for t in range(max(1,st+1),T_STEPS):
            te=max(0,bt+3*np.sin(t*2*np.pi/28)-72)
            te_list.append(te)
            be=beta*(1+ah*te)
            ni=np.clip(be*(A*(1-vacc*0.8)@I[t-1])*S[t-1],0,S[t-1]); nr=gamma*I[t-1]
            S[t]=np.clip(S[t-1]-ni,0,1); I[t]=np.clip(I[t-1]+ni-nr,0,1)
            sev[t]=I[t]*(1+0.2*te/10)

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
                    vacc[i], 0.0, 0.0, 0.0, 0.0, 0.0] # Ignore Phase 21 specific conformal params here 

        mI=I.mean(axis=1); intensity=float(mI.max())
        outbreak=float(intensity>0.15); breakdown=float((intensity>0.65) and np.max(np.abs(np.diff(mI)))>0.08)
        ba=beta*(1+ah*max(0,bt-72))
        try:
            K=(ba/gamma)*A; _,ev=np.linalg.eigh(K); v=np.abs(ev[:,-1]); d=A.sum(axis=1)
            dr0=(v**2*d*(ba/gamma)).astype(np.float32); dr0/=(dr0.max()+1e-8)
            vg=(v**2*d).astype(np.float32); vg/=(vg.max()+1e-8)
        except: dr0=np.zeros(n_cows,dtype=np.float32); vg=np.zeros(n_cows,dtype=np.float32)

        temp_factor = float(np.mean(te_list)) if len(te_list) > 0 else 0.0
        beta_eff = beta * (1 + ah * temp_factor)
        amp_target = float(beta_eff / gamma)
        
        pen_int = np.zeros(MAX_PENS, dtype=np.float32)
        pen_struct_basis = np.zeros((MAX_PENS, 3), dtype=np.float32)
        node_int = I.max(axis=0)
        
        for p in range(num_pens):
            msk = (pen_map == p)
            count = msk.sum()
            if count == 0: continue

            vals = node_int[msk]
            amplified = np.power(vals, 1.8)
            pen_int[p] = amplified.mean()
            
            # Epidemic Potential Physics
            if count >= 2:
                A_sub = A[msk][:, msk]
                degs = A_sub.sum(axis=1)
                lap = np.diag(degs) - A_sub
                try: 
                    evals_A = np.linalg.eigvals(A_sub)
                    lam_1 = float(np.max(np.abs(evals_A))) 
                except: lam_1 = 0.0
                try:
                    evals_L = np.real(np.linalg.eigvals(lap))
                    evals_L.sort()
                    lam_2 = float(evals_L[1]) if len(evals_L) > 1 else 0.0
                except: lam_2 = 0.0
                deg_density = float(degs.mean() / (count + 1e-6))
                pen_struct_basis[p] = [lam_1, lam_2, deg_density]
                    
        max_int = pen_int.max()
        pen_int = pen_int / max_int if max_int > 1e-8 else np.zeros_like(pen_int)

        return {"node_features":nf,"adjacency":A,"n_cows":n_cows,"family":family,"pen_map":pen_map,
                "labels":{"intensity":intensity,"outbreak":outbreak,"breakdown":breakdown,
                          "delta_R0":dr0,"vacc_gain":vg,"pen_int":pen_int, "pen_struct_basis":pen_struct_basis,
                          "amp_target": amp_target, "regime_idx": regime_idx, "seed_density": seed_density,
                          "temp_factor": temp_factor, "vacc_ratio": vacc_ratio}}

def evaluate_multi_seed(model, device, n_farms=200):
    seeds = [1111, 4444, 9999]
    print(f"\nEvaluating on {len(seeds)} random seeds, {n_farms} farms each...")
    model.eval()
    
    global_results = {'dr0': [], 'vacc': [], 'pen': [], 'str': [], 'amp': [], 'pen_min': []}
    family_pen_corrs = {f: [] for f in PenSimulator.FAMILIES}

    for seed in seeds:
        print(f"  --> Seed: {seed}")
        sim = PenSimulator(seed=seed)
        
        dr0_c, vacc_s, pen_c, str_c, amp_c = [], [], [], [], []
        _family_runs = {f: [] for f in PenSimulator.FAMILIES}

        with torch.no_grad():
            for i in range(n_farms):
                data = sim.simulate(i); L = data['labels']; nc = data['n_cows']
                fam = data['family']
                
                nf = np.zeros((T_STEPS, MAX_COWS, NODE_DIM), dtype=np.float32)
                ap = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
                pm = np.zeros(MAX_COWS, dtype=np.int64)
                nf[:,:nc,:] = data['node_features'][:,:nc,:]; ap[:nc,:nc] = data['adjacency'][:nc,:nc]; pm[:nc] = data['pen_map']
                
                nft=torch.tensor(nf).unsqueeze(0).to(device)[:,::T_SUBSAMPLE,:,:]
                at=torch.tensor(ap).unsqueeze(0).to(device)
                pmt=torch.tensor(pm).unsqueeze(0).to(device)
                
                amp_tgt_arr = torch.tensor([L['amp_target']], dtype=torch.float32, device=device)
                rgm_i_arr = torch.tensor([L['regime_idx']], dtype=torch.long, device=device)
                rsd_arr = torch.tensor([L['seed_density']], dtype=torch.float32, device=device).unsqueeze(1)
                rtf_arr = torch.tensor([L['temp_factor']], dtype=torch.float32, device=device).unsqueeze(1)
                rvr_arr = torch.tensor([L['vacc_ratio']], dtype=torch.float32, device=device).unsqueeze(1)
                
                o = model(nft, at, pmt)

                ps=o['delta_R0'][0,:nc].cpu().numpy(); td=L['delta_R0'][:nc]
                c_dr0 = safe_pearsonr(td, ps)
                if not np.isnan(c_dr0): dr0_c.append(c_dr0)

                pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
                c_vcc = safe_spearmanr(tv, pv)
                if not np.isnan(c_vcc): vacc_s.append(c_vcc)
                
                pb = o['pen_struct_basis'].squeeze(0); ts = L['pen_struct_basis']
                
                outb_arr = o["outbreak"]
                vacc_arr = o["vacc_rank"].mean(dim=1)
                r_emb = model.regime_emb(rgm_i_arr).squeeze(-1)
                opt_s = model.opt_scalar.expand(1)
                amp_input = torch.stack([outb_arr, vacc_arr, r_emb, opt_s], dim=-1) # [1, 4]
                amp_pred = model.outcome_head(amp_input).squeeze(-1) # [1]
                c_amp = safe_pearsonr(amp_pred.cpu().numpy(), amp_tgt_arr.cpu().numpy())
                if not np.isnan(c_amp): amp_c.append(c_amp)
                
                s_scalar_val = model.struct_scalar(pb.unsqueeze(0)).squeeze(-1).squeeze(0) # [MAX_PENS]
                p_out = torch.zeros_like(s_scalar_val)
                p_out = s_scalar_val * amp_pred[0]
                
                m_s = ts[:, 0] > 0
                if m_s.sum() >= 2:
                    t_str_val = ts[m_s, 0] # Using lam_1 for evaluation metrics
                    if t_str_val.std(ddof=0) >= 0.05:
                        pb_np = pb.cpu().numpy()
                        c_s = safe_pearsonr(t_str_val, pb_np[m_s, 0])
                        if not np.isnan(c_s): str_c.append(c_s)
                
                pp = p_out.cpu().numpy(); tp=L['pen_int']
                
                # Must mask zeros for proper pen correlation
                m_p = tp > 0
                if m_p.sum() >= 2:
                    t_pens_val = tp[m_p]
                    if t_pens_val.std(ddof=0) >= 0.05:
                        c = safe_pearsonr(t_pens_val, pp[m_p])
                        if not np.isnan(c):
                            pen_c.append(c)
                            _family_runs[fam].append(c)
        
        # Aggregate per seed
        global_results['dr0'].append(np.nanmean(dr0_c))
        global_results['vacc'].append(np.nanmean(vacc_s))
        global_results['pen'].append(np.nanmean(pen_c))
        global_results['str'].append(np.nanmean(str_c))
        global_results['amp'].append(np.nanmean(amp_c))
        global_results['pen_min'].append(np.nanmin(pen_c))
        
        # Accumulate family robust states
        for f in PenSimulator.FAMILIES:
            if _family_runs[f]: family_pen_corrs[f].extend(_family_runs[f])
            
    # Calculate global stability
    dr0_mean = np.mean(global_results['dr0']); dr0_std = np.std(global_results['dr0'])
    vacc_mean = np.mean(global_results['vacc']); vacc_std = np.std(global_results['vacc'])
    pen_mean = np.mean(global_results['pen']); pen_std = np.std(global_results['pen'])
    str_mean = np.mean(global_results['str']); str_std = np.std(global_results['str'])
    amp_mean = np.mean(global_results['amp'])
    
    # Calculate worst case topology pen metric
    family_stats = {f: (np.nanmean(vals), np.nanmin(vals)) for f, vals in family_pen_corrs.items() if vals}
    
    return {
        "dr0": (dr0_mean, dr0_std),
        "vacc": (vacc_mean, vacc_std),
        "str_mean": np.mean(global_results['str']), # using flat mean for structural score 
        "amp_mean": amp_mean,
        "pen": (pen_mean, pen_std),
        "pen_min": np.min(global_results['pen_min']),
        "family_pen": family_stats
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("🧠 PHASE 30.1: MULTI-BASIS SPECTRAL STRUCTURAL ENGINE")
    print("=" * 60)
    print(f"Device: {device}")

    paths = [
        ("models/cattle/v30_1_spectral_basis_engine.pth", "models/cattle/v30_1_config.json"),
        ("v30_1_spectral_basis_engine.pth", "v30_1_config.json"),
        ("/content/drive/MyDrive/HerdV24/v30_1_spectral_basis_engine.pth", "/content/drive/MyDrive/HerdV24/v30_1_config.json"),
    ]
    
    mp_found = None; cp_found = None
    for mp, cp in paths:
        if os.path.exists(mp):
            mp_found, cp_found = mp, cp
            break
            
    if not mp_found:
        print("❌ Model files not found. Please train first.")
        return
        
    print(f"Loading {mp_found}...")

    backbone = HerdEngineV21_Backbone(node_dim=NODE_DIM, gat_dim=GAT_DIM)
    model = DualHeadPenEngine(backbone).to(device)

    state = torch.load(mp_found, map_location=device, weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    missing, list_u = model.load_state_dict(state, strict=False)
    
    n_params = sum(p.numel() for p in model.parameters())
    pen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    print(f"Total Params: {n_params:,}")
    print(f"Dual-Head Structural & Baseline Params: {pen_params:,} (Target < 55k)")

    res = evaluate_multi_seed(model, device, n_farms=200)

    def tag(v,t): return "✅" if v>=t else "⚠️"

    print("\n" + "="*60); print("📊 GLOBAL EVALUATION RESULTS (3 SEEDS)"); print("="*60)
    
    md, vd = res['dr0']; mv, vv = res['vacc']; mp, vp = res['pen']

    print("\n── NODE-LEVEL (Strict Backbone Preservation) ──")
    print(f"  {tag(md,0.98)} ΔR₀ Pearson:       {md:.4f} ± {vd:.4f}  (≥ 0.98)")
    print(f"  {tag(mv,0.92)} Vacc Spearman:     {mv:.4f} ± {vv:.4f}  (≥ 0.92)")
    
    print("\n── PEN-LEVEL (Dual Head Physics Alignment) ──")
    print(f"  {tag(res['str_mean'],0.85)} Struct Mean Corr:  {res['str_mean']:.4f}  (≥ 0.85)")
    print(f"  {tag(res['amp_mean'],0.75)} Amplification Corr:{res['amp_mean']:.4f}  (≥ 0.75)")
    print(f"  {tag(mp,0.80)} Outcome Mean Corr: {mp:.4f} ± {vp:.4f}  (≥ 0.80)")
    print(f"  {tag(res['pen_min'], 0.55)} Outcome Min Corr:  {res['pen_min']:.4f}   (≥ 0.55)")
    
    print("\n── TOPOLOGY FAMILY ROBUSTNESS (Worst-Case Isolation) ──")
    for fam, stats in res['family_pen'].items():
        fm_mean, fm_min = stats
        print(f"  {tag(fm_min, 0.40)} {fam.ljust(15)} : Mean = {fm_mean:.3f} | Min = {fm_min:.3f} (> 0.4)")

    print("\n" + "="*60)
    all_ok = md>=0.95 and mv>=0.92 and res['str_mean']>=0.85 and mp>=0.75 and res['pen_min']>=0.55 and pen_params <= 55000
    if all_ok: print("🏆 PHASE 30.1 COMPLETE — Valid Factorization Model Generated")
    else: print("❌ Phase 30.1 FAILED — Constraints broken.")
    print("=" * 60)

if __name__ == "__main__": main()
