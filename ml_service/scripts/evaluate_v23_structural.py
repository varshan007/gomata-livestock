#!/usr/bin/env python3
"""
evaluate_v23_structural.py
PHASE 23 PRODUCTION-GRADE SPECTRAL STRUCTURAL EVALUATION
==============================================================
Validates the V23 Low-Rank Structural Engine.
Verifies metrics across 3 separate random seeds and reports
worst-case correlations per topological family.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

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
# PHASE 23 STRUCTURAL COMPLETION ENCODER
# ═══════════════════════════════════════════════════════════════

def extract_pen_structural_features(adj_b, pen_map_b, max_pens=MAX_PENS):
    P = max_pens; N = adj_b.shape[0]
    feats = torch.zeros((P, 10), device=adj_b.device, dtype=torch.float32)
    valid_mask = torch.zeros(P, device=adj_b.device, dtype=torch.bool)
    adj_bin = (adj_b > 0).float()
    
    for p in range(P):
        mask = (pen_map_b == p); num_nodes = mask.sum()
        if num_nodes < 2: continue
        valid_mask[p] = True
        
        nip = mask.nonzero(as_tuple=True)[0]
        A_sub = adj_bin[nip][:, nip]
        
        n_sub = num_nodes.float()
        deg = A_sub.sum(dim=1); edges = deg.sum() / 2.0
        density = (2.0 * edges) / (n_sub * (n_sub - 1).clamp(min=1.0))
        avg_deg = deg.mean(); deg_var = deg.var(unbiased=False)
        
        tri = torch.diag(A_sub @ A_sub @ A_sub) / 2
        pairs = deg * (deg - 1) / 2
        cc = torch.zeros_like(tri); cc_mask = pairs > 0
        cc[cc_mask] = (tri[cc_mask] / pairs[cc_mask]).to(cc.dtype)
        avg_cc = cc.mean()
        
        D = torch.diag(deg); L = D - A_sub
        try:
            evals = torch.linalg.eigvalsh(L)
            fiedler = evals[1] if n_sub > 1 else torch.tensor(0.0, device=A_sub.device)
            max_eig = evals[-1] if n_sub > 0 else torch.tensor(0.0, device=A_sub.device)
            evals_norm = evals.clamp(min=1e-8)
            evals_p = evals_norm / evals_norm.sum()
            entropy = -(evals_p * torch.log(evals_p)).sum()
        except:
            fiedler = torch.tensor(0.0, device=A_sub.device)
            max_eig = torch.tensor(0.0, device=A_sub.device)
            entropy = torch.tensor(0.0, device=A_sub.device)
            
        A_full_pen_rows = adj_bin[nip]
        total_deg = A_full_pen_rows.sum()
        internal_vol = edges * 2
        cut_size = total_deg - internal_vol
        conductance = cut_size / (internal_vol.clamp(min=1.0))
        
        feats[p] = torch.stack([
            n_sub, edges, density, avg_deg, deg_var, 
            avg_cc, conductance, fiedler, max_eig, entropy
        ])
    return feats, valid_mask

class LowRankStructuralEncoder(nn.Module):
    def __init__(self, in_features=10, hidden1=32, hidden2=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden1), nn.LayerNorm(hidden1), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden1, hidden2), nn.GELU(), nn.Linear(hidden2, 1)
        )
    def forward(self, x): return self.mlp(x)

class StructuralPenEngine(nn.Module):
    def __init__(self, backbone, max_pens=MAX_PENS):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        self.max_pens = max_pens
        self.struct_enc = LowRankStructuralEncoder(in_features=10)
        self.g_param = nn.Parameter(torch.tensor(-3.0))
        
    def forward(self, ns, adj, pen_map):
        self.backbone.eval()
        with torch.no_grad(): bb_out = self.backbone(ns, adj)
        B, N = pen_map.shape; dr0 = bb_out["delta_R0"]; pidx = pen_map.clamp(0, self.max_pens - 1)
        
        pen_sum_dr0 = torch.zeros(B, self.max_pens, device=dr0.device, dtype=dr0.dtype)
        pen_sum_dr0.scatter_add_(1, pidx, dr0)
        pen_cnt = torch.zeros(B, self.max_pens, device=dr0.device, dtype=dr0.dtype)
        pen_cnt.scatter_add_(1, pidx, torch.ones_like(dr0))
        pen_pred_bb = pen_sum_dr0 / pen_cnt.clamp(min=1.0)
        
        struct_feats = torch.zeros(B, self.max_pens, 10, device=ns.device, dtype=torch.float32)
        for b in range(B):
            s_f, _ = extract_pen_structural_features(adj[b], pen_map[b], self.max_pens)
            struct_feats[b] = s_f
            
        sf_flat = struct_feats.view(-1, 10); mask_valid = (sf_flat[:, 0] > 0)
        if mask_valid.any():
            sf_mean = sf_flat[mask_valid].mean(dim=0); sf_std = sf_flat[mask_valid].std(dim=0).clamp(min=1e-6)
            struct_feats = (struct_feats - sf_mean.view(1,1,10)) / sf_std.view(1,1,10)
            struct_feats = torch.where(struct_feats.isnan(), torch.zeros_like(struct_feats), struct_feats)
        
        structural_delta = self.struct_enc(struct_feats).squeeze(-1)
        gate_val = torch.sigmoid(self.g_param) * 0.08
        pen_final = pen_pred_bb + gate_val * structural_delta
        
        return {
            "delta_R0": dr0, "vacc_rank": bb_out["vacc_rank"], "outbreak": bb_out["outbreak"], 
            "breakdown": bb_out["breakdown"], "intensity": bb_out["intensity"], 
            "pen_pred": pen_final, "gate": gate_val
        }

# ═══════════════════════════════════════════════════════════════
# SIMULATOR & METRICS (Identical API for evaluating)
# ═══════════════════════════════════════════════════════════════

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
        clust = np.zeros_like(tri, dtype=np.float32)
        valid = pairs > 0
        clust[valid] = (tri[valid] / pairs[valid]).astype(np.float32)
        
        return deg_n,betw,eig,clust

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
        node_int = I.max(axis=0)
        for p in range(num_pens):
            msk = (pen_map == p)
            if msk.sum() > 0: pen_int[p] = node_int[msk].mean()
        pen_int = pen_int / (pen_int.max() + 1e-8)

        return {"node_features":nf,"adjacency":A,"n_cows":n_cows,"family":family,"pen_map":pen_map,
                "labels":{"intensity":intensity,"outbreak":outbreak,"breakdown":breakdown,
                          "delta_R0":dr0,"vacc_gain":vg,"pen_int":pen_int}}

def evaluate_multi_seed(model, device, n_farms=200):
    seeds = [1111, 4444, 9999]
    print(f"\nEvaluating on {len(seeds)} random seeds, {n_farms} farms each...")
    model.eval()
    
    global_results = {'dr0': [], 'vacc': [], 'pen': [], 'pen_min': []}
    family_pen_corrs = {f: [] for f in PenSimulator.FAMILIES}

    for seed in seeds:
        print(f"  --> Seed: {seed}")
        sim = PenSimulator(seed=seed)
        
        dr0_c, vacc_s, pen_c = [], [], []
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
                
                o = model(nft, at, pmt)

                ps=o['delta_R0'][0,:nc].cpu().numpy(); td=L['delta_R0'][:nc]
                if td.std()>1e-4 and ps.std()>1e-4: dr0_c.append(pearsonr(td, ps)[0])

                pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
                if tv.std()>1e-4 and pv.std()>1e-4: vacc_s.append(spearmanr(tv, pv)[0])

                pp=o['pen_pred'][0].cpu().numpy(); tp=L['pen_int']
                if tp.std()>1e-4 and pp.std()>1e-4:
                    c = pearsonr(tp, pp)[0]
                    pen_c.append(c)
                    _family_runs[fam].append(c)
        
        # Aggregate per seed
        global_results['dr0'].append(np.nanmean(dr0_c))
        global_results['vacc'].append(np.nanmean(vacc_s))
        global_results['pen'].append(np.nanmean(pen_c))
        global_results['pen_min'].append(np.nanmin(pen_c))
        
        # Accumulate family robust states
        for f in PenSimulator.FAMILIES:
            if _family_runs[f]: family_pen_corrs[f].extend(_family_runs[f])
            
    # Calculate global stability
    dr0_mean = np.mean(global_results['dr0']); dr0_std = np.std(global_results['dr0'])
    vacc_mean = np.mean(global_results['vacc']); vacc_std = np.std(global_results['vacc'])
    pen_mean = np.mean(global_results['pen']); pen_std = np.std(global_results['pen'])
    
    # Calculate worst case topology pen metric
    family_stats = {f: (np.nanmean(vals), np.nanmin(vals)) for f, vals in family_pen_corrs.items() if vals}
    
    return {
        "dr0": (dr0_mean, dr0_std),
        "vacc": (vacc_mean, vacc_std),
        "pen": (pen_mean, pen_std),
        "pen_min": np.min(global_results['pen_min']),
        "family_pen": family_stats
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("🧠 PHASE 23: PRODUCTION SPECTRAL STRUCTURAL EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")

    paths = [
        ("models/cattle/v23_structural_engine.pth", "models/cattle/v23_config.json"),
        ("v23_structural_engine.pth", "v23_config.json"),
        ("/content/drive/MyDrive/HerdV23/v23_structural_engine.pth", "/content/drive/MyDrive/HerdV23/v23_config.json"),
    ]
    
    mp_found = None; cp_found = None
    for mp, cp in paths:
        if os.path.exists(mp) and os.path.exists(cp):
            mp_found, cp_found = mp, cp
            break
            
    if not mp_found:
        print("❌ Model files not found. Please train first.")
        return
        
    print(f"Loading {mp_found}...")

    backbone = HerdEngineV21_Backbone(node_dim=NODE_DIM, gat_dim=GAT_DIM)
    model = StructuralPenEngine(backbone).to(device)

    state = torch.load(mp_found, map_location=device, weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    missing, list_u = model.load_state_dict(state, strict=False)
    
    n_params = sum(p.numel() for p in model.parameters())
    pen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gate_val = (torch.sigmoid(model.g_param) * 0.08).item()
    
    print(f"Total Params: {n_params:,}")
    print(f"Structural Encoder Params: {pen_params:,} (Target < 12k)")
    print(f"Hard Squashed Gate Target : {gate_val:.4f} (Max 0.08)")

    res = evaluate_multi_seed(model, device, n_farms=200)

    def tag(v,t): return "✅" if v>=t else "⚠️"

    print("\n" + "="*60); print("📊 GLOBAL EVALUATION RESULTS (3 SEEDS)"); print("="*60)
    
    md, vd = res['dr0']; mv, vv = res['vacc']; mp, vp = res['pen']

    print("\n── NODE-LEVEL (Strict Backbone Preservation) ──")
    print(f"  {tag(md,0.98)} ΔR₀ Pearson:       {md:.4f} ± {vd:.4f}  (≥ 0.98)")
    print(f"  {tag(mv,0.92)} Vacc Spearman:     {mv:.4f} ± {vv:.4f}  (≥ 0.92)")
    
    print("\n── PEN-LEVEL (Structural Encoder Projection) ──")
    print(f"  {tag(mp,0.82)} Pen Mean Corr:     {mp:.4f} ± {vp:.4f}  (≥ 0.82)")
    print(f"  {tag(res['pen_min'], 0.55)} Pen Min Corr:      {res['pen_min']:.4f}   (≥ 0.55)")
    print(f"  {tag(-gate_val, -0.080)} Gate Final Value:  {gate_val:.3f}   (≤ 0.080)")
    
    print("\n── TOPOLOGY FAMILY ROBUSTNESS (Worst-Case Isolation) ──")
    for fam, stats in res['family_pen'].items():
        fm_mean, fm_min = stats
        print(f"  {tag(fm_min, 0.40)} {fam.ljust(15)} : Mean = {fm_mean:.3f} | Min = {fm_min:.3f} (> 0.4)")

    print("\n" + "="*60)
    all_ok = md>=0.98 and mv>=0.92 and mp>=0.82 and res['pen_min']>=0.55 and gate_val<=0.08
    if all_ok: print("🏆 PHASE 23 COMPLETE — Production Spectral Structural Completion Engine Active")
    else: print("❌ Phase 23 FAILED — Constraints broken.")
    print("=" * 60)

if __name__ == "__main__": main()
