#!/usr/bin/env python3
"""
evaluate_v22_backbone_integrity.py (STANDALONE)
Phase 22.2 Backbone Integrity Validation Harness
======================================
Proves that the frozen Phase 21 backbone natively preserves ΔR0 >= 0.97
on the *exact same* simulation contract.
If this fails, feature ordering, normalization, or generation changed.
"""

import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

MAX_COWS = 150; MAX_PENS = 10; NODE_DIM = 22; GAT_DIM = 96; T_STEPS = 28

# ═══════════════════════════════════════════════════════════════
# PHASE 21 FROZEN BACKBONE
# ═══════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    def __init__(self, din, dout, drop=0.1):
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
        return self.dp(torch.softmax(e, dim=-1)) @ Wh

class ResGAT(nn.Module):
    def __init__(self, din, dout, nh=4, drop=0.1):
        super().__init__()
        self.hd = dout // nh; self.rem = dout - self.hd * (nh - 1)
        self.heads = nn.ModuleList([GATLayer(din, self.hd if i < nh-1 else self.rem, drop) for i in range(nh)])
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
        # NOTE: Phase 21 backbone purely takes ns and adj
        B, T, N, _ = ns.shape; herd_embs = []
        for t in range(T):
            h = self.node_enc(ns[:, t])
            h = F.elu(self.gat1(h, adj)); h = F.elu(self.gat2(h, adj))
            herd_embs.append(self.pool(h))
        H_node = h; H_herd = torch.stack(herd_embs, dim=1).mean(dim=1)
        
        return {
            "delta_R0": self.head_dr0(H_node).squeeze(-1),
            "vacc_rank": self.head_vacc(H_node).squeeze(-1),
            "outbreak": self.head_outbreak(H_herd).squeeze(-1),
            "breakdown": self.head_breakdown(H_herd).squeeze(-1),
            "intensity": self.head_intensity(H_herd).squeeze(-1),
            "HSI": self.head_hsi(H_herd).squeeze(-1),
        }

# ═══════════════════════════════════════════════════════════════
# SIMULATOR (Identical to 22.1)
# ═══════════════════════════════════════════════════════════════

class PenSimulator:
    FAMILIES = ['hub','community','small_world','scale_free', 'erdos_renyi','clustered','bipartite','multi_hub']
    def __init__(self, seed=9999): self.rng = np.random.RandomState(seed)

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
        tri=np.diag(ab@ab@ab)/2; pairs=ab.sum(axis=1)*(ab.sum(axis=1)-1)/2
        clust=np.zeros_like(tri, dtype=np.float32); valid=pairs>0
        clust[valid]=(tri[valid]/pairs[valid]).astype(np.float32)
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
        
        # NOTE: Verify order of features perfectly matches v21
        for t in range(T_STEPS):
            te=max(0,bt+3*np.sin(t*2*np.pi/28)-72)
            for i in range(n_cows):
                nf[t,i]=[
                    I[t,i],
                    float(te>5)*0.3+self.rng.normal(0,0.03),
                    I[t,i]*0.4+self.rng.normal(0,0.03),
                    0.1+self.rng.normal(0,0.03),
                    0.05+self.rng.normal(0,0.01),
                    sev[t,i],
                    float(sev[t,i]>1.5),
                    np.gradient(sev[max(0,t-3):t+1,i]).mean() if t>0 else 0,
                    sev[max(0,t-4):t+1,i].sum()*0.25,
                    float(I[t,i]>0.1 and (I[t,i]-I[max(0,t-1),i])>0),
                    float(I[t,i]>0.3 and abs(I[t,i]-I[max(0,t-1),i])<0.02),
                    float(I[t,i]>0.1 and (I[t,i]-I[max(0,t-1),i])<-0.01),
                    self.rng.uniform(1,4),max(0,1-I[t,i]),
                    max(0,30-10*I[t,i])+self.rng.normal(0,1),
                    1-sev[t,i]*0.3+self.rng.normal(0,0.03),
                    vacc[i],
                    float(pen_map[i]), # v21 used float(0.0) here if pen map wasn't injected... wait!
                    deg_n[i],
                    betw[i],
                    eig[i],
                    clust[i]
                ]

        mI=I.mean(axis=1); intensity=float(mI.max())
        outbreak=float(intensity>0.15); breakdown=float((intensity>0.65) and np.max(np.abs(np.diff(mI)))>0.08)
        ba=beta*(1+ah*max(0,bt-72))
        try:
            K=(ba/gamma)*A; _,ev=np.linalg.eigh(K); v=np.abs(ev[:,-1]); d=A.sum(axis=1)
            dr0=(v**2*d*(ba/gamma)).astype(np.float32); dr0/=(dr0.max()+1e-8)
            vg=(v**2*d).astype(np.float32); vg/=(vg.max()+1e-8)
        except: dr0=np.zeros(n_cows,dtype=np.float32); vg=np.zeros(n_cows,dtype=np.float32)

        return {"node_features":nf,"adjacency":A,"n_cows":n_cows,
                "labels":{"outbreak":outbreak,"delta_R0":dr0,"vacc_gain":vg}}


# ═══════════════════════════════════════════════════════════════
# STRICT BACKBONE INTEGRITY CHECK
# ═══════════════════════════════════════════════════════════════

def verify_backbone_integrity(n_farms=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("🔬 PHASE 22.2: BACKBONE INTEGRITY VALIDATION HARNESS")
    print("=" * 60)
    
    # 1. Load STRICTLY isolated Phase 21 Backbone
    print("Loading backbone from /content/drive/MyDrive/HerdV21/v21_backbone.pth ...")
    bb_path = "/content/drive/MyDrive/HerdV21/v21_backbone.pth"
    if not os.path.exists(bb_path): bb_path = "models/cattle/v21_backbone.pth"
    
    state = torch.load(bb_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    
    model = HerdEngineV21_Backbone(node_dim=NODE_DIM, gat_dim=GAT_DIM).to(device)
    model.load_state_dict(state)
    model.eval()

    for p in model.parameters(): p.requires_grad = False
    
    print("Backbone loaded.")
    print("Total Params:", sum(p.numel() for p in model.parameters()))
    if sum(p.numel() for p in model.parameters()) != 59335:
        print("❌ CRITICAL: Params do not match Phase 21 (59,335). Check architecture.")
        return

    # 2. Simulate Farms (Exact identical seed 9999)
    print(f"Simulating {n_farms} pure farms (seed 9999)...")
    sim = PenSimulator(seed=9999)
    
    dr0_c, vacc_s, y_ob_t, y_ob_p = [], [], [], []
    all_nf = []
    
    with torch.no_grad():
        for i in range(n_farms):
            data = sim.simulate(i)
            L = data['labels']; nc = data['n_cows']
            nf = np.zeros((T_STEPS, MAX_COWS, NODE_DIM), dtype=np.float32)
            ap = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
            nf[:,:nc,:] = data['node_features'][:,:nc,:]; ap[:nc,:nc] = data['adjacency'][:nc,:]
            
            # Record feature distributions for deep contract check
            all_nf.append(nf[:,:nc,:])
            
            # T_SUBSAMPLE=2
            nft = torch.tensor(nf).unsqueeze(0).to(device)[:,::2,:,:]
            at = torch.tensor(ap).unsqueeze(0).to(device)
            
            out = model(nft, at)
            
            ps=out['delta_R0'][0,:nc].cpu().numpy(); td=L['delta_R0'][:nc]
            if td.std()>0 and ps.std()>0:
                dr0_c.append(pearsonr(td, ps)[0])

            pv=out['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
            if tv.std()>0 and pv.std()>0:
                vacc_s.append(spearmanr(tv, pv)[0])
                
            y_ob_t.append(L['outbreak']); y_ob_p.append(out['outbreak'].item())

    # Deep Contract Check
    all_nf_concat = np.concatenate([f.flatten() for f in all_nf])
    print("\n=== DEEP DATA CONTRACT CHECK ===")
    print("Sample count (ΔR₀ node pairs):", len(dr0_c))
    print(f"Feature mean: {all_nf_concat.mean():.6f}")
    print(f"Feature std : {all_nf_concat.std():.6f}")

    delta = np.nanmean(dr0_c) if dr0_c else 0.0
    vacc = np.nanmean(vacc_s) if vacc_s else 0.0
    auc = roc_auc_score(y_ob_t, y_ob_p) if len(set(y_ob_t))>1 else 1.0

    print("\n=== BACKBONE INTEGRITY REPORT ===")
    print(f"ΔR₀ Pearson : {round(delta,4)} (Expected: ≥ 0.97)")
    print(f"Vacc Spearman: {round(vacc,4)} (Expected: ≥ 0.90)")
    print(f"Outbreak AUC: {round(auc,4)} (Expected: ≥ 0.97)")
    
    if delta >= 0.97:
        print("\n✅ PASSED: Data Pipeline aligns seamlessly with Phase 21.")
    else:
        print("\n❌ FAILED: Pipeline mismatch. 0.6131 target delta detected.")
        print("   -> Check Node Dim 17: Was it `0.0` or `pen_map[i]` in Phase 21 training?")
        
if __name__ == "__main__":
    verify_backbone_integrity()
