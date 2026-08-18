#!/usr/bin/env python3
"""
colab_train_v23_structural.py
PHASE 23 PRODUCTION-GRADE SPECTRAL STRUCTURAL COMPLETION LAYER
==============================================================
Hardening Phase 22.1 into production architecture.
Zero Backbone Drift. Zero parameter expansion.
Strict squashed gate logic (gate <= 0.08).
"""

import os, gc, json, logging, warnings, math
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

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
DRIVE_DIR   = "/content/drive/MyDrive/HerdV23"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V23.Structural")

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
# PHASE 23 STRUCTURAL COMPLETION ENCODER (PURE LOW-RANK MLP)
# ═══════════════════════════════════════════════════════════════

def extract_pen_structural_features(adj_b, pen_map_b, max_pens=MAX_PENS):
    """
    Analytically computes [10] structural descriptors per pen subgraph.
    """
    P = max_pens; N = adj_b.shape[0]
    feats = torch.zeros((P, 10), device=adj_b.device, dtype=torch.float32)
    valid_mask = torch.zeros(P, device=adj_b.device, dtype=torch.bool)
    
    adj_bin = (adj_b > 0).float()
    
    for p in range(P):
        mask = (pen_map_b == p)
        num_nodes = mask.sum()
        if num_nodes < 2: continue
        valid_mask[p] = True
        
        # Subgraph Excision
        nip = mask.nonzero(as_tuple=True)[0]
        A_sub = adj_bin[nip][:, nip]
        
        n_sub = num_nodes.float()
        deg = A_sub.sum(dim=1)
        edges = deg.sum() / 2.0
        
        density = (2.0 * edges) / (n_sub * (n_sub - 1).clamp(min=1.0))
        avg_deg = deg.mean()
        deg_var = deg.var(unbiased=False)
        
        # Clustering Coefficient
        tri = torch.diag(A_sub @ A_sub @ A_sub) / 2
        pairs = deg * (deg - 1) / 2
        cc = torch.zeros_like(tri)
        cc_mask = pairs > 0
        cc[cc_mask] = (tri[cc_mask] / pairs[cc_mask]).to(cc.dtype)
        avg_cc = cc.mean()
        
        # Spectral Analytics
        D = torch.diag(deg)
        L = D - A_sub
        
        try:
            evals = torch.linalg.eigvalsh(L)
            fiedler = evals[1] if n_sub > 1 else torch.tensor(0.0, device=A_sub.device)
            max_eig = evals[-1] if n_sub > 0 else torch.tensor(0.0, device=A_sub.device)
            
            # Spectral Entropy Proxy
            evals_norm = evals.clamp(min=1e-8)
            evals_p = evals_norm / evals_norm.sum()
            entropy = -(evals_p * torch.log(evals_p)).sum()
        except:
            fiedler = torch.tensor(0.0, device=A_sub.device)
            max_eig = torch.tensor(0.0, device=A_sub.device)
            entropy = torch.tensor(0.0, device=A_sub.device)
            
        # Approximation of Pen Conductance (Cut size over Vol)
        # For an isolated subgraph in a block diagonal matrix, this would inherently be 0
        # Compute outgoing edges vs internal volume
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
            nn.Linear(in_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, x):
        return self.mlp(x)

class StructuralPenEngine(nn.Module):
    def __init__(self, backbone, max_pens=MAX_PENS):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        
        self.max_pens = max_pens
        self.struct_enc = LowRankStructuralEncoder(in_features=10)
        
        # strict squashed gate: torch.sigmoid(g_param) * 0.08
        self.g_param = nn.Parameter(torch.tensor(-3.0))
        
    def forward(self, ns, adj, pen_map):
        self.backbone.eval()
        with torch.no_grad():
            bb_out = self.backbone(ns, adj)
            
        B, N = pen_map.shape
        dr0 = bb_out["delta_R0"]
        pidx = pen_map.clamp(0, self.max_pens - 1)
        
        # Backbone Pen Projection (Mean Pooling)
        pen_sum_dr0 = torch.zeros(B, self.max_pens, device=dr0.device, dtype=dr0.dtype)
        pen_sum_dr0.scatter_add_(1, pidx, dr0)
        pen_cnt = torch.zeros(B, self.max_pens, device=dr0.device, dtype=dr0.dtype)
        pen_cnt.scatter_add_(1, pidx, torch.ones_like(dr0))
        pen_pred_bb = pen_sum_dr0 / pen_cnt.clamp(min=1.0)
        
        # Structural Extraction
        struct_feats = torch.zeros(B, self.max_pens, 10, device=ns.device, dtype=torch.float32)
        for b in range(B):
            s_f, _ = extract_pen_structural_features(adj[b], pen_map[b], self.max_pens)
            struct_feats[b] = s_f
            
        # Global normalization to stabilize MLP
        sf_flat = struct_feats.view(-1, 10)
        mask_valid = (sf_flat[:, 0] > 0)
        if mask_valid.any():
            sf_mean = sf_flat[mask_valid].mean(dim=0)
            sf_std = sf_flat[mask_valid].std(dim=0).clamp(min=1e-6)
            struct_feats = (struct_feats - sf_mean.view(1,1,10)) / sf_std.view(1,1,10)
            struct_feats = torch.where(struct_feats.isnan(), torch.zeros_like(struct_feats), struct_feats)
        
        # Low Rank Projection
        structural_delta = self.struct_enc(struct_feats).squeeze(-1)
        
        # 0 <= gate <= 0.08 squashing operation
        gate_val = torch.sigmoid(self.g_param) * 0.08
        
        pen_final = pen_pred_bb + gate_val * structural_delta
        
        return {
            "delta_R0": dr0, "vacc_rank": bb_out["vacc_rank"],
            "outbreak": bb_out["outbreak"], "breakdown": bb_out["breakdown"], 
            "intensity": bb_out["intensity"], 
            "pen_pred": pen_final, "gate": gate_val, "bb_pen": pen_pred_bb
        }

# ═══════════════════════════════════════════════════════════════
# LOSSES & SIMULATOR
# ═══════════════════════════════════════════════════════════════

def correlation_loss(pred, target):
    B, P = pred.shape; losses = []
    for b in range(B):
        p, t = pred[b], target[b]
        if t.std() > 1e-4 and p.std() > 1e-4:
            pm = p - p.mean(); tm = t - t.mean()
            c = (pm * tm).sum() / (torch.sqrt((pm**2).sum()) * torch.sqrt((tm**2).sum()) + 1e-8)
            losses.append(1.0 - c)
    if not losses: return torch.tensor(1.0, device=pred.device, requires_grad=True)
    return sum(losses) / len(losses)

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
                    vacc[i],
                    0.0, # Zeroed per Phase 21 Conformal logic
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
# EVALUATION & TRAINING
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, device, n_farms=100, seed=7777):
    model.eval()
    sim = PenSimulator(seed=seed)
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

        ps=o['delta_R0'][0,:nc].cpu().numpy(); td=L['delta_R0'][:nc]
        if td.std()>1e-4 and ps.std()>1e-4:
            c,_=pearsonr(td,ps); dr0_c.append(c)

        pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
        if tv.std()>1e-4 and pv.std()>1e-4:
            s,_=spearmanr(tv,pv); vacc_s.append(s)

        pp=o['pen_pred'][0].cpu().numpy(); tp=L['pen_int']
        if tp.std()>1e-4 and pp.std()>1e-4:
            c,_=pearsonr(tp,pp); pen_c.append(c)

    model.train()
    return {
        'dr0': np.nanmean(dr0_c) if dr0_c else 0, 
        'vacc': np.nanmean(vacc_s) if vacc_s else 0,
        'pen_mean': np.nanmean(pen_c) if pen_c else 0, 
        'pen_min': np.nanmin(pen_c) if pen_c else 0
    }

def main():
    logger.info("="*60)
    logger.info("🧠 Phase 23 — Production-Grade Spectral Structural Completion Layer")
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

    model = StructuralPenEngine(backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Low-Rank Structural Encoder Params: {n_trainable:,} (Target < 12k)")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    amp_sc = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    STAGE_A_EP = 5; STAGE_B_EP = 8
    LAMBDA_PEN = 4.0; LR_NOW = 3e-4
    logger.info(f"Stage A [Structural Warmup]: {STAGE_A_EP}ep | LR={LR_NOW} | λ_pen={LAMBDA_PEN}")
    
    best_state = None; best_pen = 0.0
    rollbacks = 0

    for ep in range(STAGE_A_EP + STAGE_B_EP):
        is_stage_b = (ep >= STAGE_A_EP)
        if ep == STAGE_A_EP:
            LAMBDA_PEN = 3.0; LR_NOW = 1e-4
            for param_group in opt.param_groups: param_group['lr'] = LR_NOW
            logger.info("="*50)
            logger.info(f"Stage B [Joint Optimization]: {STAGE_B_EP}ep | LR={LR_NOW} | λ_pen={LAMBDA_PEN}")
            logger.info("="*50)

        model.train()
        tr_loss = 0
        for bi, batch in enumerate(loader):
            nf, adj, inten, outbreak, breakdown, pm, dr0, vg, pen_int = [b.to(device) for b in batch]
            nf_sub = nf[:, ::T_SUBSAMPLE, :, :]
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16) if amp_sc else torch.autocast("cpu", enabled=False):
                o = model(nf_sub, adj, pm)
                pen_loss = LAMBDA_PEN * correlation_loss(o['pen_pred'], pen_int)
                
                loss = pen_loss # Gate is hard bounded by torch.sigmoid(self.g_param) * 0.08

            if amp_sc:
                amp_sc.scale(loss).backward()
                amp_sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                amp_sc.step(opt); amp_sc.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                opt.step()
            tr_loss += loss.item()

        # ── Fast Validation Metrics ──
        m = evaluate_model(model, device, n_farms=100)
        gate_val = (torch.sigmoid(model.g_param) * 0.08).item()
        
        logger.info(f"EP{ep+1} | L:{tr_loss/len(loader):.3f} | ΔR₀:{m['dr0']:.4f} Vacc:{m['vacc']:.4f} Pen:{m['pen_mean']:.3f} Gate:{gate_val:.4f}")
        
        # Abort conditions & Stability Drop 
        if m['dr0'] < 0.97 or m['vacc'] < 0.90:
            logger.warning(f"🚨 Performance dip detected (ΔR₀: {m['dr0']:.4f}, Vacc: {m['vacc']:.4f}). Rolling back.")
            rollbacks += 1
            if rollbacks >= 3:
                logger.error("❌ 3 Rollbacks triggered. Aborting training to preserve Phase 21 dominance.")
                break
                
            if best_state is not None:
                model.load_state_dict(best_state)
                LR_NOW *= 0.5; LAMBDA_PEN *= 0.8
                for param_group in opt.param_groups: param_group['lr'] = LR_NOW
                logger.warning(f"   -> LR reduced to {LR_NOW:.2e}, λ_pen = {LAMBDA_PEN:.2f}")
        else:
            if m['pen_mean'] > best_pen:
                best_pen = m['pen_mean']
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
            if is_stage_b and m['pen_mean'] >= 0.82 and m['pen_min'] >= 0.55:
                logger.info("🏆 Early stopping criteria met. Production targets achieved.")
                break

    logger.info("="*60); logger.info("📊 FINAL VALIDATION"); logger.info("="*60)
    if best_state is not None: model.load_state_dict(best_state)
    m = evaluate_model(model, device, n_farms=200, seed=8888)
    
    gate_final = (torch.sigmoid(model.g_param) * 0.08).item()
    
    print(f"  ΔR₀ Pearson:       {m['dr0']:.4f}   (≥ 0.97)")
    print(f"  Vacc Spearman:     {m['vacc']:.4f}   (≥ 0.92)")
    print(f"  Pen Mean Corr:     {m['pen_mean']:.4f}   (≥ 0.82)")
    print(f"  Pen Min Corr:      {m['pen_min']:.4f}   (≥ 0.55)")
    print(f"  Gate Final Value:  {gate_final:.4f}   (≤ 0.08)")
    
    mp = f"{OUT_DIR}/v23_structural_engine.pth"; cp = f"{OUT_DIR}/v23_config.json"
    torch.save(model.to("cpu").state_dict(), mp)
    cfg = {
        "version": "23.0", 
        "architecture": "Frozen V21 Backbone + LowRankStructuralEncoder + SquashedGate",
        "dr0": float(m['dr0']), "pen_mean": float(m['pen_mean']),
        "gate_final": float(gate_final), "seed": SEED,
        "structural_features": [
            "node_count", "edge_count", "density", "avg_degree", "degree_variance", 
            "clustering_coefficient", "conductance", "fiedler_lambda2", "max_eigenvalue", "spectral_entropy"
        ],
        "gate_max": 0.08,
        "params_total": n_trainable,
        "production_ready": m['dr0'] >= 0.97 and m['pen_mean'] >= 0.82 and gate_final <= 0.08
    }
    with open(cp, "w") as f: json.dump(cfg, f, indent=2)
    os.makedirs(DRIVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{DRIVE_DIR}/v23_structural_engine.pth")
    logger.info(f"✅ V23 Engine saved to {mp} and Drive.")

if __name__ == "__main__": main()
