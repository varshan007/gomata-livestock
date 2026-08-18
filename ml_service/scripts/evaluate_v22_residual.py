#!/usr/bin/env python3
"""
evaluate_v22_residual.py (STANDALONE)
Phase 22 Residual Mesoscopic Pen Correction Evaluation
======================================
Self-contained. Only needs: v22_residual_engine.pth + v22_config.json
evaluates on 200 fresh farms (seed=9999).
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
        self.heads = nn.ModuleList([
            GATLayer(din, self.hd if i < nh-1 else self.rem, drop) for i in range(nh)])
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
            "HSI": self.head_hsi(H_herd).squeeze(-1),
        }


# ═══════════════════════════════════════════════════════════════
# PHASE 22 RESIDUAL PEN ENGINE
# ═══════════════════════════════════════════════════════════════

class PenResidualEngine(nn.Module):
    def __init__(self, backbone, max_pens=MAX_PENS, gat_dim=GAT_DIM):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        self.max_pens = max_pens; self.gat_dim = gat_dim
        
        PEN_INPUT_DIM = gat_dim + 6
        self.pen_mlp = nn.Sequential(
            nn.Linear(PEN_INPUT_DIM, 64), nn.ReLU(),
            nn.LayerNorm(64), nn.Linear(64, 32), nn.ReLU()
        )
        self.pen_corr_head = nn.Linear(32, 1)
        self.pen_risk_head = nn.Linear(32, 1)
        self.gate_w = nn.Linear(32, 1)

    def _compute_pen_features(self, H_node, adj, pen_map):
        B, N, _ = H_node.shape; P = self.max_pens
        pidx = pen_map.clamp(0, P - 1)
        idx_exp = pidx.unsqueeze(-1).expand(-1, -1, self.gat_dim)
        pen_sum = torch.zeros(B, P, self.gat_dim, device=H_node.device, dtype=H_node.dtype)
        pen_sum.scatter_add_(1, idx_exp, H_node)
        pen_cnt = torch.zeros(B, P, 1, device=H_node.device, dtype=H_node.dtype)
        pen_cnt.scatter_add_(1, pidx.unsqueeze(-1), torch.ones(B, N, 1, device=H_node.device, dtype=H_node.dtype))
        pen_emb = pen_sum / pen_cnt.clamp(min=1.0)
        
        feats = torch.zeros(B, P, 6, device=adj.device, dtype=adj.dtype)
        adj_bin = (adj > 0).float()
        for b in range(B):
            for p in range(P):
                nip = (pidx[b] == p).nonzero(as_tuple=True)[0]; nc = len(nip)
                if nc < 1: continue
                feats[b, p, 0] = nc / max(N, 1)
                if nc < 2: continue
                sa = adj_bin[b][nip][:, nip]; deg_in = sa.sum(dim=1); e_int = sa.sum() / 2
                feats[b, p, 1] = deg_in.mean() / max(nc - 1, 1)
                feats[b, p, 2] = deg_in.std() / max(nc - 1, 1) if nc > 1 else 0
                feats[b, p, 3] = (e_int / max(nc*(nc-1)/2, 1)).clamp(0, 1)
                if nc >= 3:
                    tri = torch.diag(sa @ sa @ sa) / 2
                    pairs = deg_in * (deg_in - 1) / 2
                    cc = torch.zeros_like(tri)
                    valid = pairs > 0
                    cc[valid] = (tri[valid] / pairs[valid]).to(cc.dtype)
                    feats[b, p, 4] = cc.mean().clamp(0, 1)
                    
                    try:
                        L = torch.diag(deg_in) - sa
                        ev = torch.linalg.eigvalsh(L.float())
                        feats[b, p, 5] = (ev[-1]).clamp(0, 5) / 5.0
                    except: 
                        feats[b, p, 5] = 0.0

        if not torch.isfinite(feats).all():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        mu = feats.mean(dim=(0, 1), keepdim=True)
        std = feats.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        feats = torch.nan_to_num(((feats - mu) / std).clamp(-3, 3), nan=0.0)
        return torch.cat([pen_emb, feats], dim=-1), pen_cnt.squeeze(-1)

    def forward(self, ns, adj, pen_map):
        with torch.no_grad():
            bb_out = self.backbone(ns, adj)
            H_node = bb_out["H_node"]; dr0_bb = bb_out["delta_R0"]; vacc = bb_out["vacc_rank"]
            ob = bb_out["outbreak"]; bd = bb_out["breakdown"]; inten = bb_out["intensity"]

        pen_feats, _ = self._compute_pen_features(H_node, adj, pen_map)
        pen_h = self.pen_mlp(pen_feats)
        gate = torch.clamp(torch.sigmoid(self.gate_w(pen_h)), 0.0, 0.25)
        p_corr = self.pen_corr_head(pen_h)
        corr_gated = (gate * p_corr).squeeze(-1)
        pen_risk = self.pen_risk_head(pen_h).squeeze(-1)

        B, N = dr0_bb.shape; pidx = pen_map.clamp(0, self.max_pens - 1)
        node_corr = torch.gather(corr_gated, 1, pidx)
        dr0_final = dr0_bb + node_corr

        return {
            "delta_R0": dr0_final, "vacc_rank": vacc,
            "outbreak": ob, "breakdown": bd, "intensity": inten, "HSI": bb_out["HSI"],
            "pen_risk": pen_risk, "gate": gate.mean(), "gate_max": gate.max()
        }


# ═══════════════════════════════════════════════════════════════
# SIMULATOR
# ═══════════════════════════════════════════════════════════════

class PenSimulator:
    FAMILIES = ['hub','community','small_world','scale_free',
                'erdos_renyi','clustered','bipartite','multi_hub']

    def __init__(self, seed=9999):
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

    def _graph_entropy(self, A):
        d=(A>0).sum(axis=1).astype(float); t=d.sum()
        if t==0: return 0.0
        p=d/t; p=p[p>0]; return float(-np.sum(p*np.log(p+1e-12)))

    def _compute_HSI(self, I, A, mI):
        s2=float(np.clip(np.var(mI)*10,0,1))
        dI=np.diff(mI) if len(mI)>1 else np.array([0.0])
        g=float(np.clip(np.mean(np.abs(dI))*20,0,1))
        d2I=np.diff(dI) if len(dI)>1 else np.array([0.0])
        a=float(np.clip(np.mean(np.abs(d2I))*40,0,1))
        c=float(np.clip(np.mean(1.0-I.mean(axis=1)),0,1))
        H=self._graph_entropy(A); Hm=np.log(A.shape[0]) if A.shape[0]>1 else 1.0
        base=float(np.clip(0.30*(1-s2)+0.30*(1-g)+0.20*(1-a)+0.10*c+0.10*np.clip(H/Hm,0,1),0,1))
        return float(np.clip(base*(1.0-float(np.max(mI))**2),0,1))

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
        ba=beta*(1+ah*max(0,bt-72)); hsi=self._compute_HSI(I,A,mI)
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
                "labels":{"intensity":intensity,"outbreak":outbreak,"breakdown":breakdown,"HSI":hsi,
                          "delta_R0":dr0,"vacc_gain":vg,"pen_int":pen_int}}


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_v22(model_path, config_path, n_farms=200, seed=9999):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("🧠 PHASE 22: RESIDUAL MESOSCOPIC PEN CORRECTION EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")

    # Needs to match V21 backbone + V22 Res Engine
    backbone = HerdEngineV21_Backbone(node_dim=NODE_DIM, gat_dim=GAT_DIM)
    model = PenResidualEngine(backbone).to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print(f"⚠️  Missing: {missing[:5]}")
    if unexpected: print(f"ℹ️  Extra: {len(unexpected)}")
    n_params = sum(p.numel() for p in model.parameters())
    pen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params Total: {n_params:,}, PenResidual: {pen_params:,} (Budget: 30k)")
    model.eval()

    sim = PenSimulator(seed=seed)
    y_ob_t, y_ob_p, y_bd_t, y_bd_p, y_int_t, y_int_p, y_hsi_t, y_hsi_p = [], [], [], [], [], [], [], []
    dr0_c, vacc_s, pen_c = [], [], []
    family_dr0, family_pen = {}, {}

    print(f"\nEvaluating {n_farms} farms (seed={seed})...")
    with torch.no_grad():
        for i in range(n_farms):
            data = sim.simulate(i); L = data['labels']; nc = data['n_cows']
            nf = np.zeros((T_STEPS, MAX_COWS, NODE_DIM), dtype=np.float32)
            ap = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
            pm = np.zeros(MAX_COWS, dtype=np.int64)
            nf[:,:nc,:] = data['node_features'][:,:nc,:]; ap[:nc,:nc] = data['adjacency'][:nc,:]
            pm[:nc] = data['pen_map']
            
            nft=torch.tensor(nf).unsqueeze(0).to(device)[:,::2,:,:]
            at=torch.tensor(ap).unsqueeze(0).to(device)
            pmt=torch.tensor(pm).unsqueeze(0).to(device)
            
            o = model(nft, at, pmt)

            y_ob_t.append(L['outbreak']); y_ob_p.append(o['outbreak'].item())
            y_bd_t.append(L['breakdown']); y_bd_p.append(o['breakdown'].item())
            y_int_t.append(L['intensity']); y_int_p.append(o['intensity'].item())
            y_hsi_t.append(L['HSI']); y_hsi_p.append(o['HSI'].item())

            ps=o['delta_R0'][0,:nc].cpu().numpy(); td=L['delta_R0'][:nc]
            c=np.nan; s=np.nan; pc=np.nan
            if td.std()>0 and ps.std()>0:
                c,_=pearsonr(td, ps); dr0_c.append(c)
                fam = data['family']; family_dr0.setdefault(fam, []).append(c)

            pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
            if tv.std()>0 and pv.std()>0:
                s,_=spearmanr(tv, pv); vacc_s.append(s)

            pp=o['pen_risk'][0].cpu().numpy(); tp=L['pen_int']
            if tp.std()>0 and pp.std()>0:
                pc,_=pearsonr(tp, pp); pen_c.append(pc)
                fam = data['family']; family_pen.setdefault(fam, []).append(pc)

            if (i+1)%50==0: print(f"  {i+1}/{n_farms}")

    oa = roc_auc_score(y_ob_t, y_ob_p) if len(set(y_ob_t))>1 else 1.0
    ba = roc_auc_score(y_bd_t, y_bd_p) if len(set(y_bd_t))>1 else 1.0
    ic = pearsonr(y_int_t, y_int_p)[0]
    hc = pearsonr(y_hsi_t, y_hsi_p)[0]
    md = np.mean(dr0_c) if dr0_c else 0
    mv = np.mean(vacc_s) if vacc_s else 0
    mp = np.mean(pen_c) if pen_c else 0
    min_p = np.min(pen_c) if pen_c else 0
    gate_mean = o['gate'].item()
    gate_max = o['gate_max'].item()

    def tag(v,t): return "✅" if v>=t else "⚠️"

    print("\n" + "="*60); print("📊 EVALUATION RESULTS"); print("="*60)
    print("\n── HERD-LEVEL ──")
    print(f"  {tag(oa,0.98)} Outbreak AUC:      {oa:.4f}   (≥ 0.98)")
    print(f"  {tag(ba,0.95)} Breakdown AUC:     {ba:.4f}   (≥ 0.95)")
    print(f"  {tag(ic,0.90)} Intensity Corr:    {ic:.4f}   (≥ 0.90)")
    print(f"  {tag(hc,0.90)} HSI Corr:          {hc:.4f}   (≥ 0.90)")
    
    print("\n── NODE-LEVEL (Preservation) ──")
    print(f"  {tag(md,0.97)} ΔR₀ Pearson:       {md:.4f} ± {np.std(dr0_c):.4f}  (≥ 0.97)")
    print(f"  {tag(mv,0.92)} Vacc Spearman:     {mv:.4f} ± {np.std(vacc_s):.4f}  (≥ 0.92)")
    
    print("\n── PEN-LEVEL (Residual Correction) ──")
    print(f"  {tag(mp,0.80)} Pen Mean Corr:     {mp:.4f} ± {np.std(pen_c):.4f}  (≥ 0.80)")
    print(f"  {'✅' if min_p >= 0.50 else '⚠️'} Pen Min Corr:      {min_p:.4f}   (≥ 0.50)")
    print(f"  {'✅' if gate_mean < 0.25 else '⚠️'} Gate Mean/Max:     {gate_mean:.3f} / {gate_max:.3f} (Max bounded at 0.25)")

    print("\n" + "="*60)
    all_ok = oa>=0.98 and ba>=0.95 and ic>=0.90 and hc>=0.90 and md>=0.97 and mv>=0.92 and mp>=0.80
    if all_ok: print("🏆 PHASE 22 COMPLETE — Production-Grade Residual Pen Correction")
    elif md >= 0.97 and mp >= 0.75: print("✅ Phase 22 acceptable — Pen correlation slightly under 0.80")
    else: print("❌ Phase 22 REJECTED — Targets not met")
    print("=" * 60)

if __name__ == "__main__":
    paths = [
        ("models/cattle/v22_residual_engine.pth", "models/cattle/v22_config.json"),
        ("v22_residual_engine.pth", "v22_config.json"),
        ("/content/models/cattle/v22_residual_engine.pth", "/content/models/cattle/v22_config.json"),
        ("/content/drive/MyDrive/HerdV22/v22_residual_engine.pth", "/content/drive/MyDrive/HerdV22/v22_config.json"),
    ]
    for mp, cp in paths:
        if os.path.exists(mp) and os.path.exists(cp):
            print(f"Found: {mp}"); evaluate_v22(mp, cp, n_farms=200); break
    else:
        print("❌ Model files not found.")
        for mp, cp in paths: print(f"   {mp}")
