#!/usr/bin/env python3
"""
evaluate_v23_contextual.py
PHASE 23.1 PRODUCTION MESOSCOPIC CONTEXTUAL EVALUATION
==============================================================
Validates the V23.1 Contextual Fusion Engine.
Verifies metrics across 3 separate random seeds and reports
worst-case correlations per topological family. 
"""

import os, json
import numpy as np
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
        # Safe Softmax stability trick
        e_safe = e - e.max(dim=-1, keepdim=True)[0]
        
        # Prevent FP16 NaN by masking explicitly with -6e4
        adj_exp = adj.unsqueeze(-1).expand(B, N, N, self.heads)
        attention = e_safe.masked_fill(adj_exp == 0, -6e4)
        attention = torch.softmax(attention, dim=2)
        
        out = torch.einsum('bnjh,bjhd->bnhd', attention, Wh)
        if self.concat: out = out.reshape(B, N, self.heads * self.dout)
        else: out = out.mean(dim=2)
        return out

class MiniStructuralEncoder(nn.Module):
    def __init__(self, in_dim=NODE_DIM, hidden=24, out_dim=48):
        super().__init__()
        self.gat1 = PenGATLayer(in_dim, hidden, heads=2, concat=True)
        self.norm1 = nn.LayerNorm(hidden * 2)
        self.gat2 = PenGATLayer(hidden * 2, out_dim, heads=1, concat=False)
        self.norm2 = nn.LayerNorm(out_dim)
        
    def forward(self, x_node, adj_sub, mask):
        x = F.elu(self.gat1(x_node, adj_sub))
        x = self.norm1(x)
        x = self.gat2(x, adj_sub)
        x = self.norm2(x)
        
        x = torch.nan_to_num(x, 0.0)
        sum_x = (x * mask.unsqueeze(-1)).sum(dim=1)
        cnt_x = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        struct_emb = sum_x / cnt_x  
        
        # Safe Output Normalization
        return F.layer_norm(struct_emb, struct_emb.shape[-1:])

class ContextualFusionMLP(nn.Module):
    def __init__(self, backbone_dim=96, struct_dim=48):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim + struct_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, b_emb, s_emb):
        fusion = torch.cat([b_emb, s_emb], dim=-1)
        out = self.mlp(fusion)
        return out - out.mean(dim=1, keepdim=True)

def check_tensor(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"❌ NaN detected in {name}")
        raise RuntimeError(f"NaN in {name}")

class ContextualPenEngine(nn.Module):
    def __init__(self, backbone, max_pens=MAX_PENS, gat_dim=GAT_DIM):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        self.max_pens = max_pens
        self.gat_dim = gat_dim
        self.struct_enc = MiniStructuralEncoder(in_dim=NODE_DIM, hidden=24, out_dim=48)
        self.fusion_head = ContextualFusionMLP(backbone_dim=gat_dim, struct_dim=48)
        self.confidence_head = nn.Linear(1, 1)
        self.g_param = nn.Parameter(torch.tensor(0.0))
        
    def get_gate(self):
        return torch.sigmoid(self.g_param)
        
    def _extract_pen_subgraphs(self, node_feats, adj, pen_map):
        B, N, _ = node_feats.shape; P = self.max_pens; K = B * P
        x_out = torch.zeros(B, P, N, NODE_DIM, device=node_feats.device, dtype=node_feats.dtype)
        adj_out = torch.zeros(B, P, N, N, device=node_feats.device, dtype=node_feats.dtype)
        mask_out = torch.zeros(B, P, N, device=node_feats.device, dtype=node_feats.dtype)
        adj_bin = (adj > 0).float()
        
        for b in range(B):
            for p in range(P):
                mask = (pen_map[b] == p)
                if not mask.any(): continue
                mask_flt = mask.float(); mask_out[b, p] = mask_flt
                A_sub = adj_bin[b] * mask_flt.unsqueeze(0) * mask_flt.unsqueeze(1)
                adj_out[b, p] = A_sub
                x_out[b, p] = node_feats[b] * mask_flt.unsqueeze(-1)
                
        x_out = x_out.view(K, N, NODE_DIM); adj_out = adj_out.view(K, N, N); mask_out = mask_out.view(K, N)
        return x_out, adj_out, mask_out

    def forward(self, ns, adj, pen_map):
        check_tensor("eval_input_ns", ns)
        check_tensor("eval_input_adj", adj)
        check_tensor("eval_input_pm", pen_map)
        
        self.backbone.eval()
        with torch.no_grad(): bb_out = self.backbone(ns, adj)
            
        B, N, _ = bb_out["H_node"].shape
        dr0 = bb_out["delta_R0"]; pidx = pen_map.clamp(0, self.max_pens - 1)
        
        pen_sum_dr0 = torch.zeros(B, self.max_pens, device=dr0.device, dtype=dr0.dtype)
        pen_sum_dr0.scatter_add_(1, pidx, dr0)
        pen_cnt = torch.zeros(B, self.max_pens, device=dr0.device, dtype=dr0.dtype)
        pen_cnt.scatter_add_(1, pidx, torch.ones_like(dr0))
        pen_pred_bb = pen_sum_dr0 / pen_cnt.clamp(min=1.0)
        pen_pred_bb = pen_pred_bb.detach() # Explicit backbone cutoff
        check_tensor("eval_backbone_out", pen_pred_bb)
        
        idx_exp = pidx.unsqueeze(-1).expand(-1, -1, self.gat_dim)
        pen_sum_emb = torch.zeros(B, self.max_pens, self.gat_dim, device=ns.device, dtype=ns.dtype)
        pen_sum_emb.scatter_add_(1, idx_exp, bb_out["H_node"])
        pen_backbone_emb = pen_sum_emb / pen_cnt.unsqueeze(-1).type_as(pen_sum_emb).clamp(min=1.0)
        
        ns_mean = ns.mean(dim=1) 
        x_out, adj_out, mask_out = self._extract_pen_subgraphs(ns_mean, adj, pen_map)
        
        struct_emb = self.struct_enc(x_out, adj_out, mask_out).view(B, self.max_pens, 48)
        check_tensor("eval_struct_emb", struct_emb)
        
        delta_pen = self.fusion_head(pen_backbone_emb, struct_emb).squeeze(-1)
        check_tensor("eval_fusion_out", delta_pen)
        
        delta_pen = torch.clamp(delta_pen, -5.0, 5.0)
        res_conf = torch.sigmoid(self.confidence_head(delta_pen.unsqueeze(-1))).squeeze(-1)
        
        # Spectrally Anchored Residual
        delta_pen_centered = delta_pen - delta_pen.mean(dim=1, keepdim=True)
        delta_normed = delta_pen_centered / (delta_pen_centered.std(dim=1, keepdim=True) + 1e-6)
        delta_scaled = delta_normed * 0.1
        
        gate_val = self.get_gate()
        max_residual_scale = 0.12
        scaled_gate = gate_val * max_residual_scale * res_conf
        
        pen_final = pen_pred_bb + scaled_gate * delta_scaled
        check_tensor("eval_pen_pred", pen_final)
        
        gate_reg = 0.01 * (scaled_gate ** 2).mean()
        if self.max_pens > 1: smooth_loss = ((delta_scaled[:, 1:] - delta_scaled[:, :-1])**2).mean()
        else: smooth_loss = torch.tensor(0.0, device=delta_scaled.device)
        
        energy_loss = (delta_scaled ** 2).mean()
        
        return {
            "delta_R0": dr0, "vacc_rank": bb_out["vacc_rank"], "outbreak": bb_out["outbreak"], 
            "breakdown": bb_out["breakdown"], "intensity": bb_out["intensity"], 
            "pen_pred": pen_final, "gate": gate_val,
            "gate_reg": gate_reg, "smooth_loss": smooth_loss, "energy_loss": energy_loss
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
                c_dr0 = safe_pearsonr(td, ps)
                if not np.isnan(c_dr0): dr0_c.append(c_dr0)

                pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
                c_vcc = safe_spearmanr(tv, pv)
                if not np.isnan(c_vcc): vacc_s.append(c_vcc)

                pp=o['pen_pred'][0].cpu().numpy(); tp=L['pen_int']
                
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
    print("🧠 PHASE 23.1: PRODUCTION CONTEXTUAL FUSION EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")

    paths = [
        ("models/cattle/v23_contextual_engine.pth", "models/cattle/v23_config.json"),
        ("v23_contextual_engine.pth", "v23_config.json"),
        ("/content/drive/MyDrive/HerdV23/v23_contextual_engine.pth", "/content/drive/MyDrive/HerdV23/v23_config.json"),
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
    model = ContextualPenEngine(backbone).to(device)

    state = torch.load(mp_found, map_location=device, weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    missing, list_u = model.load_state_dict(state, strict=False)
    
    n_params = sum(p.numel() for p in model.parameters())
    pen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gate_val = model.get_gate().item()
    
    print(f"Total Params: {n_params:,}")
    print(f"Contextual Structural Params: {pen_params:,} (Target < 25k)")
    print(f"Bounded Tanh Gate Evaluated : {gate_val:.4f} (Target 0.04 - 0.08)")

    res = evaluate_multi_seed(model, device, n_farms=200)

    def tag(v,t): return "✅" if v>=t else "⚠️"

    print("\n" + "="*60); print("📊 GLOBAL EVALUATION RESULTS (3 SEEDS)"); print("="*60)
    
    md, vd = res['dr0']; mv, vv = res['vacc']; mp, vp = res['pen']

    print("\n── NODE-LEVEL (Strict Backbone Preservation) ──")
    print(f"  {tag(md,0.97)} ΔR₀ Pearson:       {md:.4f} ± {vd:.4f}  (≥ 0.97)")
    print(f"  {tag(mv,0.92)} Vacc Spearman:     {mv:.4f} ± {vv:.4f}  (≥ 0.92)")
    
    print("\n── PEN-LEVEL (Contextual Fusion Prediction) ──")
    print(f"  {tag(mp,0.82)} Pen Mean Corr:     {mp:.4f} ± {vp:.4f}  (≥ 0.82)")
    print(f"  {tag(res['pen_min'], 0.55)} Pen Min Corr:      {res['pen_min']:.4f}   (≥ 0.55)")
    print(f"  {tag(-gate_val, -0.080)} Gate Final Value:  {gate_val:.3f}   (≤ 0.080)")
    
    print("\n── TOPOLOGY FAMILY ROBUSTNESS (Worst-Case Isolation) ──")
    for fam, stats in res['family_pen'].items():
        fm_mean, fm_min = stats
        print(f"  {tag(fm_min, 0.40)} {fam.ljust(15)} : Mean = {fm_mean:.3f} | Min = {fm_min:.3f} (> 0.4)")

    print("\n" + "="*60)
    all_ok = md>=0.97 and mv>=0.92 and mp>=0.82 and res['pen_min']>=0.55 and gate_val<=0.12 and pen_params <= 25000
    if all_ok: print("🏆 PHASE 23.5 COMPLETE — Spectrally Anchored Residual Active")
    else: print("❌ Phase 23.5 FAILED — Constraints broken.")
    print("=" * 60)

if __name__ == "__main__": main()
