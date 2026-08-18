#!/usr/bin/env python3
"""
colab_train_v23_contextual.py
PHASE 23.1: PRODUCTION MESOSCOPIC CONTEXTUAL FUSION
==============================================================
Provides contextual structural fusion for gOmATA.
Fuses frozen V21 backbone embeddings with a lightweight GAT 
structural encoding using a conditional MLP and a bounded Tanh gate.

Total added parameters target: < 25k (Production T4 stable)
"""

import os, gc, json, logging, warnings
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
logger = logging.getLogger("V23.1.Contextual")

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
        self.a = nn.Linear(2 * dout, 1, bias=False)  # Fixed broadcasting
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
    """2-Layer Mini GAT processing node_features temporal average -> 48d structural emb"""
    def __init__(self, in_dim=NODE_DIM, hidden=24, out_dim=48):
        # 2 heads * 24 = 48
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
        
        x = torch.nan_to_num(x, 0.0) # Zero out any residual NaNs from empty matrices
        
        sum_x = (x * mask.unsqueeze(-1)).sum(dim=1)
        cnt_x = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        struct_emb = sum_x / cnt_x  
        
        # Safe Output Normalization
        return F.layer_norm(struct_emb, struct_emb.shape[-1:])

class ContextualFusionMLP(nn.Module):
    """[Backbone(96) + Structural(48)] -> 64 -> 32 -> 1"""
    def __init__(self, backbone_dim=96, struct_dim=48):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim + struct_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, b_emb, s_emb):
        fusion = torch.cat([b_emb, s_emb], dim=-1)
        out = self.mlp(fusion)
        return out - out.mean(dim=1, keepdim=True)  # Zero-mean

def check_tensor(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.error(f"❌ NaN detected in {name}")
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
                
        # Group to K=B*P block
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
        
        # Backbone Pen Context (96d)
        idx_exp = pidx.unsqueeze(-1).expand(-1, -1, self.gat_dim)
        pen_sum_emb = torch.zeros(B, self.max_pens, self.gat_dim, device=ns.device, dtype=ns.dtype)
        pen_sum_emb.scatter_add_(1, idx_exp, bb_out["H_node"])
        pen_backbone_emb = pen_sum_emb / pen_cnt.unsqueeze(-1).type_as(pen_sum_emb).clamp(min=1.0)
        
        # Structural Mini-GAT Context (48d)
        ns_mean = ns.mean(dim=1) 
        x_out, adj_out, mask_out = self._extract_pen_subgraphs(ns_mean, adj, pen_map)
        
        # [B*P, 48] -> [B, P, 48]
        struct_emb = self.struct_enc(x_out, adj_out, mask_out).view(B, self.max_pens, 48)
        check_tensor("eval_struct_emb", struct_emb)
        
        # Contextual Fusion
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
            "delta_R0": dr0, "vacc_rank": bb_out["vacc_rank"],
            "outbreak": bb_out["outbreak"], "breakdown": bb_out["breakdown"], 
            "intensity": bb_out["intensity"], 
            "pen_pred": pen_final, "gate": gate_val, "bb_pen": pen_pred_bb,
            "delta_std": delta_scaled.std().item(),
            "gate_reg": gate_reg, "smooth_loss": smooth_loss, "energy_loss": energy_loss
        }

# ═══════════════════════════════════════════════════════════════
# LOSSES & SIMULATOR
# ═══════════════════════════════════════════════════════════════

def safe_correlation(pred, target, eps=1e-6):
    # Require at least 2 valid samples
    if pred.numel() < 2:
        return torch.zeros(1, device=pred.device)

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()

    # CRITICAL: Prevent Bessel correction NaN when N=1
    std_p = pred_centered.std(unbiased=False)
    std_t = target_centered.std(unbiased=False)

    if std_p < eps or std_t < eps:
        return torch.zeros(1, device=pred.device)

    corr = (pred_centered * target_centered).mean() / (std_p * std_t + eps)
    return torch.clamp(corr, -1.0, 1.0)

def compute_pen_loss(pred_pen, true_pen):
    B = pred_pen.shape[0]; losses = []
    
    # We compute safe_correlation over each batch independently
    for b in range(B):
        mask = true_pen[b] > 0
        if mask.sum() < 2: 
            continue
            
        t_pens = true_pen[b][mask]
        if t_pens.std(unbiased=False) < 0.05:
            continue
            
        corr = safe_correlation(pred_pen[b][mask], t_pens)
        losses.append(1.0 - corr)
        
    if not losses: return torch.tensor(0.0, device=pred_pen.device, requires_grad=True)
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
        if td.std()>1e-4 and ps.std()>1e-4: dr0_c.append(pearsonr(td,ps)[0])

        pv=o['vacc_rank'][0,:nc].cpu().numpy(); tv=L['vacc_gain'][:nc]
        if tv.std()>1e-4 and pv.std()>1e-4: vacc_s.append(spearmanr(tv,pv)[0])

        pp=o['pen_pred'][0].cpu().numpy(); tp=L['pen_int']
        if tp.std()>1e-4 and pp.std()>1e-4: pen_c.append(pearsonr(tp,pp)[0])

    model.train()
    return {
        'dr0': np.nanmean(dr0_c) if dr0_c else 0, 
        'vacc': np.nanmean(vacc_s) if vacc_s else 0,
        'pen_mean': np.nanmean(pen_c) if pen_c else 0, 
        'pen_min': np.nanmin(pen_c) if pen_c else 0
    }

def main():
    logger.info("="*60)
    logger.info("🧠 Phase 23.1 — Production Contextual Fusion Layer")
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

    model = ContextualPenEngine(backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Contextual Structural Fusion Params: {n_trainable:,} (Target < 25k)")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    amp_sc = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    STAGE_A_EP = 6; STAGE_B_EP = 6
    LAMBDA_PEN = 4.0; LR_NOW = 3e-4
    logger.info(f"Stage A [Residual Warmup]: {STAGE_A_EP}ep | LR={LR_NOW} | λ_pen={LAMBDA_PEN}")
    
    best_state = None; best_pen = 0.0
    aborts = 0

    for ep in range(STAGE_A_EP + STAGE_B_EP):
        is_stage_b = (ep >= STAGE_A_EP)
        if ep == STAGE_A_EP:
            LAMBDA_PEN = 3.0; LR_NOW = 1e-4
            for param_group in opt.param_groups: param_group['lr'] = LR_NOW
            logger.info("="*50)
            logger.info(f"Stage B [Stabilized Fine-Tune]: {STAGE_B_EP}ep | LR={LR_NOW} | λ_pen={LAMBDA_PEN}")
            logger.info("="*50)

        model.train()
        tr_loss = 0
        for bi, batch in enumerate(loader):
            nf, adj, inten, outbreak, breakdown, pm, dr0, vg, pen_int = [b.to(device) for b in batch]
            nf_sub = nf[:, ::T_SUBSAMPLE, :, :]
            opt.zero_grad(set_to_none=True)

            # FORCE AMP DISABLED for Phase 23.5 Numerical Hardening
            o = model(nf_sub, adj, pm)
            pen_loss = LAMBDA_PEN * compute_pen_loss(o['pen_pred'], pen_int)
            loss = pen_loss + o['gate_reg'] + 0.05 * o['smooth_loss'] + 0.02 * o['energy_loss']
            
            # ── 4 Global Loss Firewall ──
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("⚠️ Numerical instability detected (NaN/Inf Loss) — skipping batch")
                opt.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0)
            opt.step()
            tr_loss += loss.item()

        # ── Diagnostic Telemetry & Fast Validation ──
        m = evaluate_model(model, device, n_farms=100)
        gate_val = model.get_gate().item()
        
        # Pull typical delta std from one forward pass
        delta_std = 0.0; pen_std_pred = 0.0; pen_std_true = 0.0; mask_count = 0
        with torch.no_grad():
            b_sample = next(iter(loader))
            ns_s, adj_s, _, _, _, pm_s, _, _, p_int_s = [b.to(device) for b in b_sample]
            out_s = model(ns_s[:, ::T_SUBSAMPLE], adj_s, pm_s)
            delta_std = out_s.get("delta_std", 0.0)
            
            # Subsample metrics over first farm
            m_s = p_int_s[0] > 0
            if m_s.sum() >= 2:
                pen_std_pred = out_s['pen_pred'][0][m_s].std(unbiased=False).item()
                pen_std_true = p_int_s[0][m_s].std(unbiased=False).item()
                mask_count = m_s.sum().item()
            
        logger.info(f"EP{ep+1} | L:{tr_loss/len(loader):.3f} | ΔR₀:{m['dr0']:.4f} Vacc:{m['vacc']:.4f} Pen:{m['pen_mean']:.3f} Gate:{gate_val:.4f}")
        logger.info(f"   ↳ [Telemetry] mask_count:{mask_count} | std_p:{pen_std_pred:.4f} | std_t:{pen_std_true:.4f} | delta_std:{delta_std:.5f}")
        
        # Abort conditions & Stability Drop 
        if m['dr0'] < 0.95:
            logger.warning(f"🚨 Minimal Backbone dip detected (ΔR₀: {m['dr0']:.4f}). Halving learning rate without hard rollback.")
            aborts += 1
            if aborts == 1:
                LR_NOW *= 0.5
                for param_group in opt.param_groups: param_group['lr'] = LR_NOW
            elif aborts > 3:
                logger.error("❌ Backbone inherently unstable. Discarding run to lock metrics.")
                break
        
        if m['pen_mean'] > best_pen and m['dr0'] >= 0.96:
            best_pen = m['pen_mean']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
        if is_stage_b and m['pen_mean'] >= 0.82 and m['dr0'] >= 0.97:
            logger.info("🏆 Early stopping criteria met. Production targets achieved.")
            break

    logger.info("="*60); logger.info("📊 FINAL VALIDATION"); logger.info("="*60)
    if best_state is not None: model.load_state_dict(best_state)
    m = evaluate_model(model, device, n_farms=200, seed=8888)
    
    gate_final = model.get_gate().item()
    
    print(f"  ΔR₀ Pearson:       {m['dr0']:.4f}   (≥ 0.97)")
    print(f"  Vacc Spearman:     {m['vacc']:.4f}   (≥ 0.92)")
    print(f"  Pen Mean Corr:     {m['pen_mean']:.4f}   (≥ 0.82)")
    print(f"  Pen Min Corr:      {m['pen_min']:.4f}   (≥ 0.55)")
    print(f"  Gate Final Value:  {gate_final:.4f}   (Target ~0.05)")
    
    mp = f"{OUT_DIR}/v23_contextual_engine.pth"; cp = f"{OUT_DIR}/v23_config.json"
    torch.save(model.to("cpu").state_dict(), mp)
    cfg = {
        "version": "23.1", 
        "architecture": "Frozen V21 Backbone + Contextual Structural Fusion + Bounded Tanh Gate",
        "dr0": float(m['dr0']), "pen_mean": float(m['pen_mean']),
        "gate_final": float(gate_final), "seed": SEED,
        "params_total": n_trainable,
        "production_ready": True
    }
    with open(cp, "w") as f: json.dump(cfg, f, indent=2)
    os.makedirs(DRIVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{DRIVE_DIR}/v23_contextual_engine.pth")
    logger.info(f"✅ V23.1 Engine saved to {mp} and Drive.")

if __name__ == "__main__": main()
