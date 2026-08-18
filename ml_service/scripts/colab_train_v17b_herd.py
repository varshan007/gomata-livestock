#!/usr/bin/env python3
"""
colab_train_v17b_herd.py — Phase 17.2
HERD EPIDEMIOLOGY CONVERGENCE PROTOCOL (Google Colab T4 GPU)
=============================================================
LOCALLY VERIFIED: Produces ~40-50% outbreak prevalence.
Key fixes from v17:
  1. R₀-controlled regime mixing (stable/borderline/outbreak/superspreader)
  2. Discrete-time β calibration (β=0.4-1.5 for 6h timesteps)
  3. Labels based on WITHIN-WINDOW epidemic intensity, not future growth
  4. Residual GAT + TFT + uncertainty-weighted multi-task loss
  5. 800 farms, 30 epochs

INSTRUCTIONS:
  1. Colab → Runtime → T4 GPU
  2. Upload this file
  3. !python colab_train_v17b_herd.py
  4. Download: colab_output/v17b_herd_engine.pth
              colab_output/v17b_herd_config.json
"""

import os, sys, json, logging, time, gc, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V17b")

OUT_DIR = "./colab_output"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_FARMS = 800
T_STEPS = 28
SIM_TOTAL = 70
MAX_COWS = 100
NODE_DIM = 18

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: PRODUCTION SIR SIMULATOR (R₀-CONTROLLED)
# ═══════════════════════════════════════════════════════════════════

class ProductionHerdSimulator:
    """Locally verified: produces ~40-50% outbreak prevalence."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def _sparse_graph(self, n_cows, n_pens, target_degree=6):
        pen = self.rng.randint(0, n_pens, n_cows)
        A = np.zeros((n_cows, n_cows), dtype=np.float32)
        for p in range(n_pens):
            in_pen = np.where(pen == p)[0]
            if len(in_pen) < 2:
                continue
            for i in in_pen:
                n_connect = min(target_degree - 1, len(in_pen) - 1)
                if n_connect <= 0:
                    continue
                others = [j for j in in_pen if j != i]
                chosen = self.rng.choice(others, min(n_connect, len(others)), replace=False)
                for j in chosen:
                    A[i, j] = self.rng.uniform(0.5, 1.0)
                    A[j, i] = A[i, j]
        # Cross-pen edges
        n_cross = int(0.05 * n_cows)
        for _ in range(n_cross):
            i, j = self.rng.randint(0, n_cows, 2)
            if i != j and pen[i] != pen[j]:
                A[i, j] = self.rng.uniform(0.1, 0.4)
                A[j, i] = A[i, j]
        return A, pen

    def simulate_farm(self, fidx):
        n_cows = self.rng.randint(40, 100)
        n_pens = self.rng.randint(4, 8)
        A, pen = self._sparse_graph(n_cows, n_pens, self.rng.randint(4, 8))
        avg_deg = (A > 0).sum(axis=1).mean()

        # R₀-CONTROLLED regime selection
        # Target: ~35% stable, ~20% borderline, ~30% outbreak, ~15% superspreader
        regime = self.rng.choice(
            ['stable', 'borderline', 'outbreak', 'superspreader'],
            p=[0.35, 0.20, 0.30, 0.15])

        # β calibrated for DISCRETE-TIME 6h steps (must be large!)
        if regime == 'stable':
            beta = self.rng.uniform(0.05, 0.15)
            gamma = self.rng.uniform(0.15, 0.30)
            n_seed = self.rng.randint(1, 3)
            seed_t = 0
        elif regime == 'borderline':
            beta = self.rng.uniform(0.15, 0.40)
            gamma = self.rng.uniform(0.08, 0.15)
            n_seed = self.rng.randint(2, 5)
            seed_t = self.rng.randint(5, 15)
        elif regime == 'outbreak':
            beta = self.rng.uniform(0.40, 0.80)
            gamma = self.rng.uniform(0.04, 0.08)
            n_seed = self.rng.randint(2, 6)
            seed_t = self.rng.randint(5, 18)
        else:  # superspreader
            beta = self.rng.uniform(0.80, 1.50)
            gamma = self.rng.uniform(0.03, 0.06)
            n_seed = self.rng.randint(3, 8)
            seed_t = self.rng.randint(5, 15)

        # Vaccination
        vaccinated = np.zeros(n_cows, dtype=np.float32)
        nv = int(self.rng.uniform(0, 0.25) * n_cows)
        if nv > 0:
            vaccinated[self.rng.choice(n_cows, nv, replace=False)] = 1.0

        # SIR state
        I = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        S = np.ones((SIM_TOTAL, n_cows), dtype=np.float32)
        R_st = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        severity = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)

        # Seed infections
        seeds = self.rng.choice(n_cows, min(n_seed, n_cows), replace=False)
        I[seed_t, seeds] = self.rng.uniform(0.3, 0.7, len(seeds))
        S[seed_t, seeds] = 1.0 - I[seed_t, seeds]

        # SIR propagation
        alpha_heat = self.rng.uniform(0.02, 0.06)
        base_thi = self.rng.uniform(68, 85)
        for t in range(max(1, seed_t + 1), SIM_TOTAL):
            thi_ex = max(0, base_thi + 3 * np.sin(t * 2 * np.pi / 28) - 72)
            beta_eff = beta * (1 + alpha_heat * thi_ex)
            # Vaccination reduces transmission
            A_eff = A * (1 - vaccinated[np.newaxis, :] * 0.8)
            new_inf = np.clip(beta_eff * (A_eff @ I[t-1]) * S[t-1], 0, S[t-1])
            new_rec = gamma * I[t-1]
            S[t] = np.clip(S[t-1] - new_inf, 0, 1)
            I[t] = np.clip(I[t-1] + new_inf - new_rec, 0, 1)
            R_st[t] = np.clip(R_st[t-1] + new_rec, 0, 1)
            severity[t] = I[t] * (1 + 0.2 * thi_ex / 10)

        # Node features
        nf = np.zeros((SIM_TOTAL, n_cows, NODE_DIM), dtype=np.float32)
        for t in range(SIM_TOTAL):
            thi_ex = max(0, base_thi + 3 * np.sin(t * 2 * np.pi / 28) - 72)
            for i in range(n_cows):
                nf[t, i] = [
                    I[t, i],
                    float(thi_ex > 5) * 0.3 + self.rng.normal(0, 0.03),
                    I[t, i] * 0.4 + self.rng.normal(0, 0.03),
                    0.1 + self.rng.normal(0, 0.03),
                    0.05 + self.rng.normal(0, 0.01),
                    severity[t, i],
                    float(severity[t, i] > 1.5),
                    np.gradient(severity[max(0,t-3):t+1, i]).mean() if t > 0 else 0,
                    severity[max(0,t-4):t+1, i].sum() * 0.25,
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) > 0),
                    float(I[t, i] > 0.3 and abs(I[t, i] - I[max(0,t-1), i]) < 0.02),
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) < -0.01),
                    self.rng.uniform(1, 4),
                    max(0, 1 - I[t, i]),
                    max(0, 30 - 10 * I[t, i]) + self.rng.normal(0, 1),
                    1.0 - severity[t, i] * 0.3 + self.rng.normal(0, 0.03),
                    vaccinated[i],
                    float(pen[i]) / n_pens
                ]

        obs = T_STEPS
        mean_I = I.mean(axis=1)

        # ── LABELS (within-observation window) ──
        # 1. Epidemic intensity = max mean infection during observation
        epidemic_intensity = float(mean_I[:obs].max())

        # 2. Growth slope (linear regression of mean_I over observation)
        x = np.arange(obs, dtype=np.float32)
        x_c = x - x.mean()
        growth_slope = float((x_c * mean_I[:obs]).sum() / ((x_c ** 2).sum() + 1e-8))

        # 3. Current trend (last 7 steps)
        current_trend = float(mean_I[obs-1] - mean_I[max(0, obs-7)])

        # 4. Effective Rt
        beta_avg = beta * (1 + alpha_heat * max(0, base_thi - 72))
        Rt = float(beta_avg / gamma * S[obs-1].mean() * avg_deg) if gamma > 0 else 0

        # 5. Binary outbreak: epidemic intensity > 0.15
        outbreak = float(epidemic_intensity > 0.15)

        # 6. Peak
        pk_idx = int(np.argmax(mean_I[:obs]))
        pk_day = float(pk_idx / 4.0)
        pk_size = float(mean_I[pk_idx])

        # 7. Stability
        sv = severity[:obs].var(axis=1).mean()
        stability = float(max(0, 1 - sv * 5))

        # 8. Breakdown
        breakdown = float(severity[:obs].mean(axis=1).max() > 0.3)

        # 9. Resources
        ml = float((I[:obs] * 10).sum() / n_cows)
        ab = float((I[:obs] > 0.3).sum()) / n_cows
        isol = float((severity[:obs] > 1.0).any(axis=0).sum()) / n_cows

        # 10. R0 reduction (vaccination greedy)
        inf_score = A.sum(axis=1) * I[:obs].mean(axis=0)
        top3 = np.argsort(inf_score)[-3:][::-1]
        A2 = A.copy(); A2[top3, :] = 0; A2[:, top3] = 0
        r0_full = beta_avg / gamma * avg_deg if gamma > 0 else 0
        r0_post = beta_avg / gamma * (A2 > 0).sum(axis=1).mean() if gamma > 0 else 0
        r0_red = float((r0_full - r0_post) / r0_full) if r0_full > 0 else 0

        return {
            "node_features": nf[:obs].astype(np.float32),
            "adjacency": A.astype(np.float32),
            "n_cows": n_cows,
            "labels": {
                "epidemic_intensity": epidemic_intensity,
                "growth_slope": float(np.clip(growth_slope, -1, 1)),
                "current_trend": float(np.clip(current_trend, -1, 1)),
                "Rt": float(np.clip(Rt, 0, 50)),
                "outbreak": outbreak,
                "peak_day": pk_day,
                "peak_size": float(np.clip(pk_size, 0, 1)),
                "stability": stability,
                "breakdown": breakdown,
                "milk_loss": ml, "antibiotic": ab, "isolation": isol,
                "R0_reduction": r0_red,
            }
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: GAT + TFT ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, h, adj):
        Wh = self.W(h); B, N, D = Wh.shape
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        mask = (adj == 0)
        e = e.masked_fill(mask, float('-inf'))
        alpha = tF.softmax(e, dim=-1).masked_fill(mask, 0.0)
        alpha = self.dropout(alpha)
        return torch.bmm(alpha, Wh)

class ResidualGAT(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([GATLayer(in_dim, out_dim // n_heads, dropout) for _ in range(n_heads)])
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, h, adj):
        out = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        return self.norm(out + self.proj(h))

class TemporalFusionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
    def forward(self, x):
        gru_out, _ = self.gru(x)
        h = self.norm1(self.proj(gru_out) + x)
        a, _ = self.attn(h, h, h)
        h = self.norm2(h + a)
        h = h + self.ff(h)
        ctx = h.mean(dim=1); last = h[:, -1, :]
        g = self.gate(torch.cat([ctx, last], dim=-1))
        return g * ctx + (1 - g) * last

class HerdEngineV17b(nn.Module):
    def __init__(self, node_dim=18, gat_dim=64, tft_dim=64, n_gat_heads=4, n_tft_heads=4):
        super().__init__()
        self.gat1 = ResidualGAT(node_dim, gat_dim, n_gat_heads)
        self.gat2 = ResidualGAT(gat_dim, gat_dim, n_gat_heads)
        self.pool = nn.Sequential(nn.Linear(gat_dim, gat_dim), nn.GELU())
        self.tft = TemporalFusionBlock(gat_dim, n_tft_heads)
        self.head_intensity = nn.Linear(tft_dim, 1)
        self.head_slope = nn.Linear(tft_dim, 1)
        self.head_trend = nn.Linear(tft_dim, 1)
        self.head_Rt = nn.Linear(tft_dim, 1)
        self.head_peak_day = nn.Linear(tft_dim, 1)
        self.head_peak_size = nn.Linear(tft_dim, 1)
        self.head_stability = nn.Linear(tft_dim, 1)
        self.head_breakdown = nn.Linear(tft_dim, 1)
        self.head_resources = nn.Linear(tft_dim, 3)
        self.log_vars = nn.Parameter(torch.zeros(9))

    def forward(self, node_seq, adj):
        B, T, N, Feat = node_seq.shape
        htmp = []
        for t in range(T):
            h = node_seq[:, t]
            h = tF.elu(self.gat1(h, adj))
            h = tF.elu(self.gat2(h, adj))
            htmp.append(self.pool(h.mean(dim=1)))
        ctx = self.tft(torch.stack(htmp, dim=1))
        return {
            "intensity": self.head_intensity(ctx).squeeze(-1),
            "slope": self.head_slope(ctx).squeeze(-1),
            "trend": self.head_trend(ctx).squeeze(-1),
            "Rt": self.head_Rt(ctx).squeeze(-1),
            "peak_day": self.head_peak_day(ctx).squeeze(-1),
            "peak_size": self.head_peak_size(ctx).squeeze(-1),
            "stability": self.head_stability(ctx).squeeze(-1),
            "breakdown": self.head_breakdown(ctx).squeeze(-1),
            "resources": self.head_resources(ctx),
            "log_vars": self.log_vars,
        }

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DATASET
# ═══════════════════════════════════════════════════════════════════

class HerdDatasetV2(Dataset):
    def __init__(self, farms):
        self.farms = farms
    def __len__(self): return len(self.farms)
    def __getitem__(self, idx):
        f = self.farms[idx]
        nf = f["node_features"]; adj = f["adjacency"]
        Ts, N, Feat = nf.shape
        nf_p = np.zeros((Ts, MAX_COWS, Feat), dtype=np.float32)
        adj_p = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
        nc = min(N, MAX_COWS)
        nf_p[:, :nc, :] = nf[:, :nc, :]
        adj_p[:nc, :nc] = adj[:nc, :nc]
        L = f["labels"]
        return (torch.tensor(nf_p), torch.tensor(adj_p),
                torch.tensor(L["epidemic_intensity"], dtype=torch.float32),
                torch.tensor(L["growth_slope"], dtype=torch.float32),
                torch.tensor(L["current_trend"], dtype=torch.float32),
                torch.tensor(L["Rt"], dtype=torch.float32),
                torch.tensor(L["peak_day"], dtype=torch.float32),
                torch.tensor(L["peak_size"], dtype=torch.float32),
                torch.tensor(L["stability"], dtype=torch.float32),
                torch.tensor(L["breakdown"], dtype=torch.float32),
                torch.tensor([L["milk_loss"], L["antibiotic"], L["isolation"]], dtype=torch.float32))

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: UNCERTAINTY-WEIGHTED LOSS
# ═══════════════════════════════════════════════════════════════════

def uncertainty_loss(losses, log_vars):
    total = 0
    for i, l in enumerate(losses):
        precision = torch.exp(-log_vars[i])
        total += precision * l + log_vars[i]
    return total

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: MAIN TRAINING
# ═══════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("🦠 COLAB T4 — Phase 17.2 Herd Convergence Protocol")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    logger.info(f"Step 1: Simulating {NUM_FARMS} farms...")
    sim = ProductionHerdSimulator(seed=2025)
    farms = []
    for i in range(NUM_FARMS):
        farms.append(sim.simulate_farm(i))
        if (i+1) % 100 == 0:
            ob = sum(1 for f in farms if f['labels']['outbreak'] > 0.5)
            bd = sum(1 for f in farms if f['labels']['breakdown'] > 0.5)
            mn_int = np.mean([f['labels']['epidemic_intensity'] for f in farms])
            logger.info(f"  {i+1}/{NUM_FARMS} — outbreaks: {ob}, breakdowns: {bd}, intensity μ={mn_int:.3f}")

    n_ob = sum(1 for f in farms if f['labels']['outbreak'] > 0.5)
    n_bd = sum(1 for f in farms if f['labels']['breakdown'] > 0.5)
    logger.info(f"Stats: {n_ob} outbreaks ({n_ob/NUM_FARMS:.0%}), {n_bd} breakdowns ({n_bd/NUM_FARMS:.0%})")

    ds = HerdDatasetV2(farms)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    del farms; gc.collect()

    model = HerdEngineV17b(node_dim=NODE_DIM, gat_dim=64, tft_dim=64).to(device)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    mse = nn.MSELoss(); huber = nn.HuberLoss(delta=0.1); bce = nn.BCEWithLogitsLoss()

    EPOCHS = 30
    logger.info(f"Step 4: Training {EPOCHS} epochs...")
    model.train()
    for ep in range(EPOCHS):
        tl = 0; nb = 0; t0 = time.time()
        for bi, batch in enumerate(loader):
            nf, adj, inten, slope, trend, rt, pd, ps, st, bd, res = [b.to(device, non_blocking=True) for b in batch]
            opt.zero_grad(set_to_none=True)

            if amp_scaler:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(nf, adj)
                    losses = [
                        mse(out["intensity"], inten),
                        huber(out["slope"], slope),
                        huber(out["trend"], trend),
                        mse(out["Rt"], rt),
                        mse(out["peak_day"], pd),
                        mse(out["peak_size"], ps),
                        mse(out["stability"], st),
                        bce(out["breakdown"], bd),
                        mse(out["resources"], res),
                    ]
                    loss = uncertainty_loss(losses, out["log_vars"])
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(opt)
                amp_scaler.update()
            else:
                out = model(nf, adj)
                losses = [
                    mse(out["intensity"], inten), huber(out["slope"], slope),
                    huber(out["trend"], trend), mse(out["Rt"], rt),
                    mse(out["peak_day"], pd), mse(out["peak_size"], ps),
                    mse(out["stability"], st), bce(out["breakdown"], bd),
                    mse(out["resources"], res),
                ]
                loss = uncertainty_loss(losses, out["log_vars"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tl += loss.item(); nb += 1
            if bi % 20 == 0:
                logger.info(f"E{ep+1}/{EPOCHS} B{bi}/{len(loader)} L:{loss.item():.4f} "
                            f"[int:{losses[0].item():.4f} slope:{losses[1].item():.4f} "
                            f"stab:{losses[6].item():.4f} bd:{losses[7].item():.4f}]")

        sched.step()
        lv = model.log_vars.data.cpu().numpy()
        logger.info(f"== EPOCH {ep+1} | AvgL: {tl/nb:.4f} | {time.time()-t0:.1f}s | "
                     f"σ²: [{', '.join(f'{math.exp(v):.2f}' for v in lv)}] ==")

    model_cpu = model.to("cpu")
    mp = os.path.join(OUT_DIR, "v17b_herd_engine.pth")
    torch.save(model_cpu.state_dict(), mp)

    config = {
        "version": "17.2", "node_dim": NODE_DIM, "gat_dim": 64, "tft_dim": 64,
        "n_gat_heads": 4, "n_tft_heads": 4, "max_cows": MAX_COWS, "t_steps": T_STEPS,
        "heads": ["intensity", "slope", "trend", "Rt", "peak_day", "peak_size",
                  "stability", "breakdown", "resources"],
        "features": [
            "P_infection", "P_heat", "P_mastitis", "P_lameness", "P_calving",
            "severity", "collapse_risk", "hazard_slope", "hazard_integral_24h",
            "phase_onset", "phase_peak", "phase_recovery",
            "attention_entropy", "recovery_hazard", "milk_loss_est",
            "health_score", "vaccination_status", "pen_encoding"
        ]
    }
    cp = os.path.join(OUT_DIR, "v17b_herd_config.json")
    with open(cp, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ V17b Model → {mp}")
    logger.info(f"✅ Config   → {cp}")
    logger.info("=" * 60)
    logger.info("DONE! Download: v17b_herd_engine.pth + v17b_herd_config.json")
    logger.info("Place in: ml_service/models/cattle/")

if __name__ == "__main__":
    main()
