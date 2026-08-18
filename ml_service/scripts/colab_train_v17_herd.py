#!/usr/bin/env python3
"""
colab_train_v17_herd.py — Phase 17
HERD EPIDEMIOLOGY ENGINE (Google Colab T4 GPU)
================================================
Self-contained Colab script. Builds a GNN + TFT herd intelligence layer:
  1. Stochastic SIR Outbreak Simulator with graph topology
  2. Graph Attention Network (spatial encoding)
  3. Temporal Fusion Transformer (temporal forecasting)
  4. Multi-head herd prediction (outbreak, peak, stability, collapse, resources)
  5. Vaccination optimization module

INSTRUCTIONS:
  1. Colab → Runtime → T4 GPU
  2. Upload this file
  3. !python colab_train_v17_herd.py
  4. Download: colab_output/v17_herd_engine.pth
              colab_output/v17_herd_config.json
  5. Place in: ml_service/models/cattle/
"""

import os, sys, json, logging, time, gc, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V17Herd")

OUT_DIR = "./colab_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: STOCHASTIC SIR OUTBREAK SIMULATOR
# ═══════════════════════════════════════════════════════════════════

NUM_FARMS = 120
T_STEPS = 28          # 7 days at 6h resolution
FORECAST_7D = 28      # 7d = 28 steps
FORECAST_14D = 56     # 14d = 56 steps
SIM_TOTAL = 56        # simulate 14 days total for labels

class HerdOutbreakSimulator:
    """
    Stochastic SIR epidemic simulator on a contact graph.
    Generates farm-level training data for the GAT+TFT engine.
    """
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def _build_graph(self, n_cows, n_pens):
        """Build pen-based sparse adjacency matrix."""
        pen_assign = self.rng.randint(0, n_pens, n_cows)
        rows, cols, vals = [], [], []
        for i in range(n_cows):
            for j in range(i+1, n_cows):
                if pen_assign[i] == pen_assign[j]:
                    w = 1.0  # same pen
                elif abs(pen_assign[i] - pen_assign[j]) == 1:
                    w = 0.3 * self.rng.uniform(0.5, 1.0)  # adjacent pen
                elif self.rng.random() < 0.05:
                    w = 0.1 * self.rng.uniform(0.3, 1.0)  # random weak link
                else:
                    continue
                rows.extend([i, j]); cols.extend([j, i]); vals.extend([w, w])
        A = np.zeros((n_cows, n_cows), dtype=np.float32)
        for r, c, v in zip(rows, cols, vals):
            A[r, c] = v
        return A, pen_assign

    def simulate_farm(self, farm_idx):
        """Simulate one farm's SIR epidemic over SIM_TOTAL timesteps."""
        n_cows = self.rng.randint(30, 80)
        n_pens = self.rng.randint(3, 8)
        A, pen_assign = self._build_graph(n_cows, n_pens)

        # Epidemiological parameters (varied per farm)
        beta = self.rng.uniform(0.05, 0.20)    # transmission (higher for outbreaks)
        gamma = self.rng.uniform(0.005, 0.04)  # recovery (slower)
        sigma = self.rng.uniform(0.005, 0.02)  # noise
        alpha_heat = self.rng.uniform(0.02, 0.08)  # heat amplification

        # Environmental modulation
        base_thi = self.rng.uniform(65, 85)
        thi_series = base_thi + 5 * np.sin(np.arange(SIM_TOTAL) * 2 * np.pi / 28) \
                     + self.rng.normal(0, 2, SIM_TOTAL)

        # State arrays: S, I, R per cow per timestep
        I = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        R_state = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        severity = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)

        # Seed initial infections (more aggressive seeding)
        n_seed = max(2, self.rng.randint(2, max(3, n_cows // 5)))
        seed_cows = self.rng.choice(n_cows, n_seed, replace=False)
        I[0, seed_cows] = self.rng.uniform(0.4, 0.9, n_seed)

        # Vaccination: some cows pre-vaccinated
        vacc_rate = self.rng.uniform(0, 0.3)
        vaccinated = np.zeros(n_cows, dtype=np.float32)
        n_vacc = int(vacc_rate * n_cows)
        if n_vacc > 0:
            vacc_cows = self.rng.choice(n_cows, n_vacc, replace=False)
            vaccinated[vacc_cows] = 1.0

        # Effective adjacency (vaccination removes edges)
        A_eff = A * (1.0 - vaccinated[np.newaxis, :])

        # SIR SDE propagation
        for t in range(1, SIM_TOTAL):
            thi_excess = max(0, thi_series[t] - 72)
            beta_eff = beta * (1 + alpha_heat * thi_excess)

            # Contagion force per cow
            force = beta_eff * (A_eff @ I[t-1])
            dI = force * (1 - I[t-1] - R_state[t-1]) - gamma * I[t-1] + sigma * self.rng.normal(0, 1, n_cows)
            I[t] = np.clip(I[t-1] + dI, 0, 1)

            # Recovery transitions
            recovery_prob = gamma * I[t-1]
            recovering = (self.rng.random(n_cows) < recovery_prob).astype(float)
            R_state[t] = np.clip(R_state[t-1] + recovering * 0.1, 0, 1)

            # Severity
            severity[t] = I[t] * (1 + 0.5 * thi_excess / 10) + self.rng.normal(0, 0.05, n_cows)

        # ── Build node features (mimicking V16 cow engine outputs) ──
        node_features = np.zeros((SIM_TOTAL, n_cows, 18), dtype=np.float32)
        for t in range(SIM_TOTAL):
            thi_ex = max(0, thi_series[t] - 72)
            for i in range(n_cows):
                node_features[t, i] = [
                    I[t, i],                                    # P_infection proxy
                    float(thi_ex > 5) * 0.3 + self.rng.normal(0, 0.05),  # P_heat
                    I[t, i] * 0.4 + self.rng.normal(0, 0.05),  # P_mastitis proxy
                    0.1 + self.rng.normal(0, 0.05),             # P_lameness
                    0.05 + self.rng.normal(0, 0.02),            # P_calving
                    severity[t, i],                              # severity
                    float(severity[t, i] > 2.0),                # collapse_risk
                    np.gradient(severity[max(0,t-3):t+1, i]).mean() if t > 0 else 0,  # hazard_slope
                    severity[max(0,t-4):t+1, i].sum() * 0.25,  # hazard_integral_24h
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) > 0),  # onset
                    float(I[t, i] > 0.3 and abs(I[t, i] - I[max(0,t-1), i]) < 0.02),  # peak
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) < -0.01),  # recovery
                    self.rng.uniform(1, 4),                      # attention_entropy
                    max(0, 1 - I[t, i]),                         # recovery_hazard
                    max(0, 30 - 10 * I[t, i]) + self.rng.normal(0, 1),  # milk_loss_est
                    1.0 - severity[t, i] * 0.3 + self.rng.normal(0, 0.05),  # health_score
                    vaccinated[i],                               # vaccination status
                    float(pen_assign[i]) / n_pens               # pen encoding
                ]

        # ── Compute labels ──
        obs_window = T_STEPS  # first 28 steps = 7 days
        # Outbreak labels for next 7d and 14d (after observation)
        total_infected_at_obs = (I[obs_window-1] > 0.3).sum()
        future_7d = I[obs_window:min(obs_window+28, SIM_TOTAL)]
        future_14d = I[obs_window:]

        outbreak_7d = float((future_7d > 0.3).sum(axis=1).max() > max(total_infected_at_obs + 1, n_cows * 0.15)) if len(future_7d) > 0 else 0
        outbreak_14d = float((future_14d > 0.3).sum(axis=1).max() > max(total_infected_at_obs + 1, n_cows * 0.10)) if len(future_14d) > 0 else 0

        # Peak infection
        infected_counts = (I > 0.3).sum(axis=1)
        peak_idx = np.argmax(infected_counts)
        peak_day = peak_idx / 4.0  # convert 6h steps to days
        peak_size = float(infected_counts[peak_idx]) / n_cows  # fraction

        # Herd stability index
        sev_var = severity[:obs_window].var(axis=1).mean()
        stability = max(0, 1.0 - sev_var * 2)

        # Regulatory breakdown
        mean_sev = severity[:obs_window].mean(axis=1)
        breakdown = float(mean_sev.max() > 0.8 and sev_var > 0.1)

        # Resource stress
        milk_loss_total = (I[:obs_window] * 10).sum() / n_cows
        antibiotic_demand = float((I[:obs_window] > 0.5).sum()) / n_cows
        isolation_need = float((severity[:obs_window] > 2.0).any(axis=0).sum()) / n_cows

        # R0 estimation
        avg_degree = (A > 0).sum(axis=1).mean()
        beta_eff_avg = beta * (1 + alpha_heat * max(0, base_thi - 72))
        R0 = beta_eff_avg / gamma * avg_degree if gamma > 0 else 0

        # Super-spreader: cow with highest weighted degree × infection
        influence = (A.sum(axis=1) * I[:obs_window].mean(axis=0))
        top_spreaders = np.argsort(influence)[-3:][::-1]

        # Vaccination R0 reduction (greedy removal of top spreaders)
        A_no_top = A.copy()
        A_no_top[top_spreaders, :] = 0
        A_no_top[:, top_spreaders] = 0
        avg_degree_post = (A_no_top > 0).sum(axis=1).mean()
        R0_post = beta_eff_avg / gamma * avg_degree_post if gamma > 0 else 0
        R0_reduction = (R0 - R0_post) / R0 if R0 > 0 else 0

        # Pen-level risk
        pen_risk = np.zeros(n_pens, dtype=np.float32)
        for p in range(n_pens):
            mask = pen_assign == p
            if mask.sum() > 0:
                pen_risk[p] = I[:obs_window, mask].mean()

        labels = {
            "outbreak_7d": outbreak_7d,
            "outbreak_14d": outbreak_14d,
            "peak_day": min(peak_day, 14.0),
            "peak_size": min(peak_size, 1.0),
            "stability": stability,
            "breakdown": breakdown,
            "milk_loss": milk_loss_total,
            "antibiotic": antibiotic_demand,
            "isolation": isolation_need,
            "R0": R0,
            "R0_reduction": R0_reduction,
        }

        # Return observation window data only
        return {
            "node_features": node_features[:obs_window],  # [T, N, 18]
            "adjacency": A,                                 # [N, N]
            "n_cows": n_cows,
            "n_pens": n_pens,
            "labels": labels,
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: GAT + TFT ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        """h: [B, N, F], adj: [B, N, N] → [B, N, out_dim]"""
        Wh = self.W(h)  # [B, N, D]
        B, N, D = Wh.shape

        # Pairwise attention
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))  # [B, N, N]

        # Mask by adjacency
        mask = (adj == 0)
        e = e.masked_fill(mask, float('-inf'))
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.masked_fill(mask, 0.0)

        return torch.bmm(alpha, Wh)  # [B, N, D]


class MultiHeadGAT(nn.Module):
    """Multi-head GAT layer."""
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([GraphAttentionLayer(in_dim, out_dim // n_heads) for _ in range(n_heads)])

    def forward(self, h, adj):
        head_outs = [head(h, adj) for head in self.heads]
        return torch.cat(head_outs, dim=-1)  # [B, N, out_dim]


class TemporalFusionBlock(nn.Module):
    """
    Simplified TFT: GRU temporal encoding + temporal attention + gating.
    Operates on graph-pooled herd representations.
    """
    def __init__(self, d_model, n_heads=4, ff_dim=128):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, x):
        """x: [B, T, D] → [B, D]"""
        gru_out, _ = self.gru(x)
        h = self.proj(gru_out)  # [B, T, D]
        h = self.norm1(h + x)   # residual

        attn_out, _ = self.attn(h, h, h)
        h = self.norm2(h + attn_out)
        h = h + self.ff(h)

        # Gated fusion of temporal info
        ctx = h.mean(dim=1)      # [B, D]
        last = h[:, -1, :]       # [B, D]
        gate = self.gate(torch.cat([ctx, last], dim=-1))
        return gate * ctx + (1 - gate) * last  # [B, D]


class HerdEpidemiologyEngine(nn.Module):
    """
    V17 Herd Intelligence Engine.
    GAT (spatial) → TFT (temporal) → Multi-head predictions.
    """
    def __init__(self, node_dim=18, gat_dim=64, tft_dim=64, n_gat_heads=4, n_tft_heads=4):
        super().__init__()
        # Graph attention (per-timestep spatial encoding)
        self.gat1 = MultiHeadGAT(node_dim, gat_dim, n_gat_heads)
        self.gat2 = MultiHeadGAT(gat_dim, gat_dim, n_gat_heads)
        self.gat_norm = nn.LayerNorm(gat_dim)

        # Pool cows → herd representation
        self.graph_pool = nn.Sequential(
            nn.Linear(gat_dim, gat_dim), nn.GELU())

        # Temporal fusion
        self.tft = TemporalFusionBlock(gat_dim, n_tft_heads, ff_dim=128)

        # ── Prediction Heads ──
        self.head_outbreak_7d = nn.Linear(tft_dim, 1)
        self.head_outbreak_14d = nn.Linear(tft_dim, 1)
        self.head_peak_day = nn.Linear(tft_dim, 1)
        self.head_peak_size = nn.Linear(tft_dim, 1)
        self.head_stability = nn.Linear(tft_dim, 1)
        self.head_breakdown = nn.Linear(tft_dim, 1)
        self.head_resources = nn.Linear(tft_dim, 3)  # milk_loss, antibiotic, isolation

    def forward(self, node_seq, adj):
        """
        node_seq: [B, T, N, Feat]  — batch of temporal node features
        adj:      [B, N, N]        — adjacency (same across time for now)
        Returns dict of predictions
        """
        B, T, N, Feat = node_seq.shape

        # GAT per timestep
        herd_temporal = []
        for t in range(T):
            h = node_seq[:, t]                        # [B, N, Feat]
            h = torch.nn.functional.elu(self.gat1(h, adj))  # [B, N, D]
            h = torch.nn.functional.elu(self.gat2(h, adj))  # [B, N, D]
            h = self.gat_norm(h)
            # Mean-pool over cows → herd vector
            herd_t = self.graph_pool(h.mean(dim=1))  # [B, D]
            herd_temporal.append(herd_t)

        herd_seq = torch.stack(herd_temporal, dim=1)  # [B, T, D]

        # TFT temporal reasoning
        ctx = self.tft(herd_seq)  # [B, D]

        # Predictions
        return {
            "outbreak_7d": self.head_outbreak_7d(ctx).squeeze(-1),
            "outbreak_14d": self.head_outbreak_14d(ctx).squeeze(-1),
            "peak_day": self.head_peak_day(ctx).squeeze(-1),
            "peak_size": self.head_peak_size(ctx).squeeze(-1),
            "stability": self.head_stability(ctx).squeeze(-1),
            "breakdown": self.head_breakdown(ctx).squeeze(-1),
            "resources": self.head_resources(ctx),  # [B, 3]
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DATASET
# ═══════════════════════════════════════════════════════════════════

MAX_COWS = 80  # pad/truncate to fixed size for batching

class HerdDataset(Dataset):
    def __init__(self, farms):
        self.farms = farms

    def __len__(self):
        return len(self.farms)

    def __getitem__(self, idx):
        f = self.farms[idx]
        nf = f["node_features"]  # [T, N, 18]
        adj = f["adjacency"]     # [N, N]
        Ts, N, Feat = nf.shape

        # Pad to MAX_COWS
        nf_pad = np.zeros((Ts, MAX_COWS, Feat), dtype=np.float32)
        adj_pad = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
        nc = min(N, MAX_COWS)
        nf_pad[:, :nc, :] = nf[:, :nc, :]
        adj_pad[:nc, :nc] = adj[:nc, :nc]

        lab = f["labels"]
        return (
            torch.tensor(nf_pad),
            torch.tensor(adj_pad),
            torch.tensor(lab["outbreak_7d"], dtype=torch.float32),
            torch.tensor(lab["outbreak_14d"], dtype=torch.float32),
            torch.tensor(lab["peak_day"], dtype=torch.float32),
            torch.tensor(lab["peak_size"], dtype=torch.float32),
            torch.tensor(lab["stability"], dtype=torch.float32),
            torch.tensor(lab["breakdown"], dtype=torch.float32),
            torch.tensor([lab["milk_loss"], lab["antibiotic"], lab["isolation"]], dtype=torch.float32),
            torch.tensor(lab["R0"], dtype=torch.float32),
            torch.tensor(lab["R0_reduction"], dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("🦠 COLAB T4 — Phase 17 Herd Epidemiology Engine Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── STEP 1: Simulate Farms ──
    logger.info(f"Step 1: Simulating {NUM_FARMS} farms...")
    sim = HerdOutbreakSimulator(seed=2024)
    farms = []
    for i in range(NUM_FARMS):
        farms.append(sim.simulate_farm(i))
        if (i+1) % 20 == 0:
            ob7d = sum(1 for f in farms if f['labels']['outbreak_7d'] > 0.5)
            logger.info(f"  Generated {i+1}/{NUM_FARMS} (outbreaks: {ob7d})")

    # Stats
    n_ob7 = sum(1 for f in farms if f['labels']['outbreak_7d'] > 0.5)
    n_ob14 = sum(1 for f in farms if f['labels']['outbreak_14d'] > 0.5)
    n_bd = sum(1 for f in farms if f['labels']['breakdown'] > 0.5)
    logger.info(f"Farm stats: {n_ob7} outbreaks (7d), {n_ob14} (14d), {n_bd} breakdowns")

    # ── STEP 2: Build DataLoader ──
    dataset = HerdDataset(farms)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    del farms; gc.collect()

    # ── STEP 3: Model ──
    model = HerdEpidemiologyEngine(node_dim=18, gat_dim=64, tft_dim=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3,
                                                      steps_per_epoch=len(loader), epochs=12)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # ── STEP 4: Train ──
    EPOCHS = 12
    logger.info(f"Step 4: Training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        tl = 0; nb = 0; t0 = time.time()
        for bi, batch in enumerate(loader):
            nf, adj, ob7, ob14, pd, ps, stab, bd, res, r0, r0r = [b.to(device, non_blocking=True) for b in batch]

            optimizer.zero_grad(set_to_none=True)

            if amp_scaler:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(nf, adj)
                    loss_ob7 = bce(out["outbreak_7d"], ob7)
                    loss_ob14 = bce(out["outbreak_14d"], ob14)
                    loss_pd = mse(out["peak_day"], pd)
                    loss_ps = mse(out["peak_size"], ps)
                    loss_stab = mse(out["stability"], stab)
                    loss_bd = bce(out["breakdown"], bd)
                    loss_res = mse(out["resources"], res)

                    loss = (loss_ob7 + loss_ob14 + 0.3 * loss_pd + 0.5 * loss_ps
                            + 0.8 * loss_stab + 0.6 * loss_bd + 0.4 * loss_res)

                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                out = model(nf, adj)
                loss_ob7 = bce(out["outbreak_7d"], ob7)
                loss_ob14 = bce(out["outbreak_14d"], ob14)
                loss_pd = mse(out["peak_day"], pd)
                loss_ps = mse(out["peak_size"], ps)
                loss_stab = mse(out["stability"], stab)
                loss_bd = bce(out["breakdown"], bd)
                loss_res = mse(out["resources"], res)

                loss = (loss_ob7 + loss_ob14 + 0.3 * loss_pd + 0.5 * loss_ps
                        + 0.8 * loss_stab + 0.6 * loss_bd + 0.4 * loss_res)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            tl += loss.item(); nb += 1
            if bi % 5 == 0:
                logger.info(f"E{epoch+1}/{EPOCHS} B{bi}/{len(loader)} "
                            f"L:{loss.item():.4f} [ob7:{loss_ob7.item():.3f} "
                            f"ob14:{loss_ob14.item():.3f} pd:{loss_pd.item():.3f} "
                            f"stab:{loss_stab.item():.3f} bd:{loss_bd.item():.3f}]")

        logger.info(f"== EPOCH {epoch+1} | AvgLoss: {tl/nb:.4f} | {time.time()-t0:.1f}s ==")

    # ── STEP 5: Save ──
    model_cpu = model.to("cpu")
    mp = os.path.join(OUT_DIR, "v17_herd_engine.pth")
    torch.save(model_cpu.state_dict(), mp)

    config = {
        "node_dim": 18, "gat_dim": 64, "tft_dim": 64,
        "n_gat_heads": 4, "n_tft_heads": 4,
        "max_cows": MAX_COWS, "t_steps": T_STEPS,
        "features": [
            "P_infection", "P_heat", "P_mastitis", "P_lameness", "P_calving",
            "severity", "collapse_risk", "hazard_slope", "hazard_integral_24h",
            "phase_onset", "phase_peak", "phase_recovery",
            "attention_entropy", "recovery_hazard", "milk_loss_est",
            "health_score", "vaccination_status", "pen_encoding"
        ]
    }
    cp = os.path.join(OUT_DIR, "v17_herd_config.json")
    with open(cp, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ V17 Model → {mp}")
    logger.info(f"✅ Config  → {cp}")
    logger.info("=" * 60)
    logger.info("DONE! Download from colab_output/:")
    logger.info("  1. v17_herd_engine.pth")
    logger.info("  2. v17_herd_config.json")
    logger.info("Place in: ml_service/models/cattle/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
