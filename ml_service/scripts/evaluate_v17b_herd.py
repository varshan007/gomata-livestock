#!/usr/bin/env python3
"""
evaluate_v17b_herd.py — Phase 17.2 Part 2
V17b HERD EPIDEMIOLOGY ENGINE — OPERATIONAL VALIDATION

Evaluates V17b on 100 fresh unseen farms (seed=9999):
1. Epidemic Intensity MAE & Correlation
2. Growth Slope MAE
3. Outbreak AUC (derived from intensity)
4. Peak Day MAE & Peak Size MAE
5. Stability Correlation
6. Breakdown AUC
7. Vaccination R₀ Reduction
"""

import os, sys, json, logging, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V17b_Eval")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")
T_STEPS = 28; SIM_TOTAL = 70; MAX_COWS = 100; NODE_DIM = 18; N_EVAL = 100

# ═══════════════════════════════════════════════════════════════
# MODEL (must match Colab architecture exactly)
# ═══════════════════════════════════════════════════════════════

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
        }


# ═══════════════════════════════════════════════════════════════
# EVAL SIMULATOR (different seed, same physics)
# ═══════════════════════════════════════════════════════════════

class EvalSimulator:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def _sparse_graph(self, n_cows, n_pens, target_degree=6):
        pen = self.rng.randint(0, n_pens, n_cows)
        A = np.zeros((n_cows, n_cows), dtype=np.float32)
        for p in range(n_pens):
            in_pen = np.where(pen == p)[0]
            if len(in_pen) < 2: continue
            for i in in_pen:
                n_connect = min(target_degree - 1, len(in_pen) - 1)
                if n_connect <= 0: continue
                others = [j for j in in_pen if j != i]
                chosen = self.rng.choice(others, min(n_connect, len(others)), replace=False)
                for j in chosen:
                    A[i, j] = self.rng.uniform(0.5, 1.0); A[j, i] = A[i, j]
        n_cross = int(0.05 * n_cows)
        for _ in range(n_cross):
            i, j = self.rng.randint(0, n_cows, 2)
            if i != j and pen[i] != pen[j]:
                A[i, j] = self.rng.uniform(0.1, 0.4); A[j, i] = A[i, j]
        return A, pen

    def simulate(self, fidx):
        n_cows = self.rng.randint(40, 100)
        n_pens = self.rng.randint(4, 8)
        A, pen = self._sparse_graph(n_cows, n_pens, self.rng.randint(4, 8))
        avg_deg = (A > 0).sum(axis=1).mean()

        regime = self.rng.choice(['stable', 'borderline', 'outbreak', 'superspreader'],
                                  p=[0.35, 0.20, 0.30, 0.15])
        if regime == 'stable':
            beta = self.rng.uniform(0.05, 0.15); gamma = self.rng.uniform(0.15, 0.30)
            n_seed = self.rng.randint(1, 3); seed_t = 0
        elif regime == 'borderline':
            beta = self.rng.uniform(0.15, 0.40); gamma = self.rng.uniform(0.08, 0.15)
            n_seed = self.rng.randint(2, 5); seed_t = self.rng.randint(5, 15)
        elif regime == 'outbreak':
            beta = self.rng.uniform(0.40, 0.80); gamma = self.rng.uniform(0.04, 0.08)
            n_seed = self.rng.randint(2, 6); seed_t = self.rng.randint(5, 18)
        else:
            beta = self.rng.uniform(0.80, 1.50); gamma = self.rng.uniform(0.03, 0.06)
            n_seed = self.rng.randint(3, 8); seed_t = self.rng.randint(5, 15)

        vaccinated = np.zeros(n_cows, dtype=np.float32)
        nv = int(self.rng.uniform(0, 0.25) * n_cows)
        if nv > 0: vaccinated[self.rng.choice(n_cows, nv, replace=False)] = 1.0

        I = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        S = np.ones((SIM_TOTAL, n_cows), dtype=np.float32)
        severity = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)

        seeds = self.rng.choice(n_cows, min(n_seed, n_cows), replace=False)
        I[seed_t, seeds] = self.rng.uniform(0.3, 0.7, len(seeds))
        S[seed_t, seeds] = 1.0 - I[seed_t, seeds]

        alpha_heat = self.rng.uniform(0.02, 0.06)
        base_thi = self.rng.uniform(68, 85)
        for t in range(max(1, seed_t + 1), SIM_TOTAL):
            thi_ex = max(0, base_thi + 3 * np.sin(t * 2 * np.pi / 28) - 72)
            beta_eff = beta * (1 + alpha_heat * thi_ex)
            A_eff = A * (1 - vaccinated[np.newaxis, :] * 0.8)
            new_inf = np.clip(beta_eff * (A_eff @ I[t-1]) * S[t-1], 0, S[t-1])
            new_rec = gamma * I[t-1]
            S[t] = np.clip(S[t-1] - new_inf, 0, 1)
            I[t] = np.clip(I[t-1] + new_inf - new_rec, 0, 1)
            severity[t] = I[t] * (1 + 0.2 * thi_ex / 10)

        nf = np.zeros((SIM_TOTAL, n_cows, NODE_DIM), dtype=np.float32)
        for t in range(SIM_TOTAL):
            thi_ex = max(0, base_thi + 3 * np.sin(t * 2 * np.pi / 28) - 72)
            for i in range(n_cows):
                nf[t, i] = [
                    I[t, i], float(thi_ex > 5) * 0.3 + self.rng.normal(0, 0.03),
                    I[t, i] * 0.4 + self.rng.normal(0, 0.03), 0.1 + self.rng.normal(0, 0.03),
                    0.05 + self.rng.normal(0, 0.01), severity[t, i],
                    float(severity[t, i] > 1.5),
                    np.gradient(severity[max(0,t-3):t+1, i]).mean() if t > 0 else 0,
                    severity[max(0,t-4):t+1, i].sum() * 0.25,
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) > 0),
                    float(I[t, i] > 0.3 and abs(I[t, i] - I[max(0,t-1), i]) < 0.02),
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) < -0.01),
                    self.rng.uniform(1, 4), max(0, 1 - I[t, i]),
                    max(0, 30 - 10 * I[t, i]) + self.rng.normal(0, 1),
                    1.0 - severity[t, i] * 0.3 + self.rng.normal(0, 0.03),
                    vaccinated[i], float(pen[i]) / n_pens]

        obs = T_STEPS; mean_I = I.mean(axis=1)
        intensity = float(mean_I[:obs].max())
        x = np.arange(obs, dtype=np.float32); x_c = x - x.mean()
        slope = float((x_c * mean_I[:obs]).sum() / ((x_c ** 2).sum() + 1e-8))
        trend = float(mean_I[obs-1] - mean_I[max(0, obs-7)])
        beta_avg = beta * (1 + alpha_heat * max(0, base_thi - 72))
        Rt = float(beta_avg / gamma * S[obs-1].mean() * avg_deg) if gamma > 0 else 0
        outbreak = float(intensity > 0.15)
        pk_idx = int(np.argmax(mean_I[:obs]))
        pk_day = float(pk_idx / 4.0); pk_size = float(mean_I[pk_idx])
        sv = severity[:obs].var(axis=1).mean()
        stability = float(max(0, 1 - sv * 5))
        breakdown = float(severity[:obs].mean(axis=1).max() > 0.3)
        ml = float((I[:obs] * 10).sum() / n_cows)
        ab = float((I[:obs] > 0.3).sum()) / n_cows
        isol = float((severity[:obs] > 1.0).any(axis=0).sum()) / n_cows
        inf_score = A.sum(axis=1) * I[:obs].mean(axis=0)
        top3 = np.argsort(inf_score)[-3:][::-1]
        A2 = A.copy(); A2[top3, :] = 0; A2[:, top3] = 0
        r0_full = beta_avg / gamma * avg_deg if gamma > 0 else 0
        r0_post = beta_avg / gamma * (A2 > 0).sum(axis=1).mean() if gamma > 0 else 0
        r0_red = float((r0_full - r0_post) / r0_full) if r0_full > 0 else 0

        return {
            "node_features": nf[:obs].astype(np.float32),
            "adjacency": A.astype(np.float32),
            "labels": {
                "intensity": intensity, "slope": slope, "trend": trend,
                "Rt": Rt, "outbreak": outbreak,
                "peak_day": pk_day, "peak_size": pk_size,
                "stability": stability, "breakdown": breakdown,
                "milk_loss": ml, "antibiotic": ab, "isolation": isol,
                "R0_reduction": r0_red,
            }
        }


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("🔬 Phase 17.2 — V17b HERD ENGINE OPERATIONAL VALIDATION")
    logger.info("=" * 70)

    mp = os.path.join(MODEL_DIR, "v17b_herd_engine.pth")
    cp = os.path.join(MODEL_DIR, "v17b_herd_config.json")
    if not os.path.exists(mp):
        logger.error("V17b model not found!"); return

    with open(cp) as f:
        config = json.load(f)
    logger.info(f"V17b config: {config['version']}, heads: {config['heads']}")

    # ── STEP 1: Generate eval farms ──
    logger.info(f"Step 1: Generating {N_EVAL} unseen farms (seed=9999)...")
    sim = EvalSimulator(seed=9999)
    farms = [sim.simulate(i) for i in range(N_EVAL)]
    n_ob = sum(1 for f in farms if f['labels']['outbreak'] > 0.5)
    n_bd = sum(1 for f in farms if f['labels']['breakdown'] > 0.5)
    logger.info(f"Eval: {n_ob} outbreaks, {n_bd} breakdowns")

    # ── STEP 2: Model inference ──
    logger.info("Step 2: V17b inference...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HerdEngineV17b(node_dim=config["node_dim"], gat_dim=config["gat_dim"],
                            tft_dim=config["tft_dim"]).to(device)
    model.load_state_dict(torch.load(mp, map_location=device))
    model.eval()

    # Collect
    y_int, p_int = [], []
    y_slope, p_slope = [], []
    y_trend, p_trend = [], []
    y_pk_day, p_pk_day = [], []
    y_pk_size, p_pk_size = [], []
    y_stab, p_stab = [], []
    y_bd, p_bd = [], []
    y_ob = []
    y_r0r = []

    with torch.no_grad():
        for f in farms:
            nf = f["node_features"]; adj = f["adjacency"]
            Ts, N, Feat = nf.shape
            nf_p = np.zeros((Ts, MAX_COWS, Feat), dtype=np.float32)
            adj_p = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
            nc = min(N, MAX_COWS)
            nf_p[:, :nc, :] = nf[:, :nc, :]; adj_p[:nc, :nc] = adj[:nc, :nc]

            xn = torch.tensor(nf_p).unsqueeze(0).to(device)
            xa = torch.tensor(adj_p).unsqueeze(0).to(device)
            out = model(xn, xa)

            L = f["labels"]
            y_int.append(L["intensity"]); p_int.append(out["intensity"].item())
            y_slope.append(L["slope"]); p_slope.append(out["slope"].item())
            y_trend.append(L["trend"]); p_trend.append(out["trend"].item())
            y_pk_day.append(L["peak_day"]); p_pk_day.append(out["peak_day"].item())
            y_pk_size.append(L["peak_size"]); p_pk_size.append(out["peak_size"].item())
            y_stab.append(L["stability"]); p_stab.append(out["stability"].item())
            y_bd.append(L["breakdown"]); p_bd.append(torch.sigmoid(out["breakdown"]).item())
            y_ob.append(L["outbreak"])
            y_r0r.append(L["R0_reduction"])

    y_int=np.array(y_int); p_int=np.array(p_int)
    y_slope=np.array(y_slope); p_slope=np.array(p_slope)
    y_trend=np.array(y_trend); p_trend=np.array(p_trend)
    y_pk_day=np.array(y_pk_day); p_pk_day=np.array(p_pk_day)
    y_pk_size=np.array(y_pk_size); p_pk_size=np.array(p_pk_size)
    y_stab=np.array(y_stab); p_stab=np.array(p_stab)
    y_bd=np.array(y_bd); p_bd=np.array(p_bd)
    y_ob=np.array(y_ob); y_r0r=np.array(y_r0r)

    # ── STEP 3: Metrics ──
    logger.info("\n" + "=" * 70)
    logger.info("📊 V17b HERD ENGINE — EVALUATION RESULTS")
    logger.info("=" * 70)

    # A. Epidemic Intensity
    int_mae = np.abs(y_int - p_int).mean()
    int_corr, int_p = pearsonr(y_int, p_int) if np.std(y_int) > 0 and np.std(p_int) > 0 else (0, 1)
    logger.info(f"\n🦠 A. EPIDEMIC INTENSITY")
    logger.info(f"  MAE:         {int_mae:.4f}")
    logger.info(f"  Correlation: {int_corr:.4f} (p={int_p:.4e})")

    # B. Outbreak AUC (derived from intensity predictions)
    # Use intensity prediction as outbreak score
    ob_auc = roc_auc_score(y_ob, p_int) if len(np.unique(y_ob)) > 1 else 0
    logger.info(f"\n🚨 B. OUTBREAK DETECTION (from intensity)")
    logger.info(f"  AUC: {ob_auc:.4f}")

    # C. Growth Slope
    slope_mae = np.abs(y_slope - p_slope).mean()
    slope_corr, _ = pearsonr(y_slope, p_slope) if np.std(y_slope) > 0 and np.std(p_slope) > 0 else (0, 1)
    logger.info(f"\n📈 C. GROWTH SLOPE")
    logger.info(f"  MAE:         {slope_mae:.5f}")
    logger.info(f"  Correlation: {slope_corr:.4f}")

    # D. Peak Prediction
    pd_mae = np.abs(y_pk_day - p_pk_day).mean()
    ps_mae = np.abs(y_pk_size - p_pk_size).mean()
    logger.info(f"\n📊 D. PEAK PREDICTION")
    logger.info(f"  Peak Day MAE:  {pd_mae:.2f} days")
    logger.info(f"  Peak Size MAE: {ps_mae:.4f}")

    # E. Stability
    stab_corr, stab_p = pearsonr(y_stab, p_stab) if np.std(y_stab) > 0 and np.std(p_stab) > 0 else (0, 1)
    logger.info(f"\n🏥 E. HERD STABILITY")
    logger.info(f"  Correlation: {stab_corr:.4f} (p={stab_p:.4e})")

    # F. Breakdown
    bd_auc = roc_auc_score(y_bd, p_bd) if len(np.unique(y_bd)) > 1 else 0
    logger.info(f"\n⚠️  F. REGULATORY BREAKDOWN")
    logger.info(f"  AUC: {bd_auc:.4f}")

    # G. Vaccination R₀ Reduction
    mean_r0r = y_r0r.mean()
    logger.info(f"\n💉 G. VACCINATION R₀ REDUCTION")
    logger.info(f"  Mean (greedy top-3): {mean_r0r:.2%}")

    # ── PASS CRITERIA ──
    logger.info("\n" + "=" * 70)
    logger.info("🏁 V17b — PASS CRITERIA")
    logger.info("=" * 70)

    checks = [
        ("Outbreak AUC ≥ 0.85", ob_auc, 0.85),
        ("Intensity Corr ≥ 0.80", int_corr, 0.80),
        ("Peak Day MAE ≤ 1.5d", -pd_mae, -1.5),
        ("Stability Corr ≥ 0.80", stab_corr, 0.80),
        ("Breakdown AUC ≥ 0.85", bd_auc, 0.85),
        ("R₀ Reduction ≥ 25%", mean_r0r, 0.25),
    ]
    all_pass = True
    for name, val, target in checks:
        passed = val >= target
        st = "✅" if passed else "❌"
        if "MAE" in name: logger.info(f"  {st} {name}: {abs(val):.2f}")
        elif "R₀" in name: logger.info(f"  {st} {name}: {val:.2%}")
        else: logger.info(f"  {st} {name}: {val:.4f}")
        if not passed: all_pass = False

    if all_pass:
        logger.info("\n✅ V17b HERD ENGINE — ALL CRITERIA PASSED. PRODUCTION READY.")
    else:
        logger.info("\n⚠️ Some criteria not met. See individual heads above.")

    report = {
        "version": "17.2",
        "intensity_mae": float(int_mae), "intensity_corr": float(int_corr),
        "outbreak_auc": float(ob_auc),
        "slope_mae": float(slope_mae), "slope_corr": float(slope_corr),
        "peak_day_mae": float(pd_mae), "peak_size_mae": float(ps_mae),
        "stability_corr": float(stab_corr),
        "breakdown_auc": float(bd_auc),
        "vaccination_R0_reduction": float(mean_r0r),
        "n_eval_farms": N_EVAL,
        "n_outbreaks": int(n_ob), "n_breakdowns": int(n_bd),
    }
    rp = os.path.join(MODEL_DIR, "v17b_herd_operational_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport → {rp}")

if __name__ == "__main__":
    main()
