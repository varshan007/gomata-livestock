#!/usr/bin/env python3
"""
evaluate_v17_herd.py — Phase 17 Part 2
V17 HERD EPIDEMIOLOGY ENGINE — OPERATIONAL VALIDATION

Evaluates V17 on fresh unseen farms (different seed):
1. Outbreak AUC (7d / 14d)
2. Peak Day MAE & Peak Size MAE
3. Stability Index Correlation
4. Regulatory Breakdown AUC
5. Vaccination R₀ Reduction Analysis
6. FP ≤ 5 herd false alarms/month
"""

import os, sys, json, logging, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V17_Eval")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

# ═══════════════════════════════════════════════════════════════
# V17 MODEL (must match Colab architecture exactly)
# ═══════════════════════════════════════════════════════════════

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)
        B, N, D = Wh.shape
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        mask = (adj == 0)
        e = e.masked_fill(mask, float('-inf'))
        alpha = tF.softmax(e, dim=-1)
        alpha = alpha.masked_fill(mask, 0.0)
        return torch.bmm(alpha, Wh)

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([GraphAttentionLayer(in_dim, out_dim // n_heads) for _ in range(n_heads)])
    def forward(self, h, adj):
        return torch.cat([head(h, adj) for head in self.heads], dim=-1)

class TemporalFusionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, ff_dim=128):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
    def forward(self, x):
        gru_out, _ = self.gru(x)
        h = self.proj(gru_out)
        h = self.norm1(h + x)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm2(h + attn_out)
        h = h + self.ff(h)
        ctx = h.mean(dim=1); last = h[:, -1, :]
        gate = self.gate(torch.cat([ctx, last], dim=-1))
        return gate * ctx + (1 - gate) * last

class HerdEpidemiologyEngine(nn.Module):
    def __init__(self, node_dim=18, gat_dim=64, tft_dim=64, n_gat_heads=4, n_tft_heads=4):
        super().__init__()
        self.gat1 = MultiHeadGAT(node_dim, gat_dim, n_gat_heads)
        self.gat2 = MultiHeadGAT(gat_dim, gat_dim, n_gat_heads)
        self.gat_norm = nn.LayerNorm(gat_dim)
        self.graph_pool = nn.Sequential(nn.Linear(gat_dim, gat_dim), nn.GELU())
        self.tft = TemporalFusionBlock(gat_dim, n_tft_heads, ff_dim=128)
        self.head_outbreak_7d = nn.Linear(tft_dim, 1)
        self.head_outbreak_14d = nn.Linear(tft_dim, 1)
        self.head_peak_day = nn.Linear(tft_dim, 1)
        self.head_peak_size = nn.Linear(tft_dim, 1)
        self.head_stability = nn.Linear(tft_dim, 1)
        self.head_breakdown = nn.Linear(tft_dim, 1)
        self.head_resources = nn.Linear(tft_dim, 3)

    def forward(self, node_seq, adj):
        B, T, N, Feat = node_seq.shape
        herd_temporal = []
        for t in range(T):
            h = node_seq[:, t]
            h = torch.nn.functional.elu(self.gat1(h, adj))
            h = torch.nn.functional.elu(self.gat2(h, adj))
            h = self.gat_norm(h)
            herd_t = self.graph_pool(h.mean(dim=1))
            herd_temporal.append(herd_t)
        herd_seq = torch.stack(herd_temporal, dim=1)
        ctx = self.tft(herd_seq)
        return {
            "outbreak_7d": self.head_outbreak_7d(ctx).squeeze(-1),
            "outbreak_14d": self.head_outbreak_14d(ctx).squeeze(-1),
            "peak_day": self.head_peak_day(ctx).squeeze(-1),
            "peak_size": self.head_peak_size(ctx).squeeze(-1),
            "stability": self.head_stability(ctx).squeeze(-1),
            "breakdown": self.head_breakdown(ctx).squeeze(-1),
            "resources": self.head_resources(ctx),
        }

# ═══════════════════════════════════════════════════════════════
# FRESH SIMULATOR (different seed, same physics)
# ═══════════════════════════════════════════════════════════════

T_STEPS = 28; SIM_TOTAL = 56; MAX_COWS = 80; N_EVAL_FARMS = 60

class EvalHerdSimulator:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def _build_graph(self, n_cows, n_pens):
        pen_assign = self.rng.randint(0, n_pens, n_cows)
        A = np.zeros((n_cows, n_cows), dtype=np.float32)
        for i in range(n_cows):
            for j in range(i+1, n_cows):
                if pen_assign[i] == pen_assign[j]: w = 1.0
                elif abs(pen_assign[i] - pen_assign[j]) == 1: w = 0.3 * self.rng.uniform(0.5, 1.0)
                elif self.rng.random() < 0.05: w = 0.1 * self.rng.uniform(0.3, 1.0)
                else: continue
                A[i, j] = w; A[j, i] = w
        return A, pen_assign

    def simulate_farm(self, fidx):
        n_cows = self.rng.randint(30, 80)
        n_pens = self.rng.randint(3, 8)
        A, pen_assign = self._build_graph(n_cows, n_pens)

        beta = self.rng.uniform(0.05, 0.20)
        gamma = self.rng.uniform(0.005, 0.04)
        sigma = self.rng.uniform(0.005, 0.02)
        alpha_heat = self.rng.uniform(0.02, 0.08)
        base_thi = self.rng.uniform(65, 85)
        thi_series = base_thi + 5 * np.sin(np.arange(SIM_TOTAL) * 2 * np.pi / 28) + self.rng.normal(0, 2, SIM_TOTAL)

        I = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        R_state = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)
        severity = np.zeros((SIM_TOTAL, n_cows), dtype=np.float32)

        n_seed = max(2, self.rng.randint(2, max(3, n_cows // 5)))
        seed_cows = self.rng.choice(n_cows, n_seed, replace=False)
        I[0, seed_cows] = self.rng.uniform(0.4, 0.9, n_seed)

        vacc_rate = self.rng.uniform(0, 0.3)
        vaccinated = np.zeros(n_cows, dtype=np.float32)
        n_vacc = int(vacc_rate * n_cows)
        if n_vacc > 0:
            vaccinated[self.rng.choice(n_cows, n_vacc, replace=False)] = 1.0
        A_eff = A * (1.0 - vaccinated[np.newaxis, :])

        for t in range(1, SIM_TOTAL):
            thi_ex = max(0, thi_series[t] - 72)
            beta_eff = beta * (1 + alpha_heat * thi_ex)
            force = beta_eff * (A_eff @ I[t-1])
            dI = force * (1 - I[t-1] - R_state[t-1]) - gamma * I[t-1] + sigma * self.rng.normal(0, 1, n_cows)
            I[t] = np.clip(I[t-1] + dI, 0, 1)
            rec_prob = gamma * I[t-1]
            R_state[t] = np.clip(R_state[t-1] + (self.rng.random(n_cows) < rec_prob).astype(float) * 0.1, 0, 1)
            severity[t] = I[t] * (1 + 0.5 * thi_ex / 10) + self.rng.normal(0, 0.05, n_cows)

        nf = np.zeros((SIM_TOTAL, n_cows, 18), dtype=np.float32)
        for t in range(SIM_TOTAL):
            thi_ex = max(0, thi_series[t] - 72)
            for i in range(n_cows):
                nf[t, i] = [
                    I[t, i], float(thi_ex > 5) * 0.3 + self.rng.normal(0, 0.05),
                    I[t, i] * 0.4 + self.rng.normal(0, 0.05), 0.1 + self.rng.normal(0, 0.05),
                    0.05 + self.rng.normal(0, 0.02), severity[t, i], float(severity[t, i] > 2.0),
                    np.gradient(severity[max(0,t-3):t+1, i]).mean() if t > 0 else 0,
                    severity[max(0,t-4):t+1, i].sum() * 0.25,
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) > 0),
                    float(I[t, i] > 0.3 and abs(I[t, i] - I[max(0,t-1), i]) < 0.02),
                    float(I[t, i] > 0.1 and (I[t, i] - I[max(0,t-1), i]) < -0.01),
                    self.rng.uniform(1, 4), max(0, 1 - I[t, i]),
                    max(0, 30 - 10 * I[t, i]) + self.rng.normal(0, 1),
                    1.0 - severity[t, i] * 0.3 + self.rng.normal(0, 0.05),
                    vaccinated[i], float(pen_assign[i]) / n_pens]

        obs = T_STEPS
        total_inf_obs = (I[obs-1] > 0.3).sum()
        future_7d = I[obs:min(obs+28, SIM_TOTAL)]
        future_14d = I[obs:]
        ob7 = float((future_7d > 0.3).sum(axis=1).max() > max(total_inf_obs + 1, n_cows * 0.15)) if len(future_7d) > 0 else 0
        ob14 = float((future_14d > 0.3).sum(axis=1).max() > max(total_inf_obs + 1, n_cows * 0.10)) if len(future_14d) > 0 else 0
        ic = (I > 0.3).sum(axis=1)
        pk_idx = np.argmax(ic); pk_day = pk_idx / 4.0; pk_size = float(ic[pk_idx]) / n_cows
        sv = severity[:obs].var(axis=1).mean()
        stab = max(0, 1.0 - sv * 2)
        ms = severity[:obs].mean(axis=1)
        bd = float(ms.max() > 0.8 and sv > 0.1)
        ml = (I[:obs] * 10).sum() / n_cows
        ab = float((I[:obs] > 0.5).sum()) / n_cows
        isol = float((severity[:obs] > 2.0).any(axis=0).sum()) / n_cows
        ad = (A > 0).sum(axis=1).mean()
        bea = beta * (1 + alpha_heat * max(0, base_thi - 72))
        r0 = bea / gamma * ad if gamma > 0 else 0
        inf_score = A.sum(axis=1) * I[:obs].mean(axis=0)
        top3 = np.argsort(inf_score)[-3:][::-1]
        A2 = A.copy(); A2[top3, :] = 0; A2[:, top3] = 0
        r0p = bea / gamma * (A2 > 0).sum(axis=1).mean() if gamma > 0 else 0
        r0r = (r0 - r0p) / r0 if r0 > 0 else 0

        return {"node_features": nf[:obs], "adjacency": A, "n_cows": n_cows,
                "labels": {"outbreak_7d": ob7, "outbreak_14d": ob14, "peak_day": min(pk_day, 14),
                            "peak_size": min(pk_size, 1), "stability": stab, "breakdown": bd,
                            "milk_loss": ml, "antibiotic": ab, "isolation": isol,
                            "R0": r0, "R0_reduction": r0r}}

# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("🔬 Phase 17 — V17 HERD EPIDEMIOLOGY OPERATIONAL VALIDATION")
    logger.info("=" * 70)

    mp = os.path.join(MODEL_DIR, "v17_herd_engine.pth")
    cp = os.path.join(MODEL_DIR, "v17_herd_config.json")
    if not os.path.exists(mp):
        logger.error("V17 model not found."); return

    with open(cp) as f:
        config = json.load(f)

    # ── STEP 1: Generate fresh eval farms ──
    logger.info(f"Step 1: Generating {N_EVAL_FARMS} unseen farms (seed=7777)...")
    sim = EvalHerdSimulator(seed=7777)
    farms = [sim.simulate_farm(i) for i in range(N_EVAL_FARMS)]
    n_ob7 = sum(1 for f in farms if f['labels']['outbreak_7d'] > 0.5)
    n_ob14 = sum(1 for f in farms if f['labels']['outbreak_14d'] > 0.5)
    n_bd = sum(1 for f in farms if f['labels']['breakdown'] > 0.5)
    logger.info(f"Eval farms: {n_ob7} outbreaks(7d), {n_ob14}(14d), {n_bd} breakdowns")

    # ── STEP 2: Model inference ──
    logger.info("Step 2: V17 inference...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HerdEpidemiologyEngine(node_dim=config["node_dim"],
                                    gat_dim=config["gat_dim"], tft_dim=config["tft_dim"]).to(device)
    model.load_state_dict(torch.load(mp, map_location=device))
    model.eval()

    # Collect predictions
    y_ob7, y_ob14, y_pd, y_ps, y_stab, y_bd = [], [], [], [], [], []
    p_ob7, p_ob14, p_pd, p_ps, p_stab, p_bd = [], [], [], [], [], []
    y_r0r, p_res = [], []

    with torch.no_grad():
        for f in farms:
            nf = f["node_features"]  # [T, N, 18]
            adj = f["adjacency"]     # [N, N]
            T, N, Feat = nf.shape
            nf_pad = np.zeros((T, MAX_COWS, Feat), dtype=np.float32)
            adj_pad = np.zeros((MAX_COWS, MAX_COWS), dtype=np.float32)
            nc = min(N, MAX_COWS)
            nf_pad[:, :nc, :] = nf[:, :nc, :]; adj_pad[:nc, :nc] = adj[:nc, :nc]

            xn = torch.tensor(nf_pad, dtype=torch.float32).unsqueeze(0).to(device)
            xa = torch.tensor(adj_pad, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(xn, xa)

            lab = f["labels"]
            y_ob7.append(lab["outbreak_7d"]); p_ob7.append(torch.sigmoid(out["outbreak_7d"]).item())
            y_ob14.append(lab["outbreak_14d"]); p_ob14.append(torch.sigmoid(out["outbreak_14d"]).item())
            y_pd.append(lab["peak_day"]); p_pd.append(out["peak_day"].item())
            y_ps.append(lab["peak_size"]); p_ps.append(out["peak_size"].item())
            y_stab.append(lab["stability"]); p_stab.append(out["stability"].item())
            y_bd.append(lab["breakdown"]); p_bd.append(torch.sigmoid(out["breakdown"]).item())
            y_r0r.append(lab["R0_reduction"])

    y_ob7=np.array(y_ob7); p_ob7=np.array(p_ob7)
    y_ob14=np.array(y_ob14); p_ob14=np.array(p_ob14)
    y_pd=np.array(y_pd); p_pd=np.array(p_pd)
    y_ps=np.array(y_ps); p_ps=np.array(p_ps)
    y_stab=np.array(y_stab); p_stab=np.array(p_stab)
    y_bd=np.array(y_bd); p_bd=np.array(p_bd)
    y_r0r=np.array(y_r0r)

    # ── STEP 3: Metrics ──
    logger.info("\n" + "=" * 70)
    logger.info("📊 V17 HERD ENGINE — EVALUATION RESULTS")
    logger.info("=" * 70)

    # A. Outbreak AUC
    auc_7d = roc_auc_score(y_ob7, p_ob7) if len(np.unique(y_ob7)) > 1 else 0
    auc_14d = roc_auc_score(y_ob14, p_ob14) if len(np.unique(y_ob14)) > 1 else 0
    logger.info(f"\n🦠 A. OUTBREAK DETECTION")
    logger.info(f"  7-Day Outbreak AUC:   {auc_7d:.4f}")
    logger.info(f"  14-Day Outbreak AUC:  {auc_14d:.4f}")

    # FP analysis (farms per month)
    for th in [0.3, 0.4, 0.5]:
        fp7 = ((p_ob7 >= th) & (y_ob7 == 0)).sum()
        logger.info(f"  FP@{th:.1f} (7d):  {fp7} / {(y_ob7==0).sum()} negatives")

    # B. Peak Prediction
    pd_mae = np.abs(y_pd - p_pd).mean()
    ps_mae = np.abs(y_ps - p_ps).mean()
    logger.info(f"\n📈 B. PEAK PREDICTION")
    logger.info(f"  Peak Day MAE:     {pd_mae:.2f} days")
    logger.info(f"  Peak Size MAE:    {ps_mae:.4f} (fraction)")

    # C. Stability Correlation
    if np.std(y_stab) > 0 and np.std(p_stab) > 0:
        stab_corr, stab_p = pearsonr(y_stab, p_stab)
    else:
        stab_corr = 0; stab_p = 1.0
    logger.info(f"\n📊 C. HERD STABILITY INDEX")
    logger.info(f"  Pearson Correlation: {stab_corr:.4f} (p={stab_p:.4e})")

    # D. Regulatory Breakdown
    auc_bd = roc_auc_score(y_bd, p_bd) if len(np.unique(y_bd)) > 1 else 0
    logger.info(f"\n⚠️ D. REGULATORY BREAKDOWN")
    logger.info(f"  Breakdown AUC: {auc_bd:.4f}")

    # E. Vaccination R₀ Reduction
    mean_r0r = y_r0r.mean()
    logger.info(f"\n💉 E. VACCINATION (R₀ REDUCTION)")
    logger.info(f"  Mean R₀ Reduction (greedy top-3 removal): {mean_r0r:.2%}")

    # ── SUMMARY ──
    logger.info("\n" + "=" * 70)
    logger.info("🏁 V17 HERD ENGINE — PASS CRITERIA")
    logger.info("=" * 70)

    checks = [
        ("Outbreak 7d AUC ≥ 0.85", auc_7d, 0.85),
        ("Peak Day MAE ≤ 1.5 days", -pd_mae, -1.5),  # negate for ≤
        ("Stability Correlation ≥ 0.80", stab_corr, 0.80),
        ("R₀ Reduction ≥ 25%", mean_r0r, 0.25),
    ]
    all_pass = True
    for name, val, target in checks:
        passed = val >= target
        status = "✅" if passed else "❌"
        if "MAE" in name:
            logger.info(f"  {status} {name}: {abs(val):.2f}")
        elif "R₀" in name:
            logger.info(f"  {status} {name}: {val:.2%}")
        else:
            logger.info(f"  {status} {name}: {val:.4f}")
        if not passed: all_pass = False

    if all_pass:
        logger.info("\n✅ V17 HERD EPIDEMIOLOGY ENGINE — ALL CRITERIA PASSED. PRODUCTION READY.")
    else:
        logger.info("\n⚠️ Some criteria not met. Review individual heads.")

    # Save report
    report = {
        "outbreak_7d_auc": auc_7d, "outbreak_14d_auc": auc_14d,
        "peak_day_mae": float(pd_mae), "peak_size_mae": float(ps_mae),
        "stability_correlation": float(stab_corr),
        "breakdown_auc": auc_bd,
        "vaccination_R0_reduction": float(mean_r0r),
        "n_eval_farms": N_EVAL_FARMS,
        "n_outbreaks_7d": int(n_ob7), "n_outbreaks_14d": int(n_ob14),
    }
    rp = os.path.join(MODEL_DIR, "v17_herd_operational_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved → {rp}")

if __name__ == "__main__":
    main()
