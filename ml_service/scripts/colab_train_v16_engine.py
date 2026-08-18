#!/usr/bin/env python3
"""
colab_train_v16_engine.py — Phase 16
V16 BIOLOGICAL ENGINE (Google Colab T4 GPU)
============================================
Self-contained script. Extends V14 backbone with 4 new heads:
  1. Episode Phase (Onset/Peak/Recovery) — 3-class
  2. Stability Collapse — binary
  3. Recovery Hazard — 24h survival curve
  4. Calving Window — 72h hazard curve

INSTRUCTIONS:
  1. Colab → Runtime → T4 GPU
  2. Upload this file
  3. !python colab_train_v16_engine.py
  4. Download: colab_output/v16_shared_attention_engine.pth
              colab_output/v16_scalers.json
  5. Place in: ml_service/models/cattle/
"""

import os, sys, json, logging, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V16")

OUT_DIR = "./colab_output"
os.makedirs(OUT_DIR, exist_ok=True)

N_ANIMALS = 500
TICKS_PER_ANIMAL = 2000
TICK_MIN = 10
TICKS_PER_HOUR = 6
TICKS_PER_DAY = 144
THI_THRESHOLD = 72.0
SEQ_LEN = 288
STRIDE = 24
HAZARD_HORIZON = 144  # 24h
CALVING_HORIZON = 432 # 72h

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: DOMAIN RANDOMIZED SIMULATOR WITH V16 LABELS
# ═══════════════════════════════════════════════════════════════════

class DomainRandomizedUniverse:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def _compute_v16_labels(self, X, severity, n):
        """Compute Phase 16 extended labels from latent states."""
        # ── Episode Phase: 0=Onset, 1=Peak, 2=Recovery ──
        phase = np.zeros(n, dtype=np.int64)  # default=0 (Onset)
        sev_grad = np.gradient(severity)
        sev_smooth = np.convolve(sev_grad, np.ones(12)/12, mode='same')
        for t in range(n):
            if severity[t] < 0.3:
                phase[t] = 0  # No disease / baseline
            elif sev_smooth[t] > 0.005:
                phase[t] = 0  # Onset (rising)
            elif sev_smooth[t] < -0.005:
                phase[t] = 2  # Recovery (falling)
            else:
                phase[t] = 1  # Peak (plateau)

        # ── Stability Collapse ──
        n_diseases = np.zeros(n)
        for key in ["I", "H", "M", "L"]:
            if key in X:
                n_diseases += (np.array(X[key]) > 0.4).astype(float)
        collapse = ((severity >= 2.5) & (n_diseases >= 2)).astype(np.float32)

        # ── Recovery Hazard (24h lookahead: P(sev drops below 1.0 at h)) ──
        # Computed per-window during slicing

        # ── Calving Window (72h lookahead for calving state)
        # Computed per-window during slicing

        return phase, collapse

    def generate_native_sde_animal(self, idx):
        n = TICKS_PER_ANIMAL
        aid = f"Native_{idx:04d}"
        base_temp=38.3+self.rng.normal(0,.3); base_hr=62+self.rng.normal(0,8)
        base_act=0.65+self.rng.normal(0,.1); base_milk=30+self.rng.normal(0,6)
        heat_tol=self.rng.uniform(0.5,1.5)

        t_arr=np.arange(n)
        ambient=22+10*np.sin(t_arr*2*np.pi/TICKS_PER_DAY)+self.rng.normal(0,2,n)
        humidity=55+15*np.sin(t_arr*2*np.pi/(TICKS_PER_DAY*3))+self.rng.normal(0,5,n)
        thi=(1.8*ambient+32)-((0.55-0.0055*humidity)*(1.8*ambient-26))+self.rng.normal(0,1.5,n)
        E=np.zeros(n)
        for t in range(1,n): E[t]=E[t-1]-0.05*E[t-1]+self.rng.normal(0,0.8)

        X={k:np.zeros(n) for k in "IHMLC"}
        X["Imm"]=np.ones(n); X["Fat"]=np.zeros(n)
        has_inf=self.rng.random()<.25; has_mast=self.rng.random()<.20
        has_lame=self.rng.random()<.15; is_calv=self.rng.random()<.10
        inf_s=self.rng.randint(int(.1*n),int(.8*n)) if has_inf else -1
        mast_s=self.rng.randint(int(.1*n),int(.8*n)) if has_mast else -1
        lame_s=self.rng.randint(int(.1*n),int(.8*n)) if has_lame else -1
        calv_m=self.rng.randint(int(.4*n),int(.9*n)) if is_calv else -1
        exp=np.zeros(n)
        if has_inf: exp[inf_s:inf_s+144]=self.rng.uniform(.01,.03)

        for t in range(1,n):
            if is_calv: X["C"][t]=1/(1+np.exp(-2*((t-calv_m)/TICKS_PER_DAY)))
            thi_ex=max(thi[t]-THI_THRESHOLD,0)
            X["H"][t]=np.clip(X["H"][t-1]+(.01*thi_ex)-(.05*heat_tol)+self.rng.normal(0,.01),0,1.5)
            X["Fat"][t]=np.clip(X["Fat"][t-1]+(.02*X["H"][t])+(.05*X["I"][t-1])+(.03*X["C"][t])-.01,0,1)
            X["Imm"][t]=np.clip(X["Imm"][t-1]-(.08*X["Fat"][t])-(.05*X["H"][t])+.01,.1,1)
            X["I"][t]=np.clip(X["I"][t-1]+exp[t-1]+(.05*X["I"][t-1])-(.04*X["Imm"][t])+self.rng.normal(0,.005),0,1)
            pm=.05 if(mast_s>0 and mast_s<t<mast_s+200) else 0
            sm=.08*max(X["I"][t]-.6,0)
            X["M"][t]=np.clip(X["M"][t-1]+pm+sm+(.02*X["M"][t-1])-(.05*X["Imm"][t])+self.rng.normal(0,.005),0,1)
            pl=.03 if(lame_s>0 and lame_s<t<lame_s+500) else 0
            X["L"][t]=np.clip(X["L"][t-1]+pl+(.05*X["C"][t])+(.01*X["L"][t-1])+self.rng.normal(0,.002),0,1)

        rho=.85; tn=np.zeros(n); hn=np.zeros(n); an=np.zeros(n)
        for t in range(1,n):
            tn[t]=rho*tn[t-1]+self.rng.normal(0,.15)
            hn[t]=rho*hn[t-1]+self.rng.normal(0,1.5)
            an[t]=rho*an[t-1]+self.rng.normal(0,.03)

        temp=base_temp+(2.5*X["I"])+(1.2*X["H"])+(.4*X["M"])-(.3*X["C"])+tn+(.2*E)
        hr=base_hr+(20*X["I"])+(12*X["H"])+(8*X["C"])+hn+(2.5*E)
        resp=24+(15*X["H"])+(5*X["I"])+self.rng.normal(0,2,n)+(1.5*E)
        act=np.clip(base_act-(.5*X["L"])-(.3*X["I"])+(X["C"]*self.rng.normal(0,.2,n))+an-(.05*E),.1,1)
        milk=np.clip(base_milk-(6*X["I"])-(4*X["H"])-(12*X["M"])-(2*X["L"])+self.rng.normal(0,1.5,n)-(1.5*E),0,50)
        cond=5+(3.5*X["M"])+self.rng.normal(0,.2,n)+(.4*E)
        feed=np.clip(22-(X["I"]*5),0,50)
        sev=np.clip(X["I"]+(X["M"]*1.5)+(X["L"]*.8)+(X["H"]*.5),0,3)

        phase, collapse = self._compute_v16_labels(X, sev, n)

        return pd.DataFrame({"animalId":[aid]*n,
            "temperature_C":temp,"heartRate_bpm":hr,"respiration_bpm":resp,"activity_index":act,
            "thi":thi,"ambientTemp_C":ambient,"humidity_pct":humidity,
            "milkYield":milk,"feedIntake":feed,"conductivity":cond,
            "infectionBinary":(X["I"]>.4).astype(int),"heatStressBinary":(X["H"]>.5).astype(int),
            "mastitisBinary":(X["M"]>.4).astype(int),"lamenessBinary":(X["L"]>.4).astype(int),
            "calvingBinary":(X["C"]>.5).astype(int),"severityLevel":sev,
            "episodePhase":phase,"collapseFlag":collapse,
            "calvingState":(X["C"]).astype(np.float32)})

    def generate_alien_animal(self, idx):
        n = TICKS_PER_ANIMAL
        aid = f"Alien_{idx:04d}"
        tb=np.zeros(n); tb[0]=38.5+self.rng.normal(0,.4)
        hb=np.zeros(n); hb[0]=65+self.rng.normal(0,5)
        for t in range(1,n):
            tb[t]=tb[t-1]+self.rng.normal(0,.05)
            if self.rng.random()<.005: tb[t]+=self.rng.normal(0,.5)
            hb[t]=hb[t-1]+self.rng.normal(0,.5)
            if self.rng.random()<.01: hb[t]+=self.rng.normal(0,5)

        ambient=20+8*np.sin(np.arange(n)*2*np.pi/TICKS_PER_DAY)+self.rng.normal(0,3,n)
        humidity=60+10*np.cos(np.arange(n)*2*np.pi/TICKS_PER_DAY)+self.rng.normal(0,4,n)
        thi=(1.8*ambient+32)-((0.55-0.0055*humidity)*(1.8*ambient-26))

        inf=np.zeros(n)
        if self.rng.random()<.25:
            s=self.rng.randint(50,n-200); d=self.rng.randint(100,300); e=min(s+d,n)
            inf[s:e]=np.exp(-np.linspace(0,1.5,d))[:e-s]
        mast=np.zeros(n)
        if self.rng.random()<.20:
            s=self.rng.randint(50,n-200); d=self.rng.randint(80,250); e=min(s+d,n)
            ramp=np.cumsum(self.rng.uniform(.005,.02,d)); mast[s:e]=np.clip(ramp[:e-s],0,1)
        lame=np.zeros(n)
        if self.rng.random()<.15:
            s=self.rng.randint(50,n-400); lame[s:]=np.linspace(.2,1,n-s)
        heat=(thi>75).astype(float)
        calv=np.zeros(n)
        if self.rng.random()<.10:
            s=self.rng.randint(200,n-200)
            calv[s-100:s+100]=np.exp(-.5*((np.arange(200)-100)/35)**2)

        temp=tb+1.5*np.log1p(inf*10)+.5*np.exp(heat)+.6*mast-.4*calv+self.rng.normal(0,.2,n)
        hr=hb*(1+.2*inf)*(1+.15*heat)*(1+.12*calv)+self.rng.normal(0,2,n)
        resp=22+(12*heat)**1.2+3*inf+10*calv+self.rng.normal(0,3,n)
        cn=calv*self.rng.normal(0,.3,n)
        act=np.clip(.7*(1-.6*lame)*(1-.3*inf)+cn+self.rng.normal(0,.05,n),0,1)
        mb=32+self.rng.normal(0,4)
        milk=np.clip(mb*(1-.4*mast)*(1-.15*inf)*(1-.1*heat)+self.rng.normal(0,1.5,n),0,None)
        feed=22*(1-.25*inf)*(1-.15*lame)+self.rng.normal(0,1,n)
        cond=5+np.exp(mast*1.5)-1+self.rng.normal(0,.3,n)
        sev=inf+mast+lame+heat

        X = {"I": inf, "H": heat, "M": mast, "L": lame, "C": calv}
        phase, collapse = self._compute_v16_labels(X, sev, n)

        return pd.DataFrame({"animalId":[aid]*n,
            "temperature_C":temp,"heartRate_bpm":hr,"respiration_bpm":resp,"activity_index":act,
            "thi":thi,"ambientTemp_C":ambient,"humidity_pct":humidity,
            "milkYield":milk,"feedIntake":feed,"conductivity":cond,
            "infectionBinary":(inf>.4).astype(int),"heatStressBinary":(heat>.5).astype(int),
            "mastitisBinary":(mast>.4).astype(int),"lamenessBinary":(lame>.4).astype(int),
            "calvingBinary":(calv>.5).astype(int),"severityLevel":sev,
            "episodePhase":phase,"collapseFlag":collapse,
            "calvingState":calv.astype(np.float32)})

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: V16 MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class TemporalMHA(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        attn_out, attn_w = self.mha(x, x, x)
        return attn_out.mean(dim=1), attn_w

class V16BiologicalEngine(nn.Module):
    """
    V16 SharedAttention Livestock Operating System.
    Extends V14 backbone with Phase, Collapse, Recovery, and Calving Window heads.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        d = hidden_dim * 2  # 128

        self.attention = TemporalMHA(d, num_heads=4)

        # ── V14 Original Heads ──
        self.head_inf = nn.Linear(d, 1)
        self.head_heat = nn.Linear(d, 1)
        self.head_mast = nn.Linear(d, 1)
        self.head_lame = nn.Linear(d, 1)
        self.head_calv = nn.Linear(d, 1)
        self.head_sev = nn.Linear(d, 1)
        self.head_hazard = nn.Linear(d, 24)

        # ── V16 NEW Heads ──
        self.phase_head = nn.Linear(d, 3)          # Onset/Peak/Recovery
        self.collapse_head = nn.Linear(d, 1)        # Stability Collapse
        self.recovery_hazard_head = nn.Linear(d, 24) # Recovery Survival 24h
        self.calving_hazard_head = nn.Linear(d, 72)  # Calving Window 72h

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ctx, tw = self.attention(lstm_out)

        # Disease logits
        l_i=self.head_inf(ctx); l_h=self.head_heat(ctx)
        l_m=self.head_mast(ctx); l_l=self.head_lame(ctx); l_c=self.head_calv(ctx)

        # Biological gating
        p_i=torch.sigmoid(l_i); p_m=torch.sigmoid(l_m)
        p_l=torch.sigmoid(l_l); p_h=torch.sigmoid(l_h); p_c=torch.sigmoid(l_c)
        mod = 1 + .5*p_i + .5*p_m + .3*p_l + .2*p_h + .2*p_c

        sev = self.head_sev(ctx) * mod
        haz = self.head_hazard(ctx) * mod

        logits_cls = torch.cat([l_i, l_h, l_m, l_l, l_c], dim=1)

        # V16 new outputs
        phase_logits = self.phase_head(ctx)            # [B, 3]
        collapse_logit = self.collapse_head(ctx)       # [B, 1]
        recovery_haz = self.recovery_hazard_head(ctx)  # [B, 24]
        calving_window = self.calving_hazard_head(ctx) # [B, 72]

        return (logits_cls, sev.squeeze(1), haz, tw,
                phase_logits, collapse_logit.squeeze(1), recovery_haz, calving_window)

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: LOSSES
# ═══════════════════════════════════════════════════════════════════

class VectorizedFocalLoss(nn.Module):
    def __init__(self, alphas, gamma=3.0):
        super().__init__()
        self.register_buffer('alphas', alphas)
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alphas.unsqueeze(0) * (1-pt)**self.gamma * bce).mean()

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_features(df):
    for w, ticks in {"6h":36,"12h":72,"24h":144}.items():
        for c in ["temperature_C","heartRate_bpm","respiration_bpm","activity_index","milkYield","conductivity"]:
            df[f"{c}_mean_{w}"] = df.groupby("animalId")[c].transform(lambda x: x.rolling(ticks,min_periods=1).mean())
            df[f"{c}_std_{w}"] = df.groupby("animalId")[c].transform(lambda x: x.rolling(ticks,min_periods=1).std().fillna(0))
            df[f"{c}_delta_{w}"] = df[c] - df[f"{c}_mean_{w}"]
    df["thermal_strain_index"] = df["heartRate_bpm_delta_6h"]/(df["thi"]-72+1e-5)
    df["lameness_suppression"] = (1-df["activity_index"])*(df["feedIntake"]/22)
    df["mastitis_spike_index"] = df["conductivity_delta_12h"]*(df["milkYield"]/30)
    df["fever_decoupled"] = df["temperature_C"]-(38.5+(0.01*np.maximum(df["thi"].values-72,0)))
    df.fillna(0, inplace=True)
    return df

def get_features(cols):
    skip = {"animalId","timestamp","antibioticActive","infectionBinary","heatStressBinary",
            "mastitisBinary","lamenessBinary","calvingBinary","severityLevel","domainEngine",
            "episodePhase","collapseFlag","calvingState"}
    return [c for c in cols if c not in skip]

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: DATASET
# ═══════════════════════════════════════════════════════════════════

class V16Dataset(Dataset):
    def __init__(self, X, Yc, Yh, Yp, Ycol, Yrec, Ycw):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Yc = torch.tensor(Yc, dtype=torch.float32)   # [N, 5] disease cls
        self.Yh = torch.tensor(Yh, dtype=torch.float32)    # [N, 24] hazard
        self.Yp = torch.tensor(Yp, dtype=torch.long)       # [N] phase class
        self.Ycol = torch.tensor(Ycol, dtype=torch.float32) # [N] collapse
        self.Yrec = torch.tensor(Yrec, dtype=torch.float32) # [N, 24] recovery
        self.Ycw = torch.tensor(Ycw, dtype=torch.float32)   # [N, 72] calving window
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Yc[i], self.Yh[i], self.Yp[i], self.Ycol[i], self.Yrec[i], self.Ycw[i]

# ═══════════════════════════════════════════════════════════════════
# SECTION 6: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("🧬 COLAB T4 — Phase 16 V16 Biological Engine Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── STEP 1: Generate Data ──
    logger.info("Step 1: Generating 500 animals (70% SDE / 30% Alien)...")
    sim = DomainRandomizedUniverse(seed=2048)
    frames = []
    cn = {"N":0,"A":0}
    for i in range(N_ANIMALS):
        if sim.rng.random() < 0.30:
            frames.append(sim.generate_alien_animal(i)); cn["A"]+=1
        else:
            frames.append(sim.generate_native_sde_animal(i)); cn["N"]+=1
        if (i+1)%100==0:
            logger.info(f"  Generated {i+1}/{N_ANIMALS} (SDE:{cn['N']} Alien:{cn['A']})")
    df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    logger.info(f"Rows: {len(df):,}")

    # ── STEP 2: Features ──
    logger.info("Step 2: Extracting features...")
    df = extract_features(df)
    features = get_features(df.columns.tolist())
    logger.info(f"Feature dim: {len(features)}")

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    sp = os.path.join(OUT_DIR, "v16_scalers.json")
    with open(sp, "w") as f:
        json.dump({"features": features, "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist()}, f)

    # ── STEP 3: Slice Windows with V16 Labels ──
    logger.info("Step 3: Slicing sequence windows with V16 extended labels...")
    X_l, Yc_l, Yh_l, Yp_l, Ycol_l, Yrec_l, Ycw_l = [], [], [], [], [], [], []

    for _, grp in df.groupby("animalId"):
        vf = grp[features].values.astype(np.float32)
        vc = grp[["infectionBinary","heatStressBinary","mastitisBinary","lamenessBinary","calvingBinary"]].values.astype(np.float32)
        vs = grp["severityLevel"].values.astype(np.float32)
        vph = grp["episodePhase"].values.astype(np.int64)
        vcol = grp["collapseFlag"].values.astype(np.float32)
        vcalv = grp["calvingState"].values.astype(np.float32)
        nt = len(vf)

        # Need enough lookahead for 72h calving window
        min_len = SEQ_LEN + CALVING_HORIZON
        if nt < min_len: continue

        for s in range(0, nt - min_len, STRIDE):
            e = s + SEQ_LEN
            X_l.append(vf[s:e])

            # Disease targets (24h horizon)
            Yc_l.append(vc[e:e+HAZARD_HORIZON].max(axis=0))

            # Hazard 24h
            sh = vs[e:e+HAZARD_HORIZON]
            yh = np.zeros(24, dtype=np.float32)
            for h in range(24):
                if np.any(sh[h*6:(h+1)*6] >= 2.0): yh[h] = 1.0
            Yh_l.append(yh)

            # Phase (at sequence end)
            Yp_l.append(vph[e-1])

            # Collapse (max in next 24h)
            Ycol_l.append(float(vcol[e:e+HAZARD_HORIZON].max()))

            # Recovery hazard 24h: P(severity drops below 1.0 at hour h)
            yrec = np.zeros(24, dtype=np.float32)
            for h in range(24):
                block = vs[e+h*6:e+(h+1)*6]
                if len(block) > 0 and np.any(block < 1.0) and vs[e-1] >= 1.0:
                    yrec[h] = 1.0
            Yrec_l.append(yrec)

            # Calving window 72h
            ycw = np.zeros(72, dtype=np.float32)
            for h in range(72):
                block_c = vcalv[e+h*6:e+(h+1)*6]
                if len(block_c) > 0 and np.any(block_c > 0.5):
                    ycw[h] = 1.0
            Ycw_l.append(ycw)

    X = np.array(X_l, dtype=np.float32)
    Yc = np.array(Yc_l, dtype=np.float32)
    Yh = np.array(Yh_l, dtype=np.float32)
    Yp = np.array(Yp_l, dtype=np.int64)
    Ycol = np.array(Ycol_l, dtype=np.float32)
    Yrec = np.array(Yrec_l, dtype=np.float32)
    Ycw = np.array(Ycw_l, dtype=np.float32)
    del X_l, Yc_l, Yh_l, Yp_l, Ycol_l, Yrec_l, Ycw_l, df; gc.collect()
    logger.info(f"Training samples: {len(X):,}")

    # ── STEP 4: CUDA Training ──
    logger.info("Step 4: Training V16 on CUDA T4...")

    freq = Yc.mean(axis=0)
    logger.info(f"Disease freq: {freq}")
    alphas = torch.tensor(1.0 - freq, dtype=torch.float32)

    input_dim = len(features)
    model = V16BiologicalEngine(input_dim=input_dim).to(device)
    focal = VectorizedFocalLoss(alphas.to(device), gamma=3.0).to(device)
    haz_bce = nn.BCEWithLogitsLoss()
    phase_ce = nn.CrossEntropyLoss()
    collapse_bce = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

    dataset = V16Dataset(X, Yc, Yh, Yp, Ycol, Yrec, Ycw)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    del X, Yc, Yh, Yp, Ycol, Yrec, Ycw; gc.collect()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3,
                                                      steps_per_epoch=len(loader), epochs=6)
    amp_scaler = torch.amp.GradScaler("cuda")

    EPOCHS = 6
    model.train()
    for epoch in range(EPOCHS):
        tl=0; nb=0; t0=time.time()
        for bi, (xb, yc, yh, yp, ycol, yrec, ycw) in enumerate(loader):
            xb=xb.to(device, non_blocking=True)
            yc=yc.to(device, non_blocking=True)
            yh=yh.to(device, non_blocking=True)
            yp=yp.to(device, non_blocking=True)
            ycol=ycol.to(device, non_blocking=True)
            yrec=yrec.to(device, non_blocking=True)
            ycw=ycw.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lc, sv, hz, aw, ph_logits, col_logit, rec_hz, cw_hz = model(xb)

                # Balanced multi-head loss
                loss_disease = focal(lc, yc)                          # disease
                loss_hazard = haz_bce(hz, yh)                         # 24h hazard
                loss_phase = phase_ce(ph_logits, yp)                  # episode phase
                loss_collapse = collapse_bce(col_logit, ycol)         # collapse
                loss_recovery = haz_bce(rec_hz, yrec)                 # recovery hazard
                loss_calving_w = haz_bce(cw_hz, ycw)                  # calving window

                loss = (loss_disease
                        + 1.0 * loss_hazard
                        + 0.4 * loss_phase
                        + 0.5 * loss_collapse
                        + 0.6 * loss_recovery
                        + 0.7 * loss_calving_w)

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            scheduler.step()

            tl+=loss.item(); nb+=1
            if bi%10==0:
                logger.info(f"E{epoch+1}/{EPOCHS} B{bi}/{len(loader)} "
                            f"L:{loss.item():.4f} [dis:{loss_disease.item():.3f} "
                            f"haz:{loss_hazard.item():.3f} ph:{loss_phase.item():.3f} "
                            f"col:{loss_collapse.item():.3f} rec:{loss_recovery.item():.3f} "
                            f"cw:{loss_calving_w.item():.3f}]")

        logger.info(f"== EPOCH {epoch+1} | AvgLoss: {tl/nb:.4f} | {time.time()-t0:.1f}s ==")

    # ── STEP 5: Save ──
    model_cpu = model.to("cpu")
    mp = os.path.join(OUT_DIR, "v16_shared_attention_engine.pth")
    torch.save(model_cpu.state_dict(), mp)
    logger.info(f"✅ V16 Model → {mp}")
    logger.info(f"✅ Scalers → {sp}")
    logger.info("=" * 60)
    logger.info("DONE! Download from colab_output/:")
    logger.info("  1. v16_shared_attention_engine.pth")
    logger.info("  2. v16_scalers.json")
    logger.info("Place in: ml_service/models/cattle/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
