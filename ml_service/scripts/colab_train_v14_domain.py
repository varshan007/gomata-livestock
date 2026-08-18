#!/usr/bin/env python3
"""
colab_train_v14_domain.py — Phase 15.1 (Google Colab T4 GPU Edition)
====================================================================
SELF-CONTAINED script. Upload this single file to Google Colab.
It includes: Simulator, Model Architecture, Feature Extraction, and CUDA Training.

INSTRUCTIONS:
=============
1. Open Google Colab → Runtime → Change runtime type → T4 GPU
2. Upload this file to Colab
3. Run:  !python colab_train_v14_domain.py
4. Download the output files:
   - v14_domain_attention_model.pth  (the trained PyTorch model)
   - v14_domain_scalers.json         (the feature scaler)
5. Place both files in: ml_service/models/cattle/
"""

import os, sys, json, logging, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CoLab_V14")

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
HAZARD_HORIZON = 144

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: DOMAIN RANDOMIZED SIMULATOR (Native SDE + Alien Physics)
# ═══════════════════════════════════════════════════════════════════

class DomainRandomizedUniverse:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def generate_native_sde_animal(self, idx):
        n = TICKS_PER_ANIMAL
        aid = f"Native_{idx:04d}"
        base_temp = 38.3 + self.rng.normal(0, 0.3)
        base_hr = 62 + self.rng.normal(0, 8)
        base_act = 0.65 + self.rng.normal(0, 0.1)
        base_milk = 30 + self.rng.normal(0, 6)
        heat_tol = self.rng.uniform(0.5, 1.5)

        t_arr = np.arange(n)
        ambient = 22 + 10*np.sin(t_arr*2*np.pi/TICKS_PER_DAY) + self.rng.normal(0,2,n)
        humidity = 55 + 15*np.sin(t_arr*2*np.pi/(TICKS_PER_DAY*3)) + self.rng.normal(0,5,n)
        thi = (1.8*ambient+32) - ((0.55-0.0055*humidity)*(1.8*ambient-26)) + self.rng.normal(0,1.5,n)
        E = np.zeros(n)
        for t in range(1,n): E[t] = E[t-1] - 0.05*E[t-1] + self.rng.normal(0,0.8)

        X = {k: np.zeros(n) for k in "IHMLC"}
        X["Imm"] = np.ones(n); X["Fat"] = np.zeros(n)

        has_inf = self.rng.random()<0.25; has_mast = self.rng.random()<0.20
        has_lame = self.rng.random()<0.15; is_calv = self.rng.random()<0.10
        inf_start = self.rng.randint(int(.1*n),int(.8*n)) if has_inf else -1
        mast_start = self.rng.randint(int(.1*n),int(.8*n)) if has_mast else -1
        lame_start = self.rng.randint(int(.1*n),int(.8*n)) if has_lame else -1
        calv_mid = self.rng.randint(int(.4*n),int(.9*n)) if is_calv else -1
        exposure = np.zeros(n)
        if has_inf: exposure[inf_start:inf_start+144] = self.rng.uniform(0.01,0.03)

        for t in range(1,n):
            if is_calv: X["C"][t] = 1/(1+np.exp(-2*((t-calv_mid)/TICKS_PER_DAY)))
            thi_ex = max(thi[t]-THI_THRESHOLD,0)
            X["H"][t] = np.clip(X["H"][t-1]+(0.01*thi_ex)-(0.05*heat_tol)+self.rng.normal(0,.01),0,1.5)
            X["Fat"][t] = np.clip(X["Fat"][t-1]+(0.02*X["H"][t])+(0.05*X["I"][t-1])+(0.03*X["C"][t])-0.01,0,1)
            X["Imm"][t] = np.clip(X["Imm"][t-1]-(0.08*X["Fat"][t])-(0.05*X["H"][t])+0.01,0.1,1)
            X["I"][t] = np.clip(X["I"][t-1]+exposure[t-1]+(0.05*X["I"][t-1])-(0.04*X["Imm"][t])+self.rng.normal(0,.005),0,1)
            pm = 0.05 if(mast_start>0 and mast_start<t<mast_start+200) else 0
            sm = 0.08*max(X["I"][t]-0.6,0)
            X["M"][t] = np.clip(X["M"][t-1]+pm+sm+(0.02*X["M"][t-1])-(0.05*X["Imm"][t])+self.rng.normal(0,.005),0,1)
            pl = 0.03 if(lame_start>0 and lame_start<t<lame_start+500) else 0
            X["L"][t] = np.clip(X["L"][t-1]+pl+(0.05*X["C"][t])+(0.01*X["L"][t-1])+self.rng.normal(0,.002),0,1)

        rho=0.85; tn=np.zeros(n); hn=np.zeros(n); an=np.zeros(n)
        for t in range(1,n):
            tn[t]=rho*tn[t-1]+self.rng.normal(0,.15)
            hn[t]=rho*hn[t-1]+self.rng.normal(0,1.5)
            an[t]=rho*an[t-1]+self.rng.normal(0,.03)

        temp = base_temp+(2.5*X["I"])+(1.2*X["H"])+(0.4*X["M"])-(0.3*X["C"])+tn+(0.2*E)
        hr = base_hr+(20*X["I"])+(12*X["H"])+(8*X["C"])+hn+(2.5*E)
        resp = 24+(15*X["H"])+(5*X["I"])+self.rng.normal(0,2,n)+(1.5*E)
        act = np.clip(base_act-(0.5*X["L"])-(0.3*X["I"])+(X["C"]*self.rng.normal(0,.2,n))+an-(0.05*E),.1,1)
        milk = np.clip(base_milk-(6*X["I"])-(4*X["H"])-(12*X["M"])-(2*X["L"])+self.rng.normal(0,1.5,n)-(1.5*E),0,50)
        cond = 5+(3.5*X["M"])+self.rng.normal(0,.2,n)+(0.4*E)
        feed = np.clip(22-(X["I"]*5),0,50)
        sev = np.clip(X["I"]+(X["M"]*1.5)+(X["L"]*0.8)+(X["H"]*0.5),0,3)

        return pd.DataFrame({"animalId":[aid]*n,
            "temperature_C":temp,"heartRate_bpm":hr,"respiration_bpm":resp,"activity_index":act,
            "thi":thi,"ambientTemp_C":ambient,"humidity_pct":humidity,
            "milkYield":milk,"feedIntake":feed,"conductivity":cond,
            "infectionBinary":(X["I"]>0.4).astype(int),"heatStressBinary":(X["H"]>0.5).astype(int),
            "mastitisBinary":(X["M"]>0.4).astype(int),"lamenessBinary":(X["L"]>0.4).astype(int),
            "calvingBinary":(X["C"]>0.5).astype(int),"severityLevel":sev})

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

        return pd.DataFrame({"animalId":[aid]*n,
            "temperature_C":temp,"heartRate_bpm":hr,"respiration_bpm":resp,"activity_index":act,
            "thi":thi,"ambientTemp_C":ambient,"humidity_pct":humidity,
            "milkYield":milk,"feedIntake":feed,"conductivity":cond,
            "infectionBinary":(inf>.4).astype(int),"heatStressBinary":(heat>.5).astype(int),
            "mastitisBinary":(mast>.4).astype(int),"lamenessBinary":(lame>.4).astype(int),
            "calvingBinary":(calv>.5).astype(int),"severityLevel":sev})

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: MODEL ARCHITECTURE (SharedAttentionHazardEngine)
# ═══════════════════════════════════════════════════════════════════

class TemporalMHA(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        attn_out, attn_w = self.mha(x, x, x)
        ctx = attn_out.mean(dim=1)
        return ctx, attn_w

class SharedAttentionHazardEngine(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        d = hidden_dim * 2
        self.attention = TemporalMHA(d, num_heads=4)
        self.head_inf = nn.Linear(d, 1)
        self.head_heat = nn.Linear(d, 1)
        self.head_mast = nn.Linear(d, 1)
        self.head_lame = nn.Linear(d, 1)
        self.head_calv = nn.Linear(d, 1)
        self.head_sev = nn.Linear(d, 1)
        self.head_hazard = nn.Linear(d, 24)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ctx, tw = self.attention(lstm_out)
        l_i = self.head_inf(ctx); l_h = self.head_heat(ctx)
        l_m = self.head_mast(ctx); l_l = self.head_lame(ctx); l_c = self.head_calv(ctx)
        p_i=torch.sigmoid(l_i); p_m=torch.sigmoid(l_m); p_l=torch.sigmoid(l_l)
        p_h=torch.sigmoid(l_h); p_c=torch.sigmoid(l_c)
        mod = 1 + .5*p_i + .5*p_m + .3*p_l + .2*p_h + .2*p_c
        sev = self.head_sev(ctx) * mod
        haz = self.head_hazard(ctx) * mod
        logits_cls = torch.cat([l_i, l_h, l_m, l_l, l_c], dim=1)
        return logits_cls, sev.squeeze(1), haz, tw

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: VECTORIZED FOCAL LOSS
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
            "mastitisBinary","lamenessBinary","calvingBinary","severityLevel","domainEngine"}
    return [c for c in cols if c not in skip]

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

class SeqDataset(Dataset):
    def __init__(self, X, Yc, Yh):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.Yc=torch.tensor(Yc,dtype=torch.float32)
        self.Yh=torch.tensor(Yh,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Yc[i], self.Yh[i]

def main():
    logger.info("=" * 60)
    logger.info("🚀 COLAB T4 GPU — Phase 15.1 Domain Randomized Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── STEP 1: Generate Domain Randomized Data ──
    logger.info("Step 1: Generating 500 animals (70% SDE / 30% Alien)...")
    sim = DomainRandomizedUniverse(seed=1024)
    frames = []
    counts = {"N": 0, "A": 0}
    for i in range(N_ANIMALS):
        if sim.rng.random() < 0.30:
            frames.append(sim.generate_alien_animal(i)); counts["A"] += 1
        else:
            frames.append(sim.generate_native_sde_animal(i)); counts["N"] += 1
        if (i+1) % 100 == 0:
            logger.info(f"  Generated {i+1}/{N_ANIMALS} (SDE:{counts['N']} Alien:{counts['A']})")
    df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    logger.info(f"Total rows: {len(df):,}")

    # ── STEP 2: Extract Features ──
    logger.info("Step 2: Extracting rolling features...")
    df = extract_features(df)
    features = get_features(df.columns.tolist())
    logger.info(f"Feature dim: {len(features)}")

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    scalers_path = os.path.join(OUT_DIR, "v14_domain_scalers.json")
    with open(scalers_path, "w") as f:
        json.dump({"features": features, "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist()}, f)

    # ── STEP 3: Slice Sliding Windows ──
    logger.info("Step 3: Slicing sequence windows...")
    X_l, Yc_l, Yh_l = [], [], []
    for _, grp in df.groupby("animalId"):
        vf = grp[features].values.astype(np.float32)
        vc = grp[["infectionBinary","heatStressBinary","mastitisBinary","lamenessBinary","calvingBinary"]].values.astype(np.float32)
        vs = grp["severityLevel"].values.astype(np.float32)
        nt = len(vf)
        if nt < SEQ_LEN + HAZARD_HORIZON: continue
        for s in range(0, nt-SEQ_LEN-HAZARD_HORIZON, STRIDE):
            e = s + SEQ_LEN
            X_l.append(vf[s:e])
            Yc_l.append(vc[e:e+HAZARD_HORIZON].max(axis=0))
            sh = vs[e:e+HAZARD_HORIZON]
            yh = np.zeros(24, dtype=np.float32)
            for h in range(24):
                if np.any(sh[h*6:(h+1)*6] >= 2.0): yh[h] = 1.0
            Yh_l.append(yh)

    X = np.array(X_l, dtype=np.float32); Yc = np.array(Yc_l, dtype=np.float32); Yh = np.array(Yh_l, dtype=np.float32)
    del X_l, Yc_l, Yh_l, df; gc.collect()
    logger.info(f"Training samples: {len(X):,}")

    # ── STEP 4: CUDA T4 Training ──
    logger.info("Step 4: Training on CUDA T4...")
    freq = Yc.mean(axis=0)
    logger.info(f"Class freq: {freq}")
    alphas = torch.tensor(1.0 - freq, dtype=torch.float32)

    input_dim = len(features)
    model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
    focal = VectorizedFocalLoss(alphas.to(device), gamma=3.0).to(device)
    haz_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

    dataset = SeqDataset(X, Yc, Yh)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
    del X, Yc, Yh; gc.collect()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3,
                                                      steps_per_epoch=len(loader), epochs=4)
    scaler_amp = torch.amp.GradScaler("cuda")

    EPOCHS = 4
    model.train()
    for epoch in range(EPOCHS):
        tl = 0; nb = 0; t0 = time.time()
        for bi, (xb, yc, yh) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yc = yc.to(device, non_blocking=True)
            yh = yh.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lc, sv, hz, aw = model(xb)
                loss = focal(lc, yc) + 0.3 * haz_loss(hz, yh)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

            tl += loss.item(); nb += 1
            if bi % 10 == 0:
                logger.info(f"E{epoch+1}/4 B{bi}/{len(loader)} Loss:{loss.item():.4f}")

        logger.info(f"== EPOCH {epoch+1} | AvgLoss: {tl/nb:.4f} | {time.time()-t0:.1f}s ==")

    # ── STEP 5: Save Model ──
    model_cpu = model.to("cpu")
    mp = os.path.join(OUT_DIR, "v14_domain_attention_model.pth")
    torch.save(model_cpu.state_dict(), mp)
    logger.info(f"✅ Model saved → {mp}")
    logger.info(f"✅ Scalers saved → {scalers_path}")
    logger.info("=" * 60)
    logger.info("DONE! Download these files from colab_output/:")
    logger.info("  1. v14_domain_attention_model.pth")
    logger.info("  2. v14_domain_scalers.json")
    logger.info("Place them in: ml_service/models/cattle/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
