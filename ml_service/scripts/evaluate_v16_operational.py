#!/usr/bin/env python3
"""
evaluate_v16_operational.py — Phase 16 Part 2
V16 BIOLOGICAL ENGINE — FULL OPERATIONAL VALIDATION

Evaluates the V16 model (4 new heads) on fresh Alien Physics data:
1. Disease AUC (5 heads)
2. Episode Phase accuracy
3. Stability Collapse AUC
4. Recovery Hazard curve shape
5. Calving Window early warning
6. Isotonic recalibration + ECE
7. Per-head FP ≤ 5/week threshold optimization
"""

import os, sys, json, logging, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, accuracy_score, brier_score_loss,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("V16_Eval")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")

TICK_MIN = 10; TICKS_PER_DAY = 144
N_ANIMALS = 80; TICKS_PER_ANIMAL = 2000
SEQ_LEN = 288; STRIDE = 12; HAZARD_HORIZON = 144; CALVING_HORIZON = 432

# ═══════════════════════════════════════════════════════════════
# V16 MODEL ARCHITECTURE (must match Colab exactly)
# ═══════════════════════════════════════════════════════════════

class TemporalMHA(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        attn_out, attn_w = self.mha(x, x, x)
        return attn_out.mean(dim=1), attn_w

class V16BiologicalEngine(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        d = hidden_dim * 2
        self.attention = TemporalMHA(d, num_heads=4)
        self.head_inf = nn.Linear(d, 1); self.head_heat = nn.Linear(d, 1)
        self.head_mast = nn.Linear(d, 1); self.head_lame = nn.Linear(d, 1)
        self.head_calv = nn.Linear(d, 1)
        self.head_sev = nn.Linear(d, 1); self.head_hazard = nn.Linear(d, 24)
        self.phase_head = nn.Linear(d, 3)
        self.collapse_head = nn.Linear(d, 1)
        self.recovery_hazard_head = nn.Linear(d, 24)
        self.calving_hazard_head = nn.Linear(d, 72)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ctx, tw = self.attention(lstm_out)
        l_i=self.head_inf(ctx); l_h=self.head_heat(ctx)
        l_m=self.head_mast(ctx); l_l=self.head_lame(ctx); l_c=self.head_calv(ctx)
        p_i=torch.sigmoid(l_i); p_m=torch.sigmoid(l_m)
        p_l=torch.sigmoid(l_l); p_h=torch.sigmoid(l_h); p_c=torch.sigmoid(l_c)
        mod = 1+.5*p_i+.5*p_m+.3*p_l+.2*p_h+.2*p_c
        sev=self.head_sev(ctx)*mod; haz=self.head_hazard(ctx)*mod
        logits_cls=torch.cat([l_i,l_h,l_m,l_l,l_c], dim=1)
        return (logits_cls, sev.squeeze(1), haz, tw,
                self.phase_head(ctx), self.collapse_head(ctx).squeeze(1),
                self.recovery_hazard_head(ctx), self.calving_hazard_head(ctx))

# ═══════════════════════════════════════════════════════════════
# ALIEN PHYSICS GENERATOR
# ═══════════════════════════════════════════════════════════════

class AlienPhysicsGen:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def generate(self, idx):
        n = TICKS_PER_ANIMAL
        aid = f"AlienV16_{idx:03d}"
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
        if self.rng.random()<.35:
            s=self.rng.randint(50,n-200); d=self.rng.randint(100,300); e=min(s+d,n)
            inf[s:e]=np.exp(-np.linspace(0,1.5,d))[:e-s]
        mast=np.zeros(n)
        if self.rng.random()<.30:
            s=self.rng.randint(50,n-200); d=self.rng.randint(80,250); e=min(s+d,n)
            ramp=np.cumsum(self.rng.uniform(.005,.02,d)); mast[s:e]=np.clip(ramp[:e-s],0,1)
        lame=np.zeros(n)
        if self.rng.random()<.20:
            s=self.rng.randint(50,n-400); lame[s:]=np.linspace(.2,1,n-s)
        heat=(thi>75).astype(float)
        calv=np.zeros(n)
        if self.rng.random()<.12:
            s=self.rng.randint(200,n-200)
            calv[max(0,s-100):s+100]=np.exp(-.5*((np.arange(min(200,s+100-max(0,s-100)))-min(100,s))/35)**2)

        temp=tb+1.5*np.log1p(inf*10)+.5*np.exp(heat)+.6*mast-.4*calv+self.rng.normal(0,.2,n)
        hr=hb*(1+.2*inf)*(1+.15*heat)*(1+.12*calv)+self.rng.normal(0,2,n)
        resp=22+(12*heat)**1.2+3*inf+10*calv+self.rng.normal(0,3,n)
        act=np.clip(.7*(1-.6*lame)*(1-.3*inf)+calv*self.rng.normal(0,.3,n)+self.rng.normal(0,.05,n),0,1)
        mb=32+self.rng.normal(0,4)
        milk=np.clip(mb*(1-.4*mast)*(1-.15*inf)*(1-.1*heat)+self.rng.normal(0,1.5,n),0,None)
        feed=22*(1-.25*inf)*(1-.15*lame)+self.rng.normal(0,1,n)
        cond=5+np.exp(mast*1.5)-1+self.rng.normal(0,.3,n)
        sev=inf+mast+lame+heat

        # V16 labels
        sev_grad=np.gradient(sev); sev_sm=np.convolve(sev_grad,np.ones(12)/12,mode='same')
        phase=np.zeros(n,dtype=np.int64)
        for t in range(n):
            if sev[t]<.3: phase[t]=0
            elif sev_sm[t]>.005: phase[t]=0
            elif sev_sm[t]<-.005: phase[t]=2
            else: phase[t]=1

        nd=(inf>.4).astype(float)+(mast>.4).astype(float)+(lame>.4).astype(float)+(heat>.5).astype(float)
        collapse=((sev>=2.5)&(nd>=2)).astype(np.float32)

        return pd.DataFrame({"animalId":[aid]*n,
            "temperature_C":temp,"heartRate_bpm":hr,"respiration_bpm":resp,"activity_index":act,
            "thi":thi,"ambientTemp_C":ambient,"humidity_pct":humidity,
            "milkYield":milk,"feedIntake":feed,"conductivity":cond,
            "infectionBinary":(inf>.4).astype(int),"heatStressBinary":(heat>.5).astype(int),
            "mastitisBinary":(mast>.4).astype(int),"lamenessBinary":(lame>.4).astype(int),
            "calvingBinary":(calv>.5).astype(int),"severityLevel":sev,
            "episodePhase":phase,"collapseFlag":collapse,"calvingState":calv.astype(np.float32)})

# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1); ece = 0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.sum() == 0: continue
        ece += m.sum()/len(y_true) * abs(y_prob[m].mean() - y_true[m].mean())
    return ece

def find_threshold_fp_budget(y_true, y_prob, max_fp, windows_per_week):
    best_t, best_r = 0.5, 0
    for t in np.linspace(0.01, 0.99, 200):
        pred = (y_prob >= t).astype(int)
        fp = ((pred==1)&(y_true==0)).sum()
        fp_wk = fp/(len(y_true)/windows_per_week) if len(y_true)>0 else 0
        if fp_wk <= max_fp:
            tp = ((pred==1)&(y_true==1)).sum()
            rec = tp/y_true.sum() if y_true.sum()>0 else 0
            if rec > best_r: best_r=rec; best_t=t
    return best_t, best_r

# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("🔬 Phase 16 — V16 BIOLOGICAL ENGINE OPERATIONAL VALIDATION")
    logger.info("=" * 70)

    model_path = os.path.join(MODEL_DIR, "v16_shared_attention_engine.pth")
    scalers_path = os.path.join(MODEL_DIR, "v16_scalers.json")
    if not os.path.exists(model_path):
        logger.error("V16 model not found."); return

    # ── STEP 1: Generate Alien Data ──
    logger.info("Step 1: Generating 80 Alien Physics animals (seed=9999)...")
    sim = AlienPhysicsGen(seed=9999)
    df = pd.concat([sim.generate(i) for i in range(N_ANIMALS)], ignore_index=True)

    # ── STEP 2: Feature Extraction ──
    logger.info("Step 2: Feature extraction...")
    for w,ticks in {"6h":36,"12h":72,"24h":144}.items():
        for c in ["temperature_C","heartRate_bpm","respiration_bpm","activity_index","milkYield","conductivity"]:
            df[f"{c}_mean_{w}"]=df.groupby("animalId")[c].transform(lambda x: x.rolling(ticks,min_periods=1).mean())
            df[f"{c}_std_{w}"]=df.groupby("animalId")[c].transform(lambda x: x.rolling(ticks,min_periods=1).std().fillna(0))
            df[f"{c}_delta_{w}"]=df[c]-df[f"{c}_mean_{w}"]
    df["thermal_strain_index"]=df["heartRate_bpm_delta_6h"]/(df["thi"]-72+1e-5)
    df["lameness_suppression"]=(1-df["activity_index"])*(df["feedIntake"]/22)
    df["mastitis_spike_index"]=df["conductivity_delta_12h"]*(df["milkYield"]/30)
    df["fever_decoupled"]=df["temperature_C"]-(38.5+(0.01*np.maximum(df["thi"].values-72,0)))
    df.fillna(0, inplace=True)

    with open(scalers_path) as f: sd=json.load(f)
    features=sd["features"]
    scaler=StandardScaler(); scaler.mean_=np.array(sd["means"]); scaler.scale_=np.array(sd["scales"])
    df[features]=scaler.transform(df[features])

    # ── STEP 3: Sliding Windows with V16 Labels ──
    logger.info("Step 3: Slicing windows...")
    X_l,Yc_l,Yh_l,Yp_l,Ycol_l,Yrec_l,Ycw_l=[],[],[],[],[],[],[]
    for _,grp in df.groupby("animalId"):
        vf=grp[features].values.astype(np.float32)
        vc=grp[["infectionBinary","heatStressBinary","mastitisBinary","lamenessBinary","calvingBinary"]].values.astype(np.float32)
        vs=grp["severityLevel"].values.astype(np.float32)
        vph=grp["episodePhase"].values.astype(np.int64)
        vcol=grp["collapseFlag"].values.astype(np.float32)
        vcalv=grp["calvingState"].values.astype(np.float32)
        nt=len(vf)
        if nt<SEQ_LEN+CALVING_HORIZON: continue
        for s in range(0,nt-SEQ_LEN-CALVING_HORIZON,STRIDE):
            e=s+SEQ_LEN
            X_l.append(vf[s:e])
            Yc_l.append(vc[e:e+HAZARD_HORIZON].max(axis=0))
            sh=vs[e:e+HAZARD_HORIZON]
            yh=np.zeros(24,dtype=np.float32)
            for h in range(24):
                if np.any(sh[h*6:(h+1)*6]>=2.0): yh[h]=1.0
            Yh_l.append(yh)
            Yp_l.append(vph[e-1])
            Ycol_l.append(float(vcol[e:e+HAZARD_HORIZON].max()))
            yrec=np.zeros(24,dtype=np.float32)
            for h in range(24):
                bl=vs[e+h*6:e+(h+1)*6]
                if len(bl)>0 and np.any(bl<1.0) and vs[e-1]>=1.0: yrec[h]=1.0
            Yrec_l.append(yrec)
            ycw=np.zeros(72,dtype=np.float32)
            for h in range(72):
                bc=vcalv[e+h*6:e+(h+1)*6]
                if len(bc)>0 and np.any(bc>.5): ycw[h]=1.0
            Ycw_l.append(ycw)

    X=np.array(X_l); Yc=np.array(Yc_l); Yh=np.array(Yh_l)
    Yp=np.array(Yp_l); Ycol=np.array(Ycol_l); Yrec=np.array(Yrec_l); Ycw=np.array(Ycw_l)
    del X_l,Yc_l,Yh_l,Yp_l,Ycol_l,Yrec_l,Ycw_l,df; gc.collect()
    N=len(X)
    logger.info(f"Windows: {N}")

    # ── STEP 4: Model Inference ──
    logger.info("Step 4: V16 inference...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = V16BiologicalEngine(input_dim=len(features)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    P_cls,P_phase,P_col,P_rec,P_cw=[],[],[],[],[]
    with torch.no_grad():
        for i in range(0,N,128):
            xb=torch.tensor(X[i:i+128],dtype=torch.float32).to(device)
            lc,sv,hz,tw,ph,col,rec,cw=model(xb)
            P_cls.append(torch.sigmoid(lc).float().cpu().numpy())
            P_phase.append(ph.float().cpu().numpy())
            P_col.append(torch.sigmoid(col).float().cpu().numpy())
            P_rec.append(torch.sigmoid(rec).float().cpu().numpy())
            P_cw.append(torch.sigmoid(cw).float().cpu().numpy())

    P_cls=np.vstack(P_cls); P_phase=np.vstack(P_phase)
    P_col=np.concatenate(P_col); P_rec=np.vstack(P_rec); P_cw=np.vstack(P_cw)

    # ── STEP 5: Cal/Eval Split ──
    idx=np.arange(N); np.random.seed(42); np.random.shuffle(idx)
    cal=idx[:N//2]; evl=idx[N//2:]

    DISEASES=["Infection","HeatStress","Mastitis","Lameness","Calving"]
    WPW=588; FP_B=5

    # ═══════════════════════════════════════════════════════════
    # A. DISEASE HEADS (Isotonic + FP Budget)
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📊 A. DISEASE HEAD EVALUATION (5 Original Heads)")
    logger.info("=" * 70)

    P_cls_cal_iso=np.zeros_like(P_cls[evl])
    disease_report={}
    for i,d in enumerate(DISEASES):
        t_cal=Yc[cal,i]; p_cal=P_cls[cal,i]
        t_evl=Yc[evl,i]; p_evl=P_cls[evl,i]
        if len(np.unique(t_cal))>1:
            iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
            iso.fit(p_cal,t_cal)
            p_evl_cal=iso.predict(p_evl)
        else:
            p_evl_cal=p_evl
        P_cls_cal_iso[:,i]=p_evl_cal
        auc_r=roc_auc_score(t_evl,p_evl) if len(np.unique(t_evl))>1 else 0
        auc_c=roc_auc_score(t_evl,p_evl_cal) if len(np.unique(t_evl))>1 else 0
        ece_r=compute_ece(t_evl,p_evl); ece_c=compute_ece(t_evl,p_evl_cal)
        brier=brier_score_loss(t_evl,p_evl_cal)
        opt_t,opt_r=find_threshold_fp_budget(t_evl,p_evl_cal,FP_B,WPW)
        disease_report[d]={"AUC_raw":auc_r,"AUC_cal":auc_c,"ECE_raw":ece_r,"ECE_cal":ece_c,
                           "Brier":brier,"OptThresh":opt_t,"Recall@FP5":opt_r,"Positives":int(t_evl.sum())}

    logger.info(f"{'Disease':12} | {'AUC_raw':>8} | {'AUC_cal':>8} | {'ECE_raw':>8} | {'ECE_cal':>8} | {'Brier':>8} | {'OptThresh':>9} | {'Rec@FP≤5':>9}")
    logger.info("-"*100)
    for d,m in disease_report.items():
        logger.info(f"{d:12} | {m['AUC_raw']:8.4f} | {m['AUC_cal']:8.4f} | {m['ECE_raw']:8.4f} | {m['ECE_cal']:8.4f} | {m['Brier']:8.4f} | {m['OptThresh']:9.4f} | {m['Recall@FP5']:9.4f}")

    # ═══════════════════════════════════════════════════════════
    # B. EPISODE PHASE HEAD
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📊 B. EPISODE PHASE HEAD (Onset / Peak / Recovery)")
    logger.info("=" * 70)

    phase_pred = P_phase[evl].argmax(axis=1)
    phase_true = Yp[evl]
    phase_acc = accuracy_score(phase_true, phase_pred)
    phase_labels = ["Onset","Peak","Recovery"]
    cm = confusion_matrix(phase_true, phase_pred, labels=[0,1,2])
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    logger.info(f"Overall Phase Accuracy: {phase_acc:.4f}")
    for j,lbl in enumerate(phase_labels):
        logger.info(f"  {lbl:12}: {per_class_acc[j]:.4f} ({cm[j].sum()} samples)")

    # ═══════════════════════════════════════════════════════════
    # C. STABILITY COLLAPSE HEAD
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📊 C. STABILITY COLLAPSE HEAD")
    logger.info("=" * 70)

    col_true = Ycol[evl]; col_prob = P_col[evl]
    if len(np.unique(col_true)) > 1:
        col_auc = roc_auc_score(col_true, col_prob)
        col_ece = compute_ece(col_true, col_prob)
        col_t, col_r = find_threshold_fp_budget(col_true, col_prob, FP_B, WPW)
        logger.info(f"Collapse AUC:          {col_auc:.4f}")
        logger.info(f"Collapse ECE:          {col_ece:.4f}")
        logger.info(f"Collapse Thresh (FP≤5): {col_t:.4f}")
        logger.info(f"Collapse Recall:       {col_r:.4f}")
    else:
        col_auc = 0; col_ece = 0; col_t = 0.5; col_r = 0
        logger.info(f"Collapse: Single class in eval set (prevalence={col_true.mean():.4f})")

    # ═══════════════════════════════════════════════════════════
    # D. RECOVERY HAZARD HEAD
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📊 D. RECOVERY HAZARD HEAD (24h Curve)")
    logger.info("=" * 70)

    rec_true = Yrec[evl]; rec_prob = P_rec[evl]
    rec_aucs = []
    for h in [0, 5, 11, 17, 23]:
        if len(np.unique(rec_true[:,h])) > 1:
            a = roc_auc_score(rec_true[:,h], rec_prob[:,h])
            rec_aucs.append(a)
            logger.info(f"  Hour {h+1:2d} AUC: {a:.4f}")
        else:
            logger.info(f"  Hour {h+1:2d}: single class")
    avg_rec_auc = np.mean(rec_aucs) if rec_aucs else 0
    logger.info(f"Average Recovery AUC: {avg_rec_auc:.4f}")

    # Monotonicity check
    avg_curve = rec_prob.mean(axis=0)
    diffs = np.diff(avg_curve)
    mono_score = (diffs >= -0.01).mean()
    logger.info(f"Recovery curve monotonicity: {mono_score:.2%}")

    # ═══════════════════════════════════════════════════════════
    # E. CALVING WINDOW HEAD
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("📊 E. CALVING WINDOW 72h HEAD")
    logger.info("=" * 70)

    cw_true = Ycw[evl]; cw_prob = P_cw[evl]
    cw_aucs = []
    for h in [0, 5, 11, 23, 35, 47, 59, 71]:
        if len(np.unique(cw_true[:,h])) > 1:
            a = roc_auc_score(cw_true[:,h], cw_prob[:,h])
            cw_aucs.append(a)
            logger.info(f"  Hour {h+1:2d} AUC: {a:.4f}")
        else:
            logger.info(f"  Hour {h+1:2d}: single class")
    avg_cw_auc = np.mean(cw_aucs) if cw_aucs else 0
    logger.info(f"Average Calving Window AUC: {avg_cw_auc:.4f}")

    # Early warning check: did the 72h curve fire ≥12h before actual calving?
    calving_samples = cw_true.sum(axis=1) > 0
    if calving_samples.sum() > 0:
        early_fire = 0; total_calv = 0
        for idx_s in np.where(calving_samples)[0]:
            first_true = np.argmax(cw_true[idx_s] > 0)
            high_prob_before = cw_prob[idx_s, :max(first_true-12,0)].max() if first_true > 12 else 0
            if high_prob_before > 0.3: early_fire += 1
            total_calv += 1
        early_pct = early_fire / total_calv if total_calv > 0 else 0
        logger.info(f"Calving early warning (≥12h before): {early_pct:.2%} ({early_fire}/{total_calv})")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY VERDICT
    # ═══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("🏁 V16 OPERATIONAL VALIDATION SUMMARY")
    logger.info("=" * 70)

    avg_dis_auc = np.mean([m['AUC_cal'] for m in disease_report.values() if m['AUC_cal']>0])
    avg_dis_ece = np.mean([m['ECE_cal'] for m in disease_report.values()])
    avg_recall = np.mean([m['Recall@FP5'] for m in disease_report.values()])

    logger.info(f"Disease Avg AUC (cal):     {avg_dis_auc:.4f}")
    logger.info(f"Disease Avg ECE (cal):     {avg_dis_ece:.4f}")
    logger.info(f"Disease Avg Recall@FP≤5:   {avg_recall:.4f}")
    logger.info(f"Phase Accuracy:            {phase_acc:.4f}")
    logger.info(f"Collapse AUC:              {col_auc:.4f}")
    logger.info(f"Recovery Avg AUC:          {avg_rec_auc:.4f}")
    logger.info(f"Calving Window Avg AUC:    {avg_cw_auc:.4f}")

    all_pass = True
    checks = [
        ("Disease AUC ≥ 0.80", avg_dis_auc >= 0.80),
        ("Disease Recall ≥ 35%", avg_recall >= 0.35),
        ("Phase Acc ≥ 50%", phase_acc >= 0.50),
    ]
    for name, passed in checks:
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {name}")
        if not passed: all_pass = False

    if all_pass:
        logger.info("\n✅ V16 BIOLOGICAL ENGINE — PRODUCTION VALIDATED.")
    else:
        logger.info("\n⚠️ Some targets not met. Review individual heads.")

    # Save report
    report = {
        "disease_heads": disease_report,
        "phase_accuracy": phase_acc,
        "phase_per_class": {phase_labels[j]: float(per_class_acc[j]) for j in range(3)},
        "collapse_auc": col_auc,
        "recovery_avg_auc": avg_rec_auc,
        "calving_window_avg_auc": avg_cw_auc,
        "summary": {
            "avg_disease_auc": avg_dis_auc, "avg_disease_ece": avg_dis_ece,
            "avg_recall_fp5": avg_recall, "phase_accuracy": phase_acc,
            "collapse_auc": col_auc
        }
    }
    rp = os.path.join(MODEL_DIR, "v16_operational_report.json")
    with open(rp, "w") as f: json.dump(report, f, indent=2)
    logger.info(f"Report saved → {rp}")

if __name__ == "__main__":
    main()
