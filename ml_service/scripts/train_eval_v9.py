#!/usr/bin/env python3
"""
train_eval_v9.py — Phase 9 Unified Pipeline
Extract features from v5.2 → train disease classifier → eval early detection
→ generate pilot_readiness_v9_report.json
"""

import os, sys, json, time, traceback
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pymongo import MongoClient
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(__file__))
from split_strategy_v2 import animal_time_split

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/livestock_monitoring")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")
TICK_HOURS = 5 / 60
TICKS_PER_WEEK = 2016

out = None  # Will be set in main


def log(msg):
    print(msg)
    if out:
        out.write(msg + "\n")
        out.flush()


def vectorized_slope(s, w):
    return ((s - s.shift(w - 1)) / max(w, 1)).fillna(0)


def extract_from_v52():
    """Load v5.2 from MongoDB, compute full v6 features."""
    log("── Loading v5.2 from MongoDB ──")
    client = MongoClient(MONGO_URI)
    db = client["livestock_monitoring"]

    docs = list(db["trainingevents_v5_2"].find({}, {"_id": 0}).sort("timestamp", 1))
    rows = []
    for d in docs:
        sig = d.get("signals", {}); env = d.get("environment", {}); lab = d.get("labels", {})
        rows.append({
            "animal_id": d["animalId"], "timestamp": d["timestamp"],
            "temp_current": sig.get("temperature_C", 38.5),
            "hr_current": sig.get("heartRate_bpm", 70),
            "resp_current": sig.get("respiration_bpm", 25),
            "activity_current": sig.get("activity_index", 0.5),
            "rumination_current": sig.get("rumination_min", 35),
            "lying_current": sig.get("lying_min", 25),
            "thi": env.get("thi", 65), "ambient_temp": env.get("ambientTemp_C", 22),
            "humidity": env.get("humidity_pct", 55),
            "disease_binary": lab.get("diseaseBinary", 0),
            "severity_level": lab.get("severityLevel", 0),
        })
    dfs = pd.DataFrame(rows)
    log(f"  Sensor: {len(dfs)} rows, {dfs['animal_id'].nunique()} animals")

    # Production
    pdocs = list(db["trainingevents_v5_2_production"].find({}, {"_id": 0}).sort("timestamp", 1))
    prows = []
    for d in pdocs:
        p = d.get("production", {}); m = d.get("management", {})
        prows.append({
            "animal_id": d["animalId"], "timestamp": d["timestamp"],
            "milk_yield": p.get("milkYield", 25), "feed_intake": p.get("feedIntake", 20),
            "conductivity": p.get("conductivity", 5.0), "body_weight": p.get("bodyWeight", 550),
            "vaccination_effective": m.get("vaccinationEffective", 0),
            "antibiotic_effective": m.get("antibioticEffective", 0),
        })
    dfp = pd.DataFrame(prows)
    log(f"  Production: {len(dfp)} rows")
    client.close()
    return dfs, dfp


def compute_features(dfs, dfp):
    """Compute all v6 features per animal."""
    log("── Computing features ──")
    ml = min(len(dfs), len(dfp))
    dfs = dfs.iloc[:ml].reset_index(drop=True)
    dfp = dfp.iloc[:ml].reset_index(drop=True)
    df = dfs.copy()
    for c in ["milk_yield", "feed_intake", "conductivity", "body_weight",
              "vaccination_effective", "antibiotic_effective"]:
        if c in dfp.columns: df[c] = dfp[c]

    results = []
    for aid, g in df.groupby("animal_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        rng = np.random.RandomState(hash(aid) % 2**31)

        for sig, pfx in [("temp_current","temp"),("hr_current","hr"),
                         ("resp_current","resp"),("activity_current","activity")]:
            s = g[sig].astype(float)
            # V5 rolling
            g[f"{pfx}_1h_avg"]=s.rolling(12,1).mean()
            g[f"{pfx}_6h_avg"]=s.rolling(72,1).mean()
            g[f"{pfx}_12h_median"]=s.rolling(144,1).median()
            g[f"{pfx}_24h_std"]=s.rolling(288,1).std().fillna(0)
            g[f"{pfx}_6h_std"]=s.rolling(72,1).std().fillna(0)
            g[f"{pfx}_1h_std"]=s.rolling(12,1).std().fillna(0)
            for lh,ln in [(1,12),(3,36),(6,72),(12,144)]:
                g[f"{pfx}_lag_{lh}h"]=s.shift(ln).fillna(s.iloc[0])
            # V6 acceleration
            g[f"{pfx}_slope_1h"]=vectorized_slope(s,12)
            g[f"{pfx}_slope_3h"]=vectorized_slope(s,36)
            g[f"{pfx}_slope_6h"]=vectorized_slope(s,72)
            g[f"{pfx}_accel_3h"]=vectorized_slope(g[f"{pfx}_slope_3h"],36)
            std3=s.rolling(36,1).std().fillna(0.001)
            std24=s.rolling(288,1).std().fillna(0.001).clip(lower=0.001)
            g[f"{pfx}_instability"]=(std3/std24).round(4)
            var6=s.rolling(72,1).var().fillna(0)
            var24=s.rolling(288,1).var().fillna(0.001).clip(lower=0.001)
            g[f"{pfx}_var_ratio"]=(var6/var24).round(4)
            bl=s.rolling(288*7,288).mean().fillna(s.expanding().mean())
            g[f"{pfx}_delta_7d"]=(s-bl).round(4)
            std1=s.rolling(12,1).std().fillna(0)
            g[f"{pfx}_volatility_spike"]=(std1>2*std24).astype(int)

        g["resp_6h_avg"]=g["resp_current"].rolling(72,1).mean()
        g["resp_6h_std"]=g["resp_current"].rolling(72,1).std().fillna(0)
        g["hr_temp_ratio"]=(g["hr_current"]/g["temp_current"].clip(lower=36)).round(4)
        g["resp_activity_ratio"]=(g["resp_current"]/g["activity_current"].clip(lower=0.01)).round(4)
        g["rumination_velocity"]=g["rumination_current"].diff().fillna(0).rolling(12,1).mean()

        # Production features
        if "milk_yield" in g.columns:
            for f, c in [("milk","milk_yield"),("conductivity","conductivity"),
                         ("feed","feed_intake"),("weight","body_weight")]:
                bl=g[c].rolling(288*7,288).mean().fillna(g[c].mean())
                g[f"{f}_deviation"]=g[c]-bl
        for ev,col in [("vaccination","vaccination_effective"),("antibiotic","antibiotic_effective")]:
            le=np.zeros(len(g))
            for i in range(len(g)):
                if g[col].iloc[i]: le[i:]=i
            hs=(np.arange(len(g))-le)*TICK_HOURS
            g[f"hours_since_{ev}"]=np.where(le>0,hs,999)
            g[f"{ev[:4]}_decay"]=np.exp(-hs/168)
        g["hours_since_transport"]=999; g["hours_since_feed_change"]=999
        g["transport_decay"]=0; g["feed_decay"]=0
        g["total_antibiotic_days"]=g["antibiotic_effective"].cumsum()*TICK_HOURS/24
        g["vaccination_count_12m"]=g["vaccination_effective"].cumsum()
        g["feed_changes_30d"]=0
        g["parity"]=rng.randint(1,5); g["bcs"]=3.0+rng.normal(0,0.3); g["age"]=rng.randint(2,8)
        results.append(g)

    df = pd.concat(results, ignore_index=True)
    log(f"  Features: {len(df)} rows, {len(df.columns)} cols")
    return df


def main():
    global out
    start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    out = open("/tmp/v9_results.txt", "w")

    try:
        log("=" * 60)
        log("🧠 Phase 9 — v5.2 Preclinical Pipeline")
        log("=" * 60)

        dfs, dfp = extract_from_v52()
        df = compute_features(dfs, dfp)

        # Feature columns
        meta = {"animal_id","disease_binary","severity_level","timestamp",
                "milk_yield","feed_intake","conductivity","body_weight",
                "vaccination_effective","antibiotic_effective"}
        fcols = [c for c in df.columns if c not in meta]
        log(f"Feature cols: {len(fcols)}")

        # Save
        save_path = os.path.join(DATA_DIR, "features_v9.csv")
        df.to_csv(save_path, index=False)
        log(f"Saved: {save_path}")

        # Split
        X_tr, X_te, y_tr, y_te, _ = animal_time_split(
            df, fcols, "disease_binary",
            animal_train_ratio=0.8, time_train_ratio=0.7, seed=42)
        log(f"Train: {len(X_tr)} (pos={int(y_tr.sum())}), Test: {len(X_te)} (pos={int(y_te.sum())})")

        # Train disease classifier
        log("\n── Training disease classifier ──")
        sw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=sw,
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        prob = model.predict_proba(X_te)[:, 1]

        roc = roc_auc_score(y_te, prob)
        pr = average_precision_score(y_te, prob)
        brier = brier_score_loss(y_te, prob)
        log(f"ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}, Brier: {brier:.4f}")

        # ECE
        bins=np.linspace(0,1,11); ece=0; mce=0
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(prob>=lo)&(prob<hi)
            if m.sum()>0: g=abs(prob[m].mean()-y_te.values[m].mean()); ece+=m.sum()*g; mce=max(mce,g)
        ece/=len(y_te)
        log(f"ECE: {ece:.4f}, MCE: {mce:.4f}")

        # Feature importance
        imp=model.feature_importances_
        fi=pd.DataFrame({"f":fcols[:len(imp)],"i":imp}).sort_values("i",ascending=False)
        fi["p"]=(fi["i"]/fi["i"].sum()*100).round(2)
        ak=["slope","accel","instability","var_ratio","delta_7d","volatility","velocity","ratio"]
        at15=sum(1 for _,r in fi.head(15).iterrows() if any(k in r["f"] for k in ak))
        log(f"Accel in top15: {at15}")
        for _,r in fi.head(10).iterrows(): log(f"  {r['f']}: {r['p']:.2f}%")

        joblib.dump(model, os.path.join(MODEL_DIR, "disease_model_v9.pkl"))

        # Early detection
        log("\n── Early Detection ──")
        sev=df.loc[y_te.index,"severity_level"].values
        aids=df.loc[y_te.index,"animal_id"].values

        for thresh in [0.1, 0.2, 0.3, 0.5]:
            alerts=prob>=thresh
            d24=0;d12=0;d6=0;eps=0;fp=0;cw=0;leads=[]
            tno=0;twi=0;tr=0; decay=0.5**(1/(6/TICK_HOURS))
            for aid in np.unique(aids):
                m=aids==aid;sa=sev[m];aa=alerts[m];n=m.sum();ie=False
                for i in range(len(sa)):
                    if sa[i]>=2 and not ie:
                        ie=True;eps+=1
                        lb=min(i,576);w=aa[max(0,i-lb):i]
                        if w.any():
                            f=np.where(w)[0][0];lh=(len(w)-f)*TICK_HOURS;leads.append(lh)
                            if lh>=24:d24+=1
                            if lh>=12:d12+=1
                            if lh>=6:d6+=1
                    elif sa[i]<2:ie=False
                fp+=int((aa&(sa<0.5)).sum());cw+=n/TICKS_PER_WEEK
                # Economic
                sf=sa.astype(float);ie2=False;es=None
                for i in range(len(sf)):
                    if sf[i]>0.5 and not ie2:es=i;ie2=True
                    elif sf[i]<=0.1 and ie2:
                        if i-es>12:
                            ep=sf[es:i];nl=float((ep*0.5*TICK_HOURS).sum());tno+=nl
                            ea=aa[es:i]
                            if ea.any():
                                fi2=np.where(ea)[0][0];si=ep.copy()
                                for k2 in range(fi2,len(si)):si[k2]*=decay**(k2-fi2)
                                twi+=float((si*0.5*TICK_HOURS).sum());tr+=1
                            else:twi+=nl
                        ie2=False
            fpw=fp/max(cw,1);p24=d24/max(eps,1)*100;p12=d12/max(eps,1)*100
            p6=d6/max(eps,1)*100;avg=float(np.mean(leads)) if leads else 0
            sv=tno-twi;pct=sv/max(tno,0.001)*100
            log(f"θ={thresh}: eps={eps}, det={len(leads)}, 24h={p24:.0f}%, 12h={p12:.0f}%, "
                f"6h={p6:.0f}%, avg={avg:.1f}h, FP/wk={fpw:.1f}, econ={pct:.0f}%")

        # Pilot readiness
        log("\n── Pilot Readiness ──")
        pilot = {
            "version": "pilot_readiness_v9", "data": "v5.2_preclinical",
            "disease_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "ece": round(float(ece), 4), "mce": round(float(mce), 4),
            "brier": round(float(brier), 4),
            "accel_top15": at15,
            "top10": {r["f"]:round(float(r["p"]),2) for _,r in fi.head(10).iterrows()},
        }
        with open(os.path.join(DATA_DIR, "pilot_readiness_v9_report.json"), "w") as f:
            json.dump(pilot, f, indent=2)
        log("Saved pilot_readiness_v9_report.json")

        elapsed = time.time() - start
        log(f"\nTotal: {elapsed:.1f}s")
        log("DONE")

    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc(file=out)

    out.close()


if __name__ == "__main__":
    main()
