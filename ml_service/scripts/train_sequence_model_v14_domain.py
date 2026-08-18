#!/usr/bin/env python3
"""
train_sequence_model_v14_domain.py — Phase 15.1
REBALANCED DOMAIN-RANDOMIZED TRAINING (MPS GPU ACCELERATED)

Uses Apple MPS GPU for 10-50x speedup over CPU.
- MPS float32 (no CPU autocast)
- Batch size 512
- Vectorized Focal Loss (no Python per-head loop)
- Memmap streaming to stay under 4GB RAM
"""

import os, sys, json, logging, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Train_V14_Domain")

sys.path.append(os.path.dirname(__file__))
from train_sequence_model_v13_attn import SharedAttentionHazardEngine, get_device

DATA_DIR = os.path.join(os.path.dirname(__file__), "../training_data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/cattle")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 288
STRIDE = 24
HAZARD_HORIZON = 144

# ── VECTORIZED FOCAL LOSS (no per-head Python loop) ──
class VectorizedFocalLoss(nn.Module):
    """Computes focal loss across ALL heads in a single vectorized pass."""
    def __init__(self, alphas, gamma=3.0):
        super().__init__()
        self.register_buffer('alphas', alphas)  # [num_heads]
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: [B, num_heads], targets: [B, num_heads]
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B, H]
        pt = torch.exp(-bce)
        # alphas broadcast: [1, H] * [B, H]
        focal = self.alphas.unsqueeze(0) * (1 - pt) ** self.gamma * bce  # [B, H]
        return focal.mean()

class MemmapSequenceDataset(Dataset):
    def __init__(self, x_path, ycls_path, yhaz_path, n_samples, input_dim):
        self.X = np.memmap(x_path, dtype=np.float32, mode='r', shape=(n_samples, SEQ_LEN, input_dim))
        self.Y_cls = np.memmap(ycls_path, dtype=np.float32, mode='r', shape=(n_samples, 5))
        self.Y_haz = np.memmap(yhaz_path, dtype=np.float32, mode='r', shape=(n_samples, 24))
        self.n = n_samples
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx].copy()),
                torch.from_numpy(self.Y_cls[idx].copy()),
                torch.from_numpy(self.Y_haz[idx].copy()))

def extract_rolling_features_single(df_animal):
    windows = {"6h": 36, "12h": 72, "24h": 144}
    for w_name, w_ticks in windows.items():
        for col in ["temperature_C", "heartRate_bpm", "respiration_bpm", "activity_index", "milkYield", "conductivity"]:
            df_animal[f"{col}_mean_{w_name}"] = df_animal[col].rolling(w_ticks, min_periods=1).mean()
            df_animal[f"{col}_std_{w_name}"] = df_animal[col].rolling(w_ticks, min_periods=1).std().fillna(0)
            df_animal[f"{col}_delta_{w_name}"] = df_animal[col] - df_animal[f"{col}_mean_{w_name}"]
    df_animal["thermal_strain_index"] = df_animal["heartRate_bpm_delta_6h"] / (df_animal["thi"] - 72 + 1e-5)
    df_animal["lameness_suppression"] = (1.0 - df_animal["activity_index"]) * (df_animal["feedIntake"] / 22.0)
    df_animal["mastitis_spike_index"] = df_animal["conductivity_delta_12h"] * (df_animal["milkYield"] / 30.0)
    df_animal["fever_decoupled"] = df_animal["temperature_C"] - (38.5 + (0.01 * np.maximum(df_animal["thi"].values - 72, 0)))
    df_animal.fillna(0, inplace=True)
    return df_animal

def get_feature_names(sample_cols):
    exclude = {"animalId", "timestamp", "antibioticActive",
               "infectionBinary", "heatStressBinary", "mastitisBinary",
               "lamenessBinary", "calvingBinary", "severityLevel", "domainEngine"}
    return [c for c in sample_cols if c not in exclude]

def main():
    logger.info("============================================================")
    logger.info("🚀 Phase 15.1 — MPS GPU ACCELERATED DOMAIN TRAINING")
    logger.info("============================================================")

    # ── Detect MPS ──
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✅ Apple MPS GPU detected — using hardware acceleration")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ MPS not available — falling back to CPU")

    f_sensor = os.path.join(DATA_DIR, "trainingevents_v8_sensor.csv")
    f_prod = os.path.join(DATA_DIR, "trainingevents_v8_production.csv")
    if not os.path.exists(f_sensor):
        logger.error("V8 data missing — run simulator_v8_domain_randomized.py first.")
        return

    # ── PASS 1: Fit Scaler on Small Sample ──
    logger.info("Pass 1: Fitting scaler on 50 animals...")
    df_s = pd.read_csv(f_sensor)
    df_p = pd.read_csv(f_prod)
    df_s['timestamp'] = pd.to_datetime(df_s['timestamp'])
    df_p['timestamp'] = pd.to_datetime(df_p['timestamp'])

    animal_ids = df_s['animalId'].unique()

    sample_frames = []
    for aid in animal_ids[:50]:
        s_chunk = df_s[df_s['animalId'] == aid].sort_values('timestamp')
        p_chunk = df_p[df_p['animalId'] == aid].sort_values('timestamp')
        merged = pd.merge_asof(s_chunk, p_chunk, on="timestamp", by="animalId", direction="backward")
        merged = merged.ffill().fillna(0)
        merged = extract_rolling_features_single(merged)
        sample_frames.append(merged)

    df_sample = pd.concat(sample_frames, ignore_index=True)
    features = get_feature_names(df_sample.columns.tolist())
    logger.info(f"Feature dimension: {len(features)}")

    scaler = StandardScaler()
    scaler.fit(df_sample[features])
    with open(os.path.join(MODEL_DIR, "v14_domain_scalers.json"), "w") as f:
        json.dump({"features": features, "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist()}, f)
    del df_sample, sample_frames; gc.collect()
    logger.info("Scaler saved. Memory cleared.")

    # ── PASS 2: Stream windows → memmap ──
    logger.info("Pass 2: Streaming per-animal windows to memmap...")
    
    total_windows = 0
    for aid in animal_ids:
        n_ticks = len(df_s[df_s['animalId'] == aid])
        if n_ticks < SEQ_LEN + HAZARD_HORIZON: continue
        total_windows += len(range(0, n_ticks - SEQ_LEN - HAZARD_HORIZON, STRIDE))

    alloc_size = int(total_windows * 1.05) + 100
    input_dim = len(features)
    logger.info(f"Pre-counted {total_windows} windows (alloc={alloc_size})")

    tmp_dir = os.path.join(DATA_DIR, "memmap_v14")
    os.makedirs(tmp_dir, exist_ok=True)
    x_path = os.path.join(tmp_dir, "X.dat")
    ycls_path = os.path.join(tmp_dir, "Y_cls.dat")
    yhaz_path = os.path.join(tmp_dir, "Y_haz.dat")

    X_mm = np.memmap(x_path, dtype=np.float32, mode='w+', shape=(alloc_size, SEQ_LEN, input_dim))
    Ycls_mm = np.memmap(ycls_path, dtype=np.float32, mode='w+', shape=(alloc_size, 5))
    Yhaz_mm = np.memmap(yhaz_path, dtype=np.float32, mode='w+', shape=(alloc_size, 24))

    write_idx = 0
    processed = 0
    for aid in animal_ids:
        s_chunk = df_s[df_s['animalId'] == aid].sort_values('timestamp')
        p_chunk = df_p[df_p['animalId'] == aid].sort_values('timestamp')
        merged = pd.merge_asof(s_chunk, p_chunk, on="timestamp", by="animalId", direction="backward")
        merged = merged.ffill().fillna(0)
        merged = extract_rolling_features_single(merged)
        merged[features] = scaler.transform(merged[features])

        v_feat = merged[features].values.astype(np.float32)
        v_cls = merged[["infectionBinary", "heatStressBinary", "mastitisBinary",
                        "lamenessBinary", "calvingBinary"]].values.astype(np.float32)
        v_sev = merged["severityLevel"].values.astype(np.float32)
        n_ticks = len(v_feat)
        if n_ticks < SEQ_LEN + HAZARD_HORIZON: continue

        for start in range(0, n_ticks - SEQ_LEN - HAZARD_HORIZON, STRIDE):
            end = start + SEQ_LEN
            X_mm[write_idx] = v_feat[start:end]
            Ycls_mm[write_idx] = v_cls[end:end + HAZARD_HORIZON].max(axis=0)
            sev_h = v_sev[end:end + HAZARD_HORIZON]
            y_h = np.zeros(24, dtype=np.float32)
            for h in range(24):
                if np.any(sev_h[h*6:(h+1)*6] >= 2.0): y_h[h] = 1.0
            Yhaz_mm[write_idx] = y_h
            write_idx += 1

        processed += 1
        if processed % 100 == 0:
            X_mm.flush(); Ycls_mm.flush(); Yhaz_mm.flush(); gc.collect()
            logger.info(f"Streamed {processed}/{len(animal_ids)} animals | {write_idx} windows")

    X_mm.flush(); Ycls_mm.flush(); Yhaz_mm.flush()
    actual_windows = write_idx
    logger.info(f"✅ Memmap: {actual_windows} windows | dim={input_dim}")

    del df_s, df_p, X_mm, Ycls_mm, Yhaz_mm; gc.collect()

    # ── PASS 3: MPS GPU Training ──
    logger.info("Pass 3: MPS GPU Focal-Loss Training...")

    BATCH_SIZE = 512
    dataset = MemmapSequenceDataset(x_path, ycls_path, yhaz_path, actual_windows, input_dim)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    # Class frequencies → focal alphas
    Ycls_r = np.memmap(ycls_path, dtype=np.float32, mode='r', shape=(actual_windows, 5))
    class_freq = Ycls_r.mean(axis=0)
    logger.info(f"Class frequencies: {class_freq}")
    alphas = torch.tensor(1.0 - class_freq, dtype=torch.float32)
    del Ycls_r

    model = SharedAttentionHazardEngine(input_dim=input_dim).to(device)
    focal_loss = VectorizedFocalLoss(alphas.to(device), gamma=3.0).to(device)
    hazard_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3,
                                                      steps_per_epoch=len(loader), epochs=4)

    EPOCHS = 4
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0; n_batches = 0
        t0 = time.time()

        for batch_idx, (x_b, y_cls, y_haz) in enumerate(loader):
            x_b = x_b.to(device)
            y_cls = y_cls.to(device)
            y_haz = y_haz.to(device)

            optimizer.zero_grad(set_to_none=True)

            # NO autocast — pure float32 on MPS
            logits_cls, sev_out, hazard_logits, attn_w = model(x_b)

            # Vectorized focal loss — single tensor op, no Python loop
            loss_c = focal_loss(logits_cls, y_cls)
            loss_h = hazard_loss_fn(hazard_logits, y_haz)
            loss = loss_c + 0.3 * loss_h

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(loader)} | Loss {loss.item():.4f}")

        avg = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        logger.info(f"== EPOCH {epoch+1} DONE | Avg Loss: {avg:.4f} | Time: {elapsed:.1f}s ==")

    m_path = os.path.join(MODEL_DIR, "v14_domain_attention_model.pth")
    # Save on CPU for portability
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), m_path)
    logger.info(f"✅ V14 Domain Randomized model saved → {m_path}")

if __name__ == "__main__":
    main()
