"""
model_loader.py — Production Model & Config Loader

Loads the trained XGBoost model and model_config.json once at import time.
Thread-safe, immutable after initialization. Designed for multi-worker
deployment (uvicorn --workers N, gunicorn, Kubernetes pods).

Usage:
    from model_loader import model, config, FEATURE_ORDER
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

# ── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger("gomata.ml")

# ── Paths ────────────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).parent
_MODEL_DIR = _BASE_DIR / "models" / "cattle"
_MODEL_PATH = _MODEL_DIR / "disease_classifier_v2.json"
_CONFIG_PATH = _MODEL_DIR / "model_config.json"

# ── Strict Feature Order ────────────────────────────────────────────────────
# This MUST match the exact order used during training (train_classifier.py).
# Any misalignment silently corrupts predictions — this is the single
# source of truth for feature ordering across the entire system.

FEATURE_ORDER = [
    "temp_current",
    "temp_6h_avg",
    "temp_6h_std",
    "temp_6h_slope",
    "temp_max_6h",
    "temp_min_6h",
    "temp_range_6h",
    "hr_current",
    "hr_6h_avg",
    "hr_6h_std",
    "hr_6h_slope",
    "activity_current",
    "activity_6h_avg",
    "activity_6h_std",
    "activity_6h_slope",
    "temp_ratio",
    "hr_ratio",
    "activity_ratio",
    "temp_zscore",
    "hr_zscore",
    "temp_recent_vs_baseline",
    "hr_recent_vs_baseline",
    "activity_recent_vs_baseline",
]

FEATURE_COUNT = len(FEATURE_ORDER)  # 23

# ── Config Loader ────────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load model_config.json with all thresholds and metadata."""
    if not _CONFIG_PATH.exists():
        logger.warning("model_config.json not found at %s — using defaults", _CONFIG_PATH)
        return {
            "model_version": "unknown",
            "threshold_default": 0.5,
            "threshold_sensitive": 0.293,
            "feature_count": FEATURE_COUNT,
            "roc_auc": None,
            "pr_auc": None,
            "trained_on": "unknown",
            "deploy_ready": False,
        }

    with open(_CONFIG_PATH, "r") as f:
        raw = json.load(f)

    # Validate feature count matches
    cfg_features = raw.get("feature_count", FEATURE_COUNT)
    if cfg_features != FEATURE_COUNT:
        raise ValueError(
            f"Feature count mismatch: config says {cfg_features}, "
            f"FEATURE_ORDER has {FEATURE_COUNT}"
        )

    logger.info(
        "Config loaded — model: %s, threshold_default: %.3f, threshold_sensitive: %.3f",
        raw.get("model_version"),
        raw.get("threshold_default", 0.5),
        raw.get("threshold_sensitive", 0.293),
    )
    return raw


# ── Model Loader ─────────────────────────────────────────────────────────────


def _load_model() -> xgb.XGBClassifier:
    """Load the trained XGBoost classifier from disk."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {_MODEL_PATH}. "
            f"Run train_classifier.py first."
        )

    t0 = time.monotonic()

    clf = xgb.XGBClassifier()
    clf.load_model(str(_MODEL_PATH))

    load_ms = (time.monotonic() - t0) * 1000
    logger.info("Model loaded in %.1fms — %s", load_ms, _MODEL_PATH.name)

    return clf


# ── Warmup Inference ─────────────────────────────────────────────────────────


def _warmup(clf: xgb.XGBClassifier) -> None:
    """Run a single dummy inference to JIT-compile prediction path.
    This ensures the first real request isn't penalized by lazy init."""
    dummy = np.zeros((1, FEATURE_COUNT), dtype=np.float64)
    t0 = time.monotonic()
    clf.predict_proba(dummy)
    warmup_ms = (time.monotonic() - t0) * 1000
    logger.info("Model warmup complete — %.1fms", warmup_ms)


# ── Module-Level Initialization ──────────────────────────────────────────────
# Runs ONCE when this module is imported (at FastAPI startup).
# After this point, `model` and `config` are immutable read-only globals.

config: dict = _load_config()
model: xgb.XGBClassifier = _load_model()
_warmup(model)
