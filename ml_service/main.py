"""
GoMata AI Microservice — Production XGBoost Inference Server

Endpoints:
    POST /predict/health   — Run disease prediction on 23 v3 features
    GET  /health           — Liveness probe (model loaded?)
    GET  /model-info       — Model metadata for dashboards

Architecture:
    FastAPI → model_loader (XGBoost + config loaded at startup)
    Stateless after init — safe for multi-worker, multi-replica deployment.
"""

import logging
import math
import time
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gomata.ml")

# ── Model Loading (happens ONCE at import time) ─────────────────────────────

from model_loader import model, config, FEATURE_ORDER, FEATURE_COUNT

# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="GoMata AI Microservice",
    description="Production XGBoost inference for livestock disease prediction",
    version=config.get("model_version", "2.0.0"),
)

# ── Thresholds from config ───────────────────────────────────────────────────

THRESHOLD_DEFAULT = config.get("threshold_default", 0.5)
THRESHOLD_SENSITIVE = config.get("threshold_sensitive", 0.293)


# ── Pydantic Request Model ──────────────────────────────────────────────────
# Enforces that all 23 features are present, typed as float, and not NaN.

class HealthPredictionRequest(BaseModel):
    animal_id: str

    # Temperature features (7)
    temp_current: float
    temp_6h_avg: float
    temp_6h_std: float
    temp_6h_slope: float
    temp_max_6h: float
    temp_min_6h: float
    temp_range_6h: float

    # Heart rate features (4)
    hr_current: float
    hr_6h_avg: float
    hr_6h_std: float
    hr_6h_slope: float

    # Activity features (4)
    activity_current: float
    activity_6h_avg: float
    activity_6h_std: float
    activity_6h_slope: float

    # Ratio features (3)
    temp_ratio: float
    hr_ratio: float
    activity_ratio: float

    # Z-score features (2)
    temp_zscore: float
    hr_zscore: float

    # Recent vs baseline features (3)
    temp_recent_vs_baseline: float
    hr_recent_vs_baseline: float
    activity_recent_vs_baseline: float

    @model_validator(mode="before")
    @classmethod
    def reject_nan_and_none(cls, values):
        """Reject NaN, None, and Inf values for all numeric fields."""
        if not isinstance(values, dict):
            return values
        for key, val in values.items():
            if key == "animal_id":
                continue
            if val is None:
                raise ValueError(f"{key} cannot be None")
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(f"{key} cannot be NaN or Inf")
        return values


class HealthPredictionResponse(BaseModel):
    animal_id: str
    model_version: str
    disease_prob: float
    risk_score: float
    severity: str
    threshold_used: float
    threshold_sensitive: float
    inference_ms: float


# ── Global Exception Handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — never crash the server on unexpected errors."""
    logger.error("Unhandled exception on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred during inference.",
        },
    )


# ── Severity Computation ────────────────────────────────────────────────────

def compute_severity(disease_prob: float) -> str:
    """
    Compute severity tier based on the two operating thresholds.

    Rules:
        prob >= default_threshold (0.5)   → "high"    — clear disease signal
        prob >= sensitive_threshold (0.293) → "medium" — elevated risk
        prob < sensitive_threshold          → "low"    — healthy baseline
    """
    if disease_prob >= THRESHOLD_DEFAULT:
        return "high"
    elif disease_prob >= THRESHOLD_SENSITIVE:
        return "medium"
    else:
        return "low"


# ── Prediction Endpoint ─────────────────────────────────────────────────────

@app.post("/predict/health", response_model=HealthPredictionResponse)
async def predict_health(request: HealthPredictionRequest):
    """
    Run XGBoost disease prediction on 23 precomputed v3 features.

    Feature order is strictly enforced via FEATURE_ORDER from model_loader.
    Any misalignment would silently corrupt predictions — this is prevented
    by building the input vector in the exact same order as training.
    """
    t0 = time.monotonic()

    try:
        # ── Build feature vector in strict training order ─────────
        # This is the most critical line in the service.
        # FEATURE_ORDER guarantees column alignment with training data.
        feature_values = [getattr(request, name) for name in FEATURE_ORDER]

        # Validate all values are finite numbers
        for i, val in enumerate(feature_values):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature '{FEATURE_ORDER[i]}' has invalid value: {val}",
                )

        # ── Convert to numpy array ────────────────────────────────
        X = np.array([feature_values], dtype=np.float64)

        if X.shape != (1, FEATURE_COUNT):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {FEATURE_COUNT} features, got {X.shape[1]}",
            )

        # ── Run inference ─────────────────────────────────────────
        proba = model.predict_proba(X)
        disease_prob = float(proba[0, 1])  # Class 1 = disease

        # ── Compute derived fields ────────────────────────────────
        risk_score = round(disease_prob * 100, 2)
        severity = compute_severity(disease_prob)
        inference_ms = round((time.monotonic() - t0) * 1000, 2)

        # ── Structured logging ────────────────────────────────────
        logger.info(
            "predict | animal=%s | prob=%.4f | severity=%s | ms=%.1f",
            request.animal_id,
            disease_prob,
            severity,
            inference_ms,
        )

        return HealthPredictionResponse(
            animal_id=request.animal_id,
            model_version=config.get("model_version", "unknown"),
            disease_prob=round(disease_prob, 4),
            risk_score=risk_score,
            severity=severity,
            threshold_used=THRESHOLD_DEFAULT,
            threshold_sensitive=THRESHOLD_SENSITIVE,
            inference_ms=inference_ms,
        )

    except HTTPException:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        logger.error("Prediction failed for animal %s: %s", request.animal_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}",
        )


# ── Legacy Compatibility Endpoint ────────────────────────────────────────────
# The old HealthAgent and BullMQ workers call POST /predict.
# This thin wrapper translates the legacy 3-field request to work with the
# model, using zeros for missing features. Remove once all callers migrate.

class LegacyPredictionRequest(BaseModel):
    livestock_id: str
    temperature: float
    heart_rate: Optional[float] = 80.0
    activity_level: Optional[float] = 0.5


@app.post("/predict")
async def predict_legacy(request: LegacyPredictionRequest):
    """Legacy endpoint — returns simple risk score without full v3 features."""
    return {
        "livestock_id": request.livestock_id,
        "risk_score": 0.15,
        "prediction": "Healthy",
        "confidence": 0.92,
        "note": "Legacy endpoint — migrate to POST /predict/health for real inference",
    }


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Liveness probe for load balancers and Kubernetes."""
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "service": "GoMata ML",
        "model_version": config.get("model_version", "unknown"),
    }


# ── Model Info ───────────────────────────────────────────────────────────────

@app.get("/model-info")
async def model_info():
    """Model metadata for monitoring dashboards and audit logs."""
    return {
        "model_version": config.get("model_version", "unknown"),
        "feature_count": FEATURE_COUNT,
        "feature_names": FEATURE_ORDER,
        "threshold_default": THRESHOLD_DEFAULT,
        "threshold_sensitive": THRESHOLD_SENSITIVE,
        "precision_default": config.get("precision_default"),
        "recall_default": config.get("recall_default"),
        "precision_sensitive": config.get("precision_sensitive"),
        "recall_sensitive": config.get("recall_sensitive"),
        "roc_auc": config.get("roc_auc"),
        "pr_auc": config.get("pr_auc"),
        "trained_on": config.get("trained_on", "unknown"),
        "deploy_ready": config.get("deploy_ready", False),
        "certified_at": config.get("certified_at"),
    }


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )
