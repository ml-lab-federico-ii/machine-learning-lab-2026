"""
train.py — Model training and evaluation endpoint.
"""

from __future__ import annotations

import io
import math
import pickle
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dependencies import get_schema, get_train_df
from ml.evaluator import evaluate, evaluate_at_threshold
from ml.pipeline_builder import build_pipeline


def _sanitize(obj):
    """Recursively replace non-finite floats so json.dumps never raises."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

router = APIRouter(prefix="/api", tags=["train"])

SEED = 42


class TrainRequest(BaseModel):
    pipeline_config: dict[str, Any]


class ThresholdRequest(BaseModel):
    threshold: float
    # Cached probabilities from the last training run (passed back from frontend)
    y_true: list[int]
    y_proba: list[float]


@router.post("/train")
def train(request: TrainRequest):
    config = request.pipeline_config
    df = get_train_df()
    schema = get_schema()

    id_col = schema["id_column"]
    target_col = schema["target_column"]
    positive_class = schema.get("target_positive_class", 1)

    # ── Prepare feature matrix and target ──────────────────────────────────
    feature_cols = [c for c in df.columns if c not in (id_col, target_col)]
    X = df[feature_cols].copy()
    y_raw = df[target_col]

    # Map target to 0/1
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        y = (y_raw == str(positive_class)).astype(int).values
    else:
        y = (y_raw == positive_class).astype(int).values

    # ── Train / val / test split ────────────────────────────────────────────
    split_cfg = config.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("val", 0.15))
    # test_ratio is implicitly 1 - train_ratio - val_ratio

    rng = np.random.default_rng(SEED)
    n = len(X)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # ── Build and fit pipeline ──────────────────────────────────────────────
    try:
        pipeline = build_pipeline(config, X_train)
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Training failed: {exc}") from exc

    # ── Evaluate on validation set ──────────────────────────────────────────
    try:
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}") from exc

    val_metrics = evaluate(y_val, y_val_proba)
    test_metrics = evaluate(y_test, y_test_proba)

    # ── Serialise model to bytes for temporary caching in session ──────────
    model_bytes = io.BytesIO()
    pickle.dump(pipeline, model_bytes)
    model_b64 = __import__("base64").b64encode(model_bytes.getvalue()).decode()

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        # Probabilities returned to frontend for threshold slider re-evaluation
        # without re-training.
        "val_y_true": y_val.tolist(),
        "val_y_proba": [round(float(p), 6) for p in y_val_proba],
        # Model bytes for submission (base64-encoded, not stored server-side long-term)
        "model_b64": model_b64,
    }


@router.post("/evaluate-threshold")
def evaluate_threshold(request: ThresholdRequest):
    """Re-evaluate metrics at a different threshold without re-training."""
    y_true = np.array(request.y_true)
    y_proba = np.array(request.y_proba)
    if len(y_true) == 0 or len(y_proba) == 0:
        raise HTTPException(status_code=422, detail="Empty arrays")
    return evaluate_at_threshold(y_true, y_proba, request.threshold)
