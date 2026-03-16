"""
evaluator.py
------------
Compute all evaluation metrics from true labels and predicted probabilities.
Returns plain dicts/lists suitable for JSON serialisation.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the full evaluation suite.

    Parameters
    ----------
    y_true  : binary labels (0/1), shape (n,)
    y_proba : predicted positive-class probability, shape (n,)
    threshold : classification threshold for binary metrics

    Returns
    -------
    dict with keys:
        roc_auc, roc_curve, confusion_matrix, precision, recall, f1
    """
    roc_auc = float(roc_auc_score(y_true, y_proba))
    roc_data = _roc_curve_data(y_true, y_proba)
    cm = _confusion_matrix_data(y_true, y_proba, threshold)
    binary_metrics = _binary_metrics(y_true, y_proba, threshold)

    return {
        "roc_auc": round(roc_auc, 6),
        "roc_curve": roc_data,
        "confusion_matrix": cm,
        **binary_metrics,
    }


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Return only the threshold-dependent metrics (for the live slider)."""
    cm = _confusion_matrix_data(y_true, y_proba, threshold)
    metrics = _binary_metrics(y_true, y_proba, threshold)
    return {"confusion_matrix": cm, **metrics}


# ── Private helpers ─────────────────────────────────────────────────────────

def _safe(v: float) -> float | None:
    """Return None for NaN/Inf so the value is always JSON-serialisable."""
    try:
        f = float(v)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def _roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {
        "fpr": [_safe(round(float(v), 6)) for v in fpr],
        "tpr": [_safe(round(float(v), 6)) for v in tpr],
        # sklearn prepends max(y_score)+1 as a sentinel threshold — cap to 1.0
        "thresholds": [_safe(round(min(float(v), 1.0), 6)) for v in thresholds],
    }


def _confusion_matrix_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": _safe(round(float(precision_score(y_true, y_pred, zero_division=0)), 4)),
        "recall": _safe(round(float(recall_score(y_true, y_pred, zero_division=0)), 4)),
        "f1": _safe(round(float(f1_score(y_true, y_pred, zero_division=0)), 4)),
        "threshold": round(float(threshold), 4),
    }
