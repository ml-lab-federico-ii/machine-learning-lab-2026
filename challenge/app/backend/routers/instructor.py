"""
instructor.py — Instructor-only scoring and leaderboard endpoints.

All endpoints require a valid instructor session.
The hidden test set is uploaded once per session and held IN MEMORY only —
it is never written to disk inside the container.
"""

from __future__ import annotations

import base64
import io
import os
import pickle
import pickletools
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ml.evaluator import evaluate

router = APIRouter(prefix="/api/instructor", tags=["instructor"])

GITHUB_API_URL = "https://api.github.com"
BRANCH_NAME = "challenge-submission"

# ── Allowed sklearn top-level module prefixes for safe unpickling ───────────
_SAFE_MODULES = frozenset({
    "sklearn",
    "imblearn",
    "numpy",
    "pandas",
    "xgboost",
    "lightgbm",
    "scipy",
    "joblib",
    "builtins",
    "_abc",
    "abc",
    "collections",
})


class _RestrictedUnpickler(pickle.Unpickler):
    """Only allow classes from trusted ML libraries."""

    def find_class(self, module: str, name: str):
        root = module.split(".")[0]
        if root not in _SAFE_MODULES:
            raise pickle.UnpicklingError(
                f"Blocked: {module}.{name} is not in the trusted module list"
            )
        return super().find_class(module, name)


def _safe_loads(data: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(data)).load()


# ── Auth helpers ─────────────────────────────────────────────────────────────

def _require_instructor(request: Request):
    if not request.session.get("instructor_authenticated"):
        raise HTTPException(status_code=401, detail="Instructor authentication required")


# ── Endpoints ────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str


@router.post("/login")
def login(payload: LoginRequest, request: Request):
    expected = os.environ.get("INSTRUCTOR_PASSWORD", "")
    if not expected or payload.password != expected:
        raise HTTPException(status_code=401, detail="Invalid instructor password")
    request.session["instructor_authenticated"] = True
    return {"ok": True}


@router.post("/logout")
def logout(request: Request):
    request.session.pop("instructor_authenticated", None)
    request.session.pop("test_df", None)
    return {"ok": True}


@router.post("/upload-test")
async def upload_test(request: Request, file: UploadFile = File(...)):
    _require_instructor(request)

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    # Validate schema columns
    from dependencies import get_schema
    schema = get_schema()
    id_col = schema["id_column"]
    target_col = schema["target_column"]

    for col in (id_col, target_col):
        if col not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Expected column '{col}' not found in uploaded CSV",
            )

    # Store as oriented records in session (serialisable)
    request.session["test_df"] = df.to_json(orient="records")
    return {
        "ok": True,
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
    }


@router.get("/submissions")
async def list_submissions(request: Request):
    _require_instructor(request)

    target_repo = os.environ["TARGET_REPO"]
    token = os.environ.get("GITHUB_INSTRUCTOR_TOKEN", "")
    if not token:
        raise HTTPException(
            status_code=500,
            detail="GITHUB_INSTRUCTOR_TOKEN not set in environment",
        )

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{GITHUB_API_URL}/repos/{target_repo}/pulls",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            params={"state": "open", "per_page": 100},
        )
    resp.raise_for_status()
    prs = resp.json()

    return [
        {
            "number": pr["number"],
            "title": pr["title"],
            "user": pr["user"]["login"],
            "updated_at": pr["updated_at"],
            "head_label": pr["head"]["label"],
            "pr_url": pr["html_url"],
        }
        for pr in prs
    ]


@router.post("/score")
async def score_all(request: Request):
    _require_instructor(request)

    # ── Load hidden test set ───────────────────────────────────────────────
    test_df_json = request.session.get("test_df")
    if not test_df_json:
        raise HTTPException(status_code=422, detail="No test set uploaded. Upload test_hidden.csv first.")

    from dependencies import get_schema
    schema = get_schema()
    id_col = schema["id_column"]
    target_col = schema["target_column"]
    positive_class = schema.get("target_positive_class", 1)

    test_df = pd.read_json(io.StringIO(test_df_json), orient="records")
    feature_cols = [c for c in test_df.columns if c not in (id_col, target_col)]
    X_test = test_df[feature_cols].copy()

    y_raw = test_df[target_col]
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        y_test = (y_raw == str(positive_class)).astype(int).values
    else:
        y_test = (y_raw == positive_class).astype(int).values

    # ── Fetch submissions ──────────────────────────────────────────────────
    tok = os.environ.get("GITHUB_INSTRUCTOR_TOKEN", "")
    if not tok:
        raise HTTPException(status_code=500, detail="GITHUB_INSTRUCTOR_TOKEN not set")

    target_repo = os.environ["TARGET_REPO"]
    submissions = (await list_submissions(request))

    leaderboard = []
    for sub in submissions:
        username = sub["user"]
        head_label = sub["head_label"]          # e.g. "username:challenge-submission"
        head_owner = head_label.split(":")[0]

        # Fetch model.pkl bytes from the PR branch via GitHub API
        model_bytes = await _fetch_file_bytes(
            tok,
            owner=head_owner,
            repo=target_repo.split("/")[1],
            branch=BRANCH_NAME,
            path=f"challenge/submissions/{username}/model.pkl",
        )

        if model_bytes is None:
            leaderboard.append({
                "username": username,
                "pr_url": sub["pr_url"],
                "updated_at": sub["updated_at"],
                "roc_auc": None,
                "error": "model.pkl not found in submission branch",
                "roc_curve": None,
                "confusion_matrix": None,
            })
            continue

        # Safe-load the model
        try:
            model = _safe_loads(model_bytes)
        except Exception as exc:
            leaderboard.append({
                "username": username,
                "pr_url": sub["pr_url"],
                "updated_at": sub["updated_at"],
                "roc_auc": None,
                "error": f"Failed to load model: {exc}",
                "roc_curve": None,
                "confusion_matrix": None,
            })
            continue

        # Inference
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception as exc:
            leaderboard.append({
                "username": username,
                "pr_url": sub["pr_url"],
                "updated_at": sub["updated_at"],
                "roc_auc": None,
                "error": f"Inference failed: {exc}",
                "roc_curve": None,
                "confusion_matrix": None,
            })
            continue

        metrics = evaluate(y_test, y_proba)
        leaderboard.append({
            "username": username,
            "pr_url": sub["pr_url"],
            "updated_at": sub["updated_at"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_curve": metrics["roc_curve"],
            "confusion_matrix": metrics["confusion_matrix"],
            "error": None,
        })

    # Sort: valid entries by ROC-AUC desc; errored entries at the bottom
    leaderboard.sort(key=lambda x: (x["roc_auc"] is not None, x["roc_auc"] or 0), reverse=True)
    return leaderboard


@router.get("/export")
async def export_leaderboard(request: Request):
    """Download a simple leaderboard CSV (no ROC curve data)."""
    _require_instructor(request)

    scored = await score_all(request)
    rows = [
        {
            "rank": i + 1,
            "username": s["username"],
            "roc_auc": s["roc_auc"],
            "precision": s.get("precision"),
            "recall": s.get("recall"),
            "f1": s.get("f1"),
            "updated_at": s["updated_at"],
            "pr_url": s["pr_url"],
            "error": s.get("error"),
        }
        for i, s in enumerate(scored)
    ]
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=leaderboard.csv"},
    )


# ── GitHub fetch helper ────────────────────────────────────────────────────

async def _fetch_file_bytes(token: str, owner: str, repo: str, branch: str, path: str) -> bytes | None:
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["content"].replace("\n", ""))
