"""
main.py — FastAPI application entry point.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from routers import eda, train, submit, instructor  # noqa: E402

app = FastAPI(
    title="ML Challenge App",
    description="Interactive ML challenge platform for the Federico II lab.",
    version="1.0.0",
)

# ── Session middleware ──────────────────────────────────────────────────────
session_secret = os.environ.get("SESSION_SECRET", "dev-secret-change-in-production")
app.add_middleware(SessionMiddleware, secret_key=session_secret, max_age=60 * 60 * 8)

# ── CORS ────────────────────────────────────────────────────────────────────
frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(eda.router)
app.include_router(train.router)
app.include_router(submit.router)
app.include_router(instructor.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
