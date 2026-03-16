"""
dependencies.py
---------------
Shared application state — loaded once at startup.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def get_schema() -> dict:
    schema_path = DATA_DIR / "schema.json"
    if not schema_path.exists():
        raise RuntimeError("schema.json not found in backend/data/")
    with open(schema_path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_train_df() -> pd.DataFrame:
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise RuntimeError(
            "train.csv not found in backend/data/. "
            "Place the dataset file there before building the Docker image."
        )
    return pd.read_csv(train_path)
