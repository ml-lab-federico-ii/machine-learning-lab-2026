"""
pipeline_builder.py
--------------------
Translates a pipeline_config dict (from the frontend) into a fitted
sklearn.Pipeline that can be serialised to model.pkl.

The pipeline handles ALL preprocessing internally so that model.pkl
can accept raw feature columns (everything except id_column and
target_column) and return predict_proba probabilities directly.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42

# ── Hard caps for laptop-safe training ─────────────────────────────────────
MAX_N_ESTIMATORS = 300
MAX_DEPTH = 6


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def build_pipeline(config: dict[str, Any], X: pd.DataFrame) -> Pipeline | ImbPipeline:
    """
    Build (but do not fit) a full preprocessing + model pipeline from config.

    Parameters
    ----------
    config : dict
        The pipeline_config JSON sent from the frontend.
    X : pd.DataFrame
        Training feature matrix (used only to infer column types).

    Returns
    -------
    sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
    """
    preprocessing = config.get("preprocessing", {})
    model_cfg = config.get("model", {})

    # ── Separate numeric and categorical columns ────────────────────────────
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    # Feature selection: optionally drop columns the student deselected
    selected_features = config.get("selected_features")
    if selected_features:
        numeric_cols = [c for c in numeric_cols if c in selected_features]
        categorical_cols = [c for c in categorical_cols if c in selected_features]

    # ── Imputation ──────────────────────────────────────────────────────────
    num_impute_strategy = preprocessing.get("imputation_numeric", "mean")
    cat_impute_strategy = preprocessing.get("imputation_categorical", "most_frequent")

    valid_num_impute = {"mean", "median", "most_frequent", "constant"}
    valid_cat_impute = {"most_frequent", "constant"}
    if num_impute_strategy not in valid_num_impute:
        num_impute_strategy = "mean"
    if cat_impute_strategy not in valid_cat_impute:
        cat_impute_strategy = "most_frequent"

    # ── Scaling ─────────────────────────────────────────────────────────────
    scaling = preprocessing.get("scaling", "standard")
    if scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "none":
        scaler = "passthrough"
    else:
        scaler = StandardScaler()

    # ── Encoding ────────────────────────────────────────────────────────────
    encoding = preprocessing.get("encoding", "onehot")
    if encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif encoding == "label":
        # Label encoding per column via OrdinalEncoder (sklearn compatible)
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # ── Numeric transformer ─────────────────────────────────────────────────
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=num_impute_strategy)),
            ("scaler", scaler),
        ]
    )

    # ── Categorical transformer ─────────────────────────────────────────────
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cat_impute_strategy, fill_value="missing")),
            ("encoder", encoder),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError(
            "No features selected. Select at least one feature column in the EDA step."
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # ── Model ───────────────────────────────────────────────────────────────
    model = _build_model(model_cfg)

    # ── Imbalance handling ──────────────────────────────────────────────────
    imbalance = preprocessing.get("imbalance_handling", "none")
    class_weight = None

    if imbalance == "class_weight":
        class_weight = "balanced"
        model = _build_model(model_cfg, class_weight=class_weight)

    if imbalance == "smote":
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=SEED)),
                ("model", model),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    return pipeline


def _build_model(
    model_cfg: dict[str, Any],
    class_weight: str | None = None,
) -> Any:
    model_name = model_cfg.get("name", "logistic_regression")
    params = model_cfg.get("params", {})

    # Clamp tree hyperparams to laptop-safe limits
    n_estimators = _clamp(params.get("n_estimators", 100), 10, MAX_N_ESTIMATORS)
    max_depth = _clamp(params.get("max_depth", 4), 1, MAX_DEPTH)
    learning_rate = float(params.get("learning_rate", 0.1))
    c_param = float(params.get("C", 1.0))

    if model_name == "logistic_regression":
        return LogisticRegression(
            C=c_param,
            max_iter=1000,
            random_state=SEED,
            class_weight=class_weight,
            n_jobs=-1,
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=SEED,
            class_weight=class_weight,
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=SEED,
        )
    elif model_name == "xgboost":
        scale_pos_weight = params.get("scale_pos_weight", 1.0)
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",   # CPU histogram method — no GPU needed
            device="cpu",
            random_state=SEED,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
        )
    elif model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=SEED,
            class_weight=class_weight,
            device="cpu",
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
