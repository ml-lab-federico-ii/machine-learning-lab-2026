"""
eda.py — Data exploration endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from dependencies import get_schema, get_train_df

router = APIRouter(prefix="/api/data", tags=["eda"])


@router.get("/schema")
def schema():
    return get_schema()


@router.get("/preview")
def preview():
    df = get_train_df()
    schema = get_schema()
    target_col = schema["target_column"]

    class_counts = df[target_col].value_counts().to_dict()
    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            }
            for col in df.columns
        ],
        "class_balance": {str(k): int(v) for k, v in class_counts.items()},
        "head": df.head(5).to_dict(orient="records"),
    }


@router.get("/eda/{column}")
def column_distribution(column: str):
    df = get_train_df()
    if column not in df.columns:
        return JSONResponse(status_code=404, content={"detail": f"Column '{column}' not found"})

    col = df[column]
    if col.dtype in ("object", "category") or col.nunique() <= 20:
        counts = col.value_counts(dropna=False)
        return {
            "type": "categorical",
            "column": column,
            "data": [
                {"label": str(k), "count": int(v)}
                for k, v in counts.items()
            ],
        }
    else:
        hist_values, bin_edges = __import__("numpy").histogram(col.dropna(), bins=30)
        return {
            "type": "numeric",
            "column": column,
            "data": [
                {
                    "bin_start": round(float(bin_edges[i]), 4),
                    "bin_end": round(float(bin_edges[i + 1]), 4),
                    "count": int(hist_values[i]),
                }
                for i in range(len(hist_values))
            ],
        }


@router.get("/correlations")
def correlations():
    df = get_train_df()
    schema = get_schema()
    numeric_df = df.drop(
        columns=[schema["id_column"], schema["target_column"]], errors="ignore"
    ).select_dtypes(include="number")

    corr = numeric_df.corr()
    columns = corr.columns.tolist()
    return {
        "columns": columns,
        "matrix": [
            [round(float(v), 4) if not __import__("math").isnan(v) else None for v in row]
            for row in corr.values
        ],
    }
