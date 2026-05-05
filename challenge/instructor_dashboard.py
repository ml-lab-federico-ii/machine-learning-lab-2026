"""Dashboard interattiva per il docente — valutazione live dei gruppi durante i pitch.

Avvio:
    streamlit run challenge/instructor_dashboard.py

Workflow:
    1. Inserire la password docente nella sidebar.
    2. Tab "Submissions" → per ogni gruppo che ha fatto il pitch, cliccare "▶ Evaluta".
    3. Tab "Leaderboard" → riepilogo finale di tutti i gruppi valutati.

I .joblib consegnati dagli studenti vanno copiati in:
    challenge/submissions/model_seed_XX.joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

INSTRUCTOR_PASSWORD = "churn2026"  # ← modifica prima dell'uso se necessario

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets"
SUBMISSIONS_DIR = SCRIPT_DIR / "submissions"

TARGET_COLUMN = "Exited"

# ---------------------------------------------------------------------------
# Helpers di preprocessing + evaluation (logica identica a evaluate.py)
# ---------------------------------------------------------------------------


def _load_test_csv(seed: int) -> pd.DataFrame:
    path = DATASETS_DIR / f"seed_{seed:02d}_test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Test set non trovato: {path}\n"
            "Rigenera i dataset con: python challenge/generate_datasets.py"
        )
    return pd.read_csv(path)


def _apply_preprocessing(
    df_test: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    df = df_test.copy()
    y_test = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    if bundle.get("balance_is_zero"):
        X["balance_is_zero"] = (X["Balance"] == 0).astype(int)

    exclude = bundle.get("exclude_features", [])
    if exclude:
        X = X.drop(columns=[c for c in exclude if c in X.columns], errors="ignore")

    label_encoders: dict[str, LabelEncoder] = bundle.get("label_encoders", {})
    for col, le in label_encoders.items():
        if col in X.columns:
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])

    feature_order: list[str] = bundle.get("feature_order", list(X.columns))
    missing = set(feature_order) - set(X.columns)
    if missing:
        raise ValueError(f"Feature mancanti nel test set: {missing}")
    X = X[feature_order]

    scaler = bundle.get("scaler")
    if scaler is not None:
        X_arr = scaler.transform(X)
        X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)

    return X, y_test


def _evaluate_bundle(seed: int, path: Path) -> dict[str, Any]:
    """Carica il bundle, applica preprocessing, calcola metriche sul test set."""
    bundle: dict[str, Any] = joblib.load(path)

    df_test = _load_test_csv(seed)
    X_test, y_test = _apply_preprocessing(df_test, bundle)

    model = bundle["model"]
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc_test = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    train_metrics = bundle.get("train_metrics", {})
    auc_train = train_metrics.get("auc", float("nan"))
    delta = auc_test - auc_train

    return {
        "seed": seed,
        "componenti": bundle.get("componenti", []),
        "model_name": bundle.get("model_name", "?"),
        "auc_train": round(auc_train, 4),
        "auc_test": round(auc_test, 4),
        "delta_auc": round(delta, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
    }


def _scan_submissions() -> dict[int, Path]:
    """Restituisce {seed: path} per tutti i model_seed_XX.joblib trovati."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[int, Path] = {}
    for p in sorted(SUBMISSIONS_DIR.glob("model_seed_*.joblib")):
        try:
            seed = int(p.stem.split("_")[-1])
            found[seed] = p
        except ValueError:
            continue
    return found


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _group_label(bundle: dict[str, Any]) -> str:
    componenti = bundle.get("componenti", [])
    if not componenti:
        return "—"
    return ", ".join(c.split()[-1] for c in componenti if c.strip())


def _render_metrics(r: dict[str, Any]) -> None:
    """Mostra le metriche di un gruppo con card e grafico."""
    delta = r["delta_auc"]
    flag = "⚠️ possibile overfitting" if delta < -0.05 else "✅ generalizza bene"

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC test", f"{r['auc_test']:.4f}")
    col2.metric("AUC train", f"{r['auc_train']:.4f}", delta=f"Δ {delta:+.4f}")
    col3.metric("F1", f"{r['f1']:.4f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{r['precision']:.4f}")
    c2.metric("Recall", f"{r['recall']:.4f}")
    c3.metric("Accuracy", f"{r['accuracy']:.4f}")

    st.caption(flag)

    chart_df = pd.DataFrame(
        {"AUC": [r["auc_train"], r["auc_test"]]},
        index=["Train", "Test"],
    )
    st.bar_chart(chart_df, height=200)


# ---------------------------------------------------------------------------
# App Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Challenge — Dashboard Docente",
    page_icon="🎓",
    layout="wide",
)

# ── Password gate ────────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

with st.sidebar:
    st.title("🔐 Accesso Docente")
    pwd = st.text_input("Password", type="password", key="pwd_input")
    if st.button("Accedi"):
        if pwd == INSTRUCTOR_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Accesso consentito")
        else:
            st.error("Password errata")

if not st.session_state["authenticated"]:
    st.title("🎓 ML Challenge — Dashboard Docente")
    st.info("Inserire la password nella sidebar per accedere.")
    st.stop()

# ── Session state per i risultati accumulati ─────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"]: dict[int, dict[str, Any]] = {}

# ── Titolo principale ─────────────────────────────────────────────────────────

st.title("🎓 ML Challenge — Dashboard Docente")

tab_submissions, tab_leaderboard = st.tabs(["📋 Submissions", "🏆 Leaderboard"])

# ── Tab 1: Submissions ────────────────────────────────────────────────────────

with tab_submissions:
    st.subheader("Submissions ricevute")
    st.caption(
        f"Cartella monitorata: `{SUBMISSIONS_DIR}`  —  "
        "aggiorna la pagina per rilevare nuovi file aggiunti."
    )

    submissions = _scan_submissions()

    if not submissions:
        st.warning(
            "Nessun file trovato in `challenge/submissions/`.\n\n"
            "Copia i `.joblib` ricevuti dagli studenti in quella cartella."
        )
    else:
        for seed, path in submissions.items():
            # Leggi il nome gruppo in anteprima (senza evaluation completa)
            try:
                preview: dict[str, Any] = joblib.load(path)
                group = _group_label(preview)
                model_name = preview.get("model_name", "?")
            except Exception:
                group = "errore lettura bundle"
                model_name = "?"

            already_evaluated = seed in st.session_state["results"]
            status_icon = "✅" if already_evaluated else "⏳"

            with st.expander(
                f"{status_icon} **Seed {seed:02d}** — {group} — `{model_name}`",
                expanded=not already_evaluated,
            ):
                col_btn, col_info = st.columns([1, 3])

                with col_btn:
                    eval_btn = st.button(
                        f"▶ Evaluta Seed {seed:02d}",
                        key=f"eval_{seed}",
                        type="primary" if not already_evaluated else "secondary",
                    )

                with col_info:
                    st.caption(f"File: `{path.name}`")
                    if already_evaluated:
                        st.caption("Già valutato — premi il pulsante per rivalutare.")

                if eval_btn:
                    with st.spinner(f"Valutazione Seed {seed:02d} in corso..."):
                        try:
                            result = _evaluate_bundle(seed, path)
                            st.session_state["results"][seed] = result
                            st.success(
                                f"AUC test: **{result['auc_test']:.4f}**  |  "
                                f"Δ: **{result['delta_auc']:+.4f}**"
                            )
                        except FileNotFoundError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Errore durante la valutazione: {e}")

                if seed in st.session_state["results"]:
                    st.divider()
                    _render_metrics(st.session_state["results"][seed])

# ── Tab 2: Leaderboard ────────────────────────────────────────────────────────

with tab_leaderboard:
    st.subheader("Leaderboard finale")

    results_so_far = st.session_state["results"]

    col_rivaluta, _ = st.columns([2, 8])
    with col_rivaluta:
        if st.button("🔄 Rivaluta tutti", disabled=len(results_so_far) == 0):
            submissions = _scan_submissions()
            progress = st.progress(0, text="Rivalutazione in corso...")
            errors: list[str] = []
            seeds = list(results_so_far.keys())
            for i, seed in enumerate(seeds):
                if seed in submissions:
                    try:
                        st.session_state["results"][seed] = _evaluate_bundle(
                            seed, submissions[seed]
                        )
                    except Exception as e:
                        errors.append(f"Seed {seed:02d}: {e}")
                progress.progress((i + 1) / len(seeds), text=f"Seed {seed:02d} fatto")
            progress.empty()
            if errors:
                st.warning("\n".join(errors))
            else:
                st.success("Rivalutazione completata.")
            results_so_far = st.session_state["results"]

    if not results_so_far:
        st.info("Nessun gruppo ancora valutato. Usa la tab **Submissions** per valutare i gruppi durante i pitch.")
    else:
        rows = sorted(results_so_far.values(), key=lambda r: r["auc_test"], reverse=True)
        df_lb = pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "Seed": f"{r['seed']:02d}",
                    "Gruppo": ", ".join(
                        c.split()[-1] for c in r["componenti"] if c.strip()
                    ) or "—",
                    "Modello": r["model_name"],
                    "AUC train": r["auc_train"],
                    "AUC test": r["auc_test"],
                    "Δ": r["delta_auc"],
                    "F1": r["f1"],
                    "Accuracy": r["accuracy"],
                    "⚠️": "⚠️" if r["delta_auc"] < -0.05 else "",
                }
                for rank, r in enumerate(rows, 1)
            ]
        )

        def _highlight_overfitting(row: pd.Series) -> list[str]:
            if row["⚠️"] == "⚠️":
                return ["background-color: #fff3cd"] * len(row)
            if row["Rank"] == 1:
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        styled = df_lb.style.apply(_highlight_overfitting, axis=1).format(
            {
                "AUC train": "{:.4f}",
                "AUC test": "{:.4f}",
                "Δ": "{:+.4f}",
                "F1": "{:.4f}",
                "Accuracy": "{:.4f}",
            }
        )

        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            "🟩 1° posto  |  🟨 possibile overfitting (Δ < −0.05)  |  "
            "Δ = AUC_test − AUC_train"
        )
