"""Dashboard interattiva per il docente — valutazione live dei gruppi durante i pitch.

Avvio:
    streamlit run challenge/instructor_dashboard.py

Workflow:
    1. Inserire la password docente nella sidebar.
    2. Tab "Submissions" → per ogni gruppo che ha fatto il pitch, cliccare "▶ Evaluta".
    3. Tab "Leaderboard" → riepilogo finale di tutti i gruppi valutati.

Gli ZIP consegnati dagli studenti vanno copiati in:
    challenge/submissions/delivery_seed_XX.zip

Sono supportati anche i .joblib sciolti (compatibilità backward):
    challenge/submissions/model_seed_XX.joblib
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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

# Cheatsheet docente — baseline AUC per seed
_CHEATSHEET_PATH = SCRIPT_DIR / "instructor_cheatsheet.json"
if _CHEATSHEET_PATH.exists():
    _CHEATSHEET: list[dict] = json.loads(_CHEATSHEET_PATH.read_text(encoding="utf-8"))
    _BEST_AUC: dict[int, float] = {e["seed"]: e["best_auc"] for e in _CHEATSHEET}
else:
    _CHEATSHEET = []
    _BEST_AUC = {}

# Colori semantici (compatibili dark/light Streamlit)
COLOR_GOLD = "#f0a500"
COLOR_GREEN = "#2ecc71"
COLOR_RED = "#e74c3c"
COLOR_BLUE = "#3498db"
COLOR_GRAY = "#95a5a6"

# ---------------------------------------------------------------------------
# CSS personalizzato
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Titolo dashboard */
.dash-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

/* Card gruppo (submission) */
.group-card {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    background: rgba(255,255,255,0.03);
    transition: border-color 0.2s;
}
.group-card:hover {
    border-color: rgba(255,255,255,0.25);
}
.group-card.evaluated {
    border-left: 4px solid #2ecc71;
}
.group-card.pending {
    border-left: 4px solid #f0a500;
}

/* Header della card */
.card-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.8rem;
}
.card-seed {
    background: #3498db;
    color: white;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.2rem 0.7rem;
    border-radius: 6px;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.card-group {
    font-size: 1.05rem;
    font-weight: 600;
}
.card-model {
    font-size: 0.85rem;
    opacity: 0.6;
    font-family: monospace;
}
.status-badge {
    margin-left: auto;
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 600;
}
.status-evaluated {
    background: rgba(46,204,113,0.15);
    color: #2ecc71;
    border: 1px solid rgba(46,204,113,0.4);
}
.status-pending {
    background: rgba(240,165,0,0.15);
    color: #f0a500;
    border: 1px solid rgba(240,165,0,0.4);
}

/* Big metric display */
.big-metric {
    text-align: center;
    padding: 1rem 0.5rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.big-metric-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
}
.big-metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.55;
    margin-top: 0.3rem;
}
.big-metric-delta {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.3rem;
}
.delta-ok { color: #2ecc71; }
.delta-warn { color: #e74c3c; }

/* Divider sottile */
.thin-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1rem 0;
}

/* Leaderboard rank */
.rank-1 { color: #f0a500; font-weight: 800; font-size: 1.2rem; }
.rank-2 { color: #95a5a6; font-weight: 700; }
.rank-3 { color: #cd7f32; font-weight: 700; }

/* Overfitting badge */
.badge-overfitting {
    background: rgba(231,76,60,0.15);
    color: #e74c3c;
    border: 1px solid rgba(231,76,60,0.4);
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-ok {
    background: rgba(46,204,113,0.12);
    color: #2ecc71;
    border: 1px solid rgba(46,204,113,0.3);
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Info sidebar */
.sidebar-info {
    font-size: 0.8rem;
    opacity: 0.6;
    margin-top: 2rem;
    line-height: 1.6;
}

/* Contatore submissions */
.submissions-counter {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(52,152,219,0.12);
    border: 1px solid rgba(52,152,219,0.3);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
</style>
"""

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


def _load_bundle(path: Path) -> dict[str, Any]:
    """Carica il bundle joblib da un file .joblib o da un .zip che lo contiene."""
    if path.suffix == ".zip":
        import zipfile, io
        with zipfile.ZipFile(path, "r") as zf:
            joblib_names = [n for n in zf.namelist() if n.endswith(".joblib")]
            if not joblib_names:
                raise ValueError(f"Nessun .joblib trovato in {path.name}")
            with zf.open(joblib_names[0]) as f:
                return joblib.load(io.BytesIO(f.read()))
    else:
        return joblib.load(path)


def _evaluate_bundle(seed: int, path: Path) -> dict[str, Any]:
    """Carica il bundle, applica preprocessing, calcola metriche sul test set."""
    bundle: dict[str, Any] = _load_bundle(path)

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
    """Restituisce {seed: path} per tutti i delivery_seed_XX.zip (o model_seed_XX.joblib) trovati."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[int, Path] = {}
    # ZIP prioritario
    for p in sorted(SUBMISSIONS_DIR.glob("delivery_seed_*.zip")):
        try:
            seed = int(p.stem.split("_")[-1])
            found[seed] = p
        except ValueError:
            continue
    # Fallback: joblib sciolti (compatibilità backward)
    for p in sorted(SUBMISSIONS_DIR.glob("model_seed_*.joblib")):
        try:
            seed = int(p.stem.split("_")[-1])
            if seed not in found:  # non sovrascrivere lo ZIP
                found[seed] = p
        except ValueError:
            continue
    return found


def _extract_charts_from_zip(zip_path: Path) -> dict[str, bytes]:
    """Estrae i PNG da charts/ dentro lo ZIP. Restituisce {nome_file: bytes}."""
    charts: dict[str, bytes] = {}
    if zip_path is None or zip_path.suffix != ".zip":
        return charts
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("charts/") and name.endswith(".png"):
                    data = zf.read(name)
                    if len(data) > 0:  # salta PNG vuoti
                        charts[name] = data
    except Exception:
        pass
    return charts


def _load_manifest_from_zip(zip_path: Path) -> dict[str, Any]:
    """Legge manifest.json dallo ZIP. Restituisce dict vuoto se non trovato."""
    if zip_path is None or zip_path.suffix != ".zip":
        return {}
    try:
        import zipfile, json
        with zipfile.ZipFile(zip_path, "r") as zf:
            if "manifest.json" in zf.namelist():
                return json.loads(zf.read("manifest.json").decode("utf-8"))
    except Exception:
        pass
    return {}


def _show_chart(img_bytes: bytes, caption: str = "") -> None:
    """Mostra un PNG con st.image, gestendo errori silenziosamente."""
    if not img_bytes:
        return
    try:
        import io
        st.image(io.BytesIO(img_bytes), caption=caption, use_container_width=True)
    except Exception:
        st.caption(f"⚠️ Impossibile visualizzare: {caption}")


def _render_pitch_view(seed: int, path: Path) -> None:
    """Mostra il racconto lineare del pitch (EDA → Preprocessing → Modelli → Finale → SHAP → Note → Domande)."""
    _zip = path if path.suffix == ".zip" else None
    manifest = _load_manifest_from_zip(_zip) if _zip else {}
    charts = _extract_charts_from_zip(_zip) if _zip else {}

    eda_imgs   = {k: v for k, v in sorted(charts.items()) if "/eda_" in k}
    model_imgs = {k: v for k, v in sorted(charts.items()) if "/model_" in k}
    shap_imgs  = {k: v for k, v in sorted(charts.items()) if "/shap_" in k}

    def _chart_cap(name: str, prefix: str) -> str:
        return (name.split("/")[-1]
                .replace(".png", "").replace(prefix, "")
                .replace("_", " ").title())

    # ── 1. EDA ──────────────────────────────────────────────────────────────
    with st.expander("📊 1 — Analisi Esplorativa (EDA)", expanded=True):
        dataset_info = manifest.get("dataset_info", {})
        if dataset_info:
            cols = st.columns(4)
            cols[0].metric("Righe", f"{dataset_info.get('n_rows', '—'):,}")
            cols[1].metric("Feature", dataset_info.get('n_features', '—'))
            cols[2].metric("Churn rate", f"{dataset_info.get('churn_rate', 0):.1%}")
            cols[3].metric("SEED", seed)
        eda_notes = manifest.get("eda_notes", "")
        if eda_notes and eda_notes.strip():
            st.info(eda_notes)
        for name, img in eda_imgs.items():
            _show_chart(img, _chart_cap(name, "eda_"))
        if not eda_imgs:
            st.caption("Nessun grafico EDA nel pacchetto — re-eseguire il notebook con la versione aggiornata.")

    # ── 2. Preprocessing ────────────────────────────────────────────────────
    with st.expander("⚙️ 2 — Preprocessing", expanded=False):
        prep = manifest.get("preprocessing_config", {})
        if not prep:
            # fallback da bundle
            try:
                _b = _load_bundle(path)
                prep = {
                    "exclude_features": _b.get("exclude_features", []),
                    "balance_is_zero": _b.get("balance_is_zero", True),
                }
            except Exception:
                prep = {}
        if prep:
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Scaling**")
                st.code(prep.get("scaling", "standard"))
                st.markdown("**Encoding categoriche**")
                st.code(prep.get("encoding", "label_encoder"))
                st.markdown("**Feature escluse**")
                excl = prep.get("exclude_features", [])
                st.code(", ".join(excl) if excl else "nessuna")
            with cols[1]:
                st.markdown("**Bilanciamento classi**")
                strat = prep.get("imbalance_strategy", "class_weight")
                st.code(strat)
                st.markdown("**Balance = zero**")
                st.code(str(prep.get("balance_is_zero", True)))
                st.markdown("**Train/Val split**")
                st.code(f"{(1 - prep.get('test_size', 0.2)):.0%} / {prep.get('test_size', 0.2):.0%}")
        else:
            st.caption("Configurazione preprocessing non disponibile (pacchetto vecchio).")

    # ── 3. Tutti i modelli testati ───────────────────────────────────────────
    with st.expander("🔬 3 — Modelli Testati", expanded=False):
        runs = manifest.get("model_results_summary", [])
        final_idx = manifest.get("final_model_idx", 0)
        if runs:
            rows_html = ""
            for i, r in enumerate(runs):
                sel = "✅" if i == final_idx else ""
                bold = "font-weight:800;" if i == final_idx else ""
                rows_html += (
                    f"<tr style='{bold}background:{'rgba(52,152,219,0.08)' if i == final_idx else 'transparent'}'>"
                    f"<td style='text-align:center'>{sel} Run #{i+1}</td>"
                    f"<td>{r.get('model_name','?')}</td>"
                    f"<td style='text-align:right'>{r.get('auc',0):.4f}</td>"
                    f"<td style='text-align:right'>{r.get('f1',0):.4f}</td>"
                    f"<td style='text-align:right'>{r.get('precision',0):.4f}</td>"
                    f"<td style='text-align:right'>{r.get('recall',0):.4f}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"""<table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>
                <thead><tr style='opacity:0.5;font-size:0.75rem;text-transform:uppercase;border-bottom:1px solid #ccc;'>
                <th>Run</th><th>Modello</th><th style='text-align:right'>AUC val</th>
                <th style='text-align:right'>F1</th><th style='text-align:right'>Prec</th><th style='text-align:right'>Rec</th>
                </tr></thead><tbody>{rows_html}</tbody></table>""",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Riepilogo run non disponibile (pacchetto vecchio).")

    # ── 4. Modello finale ────────────────────────────────────────────────────
    with st.expander("🎯 4 — Modello Finale", expanded=True):
        for name, img in model_imgs.items():
            _show_chart(img, _chart_cap(name, "model_"))
        if not model_imgs:
            st.caption("Nessun grafico modello nel pacchetto.")

    # ── 5. SHAP ──────────────────────────────────────────────────────────────
    if shap_imgs:
        with st.expander("🔍 5 — Interpretabilità SHAP", expanded=False):
            for name, img in shap_imgs.items():
                _show_chart(img, _chart_cap(name, "shap_"))

    # ── 6. Interpretazione del gruppo ────────────────────────────────────────
    with st.expander("💬 6 — Interpretazione del Gruppo", expanded=False):
        notes = manifest.get("team_notes", {})
        if notes:
            labels = {
                "motivazione": "Perché questo modello?",
                "business_insight": "Insight di business",
                "strategia_retention": "Strategia di retention",
            }
            for key, label in labels.items():
                val = notes.get(key, "").strip()
                if val and "Sostituite questo testo" not in val:
                    st.markdown(f"**{label}**")
                    st.markdown(val)
                    st.divider()
        else:
            st.caption("Note del gruppo non disponibili.")

    # ── 7. Domande del docente (dal cheatsheet) ──────────────────────────────
    cheatsheet_entry = next(
        (e for e in _CHEATSHEET if e.get("seed") == seed), None
    )
    if cheatsheet_entry:
        with st.expander("❓ 7 — Domande per il Pitch", expanded=True):
            st.markdown(
                f"**Profilo dataset:** {cheatsheet_entry.get('profile_name','')}")
            st.caption(cheatsheet_entry.get("profile_desc", ""))
            st.markdown("---")
            st.markdown("**Domanda business:**")
            st.info(cheatsheet_entry.get("question_business", ""))
            st.markdown("**Domanda tecnica:**")
            st.info(cheatsheet_entry.get("question_technical", ""))
            # Baselines di riferimento
            baselines = cheatsheet_entry.get("baselines", {})
            if baselines:
                st.markdown("**Baselines istruttore:**")
                b_cols = st.columns(len(baselines))
                for i, (mname, auc_b) in enumerate(baselines.items()):
                    b_cols[i].metric(mname, f"{auc_b:.4f}")
    else:
        with st.expander("❓ 7 — Domande per il Pitch", expanded=True):
            st.caption("Scheda non trovata nel cheatsheet per questo SEED.")


def _find_html_report(seed: int, zip_path: Path | None = None) -> str | None:
    """Legge il contenuto HTML del report (dal ZIP o da file sciolto).

    Restituisce il contenuto HTML come stringa, o None se non trovato.
    """
    # Prima cerca dentro lo ZIP se disponibile
    if zip_path is not None and zip_path.suffix == ".zip":
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                html_names = [n for n in zf.namelist() if n.endswith(".html")]
                if html_names:
                    with zf.open(html_names[0]) as f:
                        return f.read().decode("utf-8", errors="replace")
        except Exception:
            pass
    # Fallback: file HTML sciolto
    candidates = [
        SUBMISSIONS_DIR / f"team_seed_{seed:02d}_report.html",
        SUBMISSIONS_DIR / f"report_seed_{seed:02d}.html",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def _delta_auc(auc_test: float, seed: int) -> float:
    """Delta AUC vs baseline istruttore: AUC_studente − AUC_baseline.

    Permette confronto equo tra seed di diversa difficoltà.
    Δ > 0 = studente batte il baseline; Δ < 0 = sotto baseline.
    """
    best = _BEST_AUC.get(seed)
    if best is None:
        return float("nan")
    return auc_test - best


def _group_label(bundle: dict[str, Any]) -> str:
    componenti = bundle.get("componenti", [])
    if not componenti:
        return "—"
    return ", ".join(c.split()[-1] for c in componenti if c.strip())


def _render_metrics(r: dict[str, Any]) -> None:
    """Mostra le metriche di un gruppo con card HTML e grafico."""
    delta = r["delta_auc"]
    is_overfit = delta < -0.05
    delta_class = "delta-warn" if is_overfit else "delta-ok"
    delta_symbol = "⚠️" if is_overfit else "✅"
    delta_label = "possibile overfitting" if is_overfit else "generalizza bene"

    # Riga principale: AUC test (prominente) + AUC train + Δ
    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; margin: 0.8rem 0;">
            <div class="big-metric">
                <div class="big-metric-value" style="color: {COLOR_BLUE};">{r['auc_test']:.4f}</div>
                <div class="big-metric-label">AUC — test set</div>
            </div>
            <div class="big-metric">
                <div class="big-metric-value">{r['auc_train']:.4f}</div>
                <div class="big-metric-label">AUC — train</div>
                <div class="big-metric-delta {delta_class}">Δ {delta:+.4f} &nbsp; {delta_symbol} {delta_label}</div>
            </div>
            <div class="big-metric">
                <div class="big-metric-value">{r['f1']:.4f}</div>
                <div class="big-metric-label">F1-Score</div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; margin-bottom: 1rem;">
            <div class="big-metric">
                <div class="big-metric-value" style="font-size:1.4rem;">{r['precision']:.4f}</div>
                <div class="big-metric-label">Precision</div>
            </div>
            <div class="big-metric">
                <div class="big-metric-value" style="font-size:1.4rem;">{r['recall']:.4f}</div>
                <div class="big-metric-label">Recall</div>
            </div>
            <div class="big-metric">
                <div class="big-metric-value" style="font-size:1.4rem;">{r['accuracy']:.4f}</div>
                <div class="big-metric-label">Accuracy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mini bar chart AUC train vs test
    chart_df = pd.DataFrame(
        {"AUC": [r["auc_train"], r["auc_test"]]},
        index=["Train", "Test"],
    )
    st.bar_chart(chart_df, height=160, color=COLOR_BLUE)


# ---------------------------------------------------------------------------
# App Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Challenge — Dashboard Docente",
    page_icon="🎓",
    layout="wide",
)

# Inietta CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Password gate ────────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

with st.sidebar:
    st.markdown("## 🔐 Accesso Docente")
    pwd = st.text_input("Password", type="password", key="pwd_input", label_visibility="collapsed", placeholder="Password docente")
    if st.button("Accedi", use_container_width=True, type="primary"):
        if pwd == INSTRUCTOR_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Password errata")

    if st.session_state["authenticated"]:
        n_sub = len(_scan_submissions())
        n_eval = len(st.session_state.get("results", {}))
        st.success("✅ Accesso consentito")
        st.markdown(
            f"""
            <div class="sidebar-info">
            📁 Submissions trovate: <strong>{n_sub}</strong><br>
            ✅ Gruppi valutati: <strong>{n_eval}</strong><br>
            ⏳ In attesa: <strong>{n_sub - n_eval}</strong><br><br>
            📂 <code>challenge/submissions/</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("🔄 Aggiorna submissions", use_container_width=True,
                     help="Rilegge i file nella cartella submissions/ senza resettare sessione e risultati"):
            st.rerun()

if not st.session_state["authenticated"]:
    st.markdown('<div class="dash-title">🎓 ML Challenge — Dashboard Docente</div>', unsafe_allow_html=True)
    st.info("Inserire la password nella sidebar per accedere.")
    st.stop()

# ── Session state per i risultati accumulati ─────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"] = {}

# ── Titolo principale ─────────────────────────────────────────────────────────

submissions_all = _scan_submissions()
n_eval = len(st.session_state["results"])
n_total = len(submissions_all)

st.markdown(
    f'<div class="dash-title">🎓 ML Challenge &nbsp; <span style="opacity:0.4;font-weight:400;font-size:1.2rem;">Dashboard Docente</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="submissions-counter">
        📬 <strong>{n_total}</strong> submission ricevute &nbsp;·&nbsp;
        ✅ <strong>{n_eval}</strong> valutate &nbsp;·&nbsp;
        ⏳ <strong>{n_total - n_eval}</strong> in attesa
    </div>
    """,
    unsafe_allow_html=True,
)

tab_submissions, tab_leaderboard = st.tabs(["📋  Submissions", "🏆  Leaderboard"])

# ── Tab 1: Submissions ────────────────────────────────────────────────────────

with tab_submissions:
    if not submissions_all:
        st.warning(
            "Nessun file trovato in `challenge/submissions/`.\n\n"
            "Copia i file `delivery_seed_XX.zip` ricevuti dagli studenti in quella cartella, "
            "poi aggiorna la pagina."
        )
    else:
        for seed, path in submissions_all.items():
            # Leggi preview bundle (nome gruppo, modello)
            try:
                preview: dict[str, Any] = _load_bundle(path)
                group = _group_label(preview)
                model_name = preview.get("model_name", "?")
            except Exception:
                group = "errore lettura bundle"
                model_name = "?"

            already_evaluated = seed in st.session_state["results"]
            card_class = "evaluated" if already_evaluated else "pending"
            status_html = (
                '<span class="status-badge status-evaluated">✅ Valutato</span>'
                if already_evaluated
                else '<span class="status-badge status-pending">⏳ In attesa</span>'
            )

            # Card header
            st.markdown(
                f"""
                <div class="group-card {card_class}">
                    <div class="card-header">
                        <span class="card-seed">SEED {seed:02d}</span>
                        <span class="card-group">{group}</span>
                        <span class="card-model">{model_name}</span>
                        {status_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Racconto lineare del pitch ───────────────────────────────────
            _render_pitch_view(seed, path)

            # ── 8. Valutazione docente ───────────────────────────────────────
            with st.expander("📈 8 — Valutazione sul Test Set", expanded=already_evaluated):
                col_btn, col_hint = st.columns([2, 5])
                with col_btn:
                    eval_btn = st.button(
                        f"▶ Evaluta Seed {seed:02d}",
                        key=f"eval_{seed}",
                        type="primary" if not already_evaluated else "secondary",
                        use_container_width=True,
                    )
                with col_hint:
                    if already_evaluated:
                        st.caption("Già valutato — clicca di nuovo per rivalutare.")
                    else:
                        st.caption(f"File: `{path.name}`")

                if eval_btn:
                    with st.spinner(f"Valutazione Seed {seed:02d}…"):
                        try:
                            result = _evaluate_bundle(seed, path)
                            st.session_state["results"][seed] = result
                            auc_te = result["auc_test"]
                            delta = result["delta_auc"]
                            flag = "⚠️ possibile overfitting" if delta < -0.05 else "✅ ok"
                            st.success(
                                f"**AUC test: {auc_te:.4f}** &nbsp;|&nbsp; "
                                f"Δ: {delta:+.4f} &nbsp; {flag}"
                            )
                        except FileNotFoundError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Errore durante la valutazione: {e}")

                if seed in st.session_state["results"]:
                    _render_metrics(st.session_state["results"][seed])
                    r = st.session_state["results"][seed]
                    delta = _delta_auc(r["auc_test"], seed)
                    best_auc = _BEST_AUC.get(seed)
                    if best_auc is not None and not (isinstance(delta, float) and np.isnan(delta)):
                        delta_str = f"{delta:+.4f}"
                        if delta > 0:
                            box_color = "#2ecc71"
                            box_bg = "rgba(46,204,113,0.1)"
                            verdict = "✅ sopra baseline"
                        elif delta < 0:
                            box_color = "#e74c3c"
                            box_bg = "rgba(231,76,60,0.1)"
                            verdict = "❌ sotto baseline"
                        else:
                            box_color = "#95a5a6"
                            box_bg = "rgba(149,165,166,0.1)"
                            verdict = "➖ pari al baseline"
                        st.markdown(
                            f"""
                            <div style='margin-top:0.5rem;padding:0.6rem 1rem;
                                background:{box_bg};border-radius:8px;
                                border-left:3px solid {box_color};'>
                            <strong>Δ AUC vs Baseline: <span style='color:{box_color}'>{delta_str}</span></strong>
                            &nbsp;·&nbsp; {verdict}
                            &nbsp;·&nbsp; Baseline istruttore: <code>{best_auc:.4f}</code>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# ── Tab 2: Leaderboard ────────────────────────────────────────────────────────

with tab_leaderboard:
    results_so_far = st.session_state["results"]

    col_title, col_btn_reval = st.columns([5, 2])
    with col_title:
        st.markdown("### 🏆 Leaderboard finale")
    with col_btn_reval:
        if st.button(
            "🔄 Rivaluta tutti",
            disabled=len(results_so_far) == 0,
            use_container_width=True,
        ):
            progress = st.progress(0, text="Rivalutazione in corso…")
            errors: list[str] = []
            seeds = list(results_so_far.keys())
            for i, seed in enumerate(seeds):
                if seed in submissions_all:
                    try:
                        st.session_state["results"][seed] = _evaluate_bundle(
                            seed, submissions_all[seed]
                        )
                    except Exception as e:
                        errors.append(f"Seed {seed:02d}: {e}")
                progress.progress((i + 1) / len(seeds), text=f"Seed {seed:02d} completato")
            progress.empty()
            if errors:
                st.warning("\n".join(errors))
            else:
                st.success("Rivalutazione completata.")
            results_so_far = st.session_state["results"]

    if not results_so_far:
        st.info(
            "Nessun gruppo ancora valutato.\n\n"
            "Usa la tab **📋 Submissions** per valutare i gruppi durante i pitch."
        )
    else:
        # Calcola Δ AUC vs baseline e ordina per delta decrescente
        for r in results_so_far.values():
            r["delta_vs_baseline"] = _delta_auc(r["auc_test"], r["seed"])

        rows = sorted(
            results_so_far.values(),
            key=lambda r: r.get("delta_vs_baseline", float("-inf")) if not (isinstance(r.get("delta_vs_baseline"), float) and np.isnan(r.get("delta_vs_baseline", 0.0))) else float("-inf"),
            reverse=True,
        )

        # Costruisci tabella con badge HTML per rank e overfitting
        rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}
        has_cheatsheet = bool(_BEST_AUC)

        table_rows_html = ""
        for rank, r in enumerate(rows, 1):
            gruppo = ", ".join(
                c.split()[-1] for c in r["componenti"] if c.strip()
            ) or "—"
            rank_display = f"{rank_icons.get(rank, str(rank))}"
            delta = r["delta_auc"]
            delta_style = f"color:{COLOR_RED};font-weight:700;" if delta < -0.05 else f"color:{COLOR_GREEN};"
            badge = '<span class="badge-overfitting">⚠️ overfitting</span>' if delta < -0.05 else '<span class="badge-ok">✅ ok</span>'
            auc_style = "font-weight:800;font-size:1.05rem;" if rank == 1 else ""
            dvb = r.get("delta_vs_baseline", float("nan"))
            if isinstance(dvb, float) and np.isnan(dvb):
                delta_vs_b_str = "—"
                delta_vs_b_style = "opacity:0.4;"
            elif dvb > 0:
                delta_vs_b_str = f"{dvb:+.4f}"
                delta_vs_b_style = f"color:{COLOR_GREEN};font-weight:700;"
            elif dvb < 0:
                delta_vs_b_str = f"{dvb:+.4f}"
                delta_vs_b_style = f"color:{COLOR_RED};font-weight:700;"
            else:
                delta_vs_b_str = "±0.0000"
                delta_vs_b_style = "opacity:0.6;"
            table_rows_html += f"""
            <tr>
                <td style="text-align:center;font-size:1.2rem;">{rank_display}</td>
                <td style="font-family:monospace;font-weight:700;">SEED {r['seed']:02d}</td>
                <td>{gruppo}</td>
                <td style="font-family:monospace;opacity:0.7;">{r['model_name']}</td>
                <td style="text-align:right;opacity:0.65;">{r['auc_train']:.4f}</td>
                <td style="text-align:right;{auc_style}color:{COLOR_BLUE};">{r['auc_test']:.4f}</td>
                <td style="text-align:right;{delta_style}">{delta:+.4f}</td>
                <td style="text-align:right;">{r['f1']:.4f}</td>
                <td style="text-align:right;{delta_vs_b_style}">{delta_vs_b_str}</td>
                <td>{badge}</td>
            </tr>"""

        score_header = "<th style='text-align:right;padding:0.5rem 0.7rem;'>Δ vs Baseline</th>" if has_cheatsheet else ""
        st.markdown(
            f"""
            <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
                <thead>
                    <tr style="border-bottom:2px solid rgba(255,255,255,0.15);opacity:0.55;
                               font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">
                        <th style="text-align:center;padding:0.5rem 0.7rem;">#</th>
                        <th style="padding:0.5rem 0.7rem;">Seed</th>
                        <th style="padding:0.5rem 0.7rem;">Gruppo</th>
                        <th style="padding:0.5rem 0.7rem;">Modello</th>
                        <th style="text-align:right;padding:0.5rem 0.7rem;">AUC train</th>
                        <th style="text-align:right;padding:0.5rem 0.7rem;">AUC test</th>
                        <th style="text-align:right;padding:0.5rem 0.7rem;">Δ</th>
                        <th style="text-align:right;padding:0.5rem 0.7rem;">F1</th>
                        {score_header}
                        <th style="padding:0.5rem 0.7rem;">Stato</th>
                    </tr>
                </thead>
                <tbody style="line-height:2.2;">
                    {table_rows_html}
                </tbody>
            </table>
            <div style="margin-top:0.8rem;font-size:0.8rem;opacity:0.5;">
                Δ vs Baseline = AUC_test − miglior modello baseline istruttore &nbsp;·&nbsp;
                Δ train/test = AUC_test − AUC_train &nbsp;·&nbsp; ⚠️ overfitting se Δ train/test &lt; −0.05
            </div>
            """,
            unsafe_allow_html=True,
        )


