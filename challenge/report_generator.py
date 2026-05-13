"""Generatore di report HTML standalone per la challenge in-class.

Produce un singolo file .html con CSS inline e grafici codificati in base64,
scaricabile dal Codespace e utilizzabile come "slide deck" per il pitch.
"""

from __future__ import annotations

import base64
import io
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    """Converte una figura matplotlib in stringa base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return encoded


def _img_tag(b64: str, alt: str = "", width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:{width};height:auto;">'


def _escape(text: str) -> str:
    """Escape HTML minimo."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1a1a2e; background: #f5f6fa; line-height: 1.6;
  }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  .cover {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white; padding: 60px 40px; text-align: center; border-radius: 12px;
    margin-bottom: 30px;
  }
  .cover h1 { font-size: 2.2em; margin-bottom: 10px; }
  .cover .subtitle { font-size: 1.1em; opacity: 0.85; margin-bottom: 20px; }
  .cover .team { font-size: 1em; opacity: 0.7; }
  .section {
    background: white; border-radius: 10px; padding: 30px;
    margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .section h2 {
    font-size: 1.4em; color: #302b63; margin-bottom: 16px;
    border-bottom: 2px solid #e8e8f0; padding-bottom: 8px;
  }
  .section h3 { font-size: 1.1em; color: #555; margin: 14px 0 8px; }
  table {
    width: 100%; border-collapse: collapse; margin: 12px 0;
    font-size: 0.92em;
  }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e8e8f0; }
  th { background: #f0f0f8; font-weight: 600; color: #302b63; }
  tr:hover { background: #fafafe; }
  .metric-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px; margin: 14px 0;
  }
  .metric-card {
    background: #f8f8ff; border-radius: 8px; padding: 16px; text-align: center;
    border: 1px solid #e0e0ee;
  }
  .metric-card .value { font-size: 1.8em; font-weight: 700; color: #302b63; }
  .metric-card .label { font-size: 0.85em; color: #666; margin-top: 4px; }
  .fig-container { text-align: center; margin: 16px 0; }
  .fig-container img { border-radius: 8px; border: 1px solid #e8e8f0; }
  .text-block {
    background: #f8f8ff; border-left: 4px solid #302b63; padding: 16px 20px;
    margin: 12px 0; border-radius: 0 8px 8px 0; white-space: pre-wrap;
  }
  .footer {
    text-align: center; padding: 20px; color: #999; font-size: 0.8em;
  }
  @media print {
    .section { break-inside: avoid; }
    .cover { break-after: page; }
  }
</style>
"""


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def generate_html_report(
    *,
    seed: int,
    componenti: list[str],
    dataset_info: dict[str, Any],
    eda_figures: list[tuple[str, plt.Figure]],
    eda_notes: str,
    preprocessing_config: dict[str, Any],
    model_results: list[dict[str, Any]],
    final_model_idx: int,
    final_model_figures: list[tuple[str, plt.Figure]],
    feature_importance_fig: plt.Figure | None,
    team_notes: dict[str, str] = None,
    output_dir: Path | str = "challenge/outputs",
) -> Path:
    """Genera un report HTML standalone e lo salva su disco.

    Parameters
    ----------
    seed : int
        SEED del gruppo (ID assegnato dal docente).
    componenti : list[str]
        Nomi e cognomi dei componenti del gruppo.
    dataset_info : dict
        Informazioni sul dataset (n_rows, n_cols, churn_rate, ecc.).
    eda_figures : list of (title, figure)
        Figure EDA con titolo.
    eda_notes : str
        Note EDA scritte dal gruppo.
    preprocessing_config : dict
        Scelte di preprocessing fatte dal gruppo.
    model_results : list of dict
        Risultati di ogni tentativo di modellazione
        (keys: model_name, params, auc, f1, precision, recall, accuracy).
    final_model_idx : int
        Indice nella lista model_results del modello selezionato.
    final_model_figures : list of (title, figure)
        Figure del modello finale (confusion matrix, ROC, ecc.).
    feature_importance_fig : Figure or None
        Bar chart feature importance del modello finale.
    team_notes : dict
        Testi liberi del gruppo: keys = 'motivazione', 'business_insight',
        'strategia_retention'.
    output_dir : Path or str
        Directory di output.

    Returns
    -------
    Path
        Percorso del file HTML generato.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if team_notes is None:
        team_notes = {}

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='it'><head>")
    parts.append("<meta charset='UTF-8'>")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append(f"<title>Challenge Report — SEED {seed:02d}</title>")
    parts.append(_CSS)
    parts.append("</head><body><div class='container'>")

    # --- Cover ---
    parts.append("<div class='cover'>")
    parts.append(f"<h1>Churn Prediction Challenge</h1>")
    parts.append(f"<div class='subtitle'>SEED {seed:02d}</div>")
    parts.append(f"<div class='team'>{'  ·  '.join(_escape(c) for c in componenti)}</div>")
    parts.append(f"<div class='team' style='margin-top:8px;'>{date.today().isoformat()}</div>")
    parts.append("</div>")

    # --- Dataset Overview ---
    parts.append("<div class='section'>")
    parts.append("<h2>1. Il Nostro Dataset</h2>")
    parts.append("<div class='metric-grid'>")
    for label, value in [
        ("Righe", f"{dataset_info.get('n_rows', '?'):,}"),
        ("Feature", str(dataset_info.get("n_features", "?"))),
        ("Churn Rate", f"{dataset_info.get('churn_rate', 0):.1%}"),
        ("Classe Maggioritaria", f"{1 - dataset_info.get('churn_rate', 0):.1%}"),
    ]:
        parts.append(
            f"<div class='metric-card'>"
            f"<div class='value'>{value}</div>"
            f"<div class='label'>{label}</div></div>"
        )
    parts.append("</div></div>")

    # --- EDA ---
    if eda_figures:
        parts.append("<div class='section'>")
        parts.append("<h2>2. Analisi Esplorativa</h2>")
        for title, fig in eda_figures:
            b64 = _fig_to_base64(fig)
            parts.append(f"<h3>{_escape(title)}</h3>")
            parts.append(f"<div class='fig-container'>{_img_tag(b64, title)}</div>")
        if eda_notes.strip():
            parts.append("<h3>Osservazioni del gruppo</h3>")
            parts.append(f"<div class='text-block'>{_escape(eda_notes)}</div>")
        parts.append("</div>")

    # --- Preprocessing ---
    parts.append("<div class='section'>")
    parts.append("<h2>3. Scelte di Preprocessing</h2>")
    parts.append("<table><tr><th>Parametro</th><th>Scelta</th></tr>")
    for k, v in preprocessing_config.items():
        parts.append(f"<tr><td>{_escape(str(k))}</td><td>{_escape(str(v))}</td></tr>")
    parts.append("</table></div>")

    # --- Model Comparison ---
    if model_results:
        parts.append("<div class='section'>")
        parts.append("<h2>4. Confronto Modelli</h2>")
        parts.append(
            "<table><tr>"
            "<th>#</th><th>Modello</th><th>Parametri</th>"
            "<th>AUC</th><th>F1</th><th>Precision</th><th>Recall</th>"
            "</tr>"
        )
        for i, res in enumerate(model_results):
            highlight = " style='background:#e8f5e9;font-weight:600;'" if i == final_model_idx else ""
            params_str = ", ".join(f"{k}={v}" for k, v in res.get("params", {}).items())
            parts.append(
                f"<tr{highlight}>"
                f"<td>{i + 1}</td>"
                f"<td>{_escape(res.get('model_name', ''))}</td>"
                f"<td style='font-size:0.85em;'>{_escape(params_str)}</td>"
                f"<td><b>{res.get('auc', 0):.4f}</b></td>"
                f"<td>{res.get('f1', 0):.4f}</td>"
                f"<td>{res.get('precision', 0):.4f}</td>"
                f"<td>{res.get('recall', 0):.4f}</td>"
                f"</tr>"
            )
        parts.append("</table>")
        parts.append("<p style='color:#666;font-size:0.85em;margin-top:8px;'>"
                     "La riga evidenziata indica il modello selezionato come finale.</p>")
        parts.append("</div>")

    # --- Final Model Details ---
    if model_results and 0 <= final_model_idx < len(model_results):
        final = model_results[final_model_idx]
        parts.append("<div class='section'>")
        parts.append("<h2>5. Modello Finale</h2>")
        parts.append(f"<h3>{_escape(final.get('model_name', ''))}</h3>")
        parts.append("<div class='metric-grid'>")
        for label, key in [("ROC-AUC", "auc"), ("F1-Score", "f1"),
                           ("Precision", "precision"), ("Recall", "recall"),
                           ("Accuracy", "accuracy")]:
            val = final.get(key)
            if val is not None:
                parts.append(
                    f"<div class='metric-card'>"
                    f"<div class='value'>{val:.4f}</div>"
                    f"<div class='label'>{label}</div></div>"
                )
        parts.append("</div>")
        for title, fig in final_model_figures:
            b64 = _fig_to_base64(fig)
            parts.append(f"<div class='fig-container'>{_img_tag(b64, title, '80%')}</div>")
        parts.append("</div>")

    # --- Feature Importance ---
    if feature_importance_fig is not None:
        parts.append("<div class='section'>")
        parts.append("<h2>6. Feature Importance</h2>")
        b64 = _fig_to_base64(feature_importance_fig)
        parts.append(f"<div class='fig-container'>{_img_tag(b64, 'Feature Importance', '90%')}</div>")
        parts.append("</div>")


    # --- Team Interpretation ---
    parts.append("<div class='section'>")
    parts.append("<h2>7. Interpretazione e Raccomandazioni</h2>")
    for label, key in [
        ("Perché abbiamo scelto questo modello", "motivazione"),
        ("Insight di business", "business_insight"),
        ("Strategia di retention proposta", "strategia_retention"),
    ]:
        text = team_notes.get(key, "").strip()
        if text:
            parts.append(f"<h3>{label}</h3>")
            parts.append(f"<div class='text-block'>{_escape(text)}</div>")
    parts.append("</div>")

    # --- Appendice Tecnica ---
    if model_results and 0 <= final_model_idx < len(model_results):
        final = model_results[final_model_idx]
        parts.append("<div class='section'>")
        parts.append("<h2>8. Appendice Tecnica</h2>")
        parts.append("<table><tr><th>Iperparametro</th><th>Valore</th></tr>")
        for k, v in final.get("params", {}).items():
            parts.append(f"<tr><td>{_escape(str(k))}</td><td>{_escape(str(v))}</td></tr>")
        parts.append("</table>")
        parts.append("<h3>Preprocessing</h3>")
        parts.append("<table><tr><th>Parametro</th><th>Scelta</th></tr>")
        for k, v in preprocessing_config.items():
            parts.append(f"<tr><td>{_escape(str(k))}</td><td>{_escape(str(v))}</td></tr>")
        parts.append("</table>")
        parts.append("</div>")

    # --- Footer ---
    parts.append(f"<div class='footer'>Report generato automaticamente — "
                 f"Machine Learning per l'Analisi Finanziaria — "
                 f"SEED {seed:02d} — {date.today().isoformat()}</div>")
    parts.append("</div></body></html>")

    html = "\n".join(parts)
    out_path = output_dir / f"team_seed_{seed:02d}_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
