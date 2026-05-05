"""Valutazione lato docente dei modelli consegnati dagli studenti.

Applica ogni modello ricevuto al test set nascosto (seed_XX_test.csv)
e produce metriche + leaderboard comparativa.

Usi tipici:
    # Singolo gruppo
    python evaluate.py --seed 5 --model submissions/model_seed_05.joblib

    # Tutti i gruppi in una cartella
    python evaluate.py --all --submissions-dir submissions/

    # Output JSON per uso programmatico
    python evaluate.py --all --submissions-dir submissions/ --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    print("❌ joblib non disponibile. Installare con: pip install joblib")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Exited"
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR / "datasets"


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------


def _load_test_csv(seed: int) -> pd.DataFrame:
    """Carica il test set nascosto per il SEED dato."""
    path = DATASETS_DIR / f"seed_{seed:02d}_test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Test set non trovato: {path}\n"
            f"Rigenerate i dataset con: python generate_datasets.py"
        )
    return pd.read_csv(path)


def _apply_preprocessing(
    df_test: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Applica la stessa pipeline di preprocessing usata dal gruppo in training.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test set grezzo (stesso schema del train).
    bundle : dict
        Bundle joblib con model, scaler, label_encoders, feature_order, ecc.

    Returns
    -------
    X_test_processed : pd.DataFrame
    y_test : pd.Series
    """
    df = df_test.copy()
    y_test = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Feature engineering (balance_is_zero)
    if bundle.get("balance_is_zero"):
        X["balance_is_zero"] = (X["Balance"] == 0).astype(int)

    # Esclusione feature
    exclude = bundle.get("exclude_features", [])
    if exclude:
        X = X.drop(columns=[c for c in exclude if c in X.columns], errors="ignore")

    # Encoding categoriche con gli stessi LabelEncoder del training
    label_encoders: dict[str, LabelEncoder] = bundle.get("label_encoders", {})
    for col, le in label_encoders.items():
        if col in X.columns:
            # Gestisce valori unseen mappando all'ultima classe nota
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])

    # Allineamento feature (stesso ordine del training)
    feature_order: list[str] = bundle.get("feature_order", list(X.columns))
    missing = set(feature_order) - set(X.columns)
    if missing:
        raise ValueError(
            f"Feature mancanti nel test set: {missing}\n"
            f"Verifica che il bundle corrisponda al SEED corretto."
        )
    X = X[feature_order]

    # Scaling con lo stesso scaler del training
    scaler = bundle.get("scaler")
    if scaler is not None:
        X_arr = scaler.transform(X)
        X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)

    return X, y_test


def evaluate_bundle(
    seed: int,
    bundle_path: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Valuta un bundle modello sul test set nascosto.

    Parameters
    ----------
    seed : int
        SEED del gruppo (identifica il test CSV da usare).
    bundle_path : Path
        Percorso al file .joblib.
    verbose : bool
        Stampa output formattato.

    Returns
    -------
    dict con keys: seed, componenti, model_name, auc_train, auc_test,
                   delta_auc, f1_test, precision_test, recall_test,
                   accuracy_test, bundle_path
    """
    bundle: dict[str, Any] = joblib.load(bundle_path)

    # Validazione bundle seed
    bundle_seed = bundle.get("seed")
    if bundle_seed is not None and bundle_seed != seed:
        print(
            f"⚠️  Attenzione: il bundle è stato creato con SEED={bundle_seed} "
            f"ma stai valutando con SEED={seed}."
        )

    # Carica test set
    df_test = _load_test_csv(seed)

    # Preprocessing
    X_test, y_test = _apply_preprocessing(df_test, bundle)

    # Predizioni
    model = bundle["model"]
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metriche test
    auc_test = roc_auc_score(y_test, y_prob)
    f1_test = f1_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred)
    rec_test = recall_score(y_test, y_pred)
    acc_test = accuracy_score(y_test, y_pred)

    # Metriche train (dal bundle)
    train_metrics = bundle.get("train_metrics", {})
    auc_train = train_metrics.get("auc", float("nan"))
    delta_auc = auc_test - auc_train

    result = {
        "seed": seed,
        "componenti": bundle.get("componenti", ["?"]),
        "model_name": bundle.get("model_name", "?"),
        "auc_train": round(auc_train, 4),
        "auc_test": round(auc_test, 4),
        "delta_auc": round(delta_auc, 4),
        "f1_test": round(f1_test, 4),
        "precision_test": round(prec_test, 4),
        "recall_test": round(rec_test, 4),
        "accuracy_test": round(acc_test, 4),
        "bundle_path": str(bundle_path),
    }

    if verbose:
        cognomi = " · ".join(
            c.split()[-1] for c in bundle.get("componenti", ["?"]) if c.strip()
        )
        overfitting = "⚠️  overfitting" if delta_auc < -0.05 else "✅ ok"
        print(f"\n{'─'*56}")
        print(f"  SEED {seed:02d} | {cognomi}")
        print(f"  Modello:       {bundle.get('model_name', '?')}")
        print(f"  AUC train:     {auc_train:.4f}")
        print(f"  AUC test:      {auc_test:.4f}   Δ={delta_auc:+.4f}  {overfitting}")
        print(f"  F1 test:       {f1_test:.4f}")
        print(f"  Precision:     {prec_test:.4f}")
        print(f"  Recall:        {rec_test:.4f}")
        print(f"{'─'*56}")

    return result


def leaderboard(results: list[dict[str, Any]]) -> None:
    """Stampa la leaderboard comparativa."""
    sorted_res = sorted(results, key=lambda r: r["auc_test"], reverse=True)

    print(f"\n{'='*76}")
    print(f"  LEADERBOARD — Challenge ML (test set nascosto)")
    print(f"{'='*76}")
    header = f"{'#':>3}  {'SEED':>4}  {'Gruppo':<22}  {'Modello':<18}  {'AUC_tr':>6}  {'AUC_te':>6}  {'Δ':>6}"
    print(header)
    print(f"{'─'*76}")
    for rank, r in enumerate(sorted_res, 1):
        cognomi = " · ".join(
            c.split()[-1] for c in r.get("componenti", ["?"]) if c.strip()
        )[:22]
        delta_str = f"{r['delta_auc']:+.4f}"
        flag = "⚠️ " if r["delta_auc"] < -0.05 else "  "
        print(
            f"{rank:>3}  {r['seed']:>4}  {cognomi:<22}  {r['model_name']:<18}  "
            f"{r['auc_train']:>6.4f}  {r['auc_test']:>6.4f}  {flag}{delta_str}"
        )
    print(f"{'='*76}")
    print(f"  Legenda: Δ = AUC_test − AUC_train  |  ⚠️ = possibile overfitting (Δ < -0.05)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valutazione docente — test set evaluation e leaderboard"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--seed",
        type=int,
        help="SEED del gruppo da valutare (richiede --model).",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Valuta tutti i .joblib trovati in --submissions-dir.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Percorso al file .joblib (richiesto con --seed).",
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=SCRIPT_DIR / "submissions",
        help="Cartella con i bundle .joblib (default: challenge/submissions/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output anche in formato JSON (challenge/submissions/leaderboard.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point CLI."""
    args = _parse_args()

    if args.seed is not None:
        # Modalità singolo gruppo
        if args.model is None:
            print("❌ Con --seed devi specificare --model <percorso.joblib>")
            sys.exit(1)
        evaluate_bundle(args.seed, args.model)

    else:
        # Modalità --all
        submissions_dir: Path = args.submissions_dir
        if not submissions_dir.exists():
            print(f"❌ Cartella non trovata: {submissions_dir}")
            print(f"   Crea la cartella e inserisci i file .joblib ricevuti dai gruppi.")
            sys.exit(1)

        bundles = sorted(submissions_dir.glob("*.joblib"))
        if not bundles:
            print(f"❌ Nessun file .joblib trovato in {submissions_dir}")
            sys.exit(1)

        print(f"🔍 Trovati {len(bundles)} bundle in {submissions_dir}")
        results: list[dict[str, Any]] = []

        for bundle_path in bundles:
            # Prova a estrarre il seed dal nome file (model_seed_05.joblib)
            parts = bundle_path.stem.split("_")
            seed_from_name: int | None = None
            for part in reversed(parts):
                if part.isdigit():
                    seed_from_name = int(part)
                    break

            if seed_from_name is None:
                # Carica il bundle per leggere il seed interno
                try:
                    b = joblib.load(bundle_path)
                    seed_from_name = b.get("seed")
                except Exception:
                    pass

            if seed_from_name is None:
                print(f"⚠️  Impossibile determinare il SEED per {bundle_path.name} — skip")
                continue

            try:
                result = evaluate_bundle(seed_from_name, bundle_path, verbose=True)
                results.append(result)
            except Exception as exc:
                print(f"❌ Errore su {bundle_path.name}: {exc}")

        if results:
            leaderboard(results)

            if args.json:
                out_path = submissions_dir / "leaderboard.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 Leaderboard JSON salvata: {out_path}")


if __name__ == "__main__":
    main()
