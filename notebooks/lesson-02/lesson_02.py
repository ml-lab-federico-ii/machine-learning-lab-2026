# %% [markdown]
# # Machine Learning per l’Analisi Finanziaria
#
# ## Lezione 02 — Dal dato grezzo al dataset modellabile
#
# **Authors:**
# - Enrico Huber
# - Pietro Soglia
#
# **Emails:**
# - enrico.huber@gmail.com
# - pietro.soglia@gmail.com
#
# **Last updated:** 2026-03-05
#
# ## Obiettivi di apprendimento
#
# - Comprendere perché il preprocessing è parte integrante del modello (e non un “passo a lato”).
# - Impostare uno split train/validation/test riproducibile e coerente con lo sbilanciamento.
# - Progettare trasformazioni (imputazione, scaling, encoding) evitando data leakage.
# - Costruire una pipeline sklearn end-to-end (`ColumnTransformer` + modello baseline).
# - Capire perché l’accuracy può essere fuorviante con classi sbilanciate.

# %% [markdown]
# ## Outline
#
# - Collegamento alla Lezione 01: cosa sappiamo già del dataset
# - Definire feature set e regole anti-leakage (ID columns, `Complain`)
# - Split train/validation/test (stratified) e rischio leakage
# - Missing values: diagnosi e imputazione (anche se missing=0)
# - Outlier: diagnosi rapida e strategie di gestione
# - Encoding categoriche + scaling numeriche (ColumnTransformer)
# - Pipeline base end-to-end (preprocess + modello)
# - Imbalance e metriche alternative all’accuracy
# - Riepilogo

# %% [markdown]
# ## Setup
#
# Definiamo dipendenze, percorsi e cartelle di output.

# %%
from __future__ import annotations

import json
import random
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.3f}".format)


def resolve_project_root() -> Path:
    """Risolve la root del repository indipendentemente dalla cwd."""
    start_points = [Path.cwd().resolve()]
    if "__file__" in globals():
        start_points.append(Path(__file__).resolve().parent)

    for start in start_points:
        for candidate in [start, *start.parents]:
            if (candidate / "data" / "archive.zip").exists() and (
                candidate / "notebooks"
            ).exists():
                return candidate

    raise FileNotFoundError(
        "Impossibile determinare la root del progetto. "
        "Atteso un percorso contenente data/archive.zip e notebooks/."
    )


ROOT = resolve_project_root()
DATA_ARCHIVE_PATH = ROOT / "data" / "archive.zip"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATA_OUT_DIR = OUTPUTS_DIR / "data"
LESSON_PLANS_DIR = OUTPUTS_DIR / "lesson_plans"

for _dir in [OUTPUTS_DIR, FIGURES_DIR, DATA_OUT_DIR, LESSON_PLANS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_current_figure(filename: str) -> None:
    """Salva la figura corrente in outputs/figures/."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=120, bbox_inches="tight")


def load_dataset_from_archive(
    archive_path: Path,
    filename_patterns: tuple[str, ...] = ("train", "test", "churn", "customer"),
) -> pd.DataFrame:
    """Carica il CSV principale dall'archivio ZIP listando i membri disponibili."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archivio dati non trovato: {archive_path}")

    with ZipFile(archive_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        print("Membri archivio:")
        for m in members:
            print(f"  - {m}")

        candidates = [
            m
            for m in members
            if any(p in m.lower() for p in filename_patterns) and m.endswith(".csv")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Nessun CSV trovato nell'archivio con i pattern {filename_patterns}. "
                f"Membri disponibili: {members}"
            )

        chosen = candidates[0]
        print(f"\nFile selezionato: {chosen}")
        with zf.open(chosen) as f:
            return pd.read_csv(f)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artefatto non trovato: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# %% [markdown]
# ## Collegamento alla Lezione 01: cosa sappiamo già del dataset
#
# La Lezione 01 ha già validato il contratto dati (dove si trova il dataset, qual è il target e come è distribuito).
# Qui riusiamo quelle evidenze per evitare lavoro duplicato e concentrarci sul preprocessing.

# %%
plan_l01_path = LESSON_PLANS_DIR / "lesson_01_plan.json"
plan_l01 = read_json_if_exists(plan_l01_path)

schema_l01 = plan_l01["schema"]
class_balance = schema_l01["target"]["class_balance"]
positive_rate = schema_l01["target"]["positive_rate"]

id_cols = schema_l01["notes"]["id_columns"]
leakage_suspects = schema_l01["notes"]["leakage_suspects"]

print("Lezione 01 — evidenze riusate")
print("Shape:", schema_l01["n_rows"], "x", schema_l01["n_cols"])
print("Target:", schema_l01["target"]["name"])
print("Class balance:", class_balance)
print("Positive rate:", positive_rate)
print("ID columns:", id_cols)
print("Leakage suspects:", leakage_suspects)

# %% [markdown]
# - Il dataset ha **10,000 righe** e **18 colonne** (contratto validato in Lezione 01).
# - Il target è `Exited` e la classe positiva vale **20.38%**: lo split deve essere **stratificato**.
# - `RowNumber`, `CustomerId`, `Surname` sono identificativi e vanno esclusi dalle feature.
# - `Complain` è un fortissimo sospetto di leakage: per default adottiamo un approccio prudente e la escludiamo.

# %% [markdown]
# ## Caricamento dati e contratto del dataset (ripresa minimale)
#
# Anche se non rifacciamo l’EDA completa, carichiamo il dataset per costruire lo split e la pipeline.

# %%
df = load_dataset_from_archive(DATA_ARCHIVE_PATH)

print("\nShape:", df.shape)
display(df.head())

print("\nDtypes:")
display(df.dtypes)

print("\nDescrittive (numeriche):")
display(df.describe(include=[np.number]).T)

# %% [markdown]
# - Confermiamo shape **10,000 × 18** e la presenza delle colonne attese.
# - Le variabili categoriche principali sono `Geography`, `Gender`, `Card Type`.
# - La presenza di colonne come `Point Earned` o `Satisfaction Score` suggerisce che alcune feature potrebbero richiedere controlli di scala/robustezza.

# %% [markdown]
# ## Definire feature set e regole anti-leakage (ID columns, `Complain`)
#
# Prima di preprocessare, decidiamo cosa entra nello “spazio delle feature”.
#
# Regola generale:
# - identificativi e variabili non disponibili *prima* della decisione (o del churn) vanno esclusi.

# %%
TARGET = "Exited"

id_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
leakage_cols = [c for c in ["Complain"] if c in df.columns]

feature_drop_cols = [*id_cols, *leakage_cols]

df_model = df.drop(columns=feature_drop_cols)

X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("Feature escluse:", feature_drop_cols)
print("Numero feature (dopo esclusioni):", X.shape[1])
print("Categoriche:", cat_cols)
print("Numeriche:", num_cols)

# %% [markdown]
# - Escludiamo **4 colonne**: 3 identificativi (`RowNumber`, `CustomerId`, `Surname`) + `Complain` (sospetta di leakage).
# - Dopo le esclusioni rimangono **13 feature**.
# - Le feature categoriche sono **3** (`Geography`, `Gender`, `Card Type`); le altre **10** sono numeriche/binarie.

# %% [markdown]
# ## Split train/validation/test (stratified) e rischio leakage
#
# Usiamo uno split 60/20/20 (train/val/test) stratificato sul target.
#
# Motivazione:
# - train: per apprendere trasformazioni e modello
# - validation: per scegliere iperparametri/decisioni di preprocessing senza “toccare” il test
# - test: per una valutazione finale non ottimistica

# %%
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
)

print("Split sizes:")
print("  train:", X_train.shape)
print("  val:  ", X_val.shape)
print("  test: ", X_test.shape)

print("\nPositive rate:")
print("  overall:", y.mean())
print("  train:  ", y_train.mean())
print("  val:    ", y_val.mean())
print("  test:   ", y_test.mean())

# %% [markdown]
# - Lo split produce **6,000** righe di train, **2,000** di validation e **2,000** di test.
# - La prevalenza di churn (`Exited=1`) resta stabile: in validation è circa **0.204**, coerente con il valore complessivo **0.2038**.
# - La stratificazione è un dettaglio fondamentale: senza, potremmo avere validation/test con percentuali diverse e metriche non confrontabili.

# %% [markdown]
# ## Missing values: diagnosi e imputazione (anche se missing=0)
#
# Anche se in questo dataset non ci sono missing values, è buona pratica progettare la pipeline in modo robusto:
# su dati futuri potremmo avere campi mancanti.

# %%
missing_rate_train = X_train.isna().mean().sort_values(ascending=False)
missing_nonzero = missing_rate_train[missing_rate_train > 0]

print("Numero colonne con missing (train):", (missing_rate_train > 0).sum())
if len(missing_nonzero) > 0:
    display(missing_nonzero.head(10))

# %% [markdown]
# - Nel train **0 colonne** hanno missing values: in questa versione del dataset l’imputazione sarebbe un no-op.
# - Nonostante ciò, includiamo `SimpleImputer` nella pipeline per rendere il workflow riutilizzabile su dati futuri.

# %% [markdown]
# ## Outlier: diagnosi rapida e strategie di gestione
#
# Una diagnosi rapida degli outlier (ad esempio con la regola IQR) aiuta a motivare lo scaling:
# - `StandardScaler` è sensibile a code pesanti
# - `RobustScaler` è più stabile in presenza di outlier
#
# Qui facciamo una diagnosi su alcune variabili numeriche comuni.

# %%
outlier_rows = []

for col in [
    c
    for c in ["Age", "Balance", "CreditScore", "EstimatedSalary"]
    if c in X_train.columns
]:
    s = X_train[col].astype(float)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    outlier_rows.append(
        {
            "feature": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lo": lo,
            "hi": hi,
            "n_out": n_out,
            "rate_out": n_out / len(s),
        }
    )

outlier_df = pd.DataFrame(outlier_rows).sort_values("rate_out", ascending=False)
display(outlier_df)

# Boxplot su Age per visualizzare le code
plt.figure(figsize=(6, 2.2))
sns.boxplot(x=X_train["Age"], color="#4C72B0")
plt.title("Age — boxplot (train)")
plt.xlabel("Age")
save_current_figure("lesson_02_age_boxplot.png")
plt.show()

# %% [markdown]
# - Per `Age` otteniamo $Q_1=32$ e $Q_3=44$ (IQR=12): la regola IQR segnala outlier fuori da **[14, 62]**.
# - In train ci sono **222** valori di `Age` fuori da questi limiti, pari a circa **3.7%**.
# - Per `CreditScore` gli outlier sono **8** (circa **0.13%**), mentre `Balance` ed `EstimatedSalary` non mostrano outlier IQR.
# - Questa diagnosi suggerisce che uno scaling robusto *potrebbe* essere utile; per semplicità partiamo da `StandardScaler` e rimandiamo confronti più approfonditi.

# %% [markdown]
# ## Encoding categoriche + scaling numeriche (ColumnTransformer)
#
# Ora traduciamo le scelte di preprocessing in un oggetto `ColumnTransformer`.
#
# Regole operative:
# - imputazione e scaling sulle numeriche
# - imputazione e one-hot sulle categoriche (con `handle_unknown="ignore"`)
# - fit delle trasformazioni **solo sul train**

# %%
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

print("Categoriche:", cat_cols)
print("Numeriche:", num_cols)

# %% [markdown]
# - Le categoriche sono: `Geography`, `Gender`, `Card Type`.
# - Le numeriche includono anche variabili binarie (`HasCrCard`, `IsActiveMember`) e discrete (`NumOfProducts`).
# - `ColumnTransformer` ci permette di evitare errori comuni (ad esempio scalare anche le stringhe) e rende la pipeline riproducibile.

# %% [markdown]
# ## Pipeline base end-to-end (preprocess + modello)
#
# Costruiamo una pipeline completa con un modello baseline (Logistic Regression).
#
# Perché una baseline?
# - ci dà un punto di partenza misurabile
# - rende immediato capire se le scelte di preprocessing stanno “rompendo” qualcosa

# %%
clf = LogisticRegression(max_iter=1000, random_state=SEED)

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
pipe.fit(X_train, y_train)

val_pred = pipe.predict(X_val)
val_proba = pipe.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, val_pred)
prec = precision_score(y_val, val_pred, zero_division=0)
rec = recall_score(y_val, val_pred, zero_division=0)
f1 = f1_score(y_val, val_pred, zero_division=0)
auc = roc_auc_score(y_val, val_proba)

print("Validation metrics (LogReg baseline)")
print("  accuracy:", acc)
print("  precision:", prec)
print("  recall:", rec)
print("  f1:", f1)
print("  roc_auc:", auc)

# %% [markdown]
# - Con la pipeline base otteniamo in validation un’**accuracy = 0.814**.
# - Tuttavia il **recall = 0.211** è basso: stiamo identificando solo ~1 churner su 5.
# - La metrica ROC-AUC è **0.786**, segnale che il ranking probabilistico è ragionevole anche se la soglia 0.5 non è ottimale.
# - Questi risultati sono un ottimo pretesto per Lezione 03: confronto modelli + tuning soglia/metriche.

# %% [markdown]
# ## Imbalance e metriche alternative all’accuracy
#
# Per capire perché l’accuracy può essere fuorviante, confrontiamo la pipeline con un baseline banale:
# predire sempre la classe maggioritaria (`Exited=0`).

# %%
majority_pred = np.zeros_like(y_val)

acc_maj = accuracy_score(y_val, majority_pred)
prec_maj = precision_score(y_val, majority_pred, zero_division=0)
rec_maj = recall_score(y_val, majority_pred, zero_division=0)
f1_maj = f1_score(y_val, majority_pred, zero_division=0)

print("Validation metrics (majority-class baseline)")
print("  accuracy:", acc_maj)
print("  precision:", prec_maj)
print("  recall:", rec_maj)
print("  f1:", f1_maj)

# %% [markdown]
# - Il baseline “tutto 0” ottiene **accuracy = 0.796**, non molto distante da 0.814.
# - Ma precision/recall/F1 sono **tutti 0**: non intercetta nessun churn.
# - Questo mostra concretamente perché con positivi ~20% l’accuracy da sola non basta: dobbiamo guardare metriche orientate alla classe positiva e al ranking (es. ROC-AUC).

# %% [markdown]
# ## Riepilogo
#
# - Abbiamo riusato le evidenze della Lezione 01 (target `Exited`, positive rate 20.38%, `Complain` sospetta di leakage).
# - Abbiamo costruito uno split **stratificato 60/20/20**: 6,000 / 2,000 / 2,000.
# - Missingness assente (0 colonne con missing), ma pipeline progettata per robustezza (`SimpleImputer`).
# - Outlier IQR: `Age` mostra **222** outlier (~3.7%); questo motiva attenzione allo scaling.
# - Pipeline baseline (LogReg) in validation: accuracy **0.814**, ROC-AUC **0.786**, recall **0.211**.
# - Baseline maggioritaria: accuracy **0.796** ma recall **0** → serve andare oltre l’accuracy.
#
# Nella Lezione 03 useremo queste basi per confrontare modelli e leggere le metriche in modo più completo.
