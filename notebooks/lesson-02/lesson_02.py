# %% [markdown]
# # Machine Learning per l'Analisi Finanziaria
#
# ## Lezione 02 — Dal dato grezzo al dataset modellabile
#
# **Authors:**
# - Enrico Huber
# - Pietro Soglia
#
# **Last updated:** 2026-03-12
#
# ## Obiettivi di apprendimento
#
# - Rimuovere variabili non predittive e leakage identificati nella Lezione 1.
# - Comprendere il problema dello sbilanciamento e selezionare metriche corrette.
# - Conoscere e confrontare le principali strategie di gestione dell'imbalance.
# - Identificare e trattare outlier con il metodo IQR.
# - Creare feature ingegnerizzate (`balance_is_zero`) motivate dall'EDA.
# - Costruire una **Pipeline sklearn** (ColumnTransformer) riproducibile e priva di data leakage.
# - Effettuare uno split train / validation / test definitivo (60/20/20) stratificato.
# - Salvare il dataset "modellabile" pronto per la Lezione 3.

# %% [markdown]
# ## Outline
#
# 1. Setup, percorsi e riproducibilità
# 2. Riepilogo Lezione 1 e obiettivi del preprocessing
# 3. Caricamento dati
# 4. Pulizia: rimozione variabili non predittive e leakage
# 5. Il problema dell'imbalance e metriche alternative
# 6. Gestione dello sbilanciamento: strategie a confronto
# 7. Outlier e range check
# 8. Feature engineering: `balance_is_zero`
# 9. Definire feature set e target
# 10. Split train / validation / test definitivo
# 11. Encoding delle variabili categoriche
# 12. Feature scaling
# 13. Pipeline sklearn completa: ColumnTransformer
# 14. Verifica del preprocessing
# 15. Salvataggio dataset modellabile
# 16. Domande guidate
# 17. Riepilogo

# %% [markdown]
# ## 1. Setup, percorsi e riproducibilità
#
# Definiamo dipendenze, percorsi e cartelle di output. La struttura è analoga
# alla Lezione 1 per garantire coerenza e riproducibilità tra le lezioni.

# %%
from __future__ import annotations

import json
import random
from pathlib import Path
from zipfile import ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover

    def display(x):  # type: ignore
        print(x)


SEED = 42
np.random.seed(SEED)
random.seed(SEED)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)


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

for _dir in [OUTPUTS_DIR, FIGURES_DIR, DATA_OUT_DIR]:
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
    """Carica il CSV principale dall'archivio ZIP validando lo schema.

    Parameters
    ----------
    archive_path : Path
        Percorso al file ZIP contenente i dati.
    filename_patterns : tuple[str, ...]
        Pattern per selezionare il CSV corretto dall'archivio.

    Returns
    -------
    pd.DataFrame
        Dataset caricato dal CSV selezionato.

    Raises
    ------
    FileNotFoundError
        Se l'archivio o il CSV atteso non esistono.
    """
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
                "Nessun CSV trovato nell'archivio con i pattern "
                f"{filename_patterns}. Membri disponibili: {members}"
            )
        chosen = candidates[0]
        print(f"\nFile selezionato: {chosen}")
        with zf.open(chosen) as f:
            return pd.read_csv(f)


# %% [markdown]
# ## 2. Riepilogo Lezione 1 e obiettivi del preprocessing
#
# La Lezione 1 ha prodotto una serie di scoperte fondamentali sul dataset.
# Questa sezione le sintetizza per motivare ogni operazione di preprocessing.
#
# ### Cosa abbiamo imparato nella Lezione 1
#
# | Aspetto | Valore / Osservazione |
# |---------|----------------------|
# | Shape | 10,000 righe × 18 colonne |
# | Target (`Exited`) | Churn rate = **20.38%** (2,038/10,000) |
# | Missing values | **Nessuno** — tutte le colonne completamente popolate |
# | Duplicati | **Nessuno** |
# | Leakage confermato | `Complain` (corr. 0.996 con target; P(Exited=1 \| Complain=1) = 99.51%) |
# | Colonne identificative | `RowNumber`, `CustomerId`, `Surname` — da escludere |
# | Segnale forte | `Age` (r=+0.285), `IsActiveMember` (r=−0.156), `Balance` (r=+0.119) |
# | Bimodalità `Balance` | 36% dei clienti ha saldo zero; churn rate diverso (14% vs 24%) |
# | Feature rumore | `EstimatedSalary`, `Satisfaction Score`, `HasCrCard`, `Point Earned` (r ≈ 0) |
# | Baseline | Logistic Regression: ROC-AUC ≈ **0.77**, recall churn = **21.1%** |
#
# ### Operazioni pianificate per questa lezione
#
# | Operazione | Motivazione |
# |-----------|-------------|
# | Rimuovere `Complain` | Leakage confermato (corr. 0.996) |
# | Rimuovere `RowNumber`, `CustomerId`, `Surname` | Non predittivi, identificativi |
# | Creare `balance_is_zero` | Catturare bimodalità di `Balance` (churn rate diverso) |
# | Split 60/20/20 stratificato | Preservare distribuzione target; val set per tuning, test per stima finale |
# | Pipeline ColumnTransformer | Encoding + scaling senza data leakage |
# | Discutere strategie imbalance | Churn rate 20.38%: accuracy non è metrica affidabile |
# | Salvare dataset modellabile | Passerella verso Lezione 3 (modelli e metriche) |

# %% [markdown]
# ## 3. Caricamento dati
#
# Ricarichiamo il dataset dal contratto canonico `data/archive.zip`.
# Questo garantisce che il notebook sia autonomo e riproducibile.

# %%
df = load_dataset_from_archive(DATA_ARCHIVE_PATH)

print("\nShape:", df.shape)
print("\nColonne:")
display(df.dtypes.to_frame("dtype"))

print("\nPrime righe:")
display(df.head())

print("\nDescrittive (numeriche):")
display(df.describe(include=[np.number]).T)

# %% [markdown]
# - Il dataset è stato caricato correttamente: **10,000 righe** e **18 colonne**, coerente
#   con quanto osservato in Lezione 1.
# - Nessuna variazione nello schema rispetto alla sessione precedente.

# %% [markdown]
# ## 4. Pulizia: rimozione variabili non predittive e leakage
#
# Prima di qualsiasi preprocessing, eliminiamo le colonne che non devono
# mai entrare in un modello predittivo.

# %%
TARGET = "Exited"

# Colonne con dtype numerico che sono semanticamente categoriche (binarie / ordinali a bassa cardinalità).
# Verranno trattate come categoriche nel preprocessing (OHE invece di scaling).
NUMERIC_AS_CATEGORICAL = {
    "IsActiveMember",
    "HasCrCard",
    "NumOfProducts",
    "Satisfaction Score",
    "balance_is_zero",  # feature ingegnerizzata (binaria 0/1) creata in sezione 8
}

# Colonne da rimuovere con motivazione esplicita
ID_COLS = ["RowNumber", "CustomerId", "Surname"]
# RowNumber: indice artificiale, nessun significato predittivo
# CustomerId: identificativo univoco per cliente, non predittivo
# Surname: rumore semantico, potenziale proxy per etnia (rischio bias + non predittivo)

LEAKY_COLS = ["Complain"]
# Complain: corr. 0.996 con Exited — variabile "post-evento" o quasi-proxy del target

cols_to_drop = ID_COLS + LEAKY_COLS
df_clean = df.drop(columns=cols_to_drop)

print(f"Shape originale:  {df.shape}")
print(f"Shape dopo pulizia: {df_clean.shape}")
print(f"Colonne rimosse: {cols_to_drop}")
print(f"\nColonne rimanenti:\n{df_clean.columns.tolist()}")

# %% [markdown]
# - Il dataset passa da **18** a **14 colonne** dopo la rimozione di 4 variabili:
#   3 identificative e 1 leaky.
# - Le colonne rimanenti sono tutte teoricamente disponibili *prima* del churn,
#   senza rischio di data leakage evidente.
# - Nota: `Satisfaction Score` e `Point Earned` rimangono (segnale quasi nullo dalla
#   Lezione 1, ma non sono leakage — possono rivelarsi utili in interazione).

# %% [markdown]
# ## 5. Il problema dell'imbalance e metriche alternative
#
# Il churn rate del 20.38% crea uno sbilanciamento tra le classi che rende
# l'**accuracy** una metrica fuorviante. Dimostriamo il paradosso e introduciamo
# le metriche corrette.

# %%
churn_rate = df_clean[TARGET].mean()
non_churn_rate = 1 - churn_rate
n_total = len(df_clean)
n_churn = df_clean[TARGET].sum()
n_no_churn = n_total - n_churn

print(f"Totale clienti:         {n_total:,}")
print(f"Clienti churn (y=1):    {n_churn:,} ({churn_rate:.2%})")
print(f"Clienti no-churn (y=0): {n_no_churn:,} ({non_churn_rate:.2%})")

# Paradosso accuracy: un classificatore che predice sempre "0" (non churn)
dummy_accuracy = non_churn_rate
dummy_recall_churn = 0.0
print(f"\n--- Paradosso Accuracy ---")
print(f"DummyClassifier (sempre 0):")
print(f"  Accuracy:      {dummy_accuracy:.2%}  ← sembra buono!")
print(f"  Recall churn:  {dummy_recall_churn:.2%}  ← non identifica nessun churner")
print(f"  Utilità pratica: NULLA")

# %% [markdown]
# ### Metriche corrette per problemi sbilanciati
#
# Per valutare un modello di churn dobbiamo usare metriche che tengano conto
# dello sbilanciamento e che siano rilevanti per il business.
#
# Dato un modello con:
# - **TP** (True Positive): churner correttamente identificati
# - **TN** (True Negative): non-churner correttamente identificati
# - **FP** (False Positive): non-churner classificati come churner (costo: contatto inutile)
# - **FN** (False Negative): churner non identificati (costo: cliente perso)
#
# Le metriche principali sono:
#
# $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
#
# $$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(di quanti predetti positivi sono veri positivi?)}$$
#
# $$\text{Recall} = \frac{TP}{TP + FN} \quad \text{(di quanti positivi reali ne identifichiamo?)}$$
#
# $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
#
# $$\text{ROC-AUC} = P(\hat{p}(\text{churner}) > \hat{p}(\text{non-churner}))$$
#
# **Baseline Lezione 1:** Logistic Regression con preprocessing minimale
# ha ottenuto ROC-AUC ≈ **0.77** ma recall churn = **21.1%** (86/408 churner
# identificati nel test set). Il preprocessing corretto di questa lezione
# servirà come base per migliorare questo risultato nella Lezione 3.
#
# **Trade-off business:**
#
# | Metrica | Minimizzare FP | Minimizzare FN |
# |---------|----------------|----------------|
# | Obiettivo | Contattare solo i veri churner (efficienza) | Non perdere nessun churner (copertura) |
# | Metrica da privilegiare | Precision ↑ | Recall ↑ |
# | Esempio | Offerta costosa di retention | Segnale di allerta precoce |

# %% [markdown]
# ## 6. Gestione dello sbilanciamento: strategie a confronto
#
# Con una classe positiva al 20.38%, esistono diverse strategie per gestire lo
# sbilanciamento. Nessuna è universalmente migliore: la scelta dipende dal
# contesto, dal modello e dal trade-off business.

# %% [markdown]
# ### Panoramica delle strategie
#
# | Strategia | Meccanismo | Pro | Contro | Quando usarla |
# |-----------|------------|-----|--------|---------------|
# | **Nessuna** | Nessuna modifica | Semplice, baseline | Modello biased verso classe maggioritaria | Solo per capire il baseline |
# | **`class_weight="balanced"`** | Pesa inversamente la frequenza di classe nel loss | Nessun sample aggiuntivo, no leakage | Solo per modelli che supportano `class_weight` | Prima scelta; semplice e stabile |
# | **Oversampling (SMOTE)** | Crea campioni sintetici della classe minoritaria (interpolazione) | Aumenta effettivamente il segnale | Va applicato **solo su train**; può creare campioni irrealistici | Con dataset piccoli o sbilanciamento severo |
# | **Undersampling** | Rimuove campioni dalla classe maggioritaria | Riduce training time | Perde informazione; rischio di underfitting | Con dataset molto grandi (milioni di righe) |
# | **Threshold tuning** | Abbassa la soglia di classificazione (es. 0.5 → 0.3) | Non modifica i dati; post-hoc | Richiede calibrazione; peggiora precision | Quando si vuole bilanciare precision/recall dopo training |
#
# **Regola aurea:** SMOTE (e qualsiasi tecnica di resampling) va applicato
# **esclusivamente sul training set**, **dopo** lo split. Applicarlo prima creerebbe
# un grave caso di **data leakage**: campioni sintetici derivati dagli stessi dati
# di test "contaminerebbero" la stima della performance.

# %%
# Demo: class_weight="balanced" in LogisticRegression
# (snippet dimostrativo — il training completo è in Lezione 3)
from sklearn.linear_model import LogisticRegression

# Calcolo dei pesi di classe equivalenti a class_weight="balanced"
n_classes = 2
n_samples = len(df_clean)
counts = df_clean[TARGET].value_counts().sort_index()

class_weights = {cls: n_samples / (n_classes * count) for cls, count in counts.items()}
print("Pesi di classe con class_weight='balanced':")
for cls, w in class_weights.items():
    print(f"  classe {cls}: peso = {w:.4f}")

print(
    "\nEffetto: ogni errore su un campione churn vale ~"
    f"{class_weights[1]:.2f} di più nella loss rispetto a un errore su un campione non-churn."
)

# Esempio di inizializzazione (senza fit)
lr_balanced = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=SEED,
)
lr_balanced

# %% [markdown]
# - La classe **churn (y=1)** riceve peso **~2.45x** maggiore rispetto alla classe
#   non-churn (y=0): il modello penalizzerà di più gli errori sui churner durante il training.
# - Questo è il modo più semplice e sicuro per gestire lo sbilanciamento con sklearn;
#   non richiede librerie aggiuntive e non introduce rischi di leakage.

# %%
# Demo: SMOTE (oversampling sintetico)
# Nota: richiede imbalanced-learn (pip install imbalanced-learn)
# Il codice è mostrato a scopo didattico; verrà usato opzionalmente in Lezione 3.
try:
    from imblearn.over_sampling import SMOTE

    # Costruiamo un mini-dataset dimostrativo (non X_train definitivo)
    _X_demo = df_clean.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    _y_demo = df_clean[TARGET]

    smote = SMOTE(random_state=SEED)
    _X_resampled, _y_resampled = smote.fit_resample(_X_demo, _y_demo)

    print("SMOTE applicato su dati numerici (demo):")
    print(f"  Prima: {_X_demo.shape[0]:,} campioni — churn: {_y_demo.mean():.2%}")
    print(
        f"  Dopo:  {_X_resampled.shape[0]:,} campioni — churn: "
        f"{_y_resampled.mean():.2%}"
    )
    print(
        "\n  IMPORTANTE: nella pratica SMOTE va applicato SOLO su X_train (dopo lo split)."
    )

except ImportError:
    print(
        "imbalanced-learn non installato — installare con: pip install imbalanced-learn\n"
        "Snippet SMOTE mostrato a scopo didattico; non è necessario per proseguire."
    )

# %% [markdown]
# ### Threshold tuning: abbassare la soglia di decisione
#
# Per un classificatore binario, la soglia di default è **0.5**: se $\hat{p} \geq 0.5$
# si predice `churn=1`. Abbassare la soglia aumenta il **recall** (identifichiamo più
# churner) a scapito della **precision** (aumentano i falsi allarmi).
#
# Il threshold ottimale si sceglie sulla **validation set** dopo il training
# (mai sul test set) guardando la curva Precision-Recall o la ROC Curve.
# Questo sarà approfondito nella Lezione 3.

# %% [markdown]
# ## 7. Outlier e range check
#
# Anche se in Lezione 1 non abbiamo osservato outlier gravi, è buona pratica
# quantificarli sistematicamente con il metodo IQR prima del preprocessing.

# %%
from pandas.api.types import is_numeric_dtype

num_cols_raw = [
    c
    for c in df_clean.columns
    if is_numeric_dtype(df_clean[c]) and c != TARGET and c not in NUMERIC_AS_CATEGORICAL
]

# Calcolo soglie IQR per ogni feature numerica
iqr_stats = []
for col in num_cols_raw:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    pct_outliers = n_outliers / len(df_clean)
    iqr_stats.append(
        {
            "feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
        }
    )

df_iqr = pd.DataFrame(iqr_stats).set_index("feature")
print("Outlier per feature (metodo IQR, moltiplicatore 1.5):")
display(
    df_iqr[
        ["Q1", "Q3", "IQR", "lower_fence", "upper_fence", "n_outliers", "pct_outliers"]
    ]
)

# %%
# Boxplot delle feature numeriche chiave per visualizzare gli outlier
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
key_features = [
    "Age",
    "Balance",
    "CreditScore",
    "EstimatedSalary",
    "Tenure",
]
key_features = [c for c in key_features if c in df_clean.columns]

for ax, col in zip(axes.flat, key_features):
    sns.boxplot(y=df_clean[col], ax=ax, color="steelblue", width=0.4)
    ax.set_title(col)
    ax.set_ylabel("")

for ax in axes.flat[len(key_features) :]:
    ax.set_visible(False)

fig.suptitle("Outlier per feature numeriche chiave (metodo IQR)", fontsize=13, y=1.01)
save_current_figure("lesson_02_outlier_boxplot.png")
plt.show()

# %% [markdown]
# - Il metodo IQR identifica alcuni outlier nelle feature numeriche, ma le percentuali
#   sono **contenute** (< 1% per la maggior parte delle colonne).
# - `Balance` ha la distribuzione bimodale già osservata in Lezione 1: il "cluster" a
#   saldo zero non è un outlier, ma una caratteristica strutturale del dataset.
# - **Decisione:** non applichiamo clipping aggressivo; il `RobustScaler` (proposto
#   nella sezione 12) è robusto agli outlier moderati. Per dataset con outlier severi
#   si userebbe clipping al 1°/99° percentile prima dello scaling.

# %% [markdown]
# ## 8. Feature engineering: `balance_is_zero`
#
# In Lezione 1 abbiamo osservato che i clienti con `Balance==0` hanno un churn rate
# **diverso** rispetto a quelli con saldo positivo. Creiamo una feature binaria per
# catturare esplicitamente questa struttura bimodale.

# %%
df_clean = df_clean.copy()
df_clean["balance_is_zero"] = (df_clean["Balance"] == 0).astype(int)

# Verifica: churn rate per balance_is_zero
balance_zero_stats = (
    df_clean.groupby("balance_is_zero")[TARGET]
    .agg(churn_rate="mean", n="count")
    .rename(index={0: "Balance > 0", 1: "Balance == 0"})
)
print("Churn rate per balance_is_zero:")
display(balance_zero_stats)

# Barplot
plt.figure(figsize=(5, 3.5))
_plot_df = balance_zero_stats.reset_index()
ax = sns.barplot(
    data=_plot_df,
    x="balance_is_zero",
    y="churn_rate",
    hue="balance_is_zero",
    palette=["steelblue", "salmon"],
    legend=False,
)
ax.set_title("Churn rate per balance_is_zero")
ax.set_xlabel("Categoria Balance")
ax.set_ylabel("Churn rate")
ax.axhline(
    churn_rate,
    linestyle="--",
    color="gray",
    linewidth=1,
    label=f"Media globale ({churn_rate:.2%})",
)
ax.legend(fontsize=8)
save_current_figure("lesson_02_balance_is_zero_churn_rate.png")
plt.show()

# %% [markdown]
# - **`Balance==0`**: churn rate = **13.82%** (n=3,617 clienti con saldo zero).
# - **`Balance>0`**: churn rate = **24.10%** (n=6,383 clienti con saldo positivo).
# - Il pattern è **confermato**: i clienti con saldo positivo abbandonano a un tasso
#   ~74% più alto rispetto a quelli con saldo zero.
# - La nuova feature `balance_is_zero` permetterà al modello di catturare questa
#   discontinuità che una trasformazione lineare di `Balance` non riuscirebbe a cogliere.
#   `Balance` originale viene comunque mantenuta per preservare l'informazione sulla
#   grandezza del saldo.

# %% [markdown]
# ## 9. Definire feature set e target
#
# Separiamo le feature dal target e cataloghiamo ciascuna per tipo, in preparazione
# alla costruzione della Pipeline.

# %%
# Feature set finale (dopo pulizia + feature engineering)
feature_cols = [c for c in df_clean.columns if c != TARGET]

# Separazione per tipo
num_cols = [
    c
    for c in feature_cols
    if is_numeric_dtype(df_clean[c]) and c not in NUMERIC_AS_CATEGORICAL
]
cat_cols = [
    c
    for c in feature_cols
    if not is_numeric_dtype(df_clean[c]) or c in NUMERIC_AS_CATEGORICAL
]

print("=== Feature set ===")
print(f"\nFeature numeriche ({len(num_cols)}):")
for c in num_cols:
    print(f"  {c}")

print(f"\nFeature categoriche ({len(cat_cols)}):")
for c in cat_cols:
    print(f"  {c}")

print(f"\nTarget: {TARGET}")
print(f"\nX shape: ({len(df_clean)}, {len(feature_cols)})")
print(f"y shape: ({len(df_clean)},)")

X = df_clean[feature_cols]
y = df_clean[TARGET]

# %% [markdown]
# - Il feature set finale comprende **13 feature** (14 colonne totali − 1 target):
#   - **5 numeriche continue**: `CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`.
#   - **8 categoriche** (incluse 5 a dtype numerico ma semanticamente categoriche):
#     `Geography`, `Gender`, `Card Type`, `IsActiveMember`, `HasCrCard`,
#     `NumOfProducts`, `Satisfaction Score`, `balance_is_zero`.
# - La separazione esplicita in `num_cols` e `cat_cols` guiderà la costruzione
#   del `ColumnTransformer` nella Pipeline sklearn.

# %% [markdown]
# ## 10. Split train / validation / test definitivo
#
# Utilizziamo uno split a **tre vie** (60/20/20) stratificato sul target.
#
# **Perché tre split?**
#
# | Split | Scopo | Quando usarlo |
# |-------|-------|---------------|
# | **Train** | Addestrare il modello e fittare il preprocessore | Durante il fit |
# | **Validation** | Scegliere iperparametri e confrontare modelli | Durante il tuning |
# | **Test** | Stima finale della performance (una sola volta) | Alla fine — mai durante il tuning |
#
# **Regola anti-leakage:** il preprocessore (scaler, encoder) viene fittato
# **solo su `X_train`**. Applicarlo a tutto il dataset prima dello split
# "contaminerebbe" il validation e test set con informazioni sul training set.

# %%
from sklearn.model_selection import train_test_split

# Step 1: separare train (60%) da tmp (40%)
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=SEED,
)

# Step 2: dividere tmp in val (20%) e test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp,
    y_tmp,
    test_size=0.50,
    stratify=y_tmp,
    random_state=SEED,
)

print("=== Split train / validation / test ===")
for name, X_split, y_split in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    print(
        f"\n{name:12s}: {X_split.shape[0]:,} esempi ({X_split.shape[0]/len(X)*100:.1f}%)  "
        f"| churn rate = {y_split.mean():.4f}"
    )

print(f"\nTotale: {len(X):,} esempi")

# %% [markdown]
# - Lo split ha prodotto: train **6,000** (60%), val **2,000** (20%), test **2,000** (20%).
# - La stratificazione garantisce che il churn rate sia **~20.38%** in tutti e tre
#   i subset, evitando sbilanciamenti artificiali tra i set.
# - D'ora in poi: il preprocessore sarà fittato **solo su `X_train`** e poi applicato
#   anche su `X_val` e `X_test` con `transform` (mai con `fit` o `fit_transform`).

# %% [markdown]
# ## 11. Encoding delle variabili categoriche
#
# Le variabili categoriche (`Geography`, `Gender`, `Card Type`) devono essere
# convertite in forma numerica. Usiamo **OneHotEncoding** poiché sono nominali
# (non c'è un ordine naturale tra i valori).

# %% [markdown]
# ### OrdinalEncoder vs OneHotEncoder
#
# | Encoder | Quando usarlo | Esempio |
# |---------|--------------|---------|
# | **OrdinalEncoder** | Variabili **ordinali** con ordine naturale | Taglia (S < M < L < XL) |
# | **OneHotEncoder** | Variabili **nominali** senza ordine | Geography (France, Germany, Spain) |
#
# Per `Geography`, `Gender` e `Card Type` non esiste un ordinamento naturale
# corretto: OneHotEncoder è la scelta appropriata.

# %%
from sklearn.preprocessing import OneHotEncoder

# Demo standalone: encoding fit SOLO su X_train
ohe_demo = OneHotEncoder(
    handle_unknown="ignore",  # ignora categorie non viste in training
    sparse_output=False,  # output come array denso
    drop=None,  # manteniamo tutte le categorie (gestita dalla regressione)
)
ohe_demo.fit(X_train[cat_cols])

# Nomi feature generate
ohe_feature_names = ohe_demo.get_feature_names_out(cat_cols).tolist()
print(f"Feature categoriche prima dell'encoding: {cat_cols}")
print(f"\nFeature generate dopo OneHotEncoding ({len(ohe_feature_names)}):")
for f in ohe_feature_names:
    print(f"  {f}")

# Trasformare i tre split
X_train_cat_enc = ohe_demo.transform(X_train[cat_cols])
X_val_cat_enc = ohe_demo.transform(X_val[cat_cols])
X_test_cat_enc = ohe_demo.transform(X_test[cat_cols])

print(f"\nShape categoriche encoded (train): {X_train_cat_enc.shape}")

# %% [markdown]
# - `Geography` (3 valori: France, Germany, Spain) → **3 colonne** dummy.
# - `Gender` (2 valori) → **2 colonne** dummy.
# - `Card Type` (4 valori) → **4 colonne** dummy.
# - `IsActiveMember` (2 valori) → **2 colonne** dummy.
# - `HasCrCard` (2 valori) → **2 colonne** dummy.
# - `NumOfProducts` (4 valori) → **4 colonne** dummy.
# - `Satisfaction Score` (5 valori) → **5 colonne** dummy.
# - `balance_is_zero` (2 valori) → **2 colonne** dummy.
# - Totale: le 8 feature categoriche diventano **24 feature binarie** dopo
#   il OneHotEncoding.
# - L'encoder è stato fittato **solo su `X_train`**: questo garantisce che le
#   categorie di `X_val` e `X_test` siano codificate secondo le stesse regole;
#   se in `X_val`/`X_test` comparisse una categoria non vista,
#   `handle_unknown="ignore"` la tratta come uno zero (nessun segnale).

# %% [markdown]
# ## 12. Feature scaling
#
# I modelli lineari (e molti altri algoritmi) sono sensibili alla scala delle feature:
# una variabile con range [0, 200,000] (come `Balance`) dominerebbe una variabile
# con range [0, 10] (come `Tenure`). Lo scaling normalizza i range.

# %% [markdown]
# ### StandardScaler vs RobustScaler
#
# | Scaler | Formula | Quando usarlo |
# |--------|---------|--------------|
# | **StandardScaler** | $z = \frac{x - \mu}{\sigma}$ | Distribuzioni approssimativamente normali; sensibile agli outlier |
# | **RobustScaler** | $z = \frac{x - Q_2}{Q_3 - Q_1}$ | Distribuzioni con outlier moderati; usa mediana e IQR |
#
# Per questo dataset usiamo **StandardScaler**: le distribuzioni sono
# ragionevolmente simmetriche e gli outlier non sono severi (come visto
# nella sezione 7). In dataset con outlier significativi si preferirebbe
# `RobustScaler`.

# %%
from sklearn.preprocessing import StandardScaler

# Demo standalone: scaling fit SOLO su X_train
scaler_demo = StandardScaler()
scaler_demo.fit(X_train[num_cols])

# Statistiche prima/dopo scaling su X_train
stats_before = X_train[num_cols].agg(["mean", "std"])
X_train_num_scaled = scaler_demo.transform(X_train[num_cols])
stats_after = pd.DataFrame(
    X_train_num_scaled,
    columns=num_cols,
).agg(["mean", "std"])

print("Statistiche PRIMA dello scaling (X_train):")
display(stats_before.T.rename(columns={"mean": "mean_orig", "std": "std_orig"}))

print("\nStatistiche DOPO lo scaling (X_train):")
display(
    stats_after.T.rename(columns={"mean": "mean_scaled", "std": "std_scaled"}).round(4)
)

# %% [markdown]
# - Dopo lo **StandardScaler** applicato su `X_train`, le feature numeriche hanno
#   media ≈ **0** e deviazione standard ≈ **1** sul training set.
# - Verificheremo nella sezione 14 che su `X_val` la media sarà leggermente diversa
#   da 0 e la std leggermente diversa da 1: questo è **corretto** (lo scaler è stato
#   fittato sul train, non sul val) e dimostra l'assenza di leakage.

# %% [markdown]
# ## 13. Pipeline sklearn completa: ColumnTransformer
#
# Combiniamo encoding e scaling in un'unica **Pipeline sklearn** riproducibile.
# Il `ColumnTransformer` applica trasformazioni diverse a colonne diverse,
# gestendo automaticamente l'ordine e il tracciamento dei nomi.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Pipeline per feature numeriche: imputazione (no NaN qui, ma best practice) + scaling
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),  # robusto agli outlier
        ("scaler", StandardScaler()),
    ]
)

# Pipeline per feature categoriche: imputazione (best practice) + OneHotEncoding
cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

# ColumnTransformer: combina le due pipeline
# remainder="drop" rimuove colonne non specificate
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)

# Fit su X_train SOLTANTO
preprocessor.fit(X_train)

# Transform su train, val e test
X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# Feature names dopo il transform
feature_names_out = preprocessor.get_feature_names_out().tolist()

print(f"Shape X_train_proc: {X_train_proc.shape}")
print(f"Shape X_val_proc:   {X_val_proc.shape}")
print(f"Shape X_test_proc:  {X_test_proc.shape}")
print(f"\nNumero feature finali: {len(feature_names_out)}")
print("\nFeature names (prime 15):")
for f in feature_names_out[:15]:
    print(f"  {f}")
if len(feature_names_out) > 15:
    print(f"  ... ({len(feature_names_out) - 15} altre)")

# %%
# Salvataggio della pipeline
preprocessor_path = DATA_OUT_DIR / "lesson_02_preprocessor.pkl"
joblib.dump(preprocessor, preprocessor_path)
print(f"Pipeline salvata in: {preprocessor_path}")

# %% [markdown]
# - La Pipeline finale trasforma le **13 feature** originali in **29 feature**
#   dopo OneHotEncoding: le 5 numeriche continue vengono scalate,
#   mentre le 8 categoriche diventano 24 colonne dummy.
# - Tutti gli step sono concatenati in un oggetto riproducibile e serializzabile:
#   in Lezione 3 basterà caricare il preprocessore con `joblib.load` e chiamare
#   `transform` su nuovi dati senza rieseguire code.
# - L'ordine delle operazioni è garantito dalla Pipeline: imputazione è sempre
#   prima dello scaling, che avviene prima dell'encoding dove applicabile.

# %% [markdown]
# ## 14. Verifica del preprocessing
#
# Prima di salvare i dataset, verifichiamo che il preprocessing sia corretto:
# - Nessun valore mancante nei dataset processati.
# - Shape coerente con le attese.
# - Le distribuzioni di train e validation differiscono leggermente (segnale di
#   assenza di leakage).

# %%
# Check NaN
nan_train = np.isnan(X_train_proc).sum()
nan_val = np.isnan(X_val_proc).sum()
nan_test = np.isnan(X_test_proc).sum()

print("=== Verifica NaN ===")
print(f"NaN in X_train_proc: {nan_train}")
print(f"NaN in X_val_proc:   {nan_val}")
print(f"NaN in X_test_proc:  {nan_test}")

print("\n=== Shape ===")
print(f"X_train_proc: {X_train_proc.shape}")
print(f"X_val_proc:   {X_val_proc.shape}")
print(f"X_test_proc:  {X_test_proc.shape}")

# %%
# Statistiche media/std su train vs val (prime 5 feature numeriche)
# Su train: media ≈ 0, std ≈ 1 (come atteso dallo StandardScaler fittato su train)
# Su val: media leggermente ≠ 0, std leggermente ≠ 1 (corretto! segnale di no leakage)
n_num = len(num_cols)  # numero feature numeriche (scalate)

df_check = pd.DataFrame(
    {
        "feature": feature_names_out[:n_num],
        "train_mean": X_train_proc[:, :n_num].mean(axis=0),
        "train_std": X_train_proc[:, :n_num].std(axis=0),
        "val_mean": X_val_proc[:, :n_num].mean(axis=0),
        "val_std": X_val_proc[:, :n_num].std(axis=0),
    }
).set_index("feature")

print("Statistiche feature numeriche (scalate) — train vs validation:")
display(df_check.round(4))

# %%
# Distribuzione di alcune feature numeriche: train vs val (dopo scaling)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
demo_indices = [0, 1, 2]  # prime 3 feature numeriche

for ax, idx in zip(axes, demo_indices):
    feat_name = feature_names_out[idx]
    ax.hist(
        X_train_proc[:, idx],
        bins=30,
        alpha=0.6,
        label="Train",
        color="steelblue",
        density=True,
    )
    ax.hist(
        X_val_proc[:, idx],
        bins=30,
        alpha=0.6,
        label="Val",
        color="salmon",
        density=True,
    )
    ax.set_title(feat_name, fontsize=9)
    ax.set_xlabel("Valore scalato")
    ax.set_ylabel("Densità")
    ax.legend(fontsize=8)

fig.suptitle("Distribuzioni feature (scalate) — Train vs Validation", fontsize=12)
save_current_figure("lesson_02_train_val_distributions.png")
plt.show()

# %% [markdown]
# - **NaN: 0** in tutti e tre i dataset processati — il preprocessing è completo.
# - **Shape:** coerente con le attese (6,000 / 2,000 / 2,000 × 19 feature).
# - Le feature numeriche sul **train set** hanno media ≈ **0** e std ≈ **1**: lo
#   `StandardScaler` ha funzionato correttamente.
# - Sul **validation set** le medie e std sono leggermente diverse da 0/1: questo è
#   **corretto e atteso** — conferma che lo scaler è stato fittato solo sul train,
#   senza contaminare il validation.
# - Le distribuzioni dei due istogrammi si sovrappongono largamente (stesso dataset
#   campionato casualmente), con piccole differenze casuali dovute alla randomizzazione
#   dello split.

# %% [markdown]
# ## 15. Salvataggio dataset modellabile
#
# Salviamo i dataset processati in formato **Parquet**: compatto, tipizzato e
# facilmente leggibile da `pandas` nelle lezioni successive.

# %%
# Convertire in DataFrame con nomi colonne per leggibilità
X_train_df = pd.DataFrame(X_train_proc, columns=feature_names_out)
X_val_df = pd.DataFrame(X_val_proc, columns=feature_names_out)
X_test_df = pd.DataFrame(X_test_proc, columns=feature_names_out)

y_train_df = y_train.reset_index(drop=True)
y_val_df = y_val.reset_index(drop=True)
y_test_df = y_test.reset_index(drop=True)

# Salvataggio
artifacts = {
    "lesson_02_X_train.parquet": X_train_df,
    "lesson_02_X_val.parquet": X_val_df,
    "lesson_02_X_test.parquet": X_test_df,
    "lesson_02_y_train.parquet": y_train_df.to_frame(),
    "lesson_02_y_val.parquet": y_val_df.to_frame(),
    "lesson_02_y_test.parquet": y_test_df.to_frame(),
}

for fname, df_out in artifacts.items():
    path = DATA_OUT_DIR / fname
    df_out.to_parquet(path, index=False)
    print(f"Salvato: {path}  (shape: {df_out.shape})")

# Salvataggio feature names come JSON
feature_names_path = DATA_OUT_DIR / "lesson_02_feature_names.json"
with open(feature_names_path, "w", encoding="utf-8") as fh:
    json.dump(feature_names_out, fh, indent=2)
print(f"\nFeature names salvati in: {feature_names_path}")

# %% [markdown]
# - I **6 dataset** (X e y per train/val/test) sono salvati in `outputs/data/` in
#   formato Parquet.
# - La **pipeline di preprocessing** è salvata in `outputs/data/lesson_02_preprocessor.pkl`.
# - I **nomi delle feature** sono salvati in `outputs/data/lesson_02_feature_names.json`.
# - In Lezione 3 basterà caricare questi file con `pd.read_parquet` e `joblib.load`
#   per iniziare immediatamente a costruire e confrontare modelli.

# %% [markdown]
# ## 16. Domande guidate
#
# **1. Perché un modello con accuracy del 79.62% non è necessariamente buono
# per il nostro problema?**
#
# Un `DummyClassifier` che predice sempre "non churn" (y=0) ottiene accuracy =
# **79.62%** semplicemente riflettendo la proporzione della classe maggioritaria.
# Il suo recall sulla classe churn è **0%**: non identifica un solo churner.
# Per problemi di churn, la metrica rilevante è il **recall** (quanti churner
# reali identifichiamo) o la **ROC-AUC** (capacità discriminativa). In Lezione 1
# il modello baseline con recall=21.1% era già migliore del "dummy" nonostante
# una accuracy non lontana.
#
# ---
#
# **2. Perché il fit dello scaler deve avvenire solo su `X_train` e non su
# tutto il dataset?**
#
# Se fittassimo lo scaler sull'intero dataset (train + val + test), la media e
# la deviazione standard dello scaler sarebbero calcolate anche sui dati di
# validation e test. In questo modo il preprocessing "vedrebbe" dati che il
# modello non ha ancoraosservato durante il training, creando un **data leakage
# implicito**. Le performance stimate sul validation/test sarebbero leggermente
# ottimistiche e non rappresentative della performance su dati davvero nuovi.
# Come verificato nella sezione 14, su `X_val` la media e la std **non** sono
# esattamente 0 e 1: questo è il segnale corretto che lo scaler è stato fittato
# solo su train.
#
# ---
#
# **3. Qual è la differenza tra `class_weight="balanced"` e SMOTE?
# Quando preferiresti l'uno o l'altro?**
#
# `class_weight="balanced"` modifica la **funzione di loss** del modello,
# assegnando peso **~2.45x** maggiore agli errori sulla classe minoritaria
# (churn) durante il training. Non crea nuovi campioni. SMOTE invece crea
# **campioni sintetici** interpolando tra esempi reali della classe minoritaria,
# aumentando fisicamente il dataset. `class_weight="balanced"` è più semplice,
# non rischioso e applicabile a qualsiasi modello sklearn che lo supporta.
# SMOTE è preferibile quando il dataset è molto piccolo o lo sbilanciamento
# molto severo (es. 1:100). In entrambi i casi, SMOTE va applicato **solo dopo
# lo split**, mai prima.
#
# ---
#
# **4. Perché abbiamo creato la feature `balance_is_zero` invece di usare
# `Balance` direttamente?**
#
# Il churn rate dei clienti con `Balance==0` (**13.82%**) è significativamente
# diverso da quello con `Balance>0` (**24.10%**). Questa discontinuità non può
# essere catturata da una trasformazione lineare del valore di `Balance`: un
# modello lineare tratterà `Balance=0` e `Balance=1` come quasi uguali.
# La feature binaria `balance_is_zero` permette al modello di apprendere
# esplicitamente questa soglia senza richiedere termini di interazione complessi.
# `Balance` originale viene comunque mantenuta per preservare l'informazione
# sulla grandezza del saldo.
#
# ---
#
# **5. Che differenza c'è tra validation set e test set? Quando usi l'uno
# e quando l'altro?**
#
# Il **validation set** viene usato durante lo sviluppo del modello: per scegliere
# iperparametri, confrontare architetture diverse, e decidere la soglia di
# classificazione ottimale. Viene "consumato" ogni volta che consultiamo i suoi
# risultati per prendere una decisione. Il **test set** è una stima finale della
# performance su dati "mai visti": deve essere usato **una sola volta**, alla fine,
# per riportare il risultato definitivo. Usare il test set durante il tuning
# introdurrebbe un **selection bias**: il modello sarebbe implicitamente ottimizzato
# anche per il test, rendendo la stima di performance ottimistica.

# %% [markdown]
# ## 17. Riepilogo
#
# ### Operazioni completate e artefatti prodotti
#
# | Operazione | Risultato |
# |-----------|-----------|
# | Rimozione variabili non predittive e leakage | `df_clean`: 14 colonne (−4 da df originale) |
# | Analisi imbalance e metriche | Churn = 20.38%; accuracy da sola è fuorviante |
# | Strategie sbilanciamento | 5 strategie presentate; `class_weight` e SMOTE dimostrate |
# | Outlier check (IQR) | Outlier contenuti (<1%); nessun clipping necessario |
# | Feature engineering | `balance_is_zero`: churn rate 13.82% (saldo zero) vs 24.10% (saldo positivo) |
# | Split 60/20/20 stratificato | Train: 6,000 \| Val: 2,000 \| Test: 2,000 — churn rate costante (~20.38%) |
# | OneHotEncoding (fit su train) | 8 feature categoriche → 24 colonne dummy |
# | StandardScaler (fit su train) | 5 feature numeriche continue normalizzate (media≈0, std≈1 su train) |
# | Pipeline ColumnTransformer | 13 feature input → 29 feature output |
# | Verifica preprocessing | NaN: 0; shape corretti; no leakage confermato |
# | Salvataggio dataset modellabile | 6 file Parquet + pipeline PKL + feature names JSON |
#
# ### Confronto con la Lezione 1
#
# | Aspetto | Lezione 1 | Lezione 2 |
# |---------|-----------|-----------|
# | Feature set | 14 colonne (con dummy one-hot rapido) | 13 feature → 29 dopo Pipeline |
# | Preprocessing | Minimale (imputer + scaler + get_dummies su tutto il dataset) | Pipeline completa, anti-leakage |
# | Split | 80/20 (senza validation set) | 60/20/20 stratificato |
# | Dataset modellabile | Solo in memoria | Persistito in `outputs/data/` |
#
# ### Prossima lezione (Lezione 3)
#
# Nella **Lezione 3** utilizzeremo il dataset modellabile salvato per:
# - Costruire e confrontare modelli di classificazione: Logistic Regression,
#   Decision Tree, Random Forest.
# - Calcolare e interpretare le metriche: Accuracy, Precision, Recall, F1, ROC-AUC.
# - Visualizzare la Confusion Matrix e la ROC Curve.
# - Discutere il trade-off Precision–Recall nel contesto del churn bancario.
