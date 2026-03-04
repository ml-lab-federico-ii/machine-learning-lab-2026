# %% [markdown]
# # Lezione 01 — Introduction to Machine Learning and the Churn Prediction Problem
#
# ## Learning objectives
#
# - Formalizzare il churn come problema di classificazione binaria.
# - Comprendere il contesto di business di una banca retail.
# - Ispezionare in modo rigoroso un dataset reale di clienti.
# - Eseguire una EDA di base orientata alla modellazione futura.
# - Identificare pattern e segnali preliminari associati al churn.

# %% [markdown]
# ## Setup
#
# In questo blocco impostiamo dipendenze, percorsi di progetto e cartelle
# di output. Usiamo `pathlib` per garantire portabilità e separiamo gli
# artefatti in `outputs/`.

# %%
from __future__ import annotations

from pathlib import Path
import json
import random
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.2f}".format)

def resolve_project_root() -> Path:
    """Resolve repository root regardless of the current working directory."""
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
CONFIG_DIR = OUTPUTS_DIR / "config"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

for directory in [
    OUTPUTS_DIR,
    FIGURES_DIR,
    CONFIG_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    SUBMISSIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_current_figure(filename: str) -> None:
    """Save the current matplotlib figure into outputs/figures."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=120, bbox_inches="tight")


def load_dataset_from_archive(
    archive_path: Path,
    filename_patterns: tuple[str, ...] = (
        "train",
        "test",
        "churn",
        "customer",
        "record",
        "label",
    ),
) -> pd.DataFrame:
    """Load the most suitable churn CSV from a ZIP archive using schema checks."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archivio dati non trovato: {archive_path}")

    expected_columns = {
        "creditscore",
        "geography",
        "gender",
        "age",
        "tenure",
        "balance",
        "numofproducts",
        "hascrcard",
        "isactivemember",
        "estimatedsalary",
    }
    target_candidates = {"exited", "churn", "target", "label"}

    with ZipFile(archive_path) as zip_file:
        members = [member for member in zip_file.namelist() if not member.endswith("/")]
        print("Membri archivio:")
        for member in members:
            print(f"- {member}")

        csv_members = [member for member in members if member.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(
                "Nessun file CSV trovato nell'archivio. "
                f"Membri disponibili: {members}"
            )

        pattern_matches = [
            member
            for member in csv_members
            if any(pattern in member.lower() for pattern in filename_patterns)
        ]
        candidates = pattern_matches if pattern_matches else csv_members

        best_member: str | None = None
        best_score = -1

        for member in candidates:
            with zip_file.open(member) as file_obj:
                preview_df = pd.read_csv(file_obj, nrows=200)

            normalized_columns = {col.strip().lower() for col in preview_df.columns}
            schema_score = len(expected_columns.intersection(normalized_columns))
            has_target = bool(target_candidates.intersection(normalized_columns))
            total_score = schema_score + (3 if has_target else 0)

            if has_target and total_score > best_score:
                best_member = member
                best_score = total_score

        if best_member is None:
            raise ValueError(
                "Nessun CSV adatto trovato nell'archivio dopo validazione schema. "
                f"CSV candidati: {candidates}. "
                "Attese colonne simili a feature churn e una colonna target "
                "(Exited/Churn/Target/Label)."
            )

        print(f"CSV selezionato: {best_member}")
        with zip_file.open(best_member) as file_obj:
            return pd.read_csv(file_obj)


print(f"Dataset archive path: {DATA_ARCHIVE_PATH}")
print(f"Figure directory: {FIGURES_DIR}")

# %% [markdown]
# Le cartelle di output sono disponibili e il setup è deterministico
# (`SEED = 42`). Questo rende la lezione riproducibile e pronta alla
# produzione di artefatti (grafici e file di sintesi).

# %% [markdown]
# ## Introduction to churn prediction
#
# Prima di leggere i dati, esplicitiamo il problema: prevedere se un
# cliente uscirà dalla banca (`Exited = 1`) oppure resterà (`Exited = 0`).

# %%
problem_definition = pd.DataFrame(
    {
        "elemento": ["unità statistica", "target", "tipo problema", "orizzonte"],
        "definizione": [
            "cliente bancario",
            "Exited (0 = resta, 1 = abbandona)",
            "classificazione binaria supervisionata",
            "decisioni di retention nel breve periodo",
        ],
    }
)
problem_definition

# %% [markdown]
# La formulazione collega chiaramente osservazione, target e obiettivo
# analitico. Il problema è supervisionato e binario: nella lezione corrente
# ci concentriamo su qualità dati e segnali esplorativi, senza ancora
# costruire modelli predittivi.

# %% [markdown]
# ## Business framing of churn
#
# Quantifichiamo un esempio semplice di costo atteso del churn per rendere
# operativo il legame tra metrica statistica e impatto economico.

# %%
business_frame = pd.DataFrame(
    {
        "voce": [
            "clienti totali",
            "valore medio annuo per cliente (€)",
            "costo retention per cliente a rischio (€)",
        ],
        "valore": [10000, 1200, 120],
    }
)

base_churn_rate = 0.2038
expected_lost_value = int(10000 * base_churn_rate * 1200)
retention_budget = int(10000 * base_churn_rate * 120)

business_frame, expected_lost_value, retention_budget

# %% [markdown]
# Con un tasso di churn di riferimento pari a circa 20.38%, il valore annuo
# potenzialmente perso è nell'ordine di €2.445.600, mentre un budget di
# retention uniforme sarebbe circa €244.560. Questo giustifica analisi
# granulari per concentrare gli interventi sui segmenti più a rischio.

# %% [markdown]
# ## Dataset overview
#
# Carichiamo il dataset e produciamo una prima scheda tecnica con forma,
# tipi e qualità generale.

# %%
df = load_dataset_from_archive(DATA_ARCHIVE_PATH)

dataset_overview = {
    "n_righe": int(df.shape[0]),
    "n_colonne": int(df.shape[1]),
    "target": "Exited",
    "missing_values_totali": int(df.isna().sum().sum()),
}

pd.Series(dataset_overview)

# %% [markdown]
# Il dataset contiene 10.000 osservazioni e 18 variabili, con target
# `Exited` e nessun valore mancante. La dimensione è adeguata per una prima
# EDA robusta e per confronti segmentati su più sottogruppi.

# %% [markdown]
# ## Data loading
#
# Salviamo una copia raw del dataset negli artefatti del progetto per
# preservare tracciabilità e riproducibilità.

# %%
raw_copy_path = OUTPUTS_DIR / "data" / "lesson_01_raw.csv"
df.to_csv(raw_copy_path, index=False)
raw_copy_path

# %% [markdown]
# La copia raw è stata salvata in `outputs/data/lesson_01_raw.csv`. Questo
# separa in modo esplicito il dato sorgente dai passaggi di analisi e riduce
# il rischio di sovrascrivere accidentalmente il file originale.

# %% [markdown]
# ## First data inspection
#
# Ispezioniamo colonne, tipi e un campione di record per valutare la natura
# delle variabili disponibili.

# %%
inspection_table = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "missing": df.isna().sum(),
    }
).sort_values("dtype")

inspection_table.head(18)

# %% [markdown]
# Le variabili miste numeriche/categoriche sono coerenti con un problema di
# churn retail. `Geography`, `Gender` e `Card Type` risultano categoriche;
# `Age`, `Balance`, `CreditScore` e `EstimatedSalary` sono quantitative e
# candidati naturali per analisi di relazione con il target.

# %% [markdown]
# ## Basic exploratory analysis
#
# Partiamo dalle statistiche descrittive numeriche per valutare ordini di
# grandezza, dispersione e possibili asimmetrie.

# %%
numeric_cols = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "Satisfaction Score",
    "Point Earned",
]

desc = df[numeric_cols].describe().T

desc

# %% [markdown]
# Si osserva una forte eterogeneità di scala: `Balance` ha mediana
# ~97.199 e massimo ~250.898, mentre `Tenure` è compresa tra 0 e 10.
# L'età ha mediana 37 anni (IQR circa 12), suggerendo una popolazione
# prevalentemente adulta ma con coda fino a 92 anni.

# %% [markdown]
# Analizziamo la distribuzione dell'età e differenziamo per stato di churn.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data=df, x="Age", bins=30, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribuzione di Age")
axes[0].set_xlabel("Età")
axes[0].set_ylabel("Frequenza")

sns.boxplot(data=df, x="Exited", y="Age", ax=axes[1])
axes[1].set_title("Age per classe di Exited")
axes[1].set_xlabel("Exited")
axes[1].set_ylabel("Età")

save_current_figure("lesson_01_age_distribution_live.png")
plt.show()

# %% [markdown]
# La mediana dell'età passa da circa 36 anni (`Exited=0`) a 45 anni
# (`Exited=1`), con differenza di 9 anni. Il segnale è consistente e indica
# che l'età potrebbe contribuire in modo rilevante alla separazione delle
# classi in una fase modellistica successiva.

# %% [markdown]
# Verifichiamo ora il comportamento del `CreditScore` rispetto al churn.

# %%
# TODO(LIVE): implement this step during lecture
# - Costruire i due grafici (histplot + boxplot) per CreditScore.
# - Salvare la figura in outputs/figures.

# %% [markdown]
# L'effetto è più debole rispetto all'età: la mediana scende da circa 653
# (`Exited=0`) a 646 (`Exited=1`), differenza di 7 punti. Il `CreditScore`
# da solo non sembra discriminante forte, ma può aggiungere informazione in
# combinazione con altre variabili.

# %% [markdown]
# Confrontiamo i tassi di churn per geografia, genere e numero di prodotti.

# %%
# TODO(LIVE): implement this step during lecture
# - Calcolare churn_by_geography, churn_by_gender, churn_by_products.
# - Visualizzare i tre barplot in una figura 1x3.
# - Mostrare le tabelle aggregate finali.

churn_by_geography = (
    df.groupby("Geography", as_index=False)["Exited"].mean()
    .assign(churn_rate_pct=lambda x: x["Exited"] * 100)
    .sort_values("churn_rate_pct", ascending=False)
)

churn_by_gender = (
    df.groupby("Gender", as_index=False)["Exited"].mean()
    .assign(churn_rate_pct=lambda x: x["Exited"] * 100)
    .sort_values("churn_rate_pct", ascending=False)
)

churn_by_products = (
    df.groupby("NumOfProducts", as_index=False)["Exited"].mean()
    .assign(churn_rate_pct=lambda x: x["Exited"] * 100)
    .sort_values("NumOfProducts")
)

churn_by_geography, churn_by_gender, churn_by_products

# %% [markdown]
# Emergono differenze marcate: Germania ~32.44% contro Francia ~16.17% e
# Spagna ~16.67%; donne ~25.07% contro uomini ~16.47%. Sul numero prodotti,
# il tasso è ~27.71% con 1 prodotto, ~7.60% con 2 prodotti, e cresce in modo
# estremo oltre 2 prodotti (segmenti però piccoli e da verificare in seguito).

# %% [markdown]
# Analizziamo la relazione tra attività cliente, reclami e churn.

# %%
# TODO(LIVE): implement this step during lecture
# - Costruire due tabelle aggregate per IsActiveMember e Complain.
# - Disegnare due barplot affiancati con etichette complete.

churn_by_activity = (
    df.groupby("IsActiveMember", as_index=False)["Exited"].mean()
    .assign(churn_rate_pct=lambda x: x["Exited"] * 100)
)

churn_by_complain = (
    df.groupby("Complain", as_index=False)["Exited"].mean()
    .assign(churn_rate_pct=lambda x: x["Exited"] * 100)
)

churn_by_activity, churn_by_complain

# %% [markdown]
# I clienti non attivi hanno churn ~26.87% contro ~14.27% degli attivi
# (gap ~12.6 punti). La variabile `Complain` mostra invece un pattern quasi
# deterministico: ~99.51% di churn quando `Complain=1` contro ~0.05% quando
# `Complain=0`, segnale molto forte da trattare con attenzione per possibili
# effetti temporali o leakage informativo.

# %% [markdown]
# Esploriamo `Balance` con due prospettive: distribuzione e tasso di churn
# tra saldo nullo e positivo.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data=df, x="Balance", bins=35, kde=True, ax=axes[0], color="#C44E52")
axes[0].set_title("Distribuzione di Balance")
axes[0].set_xlabel("Balance")
axes[0].set_ylabel("Frequenza")

sns.boxplot(data=df, x="Exited", y="Balance", ax=axes[1])
axes[1].set_title("Balance per classe di Exited")
axes[1].set_xlabel("Exited")
axes[1].set_ylabel("Balance")

save_current_figure("lesson_01_balance_distribution_live.png")
plt.show()

zero_balance_share = (df["Balance"] == 0).mean() * 100
churn_zero_balance = df.loc[df["Balance"] == 0, "Exited"].mean() * 100
churn_pos_balance = df.loc[df["Balance"] > 0, "Exited"].mean() * 100

zero_balance_share, churn_zero_balance, churn_pos_balance

# %% [markdown]
# Circa il 36.17% dei clienti ha `Balance=0`. In questo gruppo il churn è
# ~13.82%, mentre supera il 24.10% tra chi ha saldo positivo. La relazione
# suggerisce che la sola presenza di saldo non coincide con maggiore fedeltà,
# ma potrebbe riflettere segmenti con diversa dinamica di utilizzo del conto.

# %% [markdown]
# ## Target variable exploration
#
# Misuriamo prima distribuzione del target, poi relazioni numeriche globali.

# %%
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Exited", ax=ax)
ax.set_title("Distribuzione del target Exited")
ax.set_xlabel("Exited")
ax.set_ylabel("Numero clienti")
save_current_figure("lesson_01_target_count_live.png")
plt.show()

target_dist = df["Exited"].value_counts().sort_index()
target_pct = (df["Exited"].value_counts(normalize=True).sort_index() * 100).round(2)
target_dist, target_pct

# %% [markdown]
# Il target è sbilanciato ma non estremo: classe 0 pari a 7.962 casi
# (79.62%) e classe 1 pari a 2.038 casi (20.38%). Questo livello di
# imbalance va monitorato nelle fasi successive, soprattutto nella scelta
# delle metriche di valutazione.

# %% [markdown]
# Studiamo la correlazione lineare tra variabili numeriche e target.

# %%
# TODO(LIVE): implement this step during lecture
# - Calcolare corr_matrix su numeric_cols + Exited.
# - Visualizzare una heatmap annotata.
# - Estrarre corr_with_target in ordine decrescente.

corr_cols = numeric_cols + ["Exited"]
corr_matrix = df[corr_cols].corr(numeric_only=True)
corr_with_target = corr_matrix["Exited"].sort_values(ascending=False)
corr_with_target

# %% [markdown]
# La correlazione più alta con `Exited` è `Age` (~0.29), seguita da
# `Balance` (~0.12); le altre sono vicine a zero in valore assoluto.
# Questo indica che il churn non è spiegabile da una singola relazione
# lineare forte e richiederà combinazioni non banali di feature.

# %% [markdown]
# Prima prospettiva alternativa: tasso di churn per fasce di età.

# %%
# TODO(LIVE): implement this step during lecture
# - Costruire age_group con pd.cut.
# - Calcolare churn_by_age_group e visualizzare un barplot.

age_bins = [17, 30, 40, 50, 60, 100]
age_labels = ["18-30", "31-40", "41-50", "51-60", "61+"]
df["age_group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)

churn_by_age_group = (
    df.groupby("age_group", observed=False)["Exited"]
    .mean()
    .mul(100)
    .reset_index(name="churn_rate_pct")
)

churn_by_age_group

# %% [markdown]
# Il churn cresce con l'età fino alla fascia 51-60 (~56.21%), molto sopra
# 18-30 (~7.52%) e 31-40 (~12.11%). La fascia 61+ scende a ~24.78%, segnale
# di possibile non linearità che rende utile una segmentazione per classi
# d'età nelle analisi successive.

# %% [markdown]
# Seconda prospettiva alternativa: combinazione `Geography` × attività.

# %%
# TODO(LIVE): implement this step during lecture
# - Calcolare risk_cube e creare una pivot Geography x IsActiveMember.
# - Disegnare una heatmap dei tassi di churn.

risk_cube = (
    df.groupby(["Geography", "IsActiveMember"])["Exited"]
    .mean()
    .mul(100)
    .reset_index(name="churn_rate_pct")
)

risk_pivot = risk_cube.pivot(
    index="Geography", columns="IsActiveMember", values="churn_rate_pct"
)
risk_pivot

# %% [markdown]
# La combinazione mostra il segmento più critico in Germania e non attivo,
# mentre i gruppi attivi in Francia e Spagna risultano più stabili. Questo
# conferma che variabili comportamentali (`IsActiveMember`) e contesto
# geografico vanno lette congiuntamente, non in modo isolato.

# %% [markdown]
# ## Domande guidate
#
# **1) Qual è il livello di imbalance del target?**
#
# La classe `Exited=1` pesa circa il 20.38% (2.038 su 10.000), mentre
# `Exited=0` pesa il 79.62%.
#
# **2) Quale differenza geografica emerge con maggiore intensità?**
#
# La Germania ha churn ~32.44%, circa il doppio rispetto a Francia (~16.17%)
# e Spagna (~16.67%).
#
# **3) Quale segnale comportamentale è più informativo?**
#
# `IsActiveMember` mostra un gap di ~12.6 punti (26.87% vs 14.27%);
# `Complain` è quasi deterministico (~99.51% vs ~0.05%).
#
# **4) Che ruolo ha l'età nel rischio di uscita?**
#
# Il churn cresce con l'età fino a 51-60 anni (~56.21%) e poi si riduce nella
# fascia 61+ (~24.78%), indicando una relazione non lineare.
#
# **5) Quale implicazione operativa emerge dal saldo (`Balance`)?**
#
# I clienti con `Balance=0` hanno churn più basso (~13.82%) rispetto a chi ha
# saldo positivo (~24.10%), quindi la sola liquidità sul conto non è una
# garanzia di fedeltà.

# %% [markdown]
# ## Summary
#
# - Il dataset è completo (10.000 × 18) e senza missing values.
# - Il churn medio è ~20.38%, con sbilanciamento moderato della classe target.
# - Segnali forti: età, attività del cliente, geografia e reclami.
# - Segnali deboli isolati: credit score e molte variabili numeriche lineari.
# - Le analisi bivariate mostrano pattern utili per impostare il lavoro
#   successivo su preprocessing e modellazione, senza anticipare ancora
#   la costruzione dei modelli.

# %%
summary_payload = {
    "lesson": "01",
    "topic": "Introduction to Machine Learning and the Churn Prediction Problem",
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "target_rate_exited_pct": float((df["Exited"].mean() * 100).round(2)),
    "missing_total": int(df.isna().sum().sum()),
    "top_correlations_with_target": {
        key: float(round(value, 4))
        for key, value in corr_with_target.to_dict().items()
    },
}

summary_path = CONFIG_DIR / "lesson_01_eda_summary.json"
summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
summary_path

# %% [markdown]
# Il file di sintesi è stato salvato in `outputs/config/lesson_01_eda_summary.json`.
# In questo modo, i risultati principali della lezione restano riutilizzabili
# anche fuori dal notebook (reporting, dashboard o confronto tra iterazioni).
