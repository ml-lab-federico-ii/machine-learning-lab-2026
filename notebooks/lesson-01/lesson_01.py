# %% [markdown]
# # Machine Learning per l’Analisi Finanziaria
#
# ## Lezione 01 — Il churn come problema di classificazione: formulazione e analisi esplorativa
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
# - Tradurre un problema di business (churn) in un problema di classificazione binaria.
# - Identificare target e feature, distinguendo variabili predittive da identificativi.
# - Condurre un’EDA essenziale: target, statistiche descrittive, sbilanciamento, prime ipotesi.
# - Riconoscere variabili non predittive che si comportano come rumore casuale.
# - Riconoscere segnali di **data leakage** e variabili “proxy” del target.
# - Formulare ipotesi su quali variabili possano essere predittive del churn, basandosi su evidenze quantitative.

# %% [markdown]
# ## Outline
#
# - Caricamento dati e contratto del dataset
# - Definire il target e quantificare lo sbilanciamento
# - Qualità del dato: tipi, range, controlli rapidi
# - Feature numeriche: distribuzioni e differenze tra classi
# - Variabili a basso potere predittivo (Rumore statistico)
# - Feature categoriche: churn rate per gruppi
# - Segmentazioni e interazioni: pattern di churn per gruppi e fasce
# - Attenzione al leakage: il caso `Complain`
# - Riepilogo

# %% [markdown]
# ## Nota su GenAI e Code Assistants (pratica)
#
# In questo laboratorio useremo strumenti di assistenza al codice per:
# - accelerare operazioni ripetitive (EDA standard, plotting);
# - migliorare la leggibilità (refactoring, typing, docstring);
# - ridurre errori “meccanici” (es. pipeline sklearn).
#
# Buone pratiche:
# - specificare sempre obiettivo, vincoli (no leakage) e formato desiderato;
# - chiedere esplicitamente *cosa stampare/visualizzare* per verificare i risultati;
# - verificare sempre l’output con calcoli riproducibili (seed fissato).

# %% [markdown]
# ## Setup
#
# Definiamo dipendenze, percorsi e cartelle di output.

# %%
from __future__ import annotations

import random
from pathlib import Path
from zipfile import ZipFile

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
pd.set_option("display.float_format", "{:.2f}".format)


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

for _dir in [OUTPUTS_DIR, FIGURES_DIR]:
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

    La funzione:
    - controlla che l'archivio esista;
    - lista i membri;
    - sceglie il primo CSV compatibile con i pattern;
    - legge il CSV direttamente dallo ZIP.
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
# ## Caricamento dati e contratto del dataset
#
# Iniziamo validando il “contratto” del dataset: dove si trova, quali colonne contiene,
# che tipi di variabili abbiamo e quale colonna può essere il target.

# %%
df = load_dataset_from_archive(DATA_ARCHIVE_PATH)

print("\nShape:", df.shape)
display(df.head())

print("\nDtypes:")
display(df.dtypes)

print("\nDescrittive (numeriche):")
display(df.describe(include=[np.number]).T)

# %% [markdown]
# - Il dataset contiene **10,000 righe** e **18 colonne**.
# - La colonna `Exited` è un candidato naturale per il target: è binaria (0/1) e coerente con l’idea di churn.
# - Sono presenti anche colonne identificative (`RowNumber`, `CustomerId`, `Surname`) che non dovrebbero essere usate per “prevedere”, ma possono servire per controlli e join (se esistessero altre tabelle).

# %% [markdown]
# ## Definire il target e quantificare lo sbilanciamento
#
# In problemi di churn è tipico che la classe positiva (chi abbandona) sia minoritaria.
# Quantifichiamo quindi la prevalenza di `Exited=1`.

# %%
TARGET = "Exited"

target_counts = df[TARGET].value_counts().sort_index()
target_rates = df[TARGET].value_counts(normalize=True).sort_index()

print("Target counts:\n", target_counts)
print("\nTarget rates:\n", target_rates)

plt.figure(figsize=(5, 3))
ax = sns.barplot(x=target_counts.index.astype(str), y=target_counts.values)
ax.set_title("Distribuzione del target (Exited)")
ax.set_xlabel("Exited")
ax.set_ylabel("Numero clienti")

save_current_figure("lesson_01_target_distribution.png")
plt.show()

# %% [markdown]
# - `Exited=1` (churn) vale **2,038 / 10,000 = 20.38%**: la classe positiva è minoritaria ma non estremamente rara.
# - `Exited=0` vale **7,962 / 10,000 = 79.62%**.
# - Lo sbilanciamento suggerisce di usare metriche robuste (es. ROC-AUC) e split stratificati.

# %% [markdown]
# ## Qualità del dato: tipi, range, controlli rapidi
#
# Prima di qualsiasi modello, controlliamo: valori mancanti, duplicati, e la presenza di colonne “sospette”
# (es. variabili che potrebbero essere note solo dopo l’evento di churn).

# %%
missing_rate = df.isna().mean().sort_values(ascending=False)
duplicated_rows = df.duplicated().sum()
duplicated_customer_id = (
    df["CustomerId"].duplicated().sum() if "CustomerId" in df.columns else None
)

print("Duplicated rows:", duplicated_rows)
print("Duplicated CustomerId:", duplicated_customer_id)

print("\nTop missing (dovrebbe essere tutto 0):")
display(missing_rate.head(12))

id_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
print("\nColonne identificative candidate:", id_cols)

# %% [markdown]
# - Non risultano valori mancanti: il tasso di missing è **0** per tutte le colonne.
# - Non risultano righe duplicate (duplicati = **0**).
# - `CustomerId` è un identificativo: anche se unico, non è informazione “causale” e in genere va escluso dal modeling.

# %% [markdown]
# ## Feature numeriche: distribuzioni e differenze tra classi
#
# Confrontiamo alcune feature numeriche tra `Exited=0` e `Exited=1`.
# L’obiettivo non è “dimostrare causalità”, ma generare ipotesi utili per la modellazione.

# %%
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != TARGET]

age_quantiles = df.groupby(TARGET)["Age"].quantile([0.25, 0.5, 0.75]).unstack()

balance_zero_share = (df["Balance"] == 0).mean()
churn_rate_balance_zero = df.loc[df["Balance"] == 0, TARGET].mean()
churn_rate_balance_positive = df.loc[df["Balance"] > 0, TARGET].mean()

print("Age quantiles by target:\n", age_quantiles)
print("\nShare Balance==0:", balance_zero_share)
print("Churn rate Balance==0:", churn_rate_balance_zero)
print("Churn rate Balance>0:", churn_rate_balance_positive)

plt.figure(figsize=(6, 3.5))
ax = sns.boxplot(data=df, x=TARGET, y="Age")
ax.set_title("Età per classe (Exited)")
ax.set_xlabel("Exited")
ax.set_ylabel("Age")

save_current_figure("lesson_01_age_by_target.png")
plt.show()

plt.figure(figsize=(6, 3.5))
ax = sns.boxplot(data=df, x=TARGET, y="Balance")
ax.set_title("Balance per classe (Exited)")
ax.set_xlabel("Exited")
ax.set_ylabel("Balance")

save_current_figure("lesson_01_balance_by_target.png")
plt.show()

# %% [markdown]
# - L’età è più alta nei clienti churn: la **mediana** passa da **36** (`Exited=0`) a **45** (`Exited=1`).
# - La feature `Balance` ha un comportamento interessante: **36.17%** dei clienti ha `Balance==0`.
# - Il churn rate per `Balance==0` è **13.82%**, mentre per `Balance>0` è **24.10%**: il saldo non nullo sembra associato a maggior churn (ipotesi da validare).

# %% [markdown]
# ## Variabili a basso potere predittivo (Rumore statistico)
#
# Non tutte le feature raccolte offrono reale discriminazione rispetto al target. A volte, alcune variabili rappresentano puro “rumore” statistico, o sono generate in modo sostanzialmente uniforme e scorrelato dalla propensione al churn.
#
# Esaminiamo `EstimatedSalary` e `Satisfaction Score`.

# %%
salary_churn_rates = df.groupby(TARGET)["EstimatedSalary"].describe()
print("Descrittive di EstimatedSalary divise per classe target:")
display(salary_churn_rates)

# Calcoliamo il churn per punteggio di soddisfazione (1 a 5)
if "Satisfaction Score" in df.columns:
    sat_score_churn = (
        df.groupby("Satisfaction Score")[TARGET]
        .mean()
        .rename("churn_rate")
        .to_frame()
        .join(df["Satisfaction Score"].value_counts().rename("n"))
        .sort_index()
    )
    print("\nChurn rate per Satisfaction Score:")
    display(sat_score_churn)

plt.figure(figsize=(6, 3.5))
ax = sns.histplot(
    data=df,
    x="EstimatedSalary",
    hue=TARGET,
    kde=True,
    common_norm=False,
    stat="density",
)
ax.set_title("Distribuzione Densità: EstimatedSalary per Classe")
ax.set_xlabel("Salario Stimato")
ax.set_ylabel("Densità")

save_current_figure("lesson_01_salary_distribution.png")
plt.show()

# %% [markdown]
# - `EstimatedSalary` è uniformemente distribuita su tutto il dominio (da ~€11 a ~€200,000) e la sua distribuzione è praticamente **identica** nelle due popolazioni (churn e non churn). Il salario stimato, da solo, non appare un buon predittore.
# - Analogamente, il `Satisfaction Score` ha un churn rate che fluttua in modo del tutto casuale attorno al tasso medio del **20%** senza mostrare un trend monotono sensato. Variabili simili introducono unicamente rumore nei modelli e potrebbero richiedere logiche di binning o semplici scarti algoritmici.

# %% [markdown]
# ## Feature categoriche: churn rate per gruppi
#
# Alcune variabili descrivono gruppi (es. area geografica) e possono mostrare tassi di churn differenti.
# Calcoliamo il churn rate per gruppo per `Geography`, `Gender` e `Card Type`.

# %%
from pandas.api.types import is_numeric_dtype

cat_cols: list[str] = []
for c in df.columns:
    if c == TARGET:
        continue
    if c in {"RowNumber", "CustomerId", "Surname"}:
        continue
    if not is_numeric_dtype(df[c]):
        cat_cols.append(c)

print("Colonne categoriche candidate:", cat_cols)

for c in ["Geography", "Gender", "Card Type"]:
    if c in df.columns:
        out = (
            df.groupby(c)[TARGET]
            .mean()
            .rename("churn_rate")
            .to_frame()
            .join(df[c].value_counts().rename("n"))
            .sort_values("churn_rate", ascending=False)
        )
        print(f"\n{c} (n, churn_rate):")
        display(out)

geo_rates = df.groupby("Geography")[TARGET].mean().sort_values(ascending=False)
plt.figure(figsize=(6, 3))
ax = sns.barplot(x=geo_rates.index, y=geo_rates.values)
ax.set_title("Churn rate per Geography")
ax.set_xlabel("Geography")
ax.set_ylabel("Churn rate")

save_current_figure("lesson_01_churn_rate_by_geography.png")
plt.show()

# %% [markdown]
# - Il churn rate varia per gruppo: ad esempio **Germany = 32.44%** (n=2,509) è più alto di **France = 16.17%** (n=5,014) e **Spain = 16.67%** (n=2,477).
# - Anche `Gender` mostra differenze: **Female = 25.07%** (n=4,543) vs **Male = 16.47%** (n=5,457).
# - Queste differenze possono riflettere pattern reali, ma anche variabili confondenti: è utile modellare e poi interpretare con cautela.

# %% [markdown]
# ## Segmentazioni e interazioni: pattern di churn per gruppi e fasce
#
# Estendiamo l’EDA con analisi tipiche da data scientist:
# - variabili binarie (engagement/attività);
# - segmentazioni per fasce (es. età);
# - interazioni tra gruppi (es. `Geography × Gender`).

# %% [markdown]
# ### Feature binarie: attività e carta di credito
#
# Calcoliamo churn rate e numerosità per `IsActiveMember` e `HasCrCard`.

# %%
for c in ["IsActiveMember", "HasCrCard"]:
    out = (
        df.groupby(c)[TARGET]
        .mean()
        .rename("churn_rate")
        .to_frame()
        .join(df[c].value_counts().rename("n"))
        .sort_index()
    )
    print(f"\n{c} (n, churn_rate):")
    display(out)

active_rates = df.groupby("IsActiveMember")[TARGET].mean().sort_index()
plt.figure(figsize=(5, 3))
ax = sns.barplot(x=active_rates.index.astype(str), y=active_rates.values)
ax.set_title("Churn rate per IsActiveMember")
ax.set_xlabel("IsActiveMember")
ax.set_ylabel("Churn rate")

save_current_figure("lesson_01_churn_rate_by_is_active.png")
plt.show()

# %% [markdown]
# - `IsActiveMember=0` ha churn rate **26.87%** (n=4,849), mentre `IsActiveMember=1` ha churn rate **14.27%** (n=5,151): la differenza è ampia e coerente con l’idea che l’inattività sia un segnale di rischio.
# - `HasCrCard` mostra invece un effetto molto più debole: **20.81%** per `HasCrCard=0` (n=2,945) vs **20.20%** per `HasCrCard=1` (n=7,055).

# %% [markdown]
# ### Numero di prodotti: non-linearità e cautela sui piccoli gruppi
#
# `NumOfProducts` può catturare sia engagement sia complessità del rapporto.
# Verifichiamo churn rate per numero di prodotti, tenendo presente la dimensione dei gruppi.

# %%
num_products = (
    df.groupby("NumOfProducts")[TARGET]
    .mean()
    .rename("churn_rate")
    .to_frame()
    .join(df["NumOfProducts"].value_counts().rename("n"))
    .sort_index()
)
display(num_products)

plt.figure(figsize=(6, 3))
ax = sns.barplot(
    x=num_products.index.astype(str),
    y=num_products["churn_rate"].values,
)
ax.set_title("Churn rate per NumOfProducts")
ax.set_xlabel("NumOfProducts")
ax.set_ylabel("Churn rate")

save_current_figure("lesson_01_churn_rate_by_num_products.png")
plt.show()

# %% [markdown]
# - Il pattern è fortemente non lineare: `NumOfProducts=2` ha churn rate **7.60%** (n=4,590), molto più basso di `NumOfProducts=1` (**27.71%**, n=5,084).
# - I gruppi `NumOfProducts=3` (**82.71%**, n=266) e `NumOfProducts=4` (**100%**, n=60) sono estremi ma anche molto piccoli: è essenziale interpretare questi numeri con cautela (varianza alta) e verificare se esiste una spiegazione di business o di quality.

# %% [markdown]
# ### Segmentazione per fasce d’età
#
# L’età mostrava già differenze nella boxplot. Ora costruiamo fasce (bin) e calcoliamo il churn rate per ciascuna fascia.

# %%
age_bins = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 65, 100], right=True)
age_out = (
    df.groupby(age_bins, observed=True)[TARGET]
    .agg(["mean", "count"])
    .rename(columns={"mean": "churn_rate", "count": "n"})
)
display(age_out)

plt.figure(figsize=(7, 3))
ax = sns.barplot(x=age_out.index.astype(str), y=age_out["churn_rate"].values)
ax.set_title("Churn rate per fasce di età")
ax.set_xlabel("Fascia di età")
ax.set_ylabel("Churn rate")
ax.tick_params(axis="x", rotation=20)

save_current_figure("lesson_01_churn_rate_by_age_bins.png")
plt.show()

# %% [markdown]
# - Il churn aumenta nettamente passando alle fasce centrali: la fascia **(25, 35]** ha churn rate **8.50%** (n=3,542), mentre **(45, 55]** arriva a **50.57%** (n=1,311).
# - La fascia **(55, 65]** resta molto alta (**48.32%**, n=536).
# - La fascia **(65, 100]** scende a **13.26%** (n=264): il gruppo è piccolo e può riflettere selezione/censura o caratteristiche specifiche del campione.

# %% [markdown]
# ### Interazioni: Geography × Gender
#
# Le differenze per `Geography` e per `Gender` possono combinarsi. Calcoliamo quindi il churn rate per coppie (`Geography`, `Gender`).

# %%
geo_gender_rate = df.pivot_table(
    index="Geography",
    columns="Gender",
    values=TARGET,
    aggfunc="mean",
)
geo_gender_n = df.pivot_table(
    index="Geography",
    columns="Gender",
    values=TARGET,
    aggfunc="size",
)

print("Churn rate Geography x Gender:")
display(geo_gender_rate)

print("\nCounts Geography x Gender:")
display(geo_gender_n)

plt.figure(figsize=(5.5, 3.2))
ax = sns.heatmap(geo_gender_rate, annot=True, fmt=".3f", cmap="Blues")
ax.set_title("Churn rate: Geography × Gender")

save_current_figure("lesson_01_geo_gender_heatmap.png")
plt.show()

# %% [markdown]
# - In **Germany** il churn è alto per entrambi i generi: **37.55%** (Female, n=1,193) e **27.81%** (Male, n=1,316).
# - In **France** la differenza di genere è marcata: **20.35%** (Female, n=2,261) vs **12.75%** (Male, n=2,753).
# - Questa analisi suggerisce che alcune feature possano interagire; nelle lezioni successive queste interazioni verranno gestite in modo più sistematico.

# %% [markdown]
# ## Attenzione al leakage: il caso `Complain`
#
# In un progetto reale è cruciale verificare che le feature siano *disponibili al momento della previsione*.
# Una variabile come `Complain` (reclamo) potrebbe essere registrata **dopo** segnali di churn, diventando una proxy quasi deterministica.
#
# Quantifichiamo l’associazione tra `Complain` e `Exited`.

# %%
ct = pd.crosstab(df["Complain"], df[TARGET], normalize="index")
ct_counts = pd.crosstab(df["Complain"], df[TARGET])

print("P(Exited | Complain):")
display(ct)

print("\nCounts:")
display(ct_counts)

corr_num = (
    df.select_dtypes(include=[np.number])
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

print("\nTop correlazioni (numeriche) con Exited:")
display(corr_num.head(10))

# %% [markdown]
# - Se `Complain=0`, la probabilità di churn è **0.050%** (4 su 7,956).
# - Se `Complain=1`, la probabilità di churn è **99.51%** (2,034 su 2,044).
# - La correlazione `Complain`–`Exited` è **0.996**, enormemente più alta delle altre: questo è un segnale forte di variabile “proxy”/leaky.
# - In assenza di documentazione temporale, è prudente trattare `Complain` come sospetta e chiarire se sia disponibile *prima* del churn; in caso contrario, va esclusa dalle feature utilizzabili.

# %% [markdown]
# ## Riepilogo
#
# - Il churn è formulabile come classificazione binaria con target `Exited` (classe positiva = **20.38%**).
# - Non emergono problemi di missingness o duplicati nel dataset (tassi missing = **0**).
# - Alcune feature mostrano differenze marcate tra classi (es. mediana `Age` **36** vs **45**, churn rate più alto in Germany).
# - Menzione del rumore: variabili come `EstimatedSalary` e `Satisfaction Score` figurano come rumorose e casuali rispetto al target.
# - Le segmentazioni mettono in evidenza pattern forti: `IsActiveMember=0` ha churn rate **26.87%** vs **14.27%** per `IsActiveMember=1`; la fascia età **(45, 55]** arriva a **50.57%**.
# - `NumOfProducts` mostra un andamento non lineare (es. **7.60%** per 2 prodotti vs **27.71%** per 1 prodotto), con gruppi estremi ma piccoli per 3–4 prodotti.
# - `Complain` è un fortissimo sospetto di leakage/proxy: `P(Exited=1 | Complain=1)` è **99.51%**.
#
# Prossimo passo (Lezione 02): costruire preprocessing e pipeline in modo sistematico e non-leaky.
