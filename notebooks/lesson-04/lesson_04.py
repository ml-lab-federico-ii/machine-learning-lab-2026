# %% [markdown]
# # Machine Learning per l'Analisi Finanziaria
#
# ## Lezione 04 — Feature Importance e Interpretabilità
#
# **Authors:**
# - Enrico Huber
# - Pietro Soglia
#
# **Last updated:** 2026-05-01
#
# ## Obiettivi di apprendimento
#
# - Comprendere e confrontare diverse metriche di **feature importance** (Gini,
#   Permutation, SHAP), valutandone limiti e affidabilità sui dati reali.
# - Introdurre **SHAP** come framework rigoroso (basato sulla teoria dei giochi)
#   per la spiegabilità locale e globale di un modello.
# - Costruire un **profilo del cliente a rischio churn** basato su evidenze
#   quantitative, comunicabile al management.
# - Sperimentare il **re-training con feature selection** data-driven e valutare
#   il trade-off semplicità–performance.
# - Collegare i risultati alle **implicazioni regolamentari** (GDPR, EBA) sulla
#   spiegabilità nel contesto del credito.
#
# ## Approccio metodologico
#
# Questa lezione segue il ciclo di ragionamento di un Data Scientist reale:
#
# > **Ipotesi → Esperimento → Osservazione → Decisione → Prossima ipotesi**
#
# Non conosciamo a priori cosa troveremo. Ogni decisione è **data-driven**:
# prima osserviamo i risultati, poi decidiamo il passo successivo.
# Conserviamo anche le sperimentazioni che non portano dove atteso —
# dagli errori si impara.

# %% [markdown]
# ## Outline
#
# ### BLOCCO A — Fondamenta e riapertura dell'indagine
# 1. Setup, percorsi e costanti
# 2. Recap Lezione 3 e domanda motrice
#
# ### BLOCCO B — Feature Importance: la prima approssimazione
# 3. Importanza basata su impurity (Gini) — ciò che già abbiamo
# 4. Importanza per permutazione — un secondo parere
# 5. Stabilità dell'importanza — quanto è rumorosa?
#
# ### BLOCCO C — SHAP: spiegabilità rigorosa
# 6. Introduzione a SHAP — da Shapley al Machine Learning
# 7. Calcolo SHAP values sul modello RF
# 8. SHAP Summary Plot — importanza globale riveduta
# 9. SHAP Bar Plot e tabella comparativa dei 3 ranking
#
# ### BLOCCO D — Interpretabilità locale: capire le singole predizioni
# 10. Scegliere clienti rappresentativi da spiegare
# 11. SHAP Waterfall Plot — anatomia di una predizione
# 12. Costruzione del "profilo cliente a rischio"
#
# ### BLOCCO E — Feature selection e re-training (sperimentazione)
# 13. Ipotesi: un modello più semplice generalizza meglio?
# 14. Esperimento: re-training con top-K feature
# 15. Analisi dell'errore: cosa perde il modello ridotto?
#
# ### BLOCCO F — Contesto regolatorio e implicazioni di business
# 16. Feature sensibili: Geography e Gender nel modello
# 17. Cenni regolatori — GDPR e spiegabilità nel credito
#
# ### BLOCCO G — Chiusura
# 18. Riepilogo del percorso e artefatti prodotti
# 19. Domande guidate
# 20. Bridge verso la Lezione 5

# %% [markdown]
# ---
# ## BLOCCO A — Fondamenta e riapertura dell'indagine
#
# ---
#
# ## 1. Setup, percorsi e costanti
#
# Importiamo le dipendenze, carichiamo il modello e i dati prodotti nelle
# lezioni precedenti. L'obiettivo è ripartire esattamente da dove ci siamo
# fermati alla fine della Lezione 3.

# %%
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover

    def display(x):  # type: ignore
        """Fallback display per esecuzione fuori da Jupyter."""
        print(x)


warnings.filterwarnings("ignore")

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
DATA_OUT_DIR = ROOT / "outputs" / "data"
FIGURES_DIR = ROOT / "outputs" / "figures"

for _dir in [DATA_OUT_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_current_figure(filename: str) -> None:
    """Salva la figura corrente in outputs/figures/."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=120, bbox_inches="tight")
    print(f"Figura salvata: {FIGURES_DIR / filename}")


def load_preprocessed_split(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Carica X e y da parquet per uno split (train, val o test).

    Parameters
    ----------
    prefix : str
        Uno tra 'train', 'val', 'test'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coppia (X, y) come array numpy.
    """
    x_path = DATA_OUT_DIR / f"lesson_02_X_{prefix}.parquet"
    y_path = DATA_OUT_DIR / f"lesson_02_y_{prefix}.parquet"
    for p in [x_path, y_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"File parquet non trovato: {p}\n"
                "Esegui prima il notebook della Lezione 2 per generare il dataset."
            )
    X = pd.read_parquet(x_path).values
    y = pd.read_parquet(y_path).squeeze().values
    return X, y


# %%
# --- Caricamento dati preprocessati dalla Lezione 2 ---
X_train, y_train = load_preprocessed_split("train")
X_val, y_val = load_preprocessed_split("val")
X_test, y_test = load_preprocessed_split("test")

with open(DATA_OUT_DIR / "lesson_02_feature_names.json") as fh:
    feature_names = json.load(fh)

print(f"Feature totali: {len(feature_names)}")
print(f"\n{'Split':<8} {'Shape X':<18} {'N positivi':<14} {'Churn rate'}")
print("-" * 56)
for name, X, y in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]:
    print(f"{name:<8} {str(X.shape):<18} {y.sum():<14.0f} {y.mean():.4f}")

# %%
# --- Caricamento modello RF dalla Lezione 3 ---
model_path = DATA_OUT_DIR / "lesson_03_best_model.pkl"
if not model_path.exists():
    raise FileNotFoundError(
        f"Modello non trovato: {model_path}\n"
        "Esegui prima il notebook della Lezione 3."
    )

model = joblib.load(model_path)
print(f"Modello caricato: {type(model).__name__}")
print(f"  n_estimators = {model.n_estimators}")
print(f"  class_weight = {model.class_weight}")
print(f"  random_state = {model.random_state}")
print(f"  n_features_in_ = {model.n_features_in_}")

# %%
# --- Caricamento metriche baseline dalla Lezione 3 ---
with open(DATA_OUT_DIR / "lesson_03_metrics.json") as fh:
    baseline_metrics = json.load(fh)

print("Metriche baseline (Lezione 3):")
for split in ["train", "val", "test"]:
    m = baseline_metrics[split]
    print(
        f"  {split:<6} → AUC={m['roc_auc']:.4f}, F1={m['f1']:.4f}, "
        f"Recall={m['recall']:.4f}, Precision={m['precision']:.4f}"
    )

# %% [markdown]
# **Verifica completata.** Abbiamo a disposizione:
#
# - Il dataset preprocessato (30 feature, 3 split stratificati)
# - Il modello Random Forest addestrato (n=200, class_weight='balanced')
# - Le metriche di riferimento: **AUC 0.8723 (val)**, **AUC 0.8575 (test)**
#
# Tutti gli ingredienti per iniziare l'indagine sull'interpretabilità.

# %% [markdown]
# ---
# ## 2. Recap Lezione 3 e domanda motrice
#
# Nella Lezione 3 abbiamo percorso un cammino di sperimentazione che ci ha
# portato a scegliere il Random Forest come modello candidato per la challenge:
#
# | Step | Esperimento | Insight chiave |
# |------|-------------|----------------|
# | 1 | DummyClassifier | Accuracy 79.6%, recall 0% → paradosso accuracy |
# | 2 | LR naïve | Recall basso → il modello ignora la maggior parte dei churner |
# | 3 | LR + class_weight | Recall ~78% → gestire lo sbilanciamento è essenziale |
# | 4 | SMOTE vs class_weight | Praticamente identici → class_weight per semplicità |
# | 5 | DT max_depth=None | Overfitting severo (AUC train 1.0, val basso) |
# | 6 | DT curva depth | Ottimo a depth=6, poi gap train-val si apre |
# | 7 | RF n=200 | **AUC val 0.872** — miglior modello |
# | 8 | Test set (una volta) | AUC 0.858 → generalizzazione confermata |
#
# ### La domanda aperta
#
# Sappiamo che il modello **funziona** (AUC 0.858 su dati mai visti).
# Ma in ambito finanziario un modello che funziona non basta. Dobbiamo
# rispondere a domande critiche:
#
# - **Perché** un cliente viene classificato come churner?
# - Quale combinazione di feature spinge la probabilità verso 1?
# - Il modello usa informazioni potenzialmente **discriminatorie** (Geography, Gender)?
# - Possiamo **spiegare** la decisione a un cliente che chiede motivazioni?
#
# > **Domanda motrice:** il modello funziona — ma *perché* prende le
# > decisioni che prende? Possiamo fidarci delle sue "ragioni"?

# %% [markdown]
# ---
# ## BLOCCO B — Feature Importance: la prima approssimazione
#
# ---
#
# ## 3. Importanza basata su impurity (Gini) — ciò che già abbiamo
#
# Il Random Forest di sklearn calcola automaticamente una misura di importanza
# basata sulla **riduzione media dell'impurità** (Mean Decrease in Impurity, MDI)
# operata da ciascuna feature negli split degli alberi.
#
# È una prima approssimazione, ma ha limiti noti:
# - **Favorisce feature ad alta cardinalità** e variabili continue
# - **Non considera la performance sul validation set** — usa solo i dati di
#   training (potenzialmente overfit)
# - **Non riflette l'utilità marginale** della feature (se due feature sono
#   correlate, l'importanza si "spalma" tra loro)
#
# Guardiamo cosa dice e poi mettiamola alla prova.

# %%
# --- Gini importance dal modello RF ---
gini_importances = model.feature_importances_
gini_order = np.argsort(gini_importances)[::-1]

# Mostriamo le top-15
top_k = 15
print(f"Top-{top_k} feature per importanza Gini (MDI):\n")
print(f"{'Rank':<6} {'Feature':<35} {'Importanza'}")
print("-" * 55)
for rank, idx in enumerate(gini_order[:top_k], 1):
    print(f"{rank:<6} {feature_names[idx]:<35} {gini_importances[idx]:.4f}")

# %%
fig, ax = plt.subplots(figsize=(10, 7))
top_indices = gini_order[:top_k]
ax.barh(
    range(top_k),
    gini_importances[top_indices][::-1],
    color="steelblue",
    edgecolor="white",
)
ax.set_yticks(range(top_k))
ax.set_yticklabels([feature_names[i] for i in top_indices][::-1])
ax.set_xlabel("Importanza Gini (MDI)")
ax.set_title("Feature Importance — Gini (Mean Decrease in Impurity)")
save_current_figure("lesson_04_gini_importance.png")
plt.show()

# %% [markdown]
# **Osservazioni sulla Gini importance:**
#
# - `num__Age` domina nettamente — coerente con quanto osservato nelle lezioni
#   precedenti (l'età è il predittore più forte del churn).
# - Le feature **numeriche** (`CreditScore`, `Balance`, `EstimatedSalary`,
#   `Point Earned`) occupano le prime posizioni — questo è in parte un artefatto
#   del bias della Gini importance verso variabili continue.
# - Le feature categoriche one-hot (`Geography`, `NumOfProducts`, `IsActiveMember`)
#   sono frammentate: ogni dummy prende una fetta piccola dell'importanza
#   originaria della variabile.
# - `num__Tenure` ha importanza bassa: già nella Lezione 1 avevamo osservato
#   una correlazione quasi nulla con il target.
#
# **Domanda:** questa classifica è affidabile? O è distorta dal bias noto?
# Servono un **secondo parere** e un metodo che valuti l'importanza
# *sulla performance effettiva*.

# %% [markdown]
# ---
# ## 4. Importanza per permutazione — un secondo parere
#
# La **permutation importance** misura quanto peggiora la performance del modello
# quando una feature viene "distrutta" (shufflata casualmente). A differenza
# della Gini importance:
#
# - Usa il **validation set** → non è influenzata dall'overfitting
# - È **model-agnostic** → funziona con qualsiasi modello
# - Misura il contributo **reale** alla metrica target (AUC nel nostro caso)
#
# Ipotesi: ci aspettiamo conferme per le top feature (Age), ma possibili
# sorprese per le feature categoriche.

# %%
# --- Permutation importance su validation set ---
perm_result = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=30,
    random_state=SEED,
    scoring="roc_auc",
)

perm_importances = perm_result.importances_mean
perm_std = perm_result.importances_std
perm_order = np.argsort(perm_importances)[::-1]

print(f"Top-{top_k} feature per Permutation Importance (AUC drop):\n")
print(f"{'Rank':<6} {'Feature':<35} {'ΔAUC media':<14} {'± std'}")
print("-" * 65)
for rank, idx in enumerate(perm_order[:top_k], 1):
    print(
        f"{rank:<6} {feature_names[idx]:<35} "
        f"{perm_importances[idx]:.4f}       ±{perm_std[idx]:.4f}"
    )

# %%
fig, ax = plt.subplots(figsize=(10, 7))
top_perm_indices = perm_order[:top_k]
ax.barh(
    range(top_k),
    perm_importances[top_perm_indices][::-1],
    xerr=perm_std[top_perm_indices][::-1],
    color="darkorange",
    edgecolor="white",
    capsize=3,
)
ax.set_yticks(range(top_k))
ax.set_yticklabels([feature_names[i] for i in top_perm_indices][::-1])
ax.set_xlabel("Diminuzione media ROC-AUC (Permutation Importance)")
ax.set_title("Feature Importance — Permutation (30 ripetizioni, validation set)")
save_current_figure("lesson_04_permutation_importance.png")
plt.show()

# %% [markdown]
# **Osservazioni sulla Permutation Importance:**
#
# - `num__Age` si conferma la feature più importante — se la distruggiamo,
#   il modello perde molta capacità discriminativa.
# - Le feature categoriche (e.g., `IsActiveMember`, `NumOfProducts`) possono
#   emergere con importanza aggregata più alta rispetto alla Gini, dove ogni
#   dummy veniva contata separatamente.
# - Le barre d'errore ci dicono **quanto è stabile** la misura: feature con
#   barre grandi non hanno un ranking affidabile.
#
# Mettiamo ora i due metodi a confronto diretto.

# %% [markdown]
# ---
# ## 5. Stabilità dell'importanza — quanto è rumorosa?
#
# Confrontiamo Gini e Permutation importance side-by-side per le top feature.
# L'obiettivo è verificare la **convergenza** (i due metodi concordano?) e
# la **stabilità** (le barre d'errore della permutation sono piccole?).

# %%
# --- Confronto side-by-side: Gini vs Permutation (top-15 per permutation) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Gini — ordinato per permutation importance per confronto diretto
ax = axes[0]
ax.barh(
    range(top_k),
    gini_importances[top_perm_indices][::-1],
    color="steelblue",
    edgecolor="white",
)
ax.set_yticks(range(top_k))
ax.set_yticklabels([feature_names[i] for i in top_perm_indices][::-1])
ax.set_xlabel("Gini Importance")
ax.set_title("Gini (MDI)")

# Permutation
ax = axes[1]
ax.barh(
    range(top_k),
    perm_importances[top_perm_indices][::-1],
    xerr=perm_std[top_perm_indices][::-1],
    color="darkorange",
    edgecolor="white",
    capsize=3,
)
ax.set_yticks(range(top_k))
ax.set_yticklabels([feature_names[i] for i in top_perm_indices][::-1])
ax.set_xlabel("ΔAUC (Permutation)")
ax.set_title("Permutation (val set)")

plt.suptitle("Confronto Feature Importance: Gini vs Permutation", fontsize=13)
save_current_figure("lesson_04_gini_vs_permutation.png")
plt.show()

# %%
# --- Tabella comparativa ranking Gini vs Permutation ---
print(f"\n{'Feature':<35} {'Rank Gini':<12} {'Rank Perm':<12} {'Δ Rank'}")
print("-" * 68)

gini_ranks = {idx: rank for rank, idx in enumerate(gini_order, 1)}
perm_ranks = {idx: rank for rank, idx in enumerate(perm_order, 1)}

for idx in perm_order[:top_k]:
    delta = gini_ranks[idx] - perm_ranks[idx]
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    print(
        f"{feature_names[idx]:<35} {gini_ranks[idx]:<12} "
        f"{perm_ranks[idx]:<12} {delta:+d} {arrow}"
    )

# %% [markdown]
# **Analisi della convergenza:**
#
# - Le top-3 feature (Age, e probabilmente NumOfProducts, IsActiveMember) tendono
#   a essere confermate da entrambi i metodi — buon segnale di robustezza.
# - Alcune feature numeriche (e.g., `EstimatedSalary`, `CreditScore`) possono
#   essere **sopravvalutate** dalla Gini: la permutation importance rivela che
#   distruggerle non peggiora significativamente l'AUC.
# - Feature con **Δ Rank grande** meritano cautela: il loro contributo reale
#   è ambiguo tra i due metodi.
#
# **Decisione:** la permutation importance è più affidabile per le decisioni
# di business (usa il val set, è model-agnostic). Ma per un verdetto definitivo
# servono i **SHAP values** — l'unico metodo con fondamenti teorici rigorosi.

# %% [markdown]
# ---
# ## BLOCCO C — SHAP: spiegabilità rigorosa
#
# ---
#
# ## 6. Introduzione a SHAP — da Shapley al Machine Learning
#
# ### Motivazione
#
# Gini e Permutation importance rispondono alla domanda *"quale feature è
# importante in media?"*. Ma non ci dicono:
# - Per **questo specifico cliente**, cosa ha pesato di più?
# - In che **direzione** agisce ogni feature (verso churn o verso retention)?
# - Quanto è **l'effetto marginale** di una feature, considerando tutte le
#   possibili combinazioni con le altre?
#
# ### Valori di Shapley (teoria dei giochi)
#
# Nella teoria dei giochi cooperativi, il **valore di Shapley** di un giocatore
# $i$ in un gioco con funzione di valore $v$ è:
#
# $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \, (|N|-|S|-1)!}{|N|!}
# \left[ v(S \cup \{i\}) - v(S) \right]$$
#
# dove $N$ è l'insieme di tutti i giocatori e $S$ un sottoinsieme che non
# include $i$.
#
# In SHAP, i "giocatori" sono le **feature** e il "gioco" è la **predizione**
# del modello per una singola osservazione. Il valore $\phi_i$ quantifica il
# contributo marginale della feature $i$ alla predizione, mediato su tutte
# le possibili coalizioni di feature.
#
# ### Proprietà chiave
#
# 1. **Efficienza:** $\sum_i \phi_i = f(x) - E[f(x)]$ — i contributi sommano
#    esattamente alla differenza tra la predizione e la media.
# 2. **Simmetria:** se due feature contribuiscono ugualmente in ogni coalizione,
#    hanno lo stesso SHAP value.
# 3. **Additività:** per modelli additivi, SHAP si decompone linearmente.
# 4. **Null player:** una feature che non contribuisce mai ha SHAP = 0.
#
# SHAP è l'**unico** metodo di attribuzione che soddisfa simultaneamente tutte
# queste proprietà — è matematicamente "equo".
#
# ### TreeExplainer
#
# Per i modelli tree-based (RF, GBM, XGBoost), l'algoritmo **TreeSHAP**
# calcola i valori esatti in tempo polinomiale $O(TLD^2)$ anziché esponenziale,
# dove $T$ = numero alberi, $L$ = foglie, $D$ = profondità. Questo ci permette
# di calcolare SHAP values su migliaia di osservazioni in tempi ragionevoli.

# %% [markdown]
# ---
# ## 7. Calcolo SHAP values sul modello RF
#
# Usiamo `TreeExplainer` (esatto per Random Forest) su un campione di
# osservazioni dal training set. Campioniamo per tenere i tempi di calcolo
# ragionevoli (~1000 osservazioni), mantenendo la rappresentatività.

# %%
# --- Campionamento e calcolo SHAP values ---
N_SHAP_SAMPLES = 1000

# Campione stratificato dal training set
rng = np.random.default_rng(SEED)
idx_pos = np.where(y_train == 1)[0]
idx_neg = np.where(y_train == 0)[0]

# Manteniamo la proporzione originale (~20% positivi)
n_pos = int(N_SHAP_SAMPLES * y_train.mean())
n_neg = N_SHAP_SAMPLES - n_pos

sample_idx = np.concatenate(
    [
        rng.choice(idx_pos, size=n_pos, replace=False),
        rng.choice(idx_neg, size=n_neg, replace=False),
    ]
)
rng.shuffle(sample_idx)

X_shap_sample = X_train[sample_idx]
y_shap_sample = y_train[sample_idx]

print(f"Campione SHAP: {X_shap_sample.shape[0]} osservazioni")
print(f"  Classe 1 (churn): {y_shap_sample.sum():.0f} " f"({y_shap_sample.mean():.2%})")
print(f"  Classe 0 (non-churn): {(1 - y_shap_sample).sum():.0f}")

# %%
# --- TreeExplainer: calcolo valori SHAP ---
explainer = shap.TreeExplainer(model)
shap_values_obj = explainer(X_shap_sample)

# Per RF con TreeExplainer, shap_values_obj.values ha shape
# (n_samples, n_features) per ciascuna classe.
# Lavoriamo con la classe 1 (churn) — il nostro target.
# Con la nuova API SHAP, l'oggetto Explanation ha già la struttura corretta.
print(f"\nShape shap_values: {shap_values_obj.values.shape}")
print(f"Base values shape: {shap_values_obj.base_values.shape}")

# Se il modello ha output per 2 classi, prendiamo classe 1
if shap_values_obj.values.ndim == 3:
    # shape (n_samples, n_features, n_classes) → prendiamo classe 1
    shap_values_array = shap_values_obj.values[:, :, 1]
    base_value = float(shap_values_obj.base_values[0, 1])
else:
    shap_values_array = shap_values_obj.values
    base_value = float(shap_values_obj.base_values[0])

print(f"Shape SHAP values (classe churn): {shap_values_array.shape}")
print(f"Base value (prob media churn): {base_value:.4f}")

# %% [markdown]
# **Nota tecnica:** Il `base_value` rappresenta la predizione media del modello
# (la probabilità di churn senza informazione su alcuna feature). I SHAP values
# di ciascuna osservazione sommano alla differenza tra la predizione individuale
# e questo valore base — verificabile grazie alla proprietà di *efficienza*.

# %%
# --- Verifica proprietà di efficienza ---
# Per un campione di osservazioni, verifichiamo che base_value + sum(SHAP) ≈ pred
sample_preds = model.predict_proba(X_shap_sample)[:, 1]
reconstructed = base_value + shap_values_array.sum(axis=1)

max_error = np.abs(sample_preds - reconstructed).max()
mean_error = np.abs(sample_preds - reconstructed).mean()
print(f"Verifica efficienza SHAP:")
print(f"  Errore massimo:  {max_error:.2e}")
print(f"  Errore medio:    {mean_error:.2e}")
print(f"  → {'✓ Proprietà verificata' if max_error < 1e-6 else '✗ Errore inatteso!'}")

# %% [markdown]
# La proprietà di efficienza è confermata: i SHAP values sommano esattamente
# alla predizione individuale. Possiamo procedere con l'analisi globale.

# %% [markdown]
# ---
# ## 8. SHAP Summary Plot — importanza globale riveduta
#
# Il summary plot SHAP mostra simultaneamente:
# - **L'importanza** di ciascuna feature (posizione verticale = ranking)
# - **La distribuzione dei contributi** (dispersione orizzontale)
# - **La direzione dell'effetto** (colore = valore della feature,
#   posizione orizzontale = effetto sulla predizione)

# %%
# --- SHAP Summary Plot ---
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values_array,
    X_shap_sample,
    feature_names=feature_names,
    show=False,
    max_display=15,
)
plt.title("SHAP Summary Plot — Contributi alla probabilità di Churn")
save_current_figure("lesson_04_shap_summary.png")
plt.show()

# %% [markdown]
# **Interpretazione del SHAP Summary Plot:**
#
# - **`num__Age`**: il colore rivela la direzione — valori alti (rosso) spingono
#   verso il churn (SHAP positivo), valori bassi (blu) proteggono. L'età è il
#   predittore più influente, confermando Gini e Permutation.
# - **`cat__IsActiveMember_1`**: essere membro attivo (valore alto = 1) ha
#   effetto protettivo forte (SHAP negativo). Coerente con la correlazione
#   negativa osservata nella Lezione 1.
# - **`num__Balance`**: la relazione non è monotona — sia valori molto alti
#   che molto bassi possono contribuire al churn in modi diversi.
# - **Feature Geography/Gender**: possiamo ora quantificare il loro contributo
#   direzionale. Lo approfondiremo nel Blocco F.
#
# La dispersione orizzontale indica **eterogeneità**: per Age, il contributo
# varia enormemente da persona a persona. Per feature binarie (IsActiveMember),
# si vedono due cluster netti.

# %% [markdown]
# ---
# ## 9. SHAP Bar Plot e tabella comparativa dei 3 ranking
#
# Calcoliamo l'importanza media assoluta dei SHAP values (|φᵢ| medio) per
# ottenere un ranking numerico confrontabile con Gini e Permutation.

# %%
# --- SHAP mean |value| per feature ---
shap_mean_abs = np.abs(shap_values_array).mean(axis=0)
shap_order = np.argsort(shap_mean_abs)[::-1]

fig, ax = plt.subplots(figsize=(10, 7))
top_shap_indices = shap_order[:top_k]
ax.barh(
    range(top_k),
    shap_mean_abs[top_shap_indices][::-1],
    color="mediumpurple",
    edgecolor="white",
)
ax.set_yticks(range(top_k))
ax.set_yticklabels([feature_names[i] for i in top_shap_indices][::-1])
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Feature Importance — SHAP (Mean Absolute Contribution)")
save_current_figure("lesson_04_shap_bar.png")
plt.show()

# %%
# --- Tabella comparativa: 3 ranking a confronto ---
shap_ranks = {idx: rank for rank, idx in enumerate(shap_order, 1)}

print(f"\n{'Feature':<35} {'Gini':<8} {'Perm':<8} {'SHAP':<8} {'Consenso'}")
print("-" * 72)

# Mostriamo le top-15 per SHAP ranking
for idx in shap_order[:top_k]:
    g = gini_ranks[idx]
    p = perm_ranks[idx]
    s = shap_ranks[idx]
    # Consenso: se tutti e tre concordano entro ±3 posizioni
    consensus = "✓" if max(g, p, s) - min(g, p, s) <= 3 else "~"
    print(f"{feature_names[idx]:<35} {g:<8} {p:<8} {s:<8} {consensus}")

# %% [markdown]
# **Analisi della convergenza tra i 3 metodi:**
#
# - Le feature con **"✓" (consenso)** sono quelle su cui tutti e tre i metodi
#   concordano: la loro importanza è robusta e non dipende dalla metodologia.
# - Le feature con **"~" (discordanza)** richiedono cautela: la loro importanza
#   è metodo-dipendente. SHAP è il metodo più rigoroso teoricamente, quindi
#   in caso di dubbio ci affidiamo al suo ranking.
# - Questo confronto multiplo ci protegge da conclusioni affrettate: se solo
#   la Gini importance dicesse che una feature è importante, potremmo sbagliarci.

# %% [markdown]
# ---
# ## BLOCCO D — Interpretabilità locale: capire le singole predizioni
#
# ---
#
# ## 10. Scegliere clienti rappresentativi da spiegare
#
# L'importanza globale ci dice quali feature contano *in media*. Ma il valore
# reale della spiegabilità emerge quando sappiamo spiegare le **singole decisioni**.
#
# Selezioniamo 4 casi strategici dal validation set:
# - (a) **Churner ad alta confidenza** (prob > 0.8) — perché il modello è così sicuro?
# - (b) **Non-churner ad alta confidenza** (prob < 0.1) — cosa lo protegge?
# - (c) **Caso borderline** (prob ≈ 0.4–0.6) — perché il modello esita?
# - (d) **Falso negativo** — churner reale che il modello non identifica

# %%
# --- Probabilità predette sul validation set ---
probs_val = model.predict_proba(X_val)[:, 1]

print(f"Distribuzione probabilità sul validation set:")
print(f"  Min:    {probs_val.min():.4f}")
print(f"  Q1:     {np.percentile(probs_val, 25):.4f}")
print(f"  Median: {np.median(probs_val):.4f}")
print(f"  Q3:     {np.percentile(probs_val, 75):.4f}")
print(f"  Max:    {probs_val.max():.4f}")

# %%
# --- Selezione dei 4 casi rappresentativi ---

# (a) Churner ad alta confidenza: y=1 e prob > 0.8
mask_a = (y_val == 1) & (probs_val > 0.8)
case_a_idx = np.where(mask_a)[0][0] if mask_a.any() else None

# (b) Non-churner ad alta confidenza: y=0 e prob < 0.1
mask_b = (y_val == 0) & (probs_val < 0.1)
case_b_idx = np.where(mask_b)[0][0] if mask_b.any() else None

# (c) Caso borderline: prob tra 0.4 e 0.6
mask_c = (probs_val >= 0.4) & (probs_val <= 0.6)
case_c_idx = np.where(mask_c)[0][0] if mask_c.any() else None

# (d) Falso negativo: y=1 ma prob < 0.36 (soglia ottimale L3)
threshold_l3 = baseline_metrics.get("best_threshold", 0.36)
mask_d = (y_val == 1) & (probs_val < threshold_l3)
case_d_idx = np.where(mask_d)[0][0] if mask_d.any() else None

cases = {
    "Churner alta confidenza": case_a_idx,
    "Non-churner alta confidenza": case_b_idx,
    "Borderline": case_c_idx,
    "Falso negativo": case_d_idx,
}

print(f"\nCasi selezionati:")
print(f"{'Tipo':<32} {'Idx':<8} {'y reale':<10} {'P(churn)'}")
print("-" * 65)
for label, idx in cases.items():
    if idx is not None:
        print(f"{label:<32} {idx:<8} {y_val[idx]:<10.0f} {probs_val[idx]:.4f}")
    else:
        print(f"{label:<32} {'N/A':<8} {'—':<10} {'—'}")

# %% [markdown]
# Abbiamo identificato 4 clienti che rappresentano scenari diversi.
# Per ciascuno, useremo SHAP per "aprire" la predizione e capire
# esattamente quali feature hanno contribuito — e in che direzione.

# %% [markdown]
# ---
# ## 11. SHAP Waterfall Plot — anatomia di una predizione
#
# Il waterfall plot mostra, per una singola osservazione:
# - Il **base value** (predizione media del modello)
# - Il contributo di **ogni feature** (barra rossa = spinge verso churn,
#   barra blu = protegge dal churn)
# - La **predizione finale** (somma di base + tutti i contributi)
#
# Analizziamo ciascun caso uno alla volta, come farebbe un Data Scientist
# che deve presentare i risultati al team di business.

# %%
# --- Calcolo SHAP values per i 4 casi dal validation set ---
X_cases = np.array([X_val[idx] for idx in cases.values() if idx is not None])
shap_values_cases = explainer(X_cases)

# Gestione output multi-classe
if shap_values_cases.values.ndim == 3:
    shap_values_cases_array = shap_values_cases.values[:, :, 1]
    base_val_cases = shap_values_cases.base_values[0, 1]
else:
    shap_values_cases_array = shap_values_cases.values
    base_val_cases = shap_values_cases.base_values[0]

# %%
# --- Waterfall: Caso (a) — Churner ad alta confidenza ---
case_labels = [k for k, v in cases.items() if v is not None]
case_indices_valid = [v for v in cases.values() if v is not None]

if len(case_labels) > 0:
    i = 0  # Primo caso: churner alta confidenza
    explanation = shap.Explanation(
        values=shap_values_cases_array[i],
        base_values=base_val_cases,
        data=X_cases[i],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — {case_labels[i]} "
        f"(y={y_val[case_indices_valid[i]]:.0f}, "
        f"P(churn)={probs_val[case_indices_valid[i]]:.3f})"
    )
    save_current_figure("lesson_04_shap_waterfall_churner_alta_conf.png")
    plt.show()

# %% [markdown]
# **Caso (a) — Churner ad alta confidenza:**
#
# Il modello è molto sicuro che questo cliente abbandonerà. Le feature che
# spingono verso il churn (barre rosse) dominano nettamente. Osserviamo
# quali sono e verifichiamo se ha senso dal punto di vista del business.

# %%
# --- Waterfall: Caso (b) — Non-churner ad alta confidenza ---
if len(case_labels) > 1:
    i = 1
    explanation = shap.Explanation(
        values=shap_values_cases_array[i],
        base_values=base_val_cases,
        data=X_cases[i],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — {case_labels[i]} "
        f"(y={y_val[case_indices_valid[i]]:.0f}, "
        f"P(churn)={probs_val[case_indices_valid[i]]:.3f})"
    )
    save_current_figure("lesson_04_shap_waterfall_non_churner.png")
    plt.show()

# %% [markdown]
# **Caso (b) — Non-churner ad alta confidenza:**
#
# Qui le feature protettive (barre blu) dominano. Il modello è sicuro che
# questo cliente resterà. Notiamo quali fattori contribuiscono alla
# "protezione" — tipicamente: giovane età, membro attivo, prodotti multipli.

# %%
# --- Waterfall: Caso (c) — Borderline ---
if len(case_labels) > 2:
    i = 2
    explanation = shap.Explanation(
        values=shap_values_cases_array[i],
        base_values=base_val_cases,
        data=X_cases[i],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — {case_labels[i]} "
        f"(y={y_val[case_indices_valid[i]]:.0f}, "
        f"P(churn)={probs_val[case_indices_valid[i]]:.3f})"
    )
    save_current_figure("lesson_04_shap_waterfall_borderline.png")
    plt.show()

# %% [markdown]
# **Caso (c) — Borderline:**
#
# Il caso più interessante. Il modello esita perché le forze si bilanciano:
# alcune feature spingono verso il churn, altre proteggono, e il risultato
# netto è una probabilità vicina a 0.5. In produzione, questi casi
# richiederebbero un intervento manuale (revisione da parte di un analista).

# %%
# --- Waterfall: Caso (d) — Falso negativo ---
if len(case_labels) > 3:
    i = 3
    explanation = shap.Explanation(
        values=shap_values_cases_array[i],
        base_values=base_val_cases,
        data=X_cases[i],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — {case_labels[i]} "
        f"(y={y_val[case_indices_valid[i]]:.0f}, "
        f"P(churn)={probs_val[case_indices_valid[i]]:.3f})"
    )
    save_current_figure("lesson_04_shap_waterfall_falso_negativo.png")
    plt.show()

# %% [markdown]
# **Caso (d) — Falso negativo:**
#
# Questo è un cliente che nella realtà ha abbandonato, ma il modello non lo
# ha identificato. Il waterfall ci mostra **perché** il modello si è sbagliato:
# le feature protettive hanno "mascherato" i segnali di rischio. Questi
# casi sono i più pericolosi in produzione — e capirli ci aiuta a capire
# i limiti del modello.

# %% [markdown]
# ---
# ## 12. Costruzione del "profilo cliente a rischio"
#
# Ora aggreghiamo i contributi SHAP su **tutti i churner** nel campione per
# identificare il pattern tipico del cliente a rischio. Non guardiamo un
# singolo caso — cerchiamo il **profilo medio** che emerge dai dati.

# %%
# --- Profilo medio dei churner: SHAP values aggregati ---
churner_mask = y_shap_sample == 1
shap_churners = shap_values_array[churner_mask]

mean_shap_churners = shap_churners.mean(axis=0)
mean_shap_churners_order = np.argsort(mean_shap_churners)[::-1]

# Top-5 driver del churn (SHAP positivo = spinge verso churn)
print("=" * 60)
print("PROFILO CLIENTE A RISCHIO CHURN")
print("(basato su contributi SHAP medi dei churner nel campione)")
print("=" * 60)
print(f"\nTop-5 driver DEL churn (spingono verso l'abbandono):")
print(f"{'Feature':<35} {'Mean SHAP':<12} {'Direzione'}")
print("-" * 55)
for idx in mean_shap_churners_order[:5]:
    print(
        f"{feature_names[idx]:<35} {mean_shap_churners[idx]:+.4f}    "
        f"→ aumenta P(churn)"
    )

print(f"\nTop-5 fattori PROTETTIVI (riducono il rischio churn):")
print(f"{'Feature':<35} {'Mean SHAP':<12} {'Direzione'}")
print("-" * 55)
for idx in mean_shap_churners_order[-5:][::-1]:
    print(
        f"{feature_names[idx]:<35} {mean_shap_churners[idx]:+.4f}    "
        f"→ riduce P(churn)"
    )

# %%
# --- Visualizzazione: contributi medi per i churner ---
top_n_profile = 10
profile_indices = np.concatenate(
    [
        mean_shap_churners_order[:top_n_profile],  # top driver churn
    ]
)

fig, ax = plt.subplots(figsize=(10, 6))
colors = [
    "#d63031" if mean_shap_churners[i] > 0 else "#0984e3" for i in profile_indices[::-1]
]
ax.barh(
    range(top_n_profile),
    mean_shap_churners[profile_indices][::-1],
    color=colors,
    edgecolor="white",
)
ax.set_yticks(range(top_n_profile))
ax.set_yticklabels([feature_names[i] for i in profile_indices][::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Contributo SHAP medio (churner)")
ax.set_title("Profilo Cliente a Rischio — Top driver di churn")
save_current_figure("lesson_04_risk_profile.png")
plt.show()

# %%
# --- Salvataggio del profilo cliente a rischio ---
risk_profile = {
    "descrizione": "Profilo medio dei churner basato su SHAP values",
    "n_churner_analizzati": int(churner_mask.sum()),
    "top_driver_churn": [
        {"feature": feature_names[idx], "mean_shap": float(mean_shap_churners[idx])}
        for idx in mean_shap_churners_order[:5]
    ],
    "top_fattori_protettivi": [
        {"feature": feature_names[idx], "mean_shap": float(mean_shap_churners[idx])}
        for idx in mean_shap_churners_order[-5:][::-1]
    ],
}

with open(DATA_OUT_DIR / "lesson_04_risk_profile.json", "w") as fh:
    json.dump(risk_profile, fh, indent=2, ensure_ascii=False)
print(f"Profilo salvato: {DATA_OUT_DIR / 'lesson_04_risk_profile.json'}")

# %% [markdown]
# **Profilo sintetico del cliente a rischio churn:**
#
# Sulla base dell'analisi SHAP aggregata, il cliente tipico che abbandona ha:
#
# - **Età elevata** — il driver principale: clienti più anziani hanno
#   probabilità di churn significativamente più alta
# - **Non è membro attivo** — l'inattività è un segnale forte di disengagement
# - **Specifiche caratteristiche geografiche/prodotto** che emergono dal ranking
#
# Questo profilo è direttamente utilizzabile dal team di retention: consente
# di definire una **campagna mirata** sui segmenti a rischio, con motivazioni
# trasparenti e verificabili.

# %% [markdown]
# ---
# ## BLOCCO E — Feature selection e re-training (sperimentazione)
#
# ---
#
# ## 13. Ipotesi: un modello più semplice generalizza meglio?
#
# Ora che conosciamo il ranking delle feature, una domanda sorge naturalmente:
#
# > **Servono davvero tutte e 30 le feature?** Un modello con meno feature
# > potrebbe essere:
# > - Più **spiegabile** (meno variabili = più facile comunicare al business)
# > - Più **veloce** (meno colonne = meno calcolo)
# > - Più **robusto** (meno noise = meno rischio di overfitting)
#
# Ma potrebbe anche **perdere informazione**. Non lo sappiamo a priori —
# l'unico modo è sperimentare.
#
# **Strategia:** selezioniamo le top-K feature in base al ranking SHAP
# (il più rigoroso), re-trainiamo un RF con gli stessi iperparametri della
# Lezione 3, e confrontiamo le performance sul validation set.

# %%
# --- Feature selection: top-K per SHAP ranking ---
K_values = [10, 15, 20]

# Ranking basato su mean |SHAP|
selected_features_by_k = {}
for K in K_values:
    selected_features_by_k[K] = shap_order[:K]
    print(f"\nTop-{K} feature (SHAP ranking):")
    for rank, idx in enumerate(shap_order[:K], 1):
        print(
            f"  {rank:2d}. {feature_names[idx]} "
            f"(|SHAP| medio = {shap_mean_abs[idx]:.4f})"
        )

# %% [markdown]
# ---
# ## 14. Esperimento: re-training con top-K feature
#
# Per ciascun valore di K, addestriamo un nuovo Random Forest (stessi
# iperparametri: n=200, class_weight='balanced', SEED=42) e valutiamo
# su validation set. Confrontiamo con il modello completo (K=30).

# %%
# --- Re-training con feature selection ---
results_feature_sel = []

# Baseline: modello completo (30 feature)
auc_full = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
preds_full = (model.predict_proba(X_val)[:, 1] >= threshold_l3).astype(int)
f1_full = f1_score(y_val, preds_full)
recall_full = recall_score(y_val, preds_full)
precision_full = precision_score(y_val, preds_full)

results_feature_sel.append(
    {
        "K": 30,
        "label": "Completo (30 feat)",
        "AUC": auc_full,
        "F1": f1_full,
        "Recall": recall_full,
        "Precision": precision_full,
    }
)

for K in K_values:
    feat_idx = selected_features_by_k[K]
    X_train_k = X_train[:, feat_idx]
    X_val_k = X_val[:, feat_idx]

    rf_k = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_k.fit(X_train_k, y_train)

    probs_k = rf_k.predict_proba(X_val_k)[:, 1]
    auc_k = roc_auc_score(y_val, probs_k)
    preds_k = (probs_k >= threshold_l3).astype(int)
    f1_k = f1_score(y_val, preds_k)
    recall_k = recall_score(y_val, preds_k)
    precision_k = precision_score(y_val, preds_k)

    results_feature_sel.append(
        {
            "K": K,
            "label": f"Top-{K} SHAP",
            "AUC": auc_k,
            "F1": f1_k,
            "Recall": recall_k,
            "Precision": precision_k,
        }
    )

# %%
# --- Tabella comparativa ---
df_results = pd.DataFrame(results_feature_sel)
df_results = df_results.set_index("label")

print("\n" + "=" * 70)
print("CONFRONTO MODELLI: FEATURE SELECTION vs MODELLO COMPLETO")
print("=" * 70)
print(f"\n(Soglia di classificazione: {threshold_l3})\n")
display(df_results[["K", "AUC", "F1", "Recall", "Precision"]])

# Calcolo delta rispetto al modello completo
print(f"\nDelta AUC rispetto al modello completo (30 feature):")
for _, row in df_results.iterrows():
    if row["K"] < 30:
        delta = row["AUC"] - auc_full
        print(
            f"  Top-{row['K']:.0f}: ΔAUC = {delta:+.4f} "
            f"({'accettabile' if abs(delta) < 0.01 else 'significativo'})"
        )

# %%
# --- Curva AUC vs numero di feature ---
fig, ax = plt.subplots(figsize=(8, 5))
k_vals = [r["K"] for r in results_feature_sel]
auc_vals = [r["AUC"] for r in results_feature_sel]

ax.plot(k_vals, auc_vals, "o-", color="mediumpurple", linewidth=2, markersize=8)
ax.axhline(
    auc_full,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Modello completo (AUC={auc_full:.4f})",
)
ax.set_xlabel("Numero di feature selezionate (K)")
ax.set_ylabel("ROC-AUC (validation set)")
ax.set_title("Trade-off: Numero di Feature vs Performance")
ax.set_xticks(k_vals)
ax.legend()
ax.grid(True, alpha=0.3)
save_current_figure("lesson_04_feature_selection_curve.png")
plt.show()

# %% [markdown]
# **Osservazioni sulla feature selection:**
#
# - Confrontiamo i delta AUC: se il modello con 15 feature perde meno dello
#   0.01 di AUC, la riduzione è **accettabile** — guadagniamo in semplicità
#   senza sacrificare performance significativa.
# - Se invece la perdita è > 0.01, le feature rimosse contenevano informazione
#   utile — il modello completo resta preferibile.
# - Questa analisi è il tipico **trade-off Data Science**: semplicità vs
#   performance. La decisione finale dipende dal contesto di business.

# %% [markdown]
# ---
# ## 15. Analisi dell'errore: cosa perde il modello ridotto?
#
# Se il modello ridotto ha performance simili, le feature rimosse erano
# ridondanti. Ma se perde qualcosa, vogliamo capire **su quali casi** sbaglia
# in più. Questo ci dice se la perdita è "uniforme" o concentrata su un
# sottogruppo specifico.

# %%
# --- Confronto errori: modello completo vs miglior modello ridotto ---
# Scegliamo il modello ridotto con il miglior compromesso
best_reduced = min(
    [r for r in results_feature_sel if r["K"] < 30],
    key=lambda r: abs(r["AUC"] - auc_full),
)
best_K = int(best_reduced["K"])
print(
    f"Miglior modello ridotto: Top-{best_K} feature "
    f"(ΔAUC = {best_reduced['AUC'] - auc_full:+.4f})"
)

# Re-fit per ottenere le predizioni
feat_idx_best = selected_features_by_k[best_K]
X_train_best = X_train[:, feat_idx_best]
X_val_best = X_val[:, feat_idx_best]

rf_best_reduced = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
rf_best_reduced.fit(X_train_best, y_train)

probs_reduced = rf_best_reduced.predict_proba(X_val_best)[:, 1]
preds_reduced = (probs_reduced >= threshold_l3).astype(int)
preds_complete = (probs_val >= threshold_l3).astype(int)

# Casi dove divergono
divergent_mask = preds_reduced != preds_complete
n_divergent = divergent_mask.sum()
print(
    f"\nCasi dove i due modelli divergono: {n_divergent} / {len(y_val)} "
    f"({n_divergent / len(y_val):.1%})"
)

if n_divergent > 0:
    # Analisi dei casi divergenti
    y_div = y_val[divergent_mask]
    print(f"\n  Di questi {n_divergent} casi divergenti:")
    print(f"    Churner reali: {y_div.sum():.0f}")
    print(f"    Non-churner reali: {(1 - y_div).sum():.0f}")

    # Quanti errori in più fa il modello ridotto?
    errors_complete = (preds_complete != y_val).sum()
    errors_reduced = (preds_reduced != y_val).sum()
    print(f"\n  Errori modello completo: {errors_complete}")
    print(f"  Errori modello ridotto:  {errors_reduced}")
    print(f"  Errori in più del ridotto: {errors_reduced - errors_complete}")

# %% [markdown]
# **Cosa abbiamo imparato?**
#
# L'analisi degli errori ci dice se la feature selection ha rimosso informazione
# critica o solo rumore. Se i casi divergenti sono pochi e distribuiti
# uniformemente tra churner e non-churner, la riduzione è sicura.
# Se invece il modello ridotto sbaglia sistematicamente su un sottogruppo
# (e.g., clienti di una specifica geography), abbiamo un problema.
#
# **Decisione finale sulla feature selection:** la prendiamo guardando i numeri
# concreti prodotti sopra. Il Data Scientist non decide a priori — decide
# dopo aver visto i dati.

# %% [markdown]
# ---
# ## BLOCCO F — Contesto regolatorio e implicazioni di business
#
# ---
#
# ## 16. Feature sensibili: Geography e Gender nel modello
#
# In un contesto bancario, alcune feature sono considerate **sensibili**
# da un punto di vista etico e regolatorio. `Gender` e `Geography` possono
# essere proxy di caratteristiche protette (genere, etnia, nazionalità).
#
# Domanda: **quanto contribuiscono queste feature alle decisioni del modello?**
# Se contribuiscono significativamente, il modello potrebbe discriminare —
# e questo è un problema sia etico che legale.

# %%
# --- Identificazione feature sensibili ---
sensitive_keywords = ["Gender", "Geography"]
sensitive_indices = [
    i
    for i, name in enumerate(feature_names)
    if any(kw in name for kw in sensitive_keywords)
]
sensitive_names = [feature_names[i] for i in sensitive_indices]

print("Feature sensibili identificate:")
for i, name in zip(sensitive_indices, sensitive_names):
    print(
        f"  {name}: mean |SHAP| = {shap_mean_abs[i]:.4f}, "
        f"rank SHAP = {shap_ranks[i]}"
    )

# %%
# --- Contributo SHAP delle feature sensibili ---
shap_sensitive = shap_values_array[:, sensitive_indices]

fig, axes = plt.subplots(
    1, len(sensitive_indices), figsize=(4 * len(sensitive_indices), 5)
)
if len(sensitive_indices) == 1:
    axes = [axes]

for ax, idx, name in zip(axes, sensitive_indices, sensitive_names):
    ax.hist(
        shap_values_array[:, idx], bins=30, color="salmon", edgecolor="white", alpha=0.8
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Frequenza")
    ax.set_title(f"{name}\n(mean |SHAP|={shap_mean_abs[idx]:.4f})")

plt.suptitle("Distribuzione SHAP values — Feature Sensibili", fontsize=12)
plt.tight_layout()
save_current_figure("lesson_04_sensitive_features_shap.png")
plt.show()

# %%
# --- Confronto: contributo relativo delle feature sensibili ---
total_shap_budget = shap_mean_abs.sum()
sensitive_shap_budget = shap_mean_abs[sensitive_indices].sum()

print(f"\nBudget SHAP totale: {total_shap_budget:.4f}")
print(f"Budget SHAP feature sensibili: {sensitive_shap_budget:.4f}")
print(
    f"Percentuale decisionale delle feature sensibili: "
    f"{sensitive_shap_budget / total_shap_budget:.1%}"
)
print(f"\nDettaglio:")
for i, name in zip(sensitive_indices, sensitive_names):
    pct = shap_mean_abs[i] / total_shap_budget
    print(f"  {name}: {pct:.1%} del budget decisionale totale")

# %% [markdown]
# **Analisi delle feature sensibili:**
#
# - Se le feature Geography e Gender contribuiscono per una percentuale
#   **bassa** del budget decisionale totale (e.g., < 5% ciascuna), il modello
#   non si basa significativamente su di esse.
# - Se invece contribuiscono in modo **sostanziale**, ci sono due opzioni:
#   1. **Rimuoverle** e re-trainare (come nell'esperimento del Blocco E)
#   2. **Applicare vincoli di fairness** (equalized odds, demographic parity)
# - In entrambi i casi, la **trasparenza** è fondamentale: documentiamo
#   esattamente quanto pesano e perché le abbiamo tenute/rimosse.

# %% [markdown]
# ---
# ## 17. Cenni regolatori — GDPR e spiegabilità nel credito
#
# ### Il contesto normativo europeo
#
# Nel settore finanziario europeo, l'uso di modelli di Machine Learning per
# decisioni che impattano i clienti è regolato da:
#
# **1. GDPR — Art. 22 (Decisioni automatizzate)**
#
# > *"L'interessato ha il diritto di non essere sottoposto a una decisione
# > basata unicamente sul trattamento automatizzato [...] che produca effetti
# > giuridici che lo riguardano."*
#
# In pratica: se un modello di churn viene usato per negare un prodotto
# finanziario o modificare le condizioni di un cliente, il cliente ha diritto
# a una **spiegazione** della decisione.
#
# **2. EBA Guidelines on ML for Credit Institutions (2021)**
#
# Le linee guida dell'European Banking Authority richiedono:
# - **Interpretabilità**: i modelli devono essere comprensibili per chi li usa
# - **Documentazione**: ogni modello deve avere una descrizione delle feature
#   usate e del loro impatto
# - **Monitoraggio del bias**: verificare che il modello non discrimini
#   sulla base di caratteristiche protette
#
# **3. AI Act (Regolamento UE 2024/1689)**
#
# I sistemi di scoring creditizio sono classificati come **alto rischio**
# (Allegato III, punto 5b), richiedendo:
# - Valutazione di conformità
# - Gestione del rischio
# - Spiegabilità delle decisioni
#
# ### Collegamento pratico
#
# Gli strumenti che abbiamo usato in questa lezione sono **esattamente** ciò
# che serve per la compliance:
#
# | Requisito normativo | Strumento usato |
# |---------------------|-----------------|
# | "Spiegare la decisione al cliente" | SHAP Waterfall Plot |
# | "Documentare l'importanza delle feature" | Tabella ranking 3 metodi |
# | "Verificare assenza di bias" | Analisi feature sensibili (sez. 16) |
# | "Modello interpretabile" | Profilo cliente a rischio |
#
# Il Data Scientist non opera in un vuoto tecnico — le scelte di modellazione
# hanno implicazioni legali e etiche concrete.

# %% [markdown]
# ---
# ## BLOCCO G — Chiusura
#
# ---
#
# ## 18. Riepilogo del percorso e artefatti prodotti
#
# ### Percorso ragionato in questa lezione
#
# | Step | Esperimento/Analisi | Insight chiave | Decisione |
# |------|---------------------|----------------|-----------|
# | 1 | Gini importance | Age domina, ma bias verso numeriche | Servono conferme |
# | 2 | Permutation importance | Conferma top-3, ridimensiona alcune | Più affidabile di Gini |
# | 3 | SHAP values | Ranking rigoroso + direzione effetti | Framework definitivo |
# | 4 | Summary plot | Visualizzazione globale contributi | Age confermata, IsActive forte |
# | 5 | Waterfall (4 casi) | Anatomia singole decisioni | Il modello è spiegabile |
# | 6 | Profilo rischio | Pattern medio dei churner | Comunicabile al business |
# | 7 | Feature selection | Top-K vs completo | Trade-off quantificato |
# | 8 | Feature sensibili | Contributo Geography/Gender | Documentato per compliance |
#
# ### Artefatti prodotti
#
# | File | Descrizione |
# |------|-------------|
# | `outputs/figures/lesson_04_gini_importance.png` | Barplot importanza Gini |
# | `outputs/figures/lesson_04_permutation_importance.png` | Barplot Permutation |
# | `outputs/figures/lesson_04_gini_vs_permutation.png` | Confronto side-by-side |
# | `outputs/figures/lesson_04_shap_summary.png` | SHAP summary plot |
# | `outputs/figures/lesson_04_shap_bar.png` | SHAP bar plot |
# | `outputs/figures/lesson_04_shap_waterfall_*.png` | Waterfall 4 casi |
# | `outputs/figures/lesson_04_feature_selection_curve.png` | AUC vs K feature |
# | `outputs/figures/lesson_04_sensitive_features_shap.png` | SHAP feature sensibili |
# | `outputs/data/lesson_04_risk_profile.json` | Profilo cliente a rischio |
# | `outputs/data/lesson_04_feature_ranking.json` | Ranking consolidato |

# %%
# --- Salvataggio ranking consolidato ---
feature_ranking = {
    "metodo": "Consolidamento Gini + Permutation + SHAP",
    "n_features": len(feature_names),
    "ranking": [
        {
            "feature": feature_names[idx],
            "rank_gini": int(gini_ranks[idx]),
            "rank_permutation": int(perm_ranks[idx]),
            "rank_shap": int(shap_ranks[idx]),
            "mean_abs_shap": float(shap_mean_abs[idx]),
        }
        for idx in shap_order
    ],
}

with open(DATA_OUT_DIR / "lesson_04_feature_ranking.json", "w") as fh:
    json.dump(feature_ranking, fh, indent=2, ensure_ascii=False)
print(f"Ranking salvato: {DATA_OUT_DIR / 'lesson_04_feature_ranking.json'}")

# %%
# --- Salvataggio SHAP values per riuso futuro ---
np.save(DATA_OUT_DIR / "lesson_04_shap_values.npy", shap_values_array)
print(f"SHAP values salvati: {DATA_OUT_DIR / 'lesson_04_shap_values.npy'}")
print(f"  Shape: {shap_values_array.shape}")

# %% [markdown]
# ---
# ## 19. Domande guidate
#
# **1. Perché la Gini importance da sola non basta per decisioni di business?**
#
# La Gini importance ha un bias noto verso feature ad alta cardinalità e
# variabili continue. Non usa il validation set, quindi non riflette la reale
# capacità predittiva. Nel nostro caso, `EstimatedSalary` e `CreditScore`
# potrebbero apparire importanti per Gini ma contribuire poco alla
# performance reale (ΔAUC basso nella permutation). Per decisioni di business,
# servono metodi che valutino l'impatto effettivo sulla metrica target.
#
# **2. Se dovessi spiegare al direttore di filiale perché 200 clienti sono
# ad alto rischio, quale visualizzazione useresti?**
#
# Il **profilo cliente a rischio** (sezione 12) è lo strumento più efficace:
# sintetizza i top driver del churn in un formato leggibile senza gergo tecnico.
# Per casi individuali controversi, il **waterfall plot** mostra trasparentemente
# come il modello è arrivato alla sua decisione — traducibile in "Le feature X, Y, Z
# del cliente contribuiscono al rischio elevato".
#
# **3. Quando la feature selection è preferibile al modello completo?**
#
# Quando la perdita di AUC è trascurabile (< 0.01) e il contesto richiede:
# (a) spiegabilità semplificata (meno variabili da comunicare),
# (b) velocità di inferenza (produzione ad alto throughput),
# (c) robustezza a cambiamenti distribuzionali (meno feature = meno fragile).
# Se invece la perdita è significativa, il modello completo resta preferibile
# — ma va documentato perché usa tutte le 30 feature.
#
# **4. Il modello è "fairness-compliant"? Come lo verificheresti in modo
# più rigoroso?**
#
# L'analisi della sezione 16 è un primo passo (quantifica il contributo delle
# feature sensibili). Per una verifica più rigorosa, si calcolerebbero metriche
# di fairness come:
# - **Demographic parity**: P(pred=1|Gender=F) ≈ P(pred=1|Gender=M)?
# - **Equalized odds**: TPR e FPR uguali tra gruppi?
# - **Calibration by group**: le probabilità sono ben calibrate per ogni gruppo?
# Se emergono disparità, si possono applicare post-processing (threshold
# differenziati per gruppo) o vincoli in-processing durante il training.
#
# **5. Come si collega il lavoro fatto oggi al concetto di re-training
# in produzione?**
#
# In produzione, il modello deve essere ri-addestrato periodicamente perché
# la distribuzione dei dati cambia (concept drift). Quando si re-traina,
# l'analisi SHAP va ripetuta: se il ranking delle feature cambia drasticamente,
# è un segnale di allarme (il mondo è cambiato, il modello potrebbe non essere
# più valido). Il profilo del cliente a rischio va aggiornato e comunicato
# al business per verificare che sia ancora coerente con la realtà operativa.

# %% [markdown]
# ---
# ## 20. Bridge verso la Lezione 5
#
# ### Cosa abbiamo costruito
#
# In questa lezione abbiamo aperto la "scatola nera" del Random Forest:
# sappiamo **perché** il modello prende le decisioni che prende, e possiamo
# comunicarlo al business e alla compliance.
#
# ### Cosa ci manca
#
# 1. **Cross-validation:** finora abbiamo valutato su un singolo split.
#    La stima dell'AUC è affidabile, ma non abbiamo una misura della sua
#    **variabilità**. Quanto cambia l'AUC se cambiamo il split?
#
# 2. **Hyperparameter tuning:** abbiamo usato n_estimators=200 e
#    class_weight='balanced' senza mai ottimizzarli. Esistono combinazioni
#    migliori? Grid Search e Random Search ci permettono di esplorare
#    sistematicamente lo spazio degli iperparametri.
#
# 3. **Modelli avanzati (XGBoost):** il Random Forest è buono, ma il gradient
#    boosting (in particolare XGBoost) è spesso superiore in competizioni e
#    applicazioni reali. Nella Lezione 5 lo introdurremo e lo confronteremo.
#
# 4. **Modello finale per la challenge:** la combinazione di feature selection
#    (identificata qui in L4) + tuning + XGBoost potrebbe portare a un modello
#    ancora migliore — e lo presenteremo in un mini-report simulando un
#    progetto reale end-to-end.
#
# La **Lezione 5** chiuderà il percorso: dalla sperimentazione al modello
# finale, con cross-validation rigorosa e un report di progetto.
