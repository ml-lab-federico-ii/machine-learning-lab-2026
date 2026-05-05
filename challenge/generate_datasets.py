"""Generazione di dataset sintetici per la challenge in-class.

Produce 15 CSV con stesso schema (Customer Churn) ma profili bias diversi,
più un cheat-sheet docente (baseline + domande custom per SEED).

Uso:
    python generate_datasets.py              # genera tutto
    python generate_datasets.py --n-datasets 10  # solo 10 dataset
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Provare ad importare xgboost (opzionale per la generazione)
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

MASTER_SEED = 2026
N_ROWS_DEFAULT = 10_000
N_DATASETS_DEFAULT = 15

# Colonne del dataset target (schema Customer-Churn-Records, senza leakage)
FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
]
TARGET_COLUMN = "Exited"

GEOGRAPHIES = ["France", "Germany", "Spain"]
GENDERS = ["Male", "Female"]
CARD_TYPES = ["DIAMOND", "GOLD", "SILVER", "PLATINUM"]


# ---------------------------------------------------------------------------
# Profili bias — ogni profilo è un dict di parametri che controllano la
# distribuzione dei dati e la forza del segnale
# ---------------------------------------------------------------------------

PROFILES: list[dict] = [
    {
        "id": 1,
        "name": "Churn basso, segnale lineare",
        "desc": "Churn rate ~12%. Il segnale è prevalentemente lineare (Age, Balance). "
        "Modelli lineari (LogReg) sono competitivi.",
        "churn_base": 0.12,
        "age_shift": -3,
        "geo_weights": [0.50, 0.25, 0.25],
        "noise_level": 0.15,
        "interaction_strength": 0.2,
        "num_products_effect": 0.3,
        "linear_weight": 0.85,
        "question_business": "Il vostro dataset ha un churn rate molto basso. "
        "Come influenza questo la strategia di retention della banca? "
        "È più importante la precision o la recall in questo scenario?",
        "question_technical": "Avete osservato che modelli semplici performano "
        "in modo competitivo? Perché pensate che sia così, "
        "e cosa ci dice sulla natura dei dati?",
    },
    {
        "id": 2,
        "name": "Churn alto, forte effetto Age",
        "desc": "Churn rate ~28%. Forte correlazione Age-Churn: clienti over 45 "
        "hanno rischio 3x. RF e XGBoost dovrebbero catturare bene le non-linearità.",
        "churn_base": 0.28,
        "age_shift": 5,
        "geo_weights": [0.40, 0.35, 0.25],
        "noise_level": 0.10,
        "interaction_strength": 0.5,
        "num_products_effect": 0.4,
        "linear_weight": 0.25,
        "question_business": "I clienti più anziani nel vostro dataset mostrano "
        "un rischio churn significativamente più alto. Quale programma "
        "di retention proporreste specificamente per questo segmento?",
        "question_technical": "Il vostro dataset ha un churn rate elevato (~28%). "
        "Come avete gestito lo sbilanciamento delle classi? "
        "Quale impatto ha avuto sulle metriche?",
    },
    {
        "id": 3,
        "name": "Interazioni non-lineari complesse",
        "desc": "Churn rate ~20%. Il segnale dipende da interazioni tra feature "
        "(Age×IsActive, NumProducts×Balance). XGBoost è favorito.",
        "churn_base": 0.20,
        "age_shift": 0,
        "geo_weights": [0.33, 0.34, 0.33],
        "noise_level": 0.12,
        "interaction_strength": 0.8,
        "num_products_effect": 0.6,
        "linear_weight": 0.10,
        "question_business": "Nel vostro dataset, il rischio di churn dipende "
        "da combinazioni di fattori, non da singole variabili. "
        "Come comunichereste questa complessità a un manager non tecnico?",
        "question_technical": "Avete notato differenze significative tra modelli "
        "lineari e tree-based? Cosa suggerisce questo sulla struttura "
        "sottostante dei dati?",
    },
    {
        "id": 4,
        "name": "Forte rumore, modelli semplici robusti",
        "desc": "Churn rate ~22%. Molto rumore nei dati: le relazioni feature-target "
        "sono deboli. Modelli semplici (LogReg, DT shallow) sono più robusti.",
        "churn_base": 0.22,
        "age_shift": 0,
        "geo_weights": [0.45, 0.30, 0.25],
        "noise_level": 0.40,
        "interaction_strength": 0.15,
        "num_products_effect": 0.2,
        "linear_weight": 0.70,
        "question_business": "I vostri modelli raggiungono AUC relativamente basse. "
        "Come spieghereste al cliente che il modello ha comunque "
        "valore di business nonostante le performance limitate?",
        "question_technical": "Avete osservato overfitting nei modelli più "
        "complessi? Come avete affrontato il problema? "
        "Quale modello si è dimostrato più robusto al rumore?",
    },
    {
        "id": 5,
        "name": "Squilibrio geografico forte",
        "desc": "Churn rate ~18%. 70% dei clienti sono tedeschi con churn rate "
        "molto più alto. Il preprocessing delle feature categoriche è critico.",
        "churn_base": 0.18,
        "age_shift": 2,
        "geo_weights": [0.10, 0.70, 0.20],
        "noise_level": 0.15,
        "interaction_strength": 0.4,
        "num_products_effect": 0.3,
        "linear_weight": 0.40,
        "question_business": "Il vostro dataset è dominato da clienti di una "
        "specifica area geografica con alto churn. Come adattereste "
        "la strategia di retention per questo mercato?",
        "question_technical": "La forte concentrazione geografica influenza "
        "l'importanza delle feature? Come avete gestito "
        "questa asimmetria nel preprocessing?",
    },
    {
        "id": 6,
        "name": "Poche feature dominanti",
        "desc": "Churn rate ~21%. Il segnale è concentrato su 2-3 feature "
        "(NumOfProducts, IsActiveMember). Feature selection è utile.",
        "churn_base": 0.21,
        "age_shift": 0,
        "geo_weights": [0.40, 0.30, 0.30],
        "noise_level": 0.20,
        "interaction_strength": 0.3,
        "num_products_effect": 0.9,
        "linear_weight": 0.15,
        "question_business": "Poche variabili guidano quasi tutto il rischio "
        "di churn nel vostro dataset. Quali azioni concrete "
        "la banca potrebbe intraprendere su queste leve?",
        "question_technical": "Avete provato ad escludere feature poco "
        "informative? Ha migliorato le performance? "
        "Perché sì o perché no?",
    },
    {
        "id": 7,
        "name": "Forte effetto NumOfProducts",
        "desc": "Churn rate ~25%. Clienti con 3+ prodotti hanno churn rate >60%. "
        "Decision Tree con buona depth è competitivo.",
        "churn_base": 0.25,
        "age_shift": 1,
        "geo_weights": [0.35, 0.35, 0.30],
        "noise_level": 0.12,
        "interaction_strength": 0.4,
        "num_products_effect": 1.2,
        "linear_weight": 0.10,
        "question_business": "I clienti multi-prodotto nel vostro dataset "
        "hanno un rischio di churn molto elevato. È controintuitivo: "
        "perché più prodotti = più rischio? Cosa proporreste?",
        "question_technical": "Avete osservato un effetto 'a gradino' di "
        "NumOfProducts sul churn? Come si comportano i diversi "
        "modelli nel catturare questo pattern?",
    },
    {
        "id": 8,
        "name": "Pattern diffuso, modelli simili",
        "desc": "Churn rate ~20%. Segnale distribuito uniformemente su molte feature. "
        "Tutti i modelli performano in modo simile: l'argomentazione conta.",
        "churn_base": 0.20,
        "age_shift": 0,
        "geo_weights": [0.33, 0.34, 0.33],
        "noise_level": 0.18,
        "interaction_strength": 0.35,
        "num_products_effect": 0.4,
        "linear_weight": 0.50,
        "question_business": "I vostri modelli raggiungono performance simili. "
        "Come scegliereste il modello finale in uno scenario "
        "reale dove non c'è un vincitore chiaro?",
        "question_technical": "Quando diversi modelli hanno AUC comparabile, "
        "quali altri criteri si possono usare per la selezione "
        "finale (interpretabilità, costo computazionale, ecc.)?",
    },
    {
        "id": 9,
        "name": "Churn molto alto, classe bilanciata",
        "desc": "Churn rate ~35%. Quasi bilanciato. Il class_weight ha meno "
        "impatto ma la soglia di classificazione diventa critica.",
        "churn_base": 0.35,
        "age_shift": 3,
        "geo_weights": [0.30, 0.40, 0.30],
        "noise_level": 0.15,
        "interaction_strength": 0.5,
        "num_products_effect": 0.5,
        "linear_weight": 0.30,
        "question_business": "Un terzo dei clienti nel vostro dataset abbandona. "
        "Questo scenario suggerisce un problema sistemico. "
        "Quali azioni strutturali proporreste alla banca?",
        "question_technical": "Con un churn rate ~35%, il dataset è quasi "
        "bilanciato. Come influenza questo la scelta di "
        "class_weight e della soglia di classificazione?",
    },
    {
        "id": 10,
        "name": "Clienti giovani, churn da inattività",
        "desc": "Churn rate ~19%. Popolazione giovane (media 30 anni). "
        "IsActiveMember è il fattore più predittivo.",
        "churn_base": 0.19,
        "age_shift": -8,
        "geo_weights": [0.45, 0.25, 0.30],
        "noise_level": 0.14,
        "interaction_strength": 0.3,
        "num_products_effect": 0.3,
        "linear_weight": 0.65,
        "question_business": "I vostri clienti sono prevalentemente giovani. "
        "L'inattività sembra il fattore chiave di churn. "
        "Quale strategia di engagement proporreste?",
        "question_technical": "IsActiveMember è la feature più predittiva. "
        "Come si comportano i diversi modelli nel catturare "
        "l'effetto di una singola feature binaria dominante?",
    },
    {
        "id": 11,
        "name": "Segnale nel Balance",
        "desc": "Churn rate ~22%. Clienti con saldo zero hanno churn rate molto "
        "diverso. La feature engineered balance_is_zero è molto utile.",
        "churn_base": 0.22,
        "age_shift": 1,
        "geo_weights": [0.40, 0.30, 0.30],
        "noise_level": 0.15,
        "interaction_strength": 0.4,
        "num_products_effect": 0.3,
        "linear_weight": 0.20,
        "question_business": "Il saldo del conto è fortemente correlato con "
        "il churn nel vostro dataset. Come usereste questa "
        "informazione per segmentare i clienti?",
        "question_technical": "Avete provato a creare la feature 'balance_is_zero'? "
        "Quanto ha migliorato le performance? Cosa ci dice "
        "sull'importanza del feature engineering?",
    },
    {
        "id": 12,
        "name": "Churn basso con outlier",
        "desc": "Churn rate ~10%. Pochi churner ma con pattern molto distinto. "
        "Recall è la metrica critica.",
        "churn_base": 0.10,
        "age_shift": -2,
        "geo_weights": [0.50, 0.25, 0.25],
        "noise_level": 0.12,
        "interaction_strength": 0.6,
        "num_products_effect": 0.5,
        "linear_weight": 0.15,
        "question_business": "Solo il 10% dei clienti abbandona, ma ciascuno "
        "rappresenta una perdita significativa. Come bilancereste "
        "il costo di un falso allarme vs un churner perso?",
        "question_technical": "Con un churn rate così basso, l'accuracy è "
        "fuorviante (un modello 'tutto No Churn' avrebbe 90%). "
        "Quale metrica avete privilegiato e perché?",
    },
    {
        "id": 13,
        "name": "Gender gap nel churn",
        "desc": "Churn rate ~23%. Forte differenza di churn tra generi. "
        "Solleva considerazioni di fairness.",
        "churn_base": 0.23,
        "age_shift": 0,
        "geo_weights": [0.35, 0.35, 0.30],
        "noise_level": 0.15,
        "interaction_strength": 0.4,
        "num_products_effect": 0.4,
        "linear_weight": 0.55,
        "question_business": "Nel vostro dataset c'è una forte differenza "
        "di churn tra generi. Come gestireste questa informazione "
        "da un punto di vista etico e regolatorio?",
        "question_technical": "Se il modello usa Gender come feature predittiva "
        "importante, quali rischi di bias introduce? "
        "Avete provato a rimuovere la feature?",
    },
    {
        "id": 14,
        "name": "Segnale nel Tenure",
        "desc": "Churn rate ~20%. Clienti con bassa tenure hanno churn 3x. "
        "Il tempo di relazione è il predittore chiave.",
        "churn_base": 0.20,
        "age_shift": 2,
        "geo_weights": [0.40, 0.30, 0.30],
        "noise_level": 0.13,
        "interaction_strength": 0.35,
        "num_products_effect": 0.3,
        "linear_weight": 0.35,
        "question_business": "I clienti più recenti nel vostro dataset hanno "
        "un rischio di churn molto più alto. Quale strategia "
        "di onboarding proporreste per i primi 2 anni?",
        "question_technical": "Tenure è una feature ordinale (0-10). Come "
        "la trattano diversamente i modelli lineari vs "
        "tree-based? Avete notato differenze?",
    },
    {
        "id": 15,
        "name": "Multi-segnale bilanciato",
        "desc": "Churn rate ~21%. Segnale moderato su tutte le feature, "
        "nessun pattern dominante. Richiede analisi attenta.",
        "churn_base": 0.21,
        "age_shift": 1,
        "geo_weights": [0.38, 0.32, 0.30],
        "noise_level": 0.17,
        "linear_weight": 0.45,
        "interaction_strength": 0.45,
        "num_products_effect": 0.45,
        "question_business": "Nessuna singola variabile domina il rischio "
        "di churn nel vostro dataset. Come impostereste "
        "una strategia di retention multi-fattoriale?",
        "question_technical": "Quando il segnale è distribuito su molte feature, "
        "quale paradigma (bagging vs boosting) tende a "
        "funzionare meglio? Confermate sui vostri dati?",
    },
]


# ---------------------------------------------------------------------------
# Generatore sintetico
# ---------------------------------------------------------------------------

def _generate_dataset(profile: dict, n_rows: int, seed: int) -> pd.DataFrame:
    """Genera un dataset sintetico con il profilo bias specificato.

    La generazione del target usa un approccio *ibrido*: una componente
    lineare (catturabile da LogReg) e una componente non-lineare basata
    su regole ad albero (catturabile da tree-based). Il peso relativo
    è controllato dal profilo. Questo garantisce che modelli diversi
    siano favoriti su dataset diversi.
    """
    rng = np.random.RandomState(seed)

    churn_base = profile["churn_base"]
    age_shift = profile["age_shift"]
    geo_weights = profile["geo_weights"]
    noise_level = profile["noise_level"]
    interaction_strength = profile["interaction_strength"]
    num_products_effect = profile["num_products_effect"]

    # --- Feature generation ---
    credit_score = rng.normal(650, 95, n_rows).clip(300, 850).astype(int)
    geography = rng.choice(GEOGRAPHIES, n_rows, p=geo_weights)
    gender = rng.choice(GENDERS, n_rows, p=[0.5, 0.5])
    age = rng.normal(39 + age_shift, 10, n_rows).clip(18, 85).astype(int)
    tenure = rng.randint(0, 11, n_rows)
    balance = np.where(
        rng.random(n_rows) < 0.35,
        0.0,
        rng.normal(76_000, 62_000, n_rows).clip(0, 260_000),
    )
    balance = np.round(balance, 2)
    num_of_products = rng.choice([1, 2, 3, 4], n_rows, p=[0.50, 0.35, 0.10, 0.05])
    has_cr_card = rng.binomial(1, 0.70, n_rows)
    is_active_member = rng.binomial(1, 0.50, n_rows)
    estimated_salary = rng.uniform(10_000, 200_000, n_rows).round(2)
    satisfaction_score = rng.randint(1, 6, n_rows)
    card_type = rng.choice(CARD_TYPES, n_rows, p=[0.25, 0.30, 0.25, 0.20])
    point_earned = rng.normal(450, 200, n_rows).clip(50, 1200).astype(int)

    balance_is_zero = (balance == 0).astype(float)
    is_germany = (geography == "Germany").astype(float)
    is_female = (gender == "Female").astype(float)

    # ===================================================================
    # Componente LINEARE (favorisce LogReg)
    # ===================================================================
    linear = np.zeros(n_rows)
    linear += 0.02 * (age - 38)
    linear -= 0.4 * is_active_member
    linear += 0.3 * is_germany
    linear += 0.1 * is_female
    linear -= 0.001 * (credit_score - 650)
    linear -= 0.02 * tenure
    linear -= 0.2 * balance_is_zero

    # ===================================================================
    # Componente NON-LINEARE a regole (favorisce tree-based)
    # ===================================================================
    nonlinear = np.zeros(n_rows)

    # Effetto a gradino NumOfProducts (forte, non catturabile linearmente)
    nonlinear += num_products_effect * 2.0 * (num_of_products >= 3).astype(float)
    nonlinear += num_products_effect * 1.5 * (num_of_products == 4).astype(float)

    # Interazione Age × IsActiveMember (XOR-like)
    nonlinear += interaction_strength * 1.5 * (
        (age > 45).astype(float) * (1 - is_active_member)
    )

    # Interazione Age × NumOfProducts (combo pericolosa)
    nonlinear += interaction_strength * 1.0 * (
        (age > 50).astype(float) * (num_of_products >= 2).astype(float)
    )

    # Interazione Balance × Geography (solo per Germania)
    nonlinear += interaction_strength * 0.8 * (
        is_germany * (balance > 100_000).astype(float)
    )

    # Effetto soglia CreditScore (basso = rischio, ma solo sotto 500)
    nonlinear += 0.5 * interaction_strength * (credit_score < 500).astype(float)

    # Interazione Tenure × IsActive (nuovi inattivi = alto rischio)
    nonlinear += interaction_strength * 0.6 * (
        (tenure <= 2).astype(float) * (1 - is_active_member)
    )

    # ===================================================================
    # Componenti specifiche per profilo
    # ===================================================================
    profile_effect = np.zeros(n_rows)

    pid = profile["id"]

    if pid == 11:  # Segnale nel Balance
        profile_effect += 1.2 * balance_is_zero
        profile_effect += 0.6 * (balance > 120_000).astype(float)

    elif pid == 13:  # Gender gap
        profile_effect += 0.8 * is_female
        linear -= 0.1 * is_female  # riduce componente lineare per bilanciare

    elif pid == 14:  # Segnale nel Tenure
        profile_effect -= 0.2 * tenure
        profile_effect += 1.0 * (tenure <= 1).astype(float)

    elif pid == 12:  # Churn basso con outlier pattern
        profile_effect += 1.5 * (
            (age > 55).astype(float)
            * (num_of_products >= 3).astype(float)
            * (1 - is_active_member)
        )

    # ===================================================================
    # Peso relativo lineare vs non-lineare (chiave per diversità modelli)
    # ===================================================================
    # linear_weight: 0.0 = solo regole (tree wins), 1.0 = solo lineare (logreg wins)
    linear_weight = profile.get("linear_weight", 0.5)

    logit = np.full(n_rows, np.log(churn_base / (1 - churn_base)))
    logit += linear_weight * linear * 2.0
    logit += (1 - linear_weight) * nonlinear * 2.0
    logit += profile_effect

    # Rumore
    logit += rng.normal(0, noise_level * 2, n_rows)

    # Convert to probabilities
    prob = 1 / (1 + np.exp(-logit))
    exited = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Satisfaction Score": satisfaction_score,
            "Card Type": card_type,
            "Point Earned": point_earned,
            "Exited": exited,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def _evaluate_baselines(df: pd.DataFrame, seed: int) -> dict[str, float]:
    """Addestra i 4 modelli con parametri default e restituisce AUC su val."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Encoding semplice
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include="object").columns:
        X_enc[col] = LabelEncoder().fit_transform(X_enc[col])

    X_train, X_val, y_train, y_val = train_test_split(
        X_enc, y, test_size=0.2, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    results = {}

    # LogReg
    lr = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    results["LogisticRegression"] = roc_auc_score(y_val, lr.predict_proba(X_val_s)[:, 1])

    # DT
    dt = DecisionTreeClassifier(max_depth=6, random_state=seed, class_weight="balanced")
    dt.fit(X_train_s, y_train)
    results["DecisionTree"] = roc_auc_score(y_val, dt.predict_proba(X_val_s)[:, 1])

    # RF
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=seed, class_weight="balanced"
    )
    rf.fit(X_train_s, y_train)
    results["RandomForest"] = roc_auc_score(y_val, rf.predict_proba(X_val_s)[:, 1])

    # XGBoost
    if _HAS_XGB:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=spw,
            random_state=seed,
            eval_metric="auc",
            use_label_encoder=False,
        )
        xgb.fit(X_train_s, y_train)
        results["XGBoost"] = roc_auc_score(y_val, xgb.predict_proba(X_val_s)[:, 1])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera dataset sintetici per la challenge")
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=N_DATASETS_DEFAULT,
        help=f"Numero di dataset da generare (default: {N_DATASETS_DEFAULT})",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=N_ROWS_DEFAULT,
        help=f"Righe per dataset (default: {N_ROWS_DEFAULT})",
    )
    args = parser.parse_args()

    n_datasets = min(args.n_datasets, len(PROFILES))
    script_dir = Path(__file__).resolve().parent
    datasets_dir = script_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    cheatsheet: list[dict] = []

    print(f"Generazione di {n_datasets} dataset con {args.n_rows} righe ciascuno...")
    print(f"MASTER_SEED = {MASTER_SEED}\n")

    for i in range(n_datasets):
        profile = PROFILES[i]
        seed = MASTER_SEED + profile["id"]
        seed_id = profile["id"]

        print(f"  [{seed_id:02d}] {profile['name']}...", end=" ")

        df = _generate_dataset(profile, args.n_rows, seed)

        # Salva CSV
        csv_path = datasets_dir / f"seed_{seed_id:02d}.csv"
        df.to_csv(csv_path, index=False)

        churn_rate = df[TARGET_COLUMN].mean()
        print(f"churn={churn_rate:.1%}, shape={df.shape}", end=" ")

        # Baselines
        baselines = _evaluate_baselines(df, seed)
        best_model = max(baselines, key=baselines.get)
        print(f"→ best={best_model} (AUC={baselines[best_model]:.4f})")

        cheatsheet.append(
            {
                "seed": seed_id,
                "profile_name": profile["name"],
                "profile_desc": profile["desc"],
                "n_rows": len(df),
                "churn_rate": round(churn_rate, 4),
                "baselines": {k: round(v, 4) for k, v in baselines.items()},
                "best_model": best_model,
                "best_auc": round(baselines[best_model], 4),
                "question_business": profile["question_business"],
                "question_technical": profile["question_technical"],
            }
        )

    # Salva cheat-sheet (gitignored)
    cheatsheet_path = script_dir / "instructor_cheatsheet.json"
    with open(cheatsheet_path, "w", encoding="utf-8") as f:
        json.dump(cheatsheet, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Dataset salvati in:     {datasets_dir}/")
    print(f"  Cheat-sheet docente:    {cheatsheet_path}")
    print(f"  (il cheat-sheet è gitignored)")
    print(f"{'='*60}")

    # Riepilogo diversità
    print(f"\nRiepilogo diversità modelli:")
    best_counts: dict[str, int] = {}
    for entry in cheatsheet:
        bm = entry["best_model"]
        best_counts[bm] = best_counts.get(bm, 0) + 1
    for model, count in sorted(best_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: migliore in {count}/{n_datasets} dataset")


if __name__ == "__main__":
    main()
