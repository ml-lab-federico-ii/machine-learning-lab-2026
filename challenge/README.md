# Challenge In-Class — Churn Prediction

## Panoramica

Challenge a gruppi (~3 studenti) sulla predizione del churn bancario.
Ogni gruppo riceve un dataset sintetico unico (stesso schema, bias diversi)
e deve produrre un report HTML da presentare in un pitch di 3 minuti.

## Struttura

```
challenge/
├── challenge_notebook.ipynb   # Notebook studenti
├── generate_datasets.py       # Script generazione dataset (docente)
├── report_generator.py        # Generatore report HTML
├── evaluate.py                # Valutazione modelli + leaderboard (docente)
├── datasets/                  # 15 CSV train (seed_01..seed_15) + test gitignored
├── submissions/               # Bundle .joblib ricevuti (gitignored)
├── outputs/                   # Report HTML e modelli generati (gitignored)
├── instructor_cheatsheet.json # Risposte attese (gitignored)
└── README.md                  # Questo file
```

## Preparazione (Docente)

### 1. Generazione dataset (una tantum)

```bash
python challenge/generate_datasets.py --n-datasets 15 --n-rows 10000
```

I dataset vengono salvati in `challenge/datasets/seed_01.csv` ... `seed_15.csv`.
Viene generato anche `instructor_cheatsheet.json` con le metriche baseline
per ogni dataset (gitignored).

### 2. Assegnazione SEED

Assegnate un SEED (1–15) a ciascun gruppo. Ogni SEED corrisponde a un
dataset con caratteristiche diverse (churn rate, bias, pattern).

| SEED | Profilo | Churn approx |
|------|---------|--------------|
| 01 | Churn basso, segnale lineare | ~12% |
| 02 | Churn alto, effetto Age | ~28% |
| 03 | Interazioni non-lineari | ~20% |
| 04 | Forte rumore | ~22% |
| 05 | Squilibrio geografico | ~18% |
| 06 | Poche feature dominanti | ~21% |
| 07 | Forte effetto NumOfProducts | ~25% |
| 08 | Pattern diffuso | ~20% |
| 09 | Churn molto alto | ~35% |
| 10 | Clienti giovani | ~19% |
| 11 | Segnale nel Balance | ~22% |
| 12 | Churn basso con outlier | ~10% |
| 13 | Gender gap nel churn | ~23% |
| 14 | Segnale nel Tenure | ~20% |
| 15 | Multi-segnale bilanciato | ~21% |

## Timeline

| Fase | Durata | Attività |
|------|--------|----------|
| Setup | 2 min | SEED + nomi, caricamento dati |
| EDA | 10 min | Analisi esplorativa, osservazioni |
| Preprocessing | 10 min | Scaling, feature, split |
| Modellazione | 25 min | 3–4 modelli, tuning, confronto |
| Selezione | 10 min | Modello finale, interpretazione |
| Report | 3 min | Generazione HTML, invio mail |
| **Totale lavoro** | **60 min** | |
| Pitch | 3 min/gruppo | Presentazione al "cliente" |
| Q&A | 2 min/gruppo | Domande dal docente |

## Istruzioni per gli Studenti (copiabile)

> ### Challenge — Istruzioni
>
> 1. Aprite `challenge/challenge_notebook.ipynb`
> 2. Nella prima cella, inserite il SEED assegnato e i vostri nomi
> 3. Seguite il notebook blocco per blocco:
>    - **Blocco 1**: Analisi esplorativa (EDA)
>    - **Blocco 2**: Preprocessing (scegliete scaling, feature, split)
>    - **Blocco 3**: Modellazione (provate più modelli!)
>    - **Blocco 4**: Selezionate il modello migliore e scrivete le motivazioni
>    - **Blocco 5**: Generate il report HTML
> 4. Scaricate il file HTML dal file explorer (click destro → Download)
> 5. Inviatelo a **enrico.huber@bip-group.com** con oggetto:
>    `Challenge ML — SEED XX — Cognomi`
>
> **Tempo**: 60 minuti. Poi ogni gruppo presenta un pitch di 3 minuti.

## Rubrica di Valutazione (Pitch)

| Dimensione | Peso | Eccellente | Sufficiente | Insufficiente |
|------------|------|-----------|-------------|---------------|
| **Comprensione dati** | 25% | Osservazioni precise e quantitative | Osservazioni generiche | Nessuna analisi |
| **Scelte tecniche** | 25% | Scelte motivate e coerenti | Scelte ragionevoli senza motivazione | Scelte casuali |
| **Qualità modello** | 25% | Confronto rigoroso, metriche corrette | Un solo modello provato | Errori nelle metriche |
| **Comunicazione** | 25% | Pitch chiaro, insight originali | Pitch comprensibile | Confuso o incompleto |

## Troubleshooting

### "Dataset non trovato"
Verificate che `challenge/datasets/seed_XX.csv` esista.
Rigenerate con `python challenge/generate_datasets.py`.

### "ipywidgets non disponibile"
Il notebook funziona anche senza widget. Gli studenti modificano
direttamente le celle di configurazione Python (dizionari).

### "report_generator non trovato"
Il notebook cerca `report_generator.py` risalendo le cartelle.
Assicuratevi di eseguire il notebook dalla root del repository.

### "XGBoost non disponibile"
Gli studenti possono usare i 3 modelli restanti (LogReg, DT, RF).
Per installare: `pip install xgboost`.

---

## Valutazione Lato Docente (Test Set Nascosto)

Gli studenti consegnano **due file** via email:
- `report_seed_XX.html` — report con grafici e analisi
- `model_seed_XX.joblib` — bundle con modello + pipeline preprocessing

Il test set (15% del dataset originale, ~1500 righe) è salvato in
`challenge/datasets/seed_XX_test.csv` ed è gitignored — gli studenti
non lo vedono mai.

### 1. Raccogliere i file ricevuti

Salvate i `.joblib` ricevuti in `challenge/submissions/`:

```
challenge/submissions/
├── model_seed_03.joblib   # gruppo A
├── model_seed_07.joblib   # gruppo B
└── ...
```

### 2. Valutare un singolo gruppo

```bash
python challenge/evaluate.py --seed 3 --model challenge/submissions/model_seed_03.joblib
```

Output:
```
────────────────────────────────────────────────────────
  SEED 03 | Rossi · Bianchi · Ferrari
  Modello:       RandomForest
  AUC train:     0.8123
  AUC test:      0.7891   Δ=-0.0232  ✅ ok
  F1 test:       0.6544
  Precision:     0.7012
  Recall:        0.6142
────────────────────────────────────────────────────────
```

### 3. Leaderboard di tutti i gruppi

```bash
python challenge/evaluate.py --all --submissions-dir challenge/submissions/
```

Output:
```
============================================================
  LEADERBOARD — Challenge ML (test set nascosto)
============================================================
  #  SEED  Gruppo                  Modello            AUC_tr  AUC_te       Δ
  ─────────────────────────────────────────────────────────────────────────
  1     3  Rossi · Bianchi         RandomForest       0.8123  0.7891  -0.0232
  2     7  Neri · Verdi · Blu      XGBoost            0.7956  0.7712  -0.0244
  ...
```

Aggiungete `--json` per salvare la leaderboard in `submissions/leaderboard.json`.

### Nota su overfitting

Il flag ⚠️ appare quando `Δ AUC < -0.05`. Utile come spunto di discussione
durante il debriefing: "il vostro modello funzionava su dati che avete già visto?"
