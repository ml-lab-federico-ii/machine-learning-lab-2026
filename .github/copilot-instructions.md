# GitHub Copilot — Istruzioni Workspace (Machine Learning Lab 2026)

Queste istruzioni sono **always-on** e valgono per tutto il repository. Devono restare concise, ma coprire i vincoli non negoziabili del lab.

## Vincoli non negoziabili

- Non modificare la struttura del repository (cartelle chiave: `notebooks/`, `challenge/`, `outputs/`, `src/`, `docs/`) a meno di richiesta esplicita.
- I notebook di lezione si scrivono come **script Python Jupytext percent-format** (`.py`), non come `.ipynb`.
- I notebook vivono in `notebooks/lesson-0N/` con naming:
  - Studente: `lesson_NN.py`
  - Live coding: `lesson_NN_live_coding.py`
- Tutta la narrativa (celle markdown, commenti, docstring) deve essere **in italiano**.
- Le interpretazioni devono citare **valori realmente calcolati** (niente testo generico): seguire sempre ciclo *Markdown → Codice → Esecuzione → Interpretazione*.

## Convenzioni di naming (didattica)

- Funzioni: `snake_case`.
- Classi: `PascalCase`.
- Variabili: usare pattern didattici coerenti (`X, y`, `X_train, X_val, X_test`, `df_raw, df_clean, df_feat`).

## Formato notebook (Jupytext)

- Celle codice: `# %%`
- Celle markdown: `# %% [markdown]`
- Non generare `.ipynb` direttamente: la conversione è demandata a `tools/build_notebooks.sh` (eseguirlo solo se l’utente lo chiede o come validazione esplicita concordata).

## Data source contract (obbligatorio)

- Dataset canonico: `data/archive.zip`.
- Prima di caricare dati: elencare i membri dello ZIP e selezionare i CSV corretti validando lo schema.
- Non assumere l’esistenza di CSV “sciolti” sotto `data/`.
- Non estrarre raw sotto `data/`; se serve estrarre, usare `outputs/data/`.
- Ogni notebook deve definire **una singola helper function**:
  - `load_dataset_from_archive(archive_path: Path, ...) -> pd.DataFrame`
  - Deve: verificare esistenza archive, listare membri, caricare il/i CSV selezionati, e alzare un errore chiaro se il file atteso non c’è.

## Esecuzione controllata

- Eseguire codice **solo** nel contesto di generazione/validazione dei notebook di lezione (ispezione df, statistiche, training, plot, metriche).
- Evitare comandi potenzialmente distruttivi (git, pulizie, script generici) senza richiesta esplicita.

## Riproducibilità

- Ogni notebook deve definire una costante `SEED = 42` e usarla coerentemente.

## Workflow standard: “crea notebook per lezione NN”

1. **Discovery (analisi + planning):**
  - Usare la skill `lesson-discovery`

2. **Notebook writing:**
  - Generare `notebooks/lesson-0N/lesson_NN.py` usando la skill `lesson-notebook` con il piano creato.

## Skills del repo

- Usare `lesson-discovery` per creare plan `.md`.
- Usare `lesson-notebook` per scrivere il notebook studente.
