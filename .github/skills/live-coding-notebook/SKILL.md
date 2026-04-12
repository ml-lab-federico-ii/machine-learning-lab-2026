---
name: live-coding-notebook
description: Generates the live coding version of a lesson notebook (.py Jupytext percent-format) by replacing 3–6 carefully chosen code blocks with TODO(LIVE) markers. Use this skill whenever the user asks to create a live coding notebook, a classroom version of a notebook, a version with TODO blocks for class, or wants to prepare a lesson for live demonstration with students. Also use it when the user says things like "voglio fare live coding sulla lezione X", "crea la versione per la classe", or "prepara il notebook con i TODO per la lezione".
---

# Skill: Create Live Coding Notebook

A live coding notebook is used during class. The teacher types the missing pieces
in real time while students observe and reason along. The goal is **not** to
create a fill-in-the-blanks exercise — it is to give the teacher 3–6 natural
pause points where live reasoning and typing makes the lesson more engaging and
helps students internalize key concepts by watching them be built from scratch.

---

## Output

- **File:** `notebooks/lesson-0N/lesson_NN_live_coding.py`
- **Format:** Jupytext percent-format (identical file header and metadata to the
  student notebook)

---

## Step-by-step Process

### Step 1 — Read the student notebook completely

Before selecting any TODO block, read the entire student notebook
(`notebooks/lesson-0N/lesson_NN.py`). Understand:

- The learning objectives stated at the top
- The narrative arc (which concepts build on which)
- Which code cells do the "heavy intellectual lifting"

### Step 2 — Build a candidate list

Identify code cells that are **conceptually central**, **independently
implementable in 2–5 minutes**, and **dense but not huge** (typically 3–15 lines
of substantive logic).

Prefer cells that involve:

- Building or fitting a model (`fit`, `Pipeline`, estimator instantiation)
- Splitting data (train/val/test, SMOTE resampling)
- Computing key metrics (`roc_auc_score`, `classification_report`,
  `precision_recall_curve`)
- Running search or validation (`RandomizedSearchCV`, `GridSearchCV`,
  `cross_val_score` with `StratifiedKFold`)
- Engineering a meaningful feature
- Applying a threshold or scoring decision for business impact

### Step 3 — Apply filters

Discard candidates that are:

- **Boilerplate** — imports, SEED definition, path setup (`ROOT`, `DATA_OUT_DIR`),
  `warnings.filterwarnings`. These must never be blanked because they are
  the non-negotiable scaffolding every cell afterwards depends on.
- **Already fully explained in the preceding markdown cell** with the exact
  signature — trivial to copy-paste does not make for a interesting live moment.
- **Repetitive within this notebook** — if fitting a LogisticRegression is
  already a TODO, do not blank a second logistic regression fit. If computing
  `roc_auc_score` is a TODO, do not blank it again. Every TODO must exercise
  a *distinct* concept.
- **Pure print/display statements** — blanking `print(df.head())` is trivial
  and adds no learning value.
- **Would break all downstream cells unrecoverably** — avoid blanking a variable
  definition that is used 15 cells later with no intermediate checkpoint.

### Step 4 — Rank and select

From the filtered candidates, select **3 to 6** blocks. Apply these preferences:

1. **Spread across sections** — prefer TODO blocks that are distributed across
   at least 2 different BLOCCOs or sections, not all clustered in one area.
2. **Require out-loud reasoning** — prioritize cells where a teacher would
   naturally say "and why do we do it this way?" (e.g., choosing
   `class_weight='balanced'`, choosing `StratifiedKFold`, setting a
   hyperparameter range).
3. **Mark narrative turning points** — moments in the notebook where the lesson
   shifts from theory to first empirical evidence, or from one model to the next.

### Step 5 — Write the TODO blocks

Replace each selected cell's logic with a `# TODO(LIVE):` comment block.

**Format (standard):**

```python
# TODO(LIVE): <one-line imperative describing what to implement>
# Hint: <one sentence — which variables are in scope, what output is expected>
```

The first line tells **what** to do. The second line (optional but recommended)
gives just enough scaffolding so the teacher can start typing immediately without
needing to scroll back to find variable names. Do **not** give away the
implementation — the hint is a direction, not a solution.

**Example 1 — model training:**

Before:
```python
lr_cw = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr_cw.fit(X_train, y_train)
y_pred_lr_cw = lr_cw.predict(X_val)
y_prob_lr_cw = lr_cw.predict_proba(X_val)[:, 1]
```

After:
```python
# TODO(LIVE): addestrare LogisticRegression con class_weight='balanced'
# Hint: usa gli stessi parametri dell'esperimento precedente; output atteso: y_pred_lr_cw, y_prob_lr_cw
```

**Example 2 — cross-validation:**

Before:
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(rf_baseline, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

After:
```python
# TODO(LIVE): eseguire 5-fold stratified cross-validation sul random forest baseline
# Hint: usa scoring='roc_auc'; stampa mean ± std di cv_scores
```

**Example 3 — hyperparameter search:**

Before:
```python
search = RandomizedSearchCV(
    rf_baseline, param_dist, n_iter=30, cv=cv, scoring="roc_auc",
    n_jobs=-1, random_state=SEED, verbose=0,
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_
```

After:
```python
# TODO(LIVE): eseguire RandomizedSearchCV con n_iter=30 sul random forest
# Hint: param_dist e cv già definiti sopra; salva il miglior modello in best_rf
```

---

## What Must Never Be Blanked

1. All `import` and `from ... import` statements
2. `SEED = 42` and all seed-setting calls (`np.random.seed`, `random.seed`)
3. Path setup (`ROOT`, `DATA_OUT_DIR`, `FIGURES_DIR`, directory creation)
4. Helper function definitions (`load_preprocessed_split`, `save_current_figure`,
   `load_dataset_from_archive`, etc.)
5. Data loading cells — the notebook must be fully runnable up to the first TODO
6. All markdown cells, including interpretations, theory blocks, and section headers
7. Saving and export cells (last section — results must persist for downstream lessons)

---

## Structural Preservation

The live notebook must preserve **70–85% of the student notebook untouched**.

Both files must be identical in:

- File header (title, authors, emails, last updated date)
- All markdown cells
- Section order and headings
- `SEED` value and reproducibility setup
- Import block
- Helper function definitions
- Artifact paths and output filenames

---

## Validation Checklist

Before outputting the file, verify:

- [ ] 3–6 `TODO(LIVE)` blocks total
- [ ] Each TODO comment describes a concrete, single implementation step
  solvable in 2–5 minutes
- [ ] No two TODOs cover the same concept (no duplicate model fits, no repeat
  of the same metric computation, no blanking the same class twice)
- [ ] TODO blocks are spread across at least 2 different sections/BLOCCOs
- [ ] All cells *before* the first TODO are complete and individually runnable
- [ ] All markdown interpretation cells are untouched
- [ ] File is named `lesson_NN_live_coding.py` in `notebooks/lesson-0N/`
- [ ] `SEED` usage is preserved unchanged