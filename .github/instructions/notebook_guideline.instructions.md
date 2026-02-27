---
description: Load these instructions whenever generating or editing content for the "Machine Learning for Financial Analysis" lab (notebooks, assignments, challenges, utilities, docs). Use them for all code generation, reviews, and refactors.
applyTo: "**/*"
---

# Project context

This repository contains the teaching material for a 5-lesson Machine Learning lab focused on financial analysis,
for the MSc in Mathematics at University of Naples Federico II.

Instructors:
- Enrico Huber
- Pietro Soglia

Primary outputs:
- Jupyter notebooks for lessons (theory + guided practice)
- Assignments (individual/group)
- A final challenge (hidden test set, CSV submission, leaderboard)
- Minimal reusable utilities to avoid duplication (data loading, metrics, plotting helpers)

Goal: maintain a consistent look & feel across all materials, regardless of who creates them.

## Current repository scaffold (authoritative)

The repository is intentionally lightweight: it contains **structure + stubs**, and does **not** ship real datasets.

High-level layout (key files only):

```text
.
├─ README.md
├─ requirements.txt
├─ environment/
│  └─ README.md
├─ lessons/
│  ├─ lesson-01-churn-classification-eda/
│  │  └─ README.md
│  ├─ lesson-02-preprocessing/
│  │  └─ README.md
│  ├─ lesson-03-models-metrics/
│  │  └─ README.md
│  ├─ lesson-04-interpretability/
│  │  └─ README.md
│  └─ lesson-05-advanced-modeling/
│     └─ README.md
├─ notebooks/
│  ├─ lesson-01/eda.ipynb
│  ├─ lesson-02/preprocessing.ipynb
│  ├─ lesson-03/model_comparison.ipynb
│  ├─ lesson-04/feature_importance.ipynb
│  └─ lesson-05/tuning.ipynb
├─ assignments/
│  ├─ assignment-01/
│  └─ assignment-02/
├─ challenge/
│  ├─ overview/README.md
│  ├─ rules/README.md
│  ├─ baseline/README.md
│  ├─ evaluation/README.md
│  └─ submissions/
├─ data/
│  └─ sample/
├─ docs/
└─ syllabus/
```

Notes:
- Some folders contain a `.gitkeep` file only to make Git track empty directories.
- Reusable helpers may later be added under `src/` (if/when introduced, notebooks should import from there instead of duplicating code).

## Current notebooks status (IMPORTANT)

The lesson notebooks currently in `notebooks/` are **placeholders** and may not yet follow the full standard template (sections 0–10).

Rule:
- When expanding or refactoring any lesson notebook, **first align it to the standard template sections (0–10) in order**, then add content.

---

# Audience & tone

Assume students are:
- strong in math, mixed coding experience
- new to ML engineering best practices

Therefore:
- explain steps clearly
- avoid "magic": every transformation is explicit
- keep code readable and heavily commented where it teaches something
- prefer practical intuition + minimal necessary theory

---

# Global conventions (MUST)

## Naming
- Notebooks are stored under `notebooks/lesson-0N/`.
- **Naming policy going forward**: notebook filenames must reference the **lesson number** and not the content/topic.
  - Keep existing placeholder filenames as-is (legacy), but **do not copy the pattern** for new notebooks.
  - New lesson notebooks must be named:
    - `lesson_{NN}.ipynb` (e.g., `lesson_01.ipynb`) inside `notebooks/lesson-0N/`
    - If multiple notebooks are needed for a lesson: `lesson_{NN}_part_{MM}.ipynb` (e.g., `lesson_01_part_01.ipynb`)
- Graded work (formerly “assignments”) is treated as part of the **challenge** concept (see below). Avoid creating new files under `assignments/`.
- Functions: `snake_case`, classes `PascalCase`
- Variables:
  - `X` features, `y` target
  - `X_train, X_val, X_test`, `y_train, y_val, y_test`
  - `df_raw, df_clean, df_feat`

## Reproducibility
- Always set random seeds:
  - `numpy`, `random`, model `random_state`
- Always print environment info when relevant (python version optional).

Additional rule:
- When a notebook includes any stochastic component (splits, models, CV, sampling), define a single constant `SEED = <int>` near the top and reuse it consistently.

## Outputs & artifacts
- Save important outputs to `outputs/` (create if missing):
  - data: `outputs/data/`
  - config: `outputs/config/`
  - figures: `outputs/figures/`
  - models: `outputs/models/`
  - predictions: `outputs/predictions/`
  - submissions: `outputs/submissions/`
- Filenames include timestamp or version when appropriate.

Repo hygiene rules:
- Avoid committing very large files or binaries.
- Datasets used for the course may come from **Kaggle (educational purpose)**.
  - If you decide to version datasets inside the repository, place them under `outputs/data/` and keep sizes reasonable.
  - Prefer saving only what is necessary for teaching (e.g., a sample/small subset) when possible.

## Minimal dependencies
Prefer:
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`
Optional only if already in repo:
- `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `shap`

Do NOT introduce new heavy dependencies unless asked.

---

# Notebook template (MUST follow)

Every lesson notebook must have the same skeleton and headings, in this order.

Checklist (enforceable):
- Cell 0 is a **Title markdown cell** including: course name, lesson number & title, authors, emails, last updated date (YYYY-MM-DD).
- Sections 1–10 exist and appear **in order**, with the exact heading numbers.

## Multi-notebook workflow (MUST)

The 0–10 structure below can look like a single end-to-end notebook, but the course may use **multiple notebooks per lesson** (or across lessons).

Rules for splitting work across notebooks:
- Each notebook must still include sections **0–10 in order**, but sections can be **brief** and may reference previously generated artifacts.
- Notebooks must **communicate via files on disk**, not via shared kernel state.
  - Assume a notebook can be run independently on a fresh kernel.
- In **Section 2 (Setup)** each notebook must clearly define:
  - `SEED` (when stochastic steps exist)
  - project paths using `pathlib.Path`
  - the **input artifacts** it expects and the **output artifacts** it writes
- Use `outputs/` as the single handoff location for intermediate results:
  - `outputs/data/` (intermediate datasets)
  - `outputs/config/` (params/config as JSON)
  - `outputs/models/` (serialized models)
  - `outputs/predictions/` (OOF/val/test predictions)
  - `outputs/figures/` (figures)
  - `outputs/submissions/` (challenge submissions)
- Prefer stable, lesson-number-based filenames for artifacts (examples):
  - `outputs/data/lesson_01_raw.parquet`, `outputs/data/lesson_02_model_matrix.parquet`
  - `outputs/config/lesson_03_params.json`
  - `outputs/models/lesson_03_baseline.joblib`
  - `outputs/predictions/lesson_05_test_proba.csv`
- When reading artifacts, validate existence and raise a friendly, actionable error (e.g., “Run Lesson 02 notebook first to generate …”).
- Avoid leakage across steps: if you persist splits or preprocessing, store the split indices/IDs and fit preprocessing **only on train**.

## 0. Title cell (Markdown)
Include:
- Course name
- Lesson number and title
- Authors: "Enrico Huber, Pietro Soglia"
- Emails: "enricohuber5@gmail.com", "pietro.soglia@gmail.com"
- Last updated date (YYYY-MM-DD)

## 1. Learning objectives (Markdown)
Bullet list (3–6 items), measurable.

## 2. Setup (Code + Markdown)
- Imports
- Global constants
- Random seed
- Matplotlib defaults (simple)
- Paths (use `pathlib.Path`)
- Artifact I/O contract (expected inputs + produced outputs under `outputs/`)
- Helper functions import from `src/` if present

## 3. Data loading (Code + Markdown)
- Load dataset via a single function (prefer `src/data.py`)
- Show:
  - shape
  - head
  - dtypes
  - missingness overview

## 4. Exploratory analysis (EDA) (Mixed)
- Target distribution
- Key univariate plots
- Correlations only when meaningful (avoid overdoing)
- Clear takeaways as short markdown notes

## 5. Preprocessing (Mixed)
- Train/val/test split (explain rationale)
- Handle:
  - missing values
  - categorical encoding
  - scaling when needed
- Use `Pipeline` + `ColumnTransformer` by default
- Avoid leakage: fit only on train, transform val/test

## 6. Modeling (Mixed)
- Start from a baseline model
- Add at least one stronger model
- Use consistent evaluation function
- Explain trade-offs (interpretability vs performance)

## 7. Evaluation (Mixed)
For classification:
- ROC-AUC as primary
- Also report: precision/recall, confusion matrix at a chosen threshold
- Discuss thresholding and costs (financial context)
For regression:
- MAE/RMSE, residual analysis

## 8. Interpretation (Mixed)
- Feature importance (model-based or permutation)
- Optional: SHAP if dependency exists
- Keep narrative focused: “what would we tell a stakeholder?”

## 9. Exercises (Markdown)
- 3–6 exercises
- Tag difficulty: `[easy] [medium] [hard]`
- Provide TODO hints, not full solutions

## 10. Summary (Markdown)
- 5–10 bullets: what we learned + common pitfalls

---

# Graded tasks / Challenge template (MUST follow)

This repository treats graded work as part of the **challenge** concept.

Rule:
- Do not create new “assignment” materials under `assignments/` going forward.
- If you need a structured graded deliverable, create it under `challenge/` (e.g., a baseline notebook, task notebook, or README) and follow the same structure below.

If a graded notebook/task is created (under `challenge/`), it must include:

1) **Brief**
- Context
- Tasks
- Deliverables (explicit)
- Evaluation criteria (rubric-like bullets)

2) **Starter code**
- Data loading already provided (students don’t waste time)
- TODO blocks clearly marked (use a consistent tag, e.g. `# TODO(C1.1): ...`)
- Optional “sanity check” cells

3) **Submission**
- What to submit (file name)
- How to export (notebook or script)
- Plus required CSV if relevant

Do NOT include solutions unless explicitly requested (create a separate `*_solution.ipynb` if needed).

---

# Challenge/Leaderboard rules (MUST align)

When generating challenge material:
- Training set provided
- Test set hidden
- Students submit CSV with:
  - `id` column (or `customer_id`, depending on dataset)
  - `churn_probability` column (probability in [0, 1])
- Metric: ROC-AUC
- Deadline: before Lesson 5
- Leaderboard anonymized (use nicknames or hashed IDs)
- Provide:
  - baseline notebook
  - clear submission format check

Canonical specification (single source of truth):
- **Metric**: ROC-AUC
- **Submission columns**: `id`, `churn_probability`
- Do **not** rename these columns in notebooks, docs, or evaluation scripts.

---

# Coding guidelines (MUST)

## Style
- Code is teaching material: prioritize clarity over cleverness
- Prefer small, named functions over long cells
- Add comments explaining *why*, not only *what*
- Use type hints in `src/` modules when easy

## Error handling
- Validate inputs in reusable functions (`src/`)
- Raise friendly errors with actionable messages

## Plotting
- Use matplotlib
- One plot per cell (avoid clutter)
- Titles/labels always present
- If plotting distributions: show counts or percentages when relevant

## Metrics utilities
Centralize evaluation logic in `src/metrics.py`:
- `evaluate_classifier(model, X_train, y_train, X_val, y_val) -> dict`
- consistent printed table
- return metrics dict for later comparison

---

# Data & ethics (MUST)

- Never invent dataset columns. Inspect and adapt to actual schema.
- Avoid including any sensitive personal data in examples.
- When discussing financial impacts:
  - mention costs of false positives/false negatives
  - avoid giving real investment advice

---

# Collaboration rules (MUST)

To keep materials consistent between Enrico and Pietro:
- Reuse the same headings and cell order
- Reuse shared helpers from `src/` instead of duplicating code
- If creating new helpers, document them at the top of the file
- Keep terminology consistent:
  - “validation” means held-out set used during development
  - “test” means final/hidden evaluation
- Leave all the git activities to the owners (Enrico and Pietro) to avoid merge conflicts and maintain a single source of truth. Owners will make add, commit, push, and merge all changes to the repository.
- Avoid regression: when editing existing notebooks, keep the same structure and consistency and avoid breaking already working code.

## Language & tone consistency

- Student-facing materials (README, lesson notes, notebook markdown) should be written in **Italian**.
- Code should be readable for Italian-speaking students:
  - Prefer clear variable/function names; English names are acceptable and expected in ML contexts (features, model names, library APIs).
  - Comments should explain *why* (and can be in Italian), not only *what*.

---

# Review checklist (for PRs / edits)

When asked to review changes, check:
- Notebook follows the template sections and order
- No data leakage
- Random seeds set
- Pipelines used (not manual fit/transform per split)
- Outputs saved in the right folders
- Exercise prompts are clear and scoped
- No solutions leaked in student-facing files