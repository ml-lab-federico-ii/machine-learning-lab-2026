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

# Repository structure (recommended)

When creating new files, use (or respect) this structure:

- `notebooks/`
  - `L1_*.ipynb`
  - `L2_*.ipynb`
  - `L3_*.ipynb`
  - `L4_*.ipynb`
  - `L5_*.ipynb`
- `assignments/`
  - `A1_*.ipynb` / `A1_*.md`
  - `A2_*.ipynb` / `A2_*.md`
- `challenge/`
  - `challenge_brief.md`
  - `baseline.ipynb`
  - `submission_template.csv`
- `src/`
  - `data.py` (loaders)
  - `preprocessing.py`
  - `metrics.py`
  - `plots.py`
  - `utils.py`
- `data/`
  - `raw/` (do not commit sensitive/private data)
  - `processed/`
- `docs/`
  - `syllabus.md`
  - `grading.md`
  - `faq.md`

If a different structure already exists, follow the existing one and keep naming conventions consistent.

---

# Global conventions (MUST)

## Naming
- Notebooks: `L{N}_{topic}_{short}.ipynb` (e.g., `L1_eda_churn_intro.ipynb`)
- Assignments: `A{N}_{topic}.ipynb` (e.g., `A2_preprocessing_pipeline.ipynb`)
- Functions: `snake_case`, classes `PascalCase`
- Variables:
  - `X` features, `y` target
  - `X_train, X_val, X_test`, `y_train, y_val, y_test`
  - `df_raw, df_clean, df_feat`

## Reproducibility
- Always set random seeds:
  - `numpy`, `random`, model `random_state`
- Always print environment info when relevant (python version optional).

## Outputs & artifacts
- Save important outputs to `outputs/` (create if missing):
  - figures: `outputs/figures/`
  - models: `outputs/models/`
  - submissions: `outputs/submissions/`
- Filenames include timestamp or version when appropriate.

## Minimal dependencies
Prefer:
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`
Optional only if already in repo:
- `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `shap`

Do NOT introduce new heavy dependencies unless asked.

---

# Notebook template (MUST follow)

Every lesson notebook must have the same skeleton and headings, in this order.

## 0. Title cell (Markdown)
Include:
- Course name
- Lesson number and title
- Authors: "Enrico Huber, Pietro Soglia"
- Last updated date (YYYY-MM-DD)

## 1. Learning objectives (Markdown)
Bullet list (3–6 items), measurable.

## 2. Setup (Code + Markdown)
- Imports
- Global constants
- Random seed
- Matplotlib defaults (simple)
- Paths (use `pathlib.Path`)
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

# Assignments template (MUST follow)

Each assignment file must include:

1) **Brief**
- Context
- Tasks
- Deliverables (explicit)
- Evaluation criteria (rubric-like bullets)

2) **Starter code**
- Data loading already provided (students don’t waste time)
- TODO blocks clearly marked:
  - `# TODO(A1.1): ...`
- Optional “sanity check” cells

3) **Submission**
- What to submit (file name)
- How to export:
  - notebook or script
  - plus required CSV if relevant

Do NOT include solutions unless explicitly requested (create a separate `*_solution.ipynb` if needed).

---

# Challenge/Leaderboard rules (MUST align)

When generating challenge material:
- Training set provided
- Test set hidden
- Students submit CSV with:
  - `id` column (or `customer_id`, depending on dataset)
  - `prediction` column (probability in [0, 1] for churn)
- Metric: ROC-AUC
- Deadline: before Lesson 5
- Leaderboard anonymized (use nicknames or hashed IDs)
- Provide:
  - baseline notebook
  - clear submission format check

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