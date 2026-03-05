---
name: lesson-discovery
description: Produces a data-driven lesson plan (scope + EDA findings + ordered section outline) for the "Machine Learning for Financial Analysis" lab by reading `syllabus/README.MD` and inspecting/loading the dataset from `data/archive.zip`. Use this skill whenever the user asks to create a lesson notebook "from scratch", asks for "outline/plan" for a lesson, or wants Copilot to decide notebook sections from the dataset before writing the notebook.
---

# Skill: Lesson Discovery (Scope → EDA → Plan Artifacts)

This skill is the **analysis-first** phase that runs *before* the notebook-writing phase.

Goal:
- translate generic syllabus bullets into a **concrete, executable notebook structure**;
- decide **what operations are required** (data inspection, EDA, preprocessing, modeling, evaluation, interpretability, etc.) based on the lesson outline;
- reuse evidence and artifacts from **previous lessons** when appropriate;
- validate the dataset contract by inspecting/loading data **only when needed**;
- persist results into two intermediate artifacts:
  - `outputs/lesson_plans/lesson_NN_plan.md`
  - `outputs/lesson_plans/lesson_NN_plan.json`

---

## Hard Constraints (Project Contracts)

- Dataset canonical location: `data/archive.zip`.
- First action when loading data: **list archive members**.
- Prefer reading CSV directly from the ZIP.
- Never write extracted raw files under `data/`; if needed, extract to `outputs/data/`.
- All narrative content written for humans must be **in Italian**.
- Do not invent dataset schema, target column, or label meanings: validate via EDA.

---

## Inputs to Confirm (Minimal)

- Lesson number `NN` (two digits).
- Lesson title (from `syllabus/README.MD` if not provided).
- Authors, Emails (do not guess), Last updated (YYYY-MM-DD; default: today).

If emails are missing, keep explicit `TODO` placeholders in plan artifacts.

---

## Workflow

### Step A — Read Syllabus Scope

- Open `syllabus/README.MD` and extract:
  - lesson title
  - intended learning objectives
  - any explicit sections/topics

Write a short Italian summary of the scope (3–6 bullets) to include in `plan.md`.

### Step B — Gather Existing Evidence (Previous Lessons)

Before running new analyses, look for existing artifacts that can be reused.
Examples of useful evidence:

- Previous lesson plan(s): `outputs/lesson_plans/lesson_??_plan.md|json`
- Previous student notebook(s): `notebooks/lesson-??/lesson_??.py`
- Saved datasets: `outputs/data/*.parquet` or `outputs/data/*.csv`
- Saved figures: `outputs/figures/*.png`

Rules:
- Prefer reusing prior computed insights when the syllabus indicates continuity.
  Example: if Lesson 2 says “missing values / outlier / split”, reuse Lesson 1
  dataset understanding and focus on transformations.
- If the required artifacts do not exist yet, fall back to the minimal analyses
  needed to produce the lesson safely.

Record in the plan:
- which prior artifacts were found and used
- which ones were missing (and how you compensated)

### Step C — Establish Data Context (When Needed)

- Check that `data/archive.zip` exists.
- List members (filenames).
- Select the most relevant CSV(s) based on:
  - name patterns (e.g. `train`, `test`, `churn`, `labels`, `customer`)
  - schema validation after loading (target presence, expected columns)

Only perform this step if the lesson requires interacting with raw data (e.g.
EDA in Lesson 1, preprocessing in Lesson 2, modeling in Lesson 3). If the lesson
is purely theoretical, skip loading data and focus on structure + examples.

Record:
- chosen file(s)
- rationale for the choice

### Step D — Execute Only the Required Analyses

Derive the required operations from the syllabus, then execute the minimum
analyses needed to support those operations with evidence.

Examples (non-exhaustive):

- If the lesson includes **EDA** topics: compute target distribution, summary
  stats, missingness, and a small set of feature analyses.
- If the lesson is about **preprocessing**: compute missingness/outliers
  diagnostics, define train/val/test split strategy, validate leakage risks,
  and design the preprocessing pipeline.
- If the lesson is about **metrics/imbalance**: compute baseline metrics and
  class balance; select suitable metrics and validation approach.

When the syllabus suggests continuity, reuse prior outputs instead of repeating
EDA. For example:
- Lesson 2 can reference Lesson 1’s dataset understanding and focus on cleaning
  + pipeline building.

If the target is ambiguous:
- propose 1–2 candidates and justify using evidence (column name + value set)
- do **not** proceed to modeling decisions without resolving the target

### Step E — Derive Notebook Sections + Concrete Operations

Translate scope + EDA into an ordered section list.

Each section must:
- have a clear goal (Italian)
- specify which computations/plots are needed
- specify what outputs/figures (if any) will be saved under `outputs/`

Keep the outline **lean** and consistent with the syllabus (no extra topics).

---

## Output Artifacts

### 1) Markdown plan

Write: `outputs/lesson_plans/lesson_NN_plan.md`

Must include (in Italian):
- lesson metadata (title, authors, emails, last updated)
- dataset provenance (archive members, selected file(s))
- EDA highlights with real computed numbers
- ordered outline (the exact section titles to use in the notebook)

### 2) JSON plan

Write: `outputs/lesson_plans/lesson_NN_plan.json`

Requirements:
- valid JSON
- stable keys (so the notebook-writer can consume it)

Suggested schema:

```json
{
  "lesson": {
    "number": "02",
    "title": "Preprocessing e Ingegneria delle Feature",
    "authors": ["Enrico Huber", "Pietro Soglia"],
    "emails": ["TODO", "TODO"],
    "last_updated": "2026-03-05"
  },
  "syllabus": {
    "source": "syllabus/README.MD",
    "theory_topics": ["..."],
    "practice_topics": ["..."]
  },
  "dependencies": {
    "previous_lessons_considered": ["01"],
    "artifacts_used": [
      "outputs/lesson_plans/lesson_01_plan.json",
      "notebooks/lesson-01/lesson_01.py"
    ],
    "artifacts_missing": []
  },
  "data": {
    "archive_path": "data/archive.zip",
    "members": ["..."],
    "selected_files": ["..."],
    "load_notes": "Perché questi file sono quelli corretti"
  },
  "evidence": {
    "n_rows": 10000,
    "n_cols": 14,
    "target": {
      "name": "Exited",
      "type": "binary",
      "positive_label": 1,
      "class_balance": {"0": 6064, "1": 3936}
    },
    "top_missing": [
      {"column": "Tenure", "missing_rate": 0.012}
    ]
  },
  "operations": [
    {
      "id": "split_strategy",
      "type": "data_preparation",
      "goal": "Definire train/validation/test evitando leakage.",
      "inputs": ["data/archive.zip"],
      "outputs": [],
      "notes": "Specificare criteri e seed."
    }
  ],
  "sections": [
    {
      "id": "eda_target",
      "title": "Comprendere il target e lo sbilanciamento",
      "goal": "Definire la variabile di interesse e quantificare la prevalenza.",
      "computations": ["value_counts(normalize=True)", "barplot"],
      "artifacts": []
    }
  ],
  "open_questions": [
    "TODO: confermare email autori"
  ]
}
```

Populate numeric fields with **real computed values** from the executed EDA.

If a lesson does not require EDA, keep `evidence` minimal and focus on
`operations` + `sections`.

---

## Handoff to Notebook Writing

After producing both artifacts:
- instruct the notebook-writing phase to:
  - generate `notebooks/lesson-NN/lesson_NN.py` (Jupytext percent format)
  - include a `## Outline` section whose bullet list matches `sections[].title`
  - ensure all later interpretations cite real computed values

Use the `lesson-notebook` skill for the writing phase.
