# Skill: Create Lesson Notebook

This skill defines the standard workflow to create a new lesson notebook.

The workflow is topic-agnostic and must be reused for any lesson.

---

# Step 1 – Clarify Scope

Before generating the notebook:

- Confirm lesson number
- Confirm topic/title
- Confirm which sections must be included
- Confirm which sections must be excluded
- Confirm whether interactive reasoning is required
- Confirm whether modeling is part of the lesson

The syllabus (`syllabus/README.md`) defines the allowed scope of the lesson.

---

# Step 2 – Create Student Notebook

Notebooks must be authored as `.py` Jupytext percent-format files.

They will later be converted to `.ipynb` by an external script.

---

## Jupytext Cell Authoring (MANDATORY)

Markdown cell format:

# %% [markdown]
# ## Section Title
# Markdown explanation written in Italian
# - bullet point
# - bullet point

Code cell format:

# %%
import pandas as pd

All notebooks must follow this structure.

---

# Step 3 – Notebook Metadata

The notebook must start with a title block containing:

- Course name
- Lesson number and title
- Authors
- Emails
- Last updated (YYYY-MM-DD)

Example:

# %% [markdown]
# Machine Learning for Financial Analysis
#
# Lesson 02 — Exploratory Data Analysis
#
# Authors:
# Enrico Huber
# Pietro Soglia
#
# Last updated: 2026-03-04

---

# Step 4 – Required Structural Sections

Every notebook must contain:

1. Title block  
2. Learning objectives (3–6 measurable objectives)  
3. Setup  
4. Requested analytical sections  
5. Summary  

---

# Step 5 – Setup Section

The setup section must include:

- imports
- SEED definition (if stochastic processes appear)
- project root path definition
- artifact I/O contract

Example:

# %%
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
PROJECT_ROOT = Path.cwd()

---

# Step 6 – Incremental Analytical Workflow (MANDATORY)

Every analytical block must follow this pedagogical structure:

1) Markdown — objective of the analysis  
2) Code — focused computation or visualization  
3) Markdown — interpretation of results  

Structure:

Markdown → Code → Markdown interpretation

Rules:

- Each code cell performs one focused task
- Avoid mixing unrelated analyses
- Interpretation must reference actual computed outputs
- Avoid generic commentary

Interpretation cells must contain 3–7 bullet points explaining:

- magnitude of results
- patterns or trends
- comparisons between variables
- implications for modeling when relevant

---

# Step 7 – Execution-Aware Generation

Notebook generation must be outcome-driven.

Copilot may execute notebook code cells during generation in order to:

- inspect dataframe structure
- compute statistics
- generate plots
- validate code correctness
- extract real values for interpretation

Markdown explanations must always reflect the actual observed outputs.

Placeholder interpretations are not allowed.

---

# Step 8 – Code Standards

Code must be:

- readable
- educational
- minimal but rigorous

Guidelines:

- Use clear variable names
- Avoid overly long cells
- One plot per cell
- Always include plot titles and axis labels
- Avoid unnecessary scientific notation in outputs

When preprocessing appears:

- use Pipeline
- use ColumnTransformer

Avoid data leakage.

---

# Step 9 – Content Density Principle

Lesson notebooks must prioritize analytical richness.

Prefer generating more analytical blocks than strictly necessary.

Instructors will trim content if needed.

Never under-generate analysis.

---

# Step 10 – Artifact Discipline

If the notebook produces outputs, they must be saved under:

outputs/

Use lesson-specific filenames.

Example:

outputs/data/lesson_02_clean.parquet

When reading artifacts:

- validate file existence
- avoid silent failures

---

# Step 11 – Interactive Reasoning Blocks (Optional)

If interactive reasoning is required, include a section:

## Domande guidate

Rules:

- 4–6 questions
- Questions must refer to computed outputs
- Each question must be immediately followed by its answer
- Answers must be concise and data-driven

Avoid speculation.

---

# Step 12 – Summary Section

Every notebook must end with a summary section.

This section must:

- recap the main analytical findings
- highlight implications for modeling
- connect with future lessons when relevant

---

# Step 13 – Validation Checklist

Before finishing generation verify:

- notebook sections match requested ones
- no excluded sections appear
- markdown is written in Italian
- no invented dataset schema
- no teacher-directed language
- notebook structure follows Markdown → Code → Markdown
- code executes without errors
- repository structure has not been modified