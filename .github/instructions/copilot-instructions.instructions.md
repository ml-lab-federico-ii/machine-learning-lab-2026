---
description: Global standards for generating or editing any material for the "Machine Learning for Financial Analysis" lab.
applyTo: "**/*"
---

# Global Project Standards

This repository contains teaching material for a 5-lesson Machine Learning lab 
for the MSc in Mathematics at University of Naples Federico II.

Instructors:
- Enrico Huber
- Pietro Soglia

All generated material MUST follow the standards below.

---

# Repository Architecture

Key folders:

- notebooks/
- challenge/
- outputs/
- src/
- docs/

Never change the repository structure unless explicitly requested.

---

# Naming Conventions

- Notebooks live under: `notebooks/lesson-0N/`
- Student notebook:
  - `lesson_{NN}.ipynb`
- Live coding notebook:
  - `lesson_{NN}_live_coding.ipynb`

Functions: snake_case  
Classes: PascalCase  

Variables:
- X, y
- X_train, X_val, X_test
- df_raw, df_clean, df_feat

---

# Reproducibility (MANDATORY)

- Define a single constant: `SEED = <int>`
- Reuse SEED consistently
- Set numpy, random, and model random_state

---

# Artifact-Based Communication

Notebooks must communicate only through disk artifacts.

All outputs go to:

outputs/
├── data/
├── config/
├── models/
├── predictions/
├── figures/
├── submissions/

Never rely on shared kernel state between notebooks.

---

# Coding Philosophy

- Code is teaching material.
- Clarity > cleverness.
- Prefer small functions.
- Avoid data leakage.
- Use Pipeline + ColumnTransformer by default.

---

# Language Rules

- Markdown must be written in Italian.
- Code can use standard ML English naming.
- Avoid teacher-directed language.
- Material must be self-contained.

---

# Challenge Canonical Rules

- Metric: ROC-AUC
- Submission columns: `id`, `churn_probability`
- Never rename them.

---

# Safety Rules

- Never invent dataset columns.
- Always adapt markdown to actual data output.
- Avoid financial advice.
- Discuss cost of false positives/false negatives when relevant.

---

# Collaboration

- Do not perform git operations.
- Keep structure consistent.
- Avoid regressions.
- Reuse helpers from src/ when possible.