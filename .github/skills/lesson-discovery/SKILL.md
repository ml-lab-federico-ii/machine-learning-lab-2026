---
name: lesson-discovery
description: Produces a lesson plan in markdown format based on the syllabus scope and the dataset understanding. This plan will guide the notebook-writing phase and ensure that all necessary analyses are performed in a logical order Use this skill whenever the user asks to create a lesson notebook "from scratch", asks for "outline/plan" for a lessonm, even without creating the notebook immediately.
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

### Step B — Syllabus declares the main topics, you explode them into a deep structured lesson plan

Derive the main topic of the lesson from the syllabus but explode it into a more detailed structure of sections and operations. For example, if the syllabus says that lesson 1 is about “formulating churn as a classification problem and performing EDA”, you should plan any possible exploratory analysis that can be done on the dataset to understand the target, feature distributions, feature correlation, missingness, imbalance, and any other relevant patterns. Then, you should organize these analyses into a logical sequence of sections that will form the backbone of the notebook. Each section should have a clear goal and specify which computations and outputs will be produced. This applies to all lessons: the syllabus gives you the high-level topics, but you need to decide the exact operations and their order based on the dataset and the learning objectives.

Write the section outline in Italian, with clear goals and specified computations/outputs, to include in `plan.md`.


### STEP C — Gather Existing Evidence (Previous Lessons)
Lessons should always be continuative: for example, if Lesson 2 is about preprocessing, you should reuse the dataset understanding from Lesson 1 and focus on cleaning, encoding, scaling, and splitting, rather than repeating the EDA. If Lesson 3 is about modeling, you should build on the cleaned dataset from Lesson 2 and focus on training models and evaluating them, rather than redoing the preprocessing. This way, each lesson builds on the previous ones and creates a coherent learning path for the students.

Before running new analyses, look for existing artifacts that can be reused.
Examples of useful evidence:

- Previous lesson plan(s): `outputs/lesson_plans/lesson_??_plan.md`
- Previous student notebook(s): `notebooks/lesson-??/lesson_??.py`
- Saved datasets: `outputs/data/*.parquet` or `outputs/data/*.csv`
- Saved figures: `outputs/figures/*.png`

Record in `plan.md` artifacts that you need to reuse these materials and which specific pieces of evidence you will leverage (e.g., “reusing the target distribution plot from Lesson 1”).

### Step D — Derive Notebook Sections + Concrete Operations

Based on the syllabus scope and the dataset understanding, derive a concrete structure of notebook sections and operations. For example, if the lesson is about EDA, you might have sections like “Understanding the target distribution”, “Analyzing feature distributions”, “Checking for missing values”, “Exploring feature correlations”, etc. Each section should have a clear goal (e.g., “Define the variable of interest and quantify its prevalence”) and specify which computations will be performed (e.g., `value_counts(normalize=True)`, `histogram`, `heatmap`, etc.) and which artifacts will be produced (e.g., saved figures, tables, etc.). This structured outline will guide the notebook-writing phase and ensure that all necessary analyses are performed in a logical order.

Write the section outline in Italian, with clear goals and specified computations/outputs, to include in `plan.md`.

---

## Output Artifacts

### 1) Markdown plan

Write: `outputs/lesson_plans/lesson_NN_plan.md`

Must include:
- Title, Authors, Emails, Last updated
- Italian summary of the lesson scope (3–6 bullets).
- List of existing evidence/artifacts that will be reused (with specific details).
- Detailed section outline with goals and specified computations/outputs.
- Clear formatting and structure for readability.

---

## Handoff to Notebook Writing

After producing both artifacts:
- instruct the notebook-writing phase to:
  - generate `notebooks/lesson-NN/lesson_NN.py` (Jupytext percent format)
  - include a `## Outline` section whose bullet list matches `sections[].title`
  - ensure all later interpretations cite real computed values

Use the `lesson-notebook` skill for the writing phase.
