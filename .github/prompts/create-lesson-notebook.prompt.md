# Create Lesson Notebook

Generate a lesson notebook for the course 
"Machine Learning for Financial Analysis" (MSc Mathematics, Federico II).

Follow all global copilot instructions strictly.

---

## Required Parameters

Lesson number: {{lesson_number}}
Topic/title: {{topic_title}}

Sections to include:
{{sections_to_include}}

Sections to exclude:
{{sections_to_exclude}}

Include interactive reasoning section: {{yes/no}}

Is graded/challenge-related material: {{yes/no}}

---

## Structural Requirements

- Create notebook under:
  notebooks/lesson-0{{lesson_number}}/

- Student notebook:
  lesson_{{lesson_number}}.ipynb

- Also generate live coding version:
  lesson_{{lesson_number}}_live_coding.ipynb

Both notebooks must:
- Follow repository naming conventions
- Respect artifact-based communication
- Define SEED when stochastic components exist
- Use pathlib for paths
- Save artifacts in outputs/

---

## Content Rules

- Markdown must be in Italian.
- Markdown must be data and output-driven, so they must be created to explain in human language the outcomes of executed code.
- Code must be clear, readable, educational.
- Do not invent dataset columns.
- Adapt commentary to actual outputs.
- Avoid teacher-directed phrasing.
- Course syllabus is in syllabus/README.md and is the source of truth for lesson scope
- When generating lesson material, ensure content does not anticipate future lessons beyond brief pointers

---

## Section Logic

Only generate sections explicitly requested in:
"Sections to include".

Do NOT generate sections listed in:
"Sections to exclude".

Examples of sections:
- Learning objectives
- Setup
- Data loading
- Data inspection
- EDA
- Preprocessing
- Modeling
- Evaluation
- Interpretation
- Exercises
- Summary

---

## Interactive Reasoning (if enabled)

Add a markdown section:

## Domande guidate

Rules:
- 4–6 questions
- Each question must refer to computed results
- Each answer must be immediately below the question
- Answers must be data-driven

---

## Live Coding Notebook Rules

- Structure identical to student notebook
- Same headings and order
- Same setup/imports
- Replace selected key logic with:

# TODO(LIVE): ...

- Leave enough content for meaningful live development
- Do not remove sections

---

## Quality Checklist (must self-verify)

- No data leakage
- Pipelines used where needed
- SEED defined when relevant
- Artifacts saved correctly
- Plots have title and labels
- No regression of repository structure