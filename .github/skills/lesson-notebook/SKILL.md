# Skill: Create Lesson Notebook

This skill defines the standard workflow to create a new lesson notebook.

This is topic-agnostic and must be reused for any lesson.

---

## Step 1 – Clarify Scope

Before generating:

- Confirm lesson number
- Confirm topic/title
- Confirm which sections are required
- Confirm whether interactive reasoning is required
- Confirm whether modeling is included

---

## Step 2 – Create Student Notebook

Path:
notebooks/lesson-0N/lesson_NN.ipynb

Must include:

1. Title cell
   - Course name
   - Lesson number & title
   - Authors
   - Emails
   - Last updated (YYYY-MM-DD)

2. Learning objectives (3–6 measurable)

3. Setup
   - Imports
   - SEED (if needed)
   - Path definitions using pathlib
   - Artifact I/O contract

4. Requested analytical sections only

5. Summary

---

## Step 3 – Code Standards

- Use clear variable names
- Use markdown to introduce code sections (e.g. "We are going to look at ...")
- Use markdown to summarize results (e.g. "We can see that ...")
- Avoid scientific notation in outputs when not strictly necessary
- Avoid overly long cells
- One plot per cell
- Titles and labels mandatory
- Use Pipeline + ColumnTransformer when preprocessing appears
- Avoid leakage

---

## Step 4 – Artifact Discipline

If notebook produces outputs:

- Save under outputs/
- Use lesson-number-based filenames
- Validate artifact existence when reading

Example:

outputs/data/lesson_02_clean.parquet

---

## Step 5 – Interactive Blocks (Optional)

If required:

Add section:

## Domande guidate

Each question must:
- Refer to computed output
- Be answered immediately
- Be concise
- Avoid speculation

---

## Step 6 – Validation Checklist

Before finishing:

- Sections match requested ones
- No unwanted sections included
- Markdown in Italian
- No invented schema
- No teacher-directed language
- No repository structure changes