# Skill: Create Challenge Material

This skill defines how to generate graded/challenge content.

All graded content must live under:

challenge/

Never create new materials under assignments/.

---

## Required Components

1) Brief section
   - Context
   - Task description
   - Deliverables
   - Evaluation criteria

2) Starter Code
   - Data loading provided
   - Clear TODO markers:
     # TODO(C1.1): ...
   - Sanity check cells optional

3) Submission Instructions
   - Exact filename required
   - Required CSV format:
       id
       churn_probability
   - Export instructions

4) Evaluation
   - Metric: ROC-AUC
   - No renaming columns
   - Leaderboard anonymized

---

## Structure Rules

- Clear markdown sections
- No solution included
- If solution needed:
  create separate *_solution.ipynb

---

## Artifact Rules

- Save predictions to:
  outputs/submissions/

Example:
outputs/submissions/lesson_05_submission.csv

---

## Validation Checklist

- ROC-AUC explicitly stated
- Submission format correct
- No solutions leaked
- No schema invented
- Clear grading rubric
- Consistent terminology:
    validation = development holdout
    test = final hidden set