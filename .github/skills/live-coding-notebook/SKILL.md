# Skill: Create Live Coding Notebook

This skill defines how to generate the live coding version of a lesson notebook.

---

# Core Principle

The live coding notebook must be structurally identical to the student notebook.

Same:

- headings
- section order
- setup code
- imports
- artifact paths
- reproducibility settings

The pedagogical structure must remain intact.

---

# Allowed Differences

Selected logic blocks must be replaced with TODO markers.

Example:

# TODO(LIVE): implement train/validation split
# TODO(LIVE): build preprocessing pipeline
# TODO(LIVE): compute ROC-AUC

Rules:

- gaps must be meaningful
- gaps must support live reasoning
- do not create trivial TODO tasks

---

# TODO Design Constraints

- Maximum 6 TODO(LIVE) blocks per notebook
- Each TODO must be solvable in 2–5 minutes
- TODO blocks must appear in key analytical steps

Typical examples:

- feature engineering step
- train/test split
- pipeline creation
- model training
- metric computation

---

# Structural Preservation

The live notebook must preserve 70–85% of the original student notebook.

Do NOT:

- remove entire sections
- alter headings
- change metadata
- change artifact paths

Markdown interpretation cells must remain intact.

---

# Validation

Before finalizing verify:

- headings identical to student notebook
- section order identical
- SEED usage preserved
- artifact paths unchanged
- only selected logic cells replaced with TODO(LIVE)