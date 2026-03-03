# Skill: Create Live Coding Notebook

This skill defines how to generate the live coding version 
of any lesson notebook.

---

## Core Principle

Live coding notebook must be structurally identical to the student notebook.

Same:
- Headings
- Section order
- Setup
- Imports
- Artifact definitions

---

## Differences

Replace selected logic blocks with TODO markers:

# TODO(LIVE): implement train/validation split
# TODO(LIVE): build preprocessing pipeline
# TODO(LIVE): compute ROC-AUC

Guidelines:

- Leave meaningful gaps (not trivial ones)
- Keep structure complete
- Do not remove sections
- Do not change metadata
- Do not change artifact logic

---

## What Not To Do

- Do not simplify structure
- Do not remove sections
- Do not alter naming conventions
- Do not break reproducibility rules

---

## Validation

Before finalizing:

- Same headings as student version
- Same section order
- Same SEED usage
- Same artifact paths
- Only logic cells partially removed