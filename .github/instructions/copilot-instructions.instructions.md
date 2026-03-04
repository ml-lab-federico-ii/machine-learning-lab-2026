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

Copilot must never modify the repository structure unless explicitly requested.

---

# Naming Conventions

Notebooks live under:

notebooks/lesson-0N/

Files:

Student notebook  
lesson_NN.py

Live coding notebook  
lesson_NN_live_coding.py

Naming conventions:

Functions → snake_case  
Classes → PascalCase  

Variables:

X, y  
X_train, X_val, X_test  
df_raw, df_clean, df_feat

---

# Notebook Authoring Format (MANDATORY)

Lesson notebooks are written as **Jupytext Python scripts** and later converted
to `.ipynb` using an external shell script.

Copilot MUST therefore generate `.py` files using **Jupytext percent format**.

Cell syntax:

Code cell

# %%

Markdown cell

# %% [markdown]

Example structure:

# %% [markdown]
## Section title

Explanation of the analytical step.

# %%
# code block

# %% [markdown]
Interpretation of results.

Rules:

- Never generate `.ipynb` directly
- Always generate `.py` notebook sources
- Conversion to notebook is handled externally

---

# Incremental Notebook Construction (MANDATORY)

Lesson notebooks must be constructed **incrementally and outcome-driven**.

Copilot must follow this iterative workflow:

1. Write a markdown cell describing the analytical step.
2. Write the corresponding code cell.
3. Execute the code cell.
4. Inspect the produced outputs.
5. Write a markdown interpretation that refers to the actual results.

This cycle must repeat for every analytical block.

Structure:

Markdown (analysis intent)  
Code (computation)  
Execution  
Markdown (data-driven interpretation)

Markdown explanations must always be **based on real computed values**.

Generic interpretations are not allowed.

---

# Data Source Contract (MANDATORY)

The canonical dataset location is:

- data/archive.zip

Rules:

- Do NOT assume the presence of raw `.csv` files under `data/` unless they are explicitly present.
- When loading data, first inspect `data/archive.zip` content (list filenames).
- Select the correct source files from the archive by matching common patterns
  (e.g. `train`, `test`, `churn`, `labels`) and by validating the schema.
- Prefer reading directly from the zip when possible; otherwise extract to:

outputs/data/

- Never write extracted raw files back under `data/`.
- All notebooks must define a single helper function:

`load_dataset_from_archive(archive_path: Path, ...) -> pd.DataFrame`

that:
  - checks archive existence
  - lists members
  - loads the selected CSV(s)
  - raises a clear error if the expected file is not found


---

# Controlled Code Execution

Copilot may execute code **only within lesson notebook generation**.

Execution is allowed for the following purposes:

- Inspect dataframe structure
- Compute statistics
- Train models
- Generate plots
- Validate code correctness
- Extract real values for markdown interpretation

Execution must be limited to **local notebook cells**.

Copilot must NOT:

- run shell scripts
- modify the repository structure
- run git commands
- execute unrelated scripts

---

# Reproducibility (MANDATORY)

All notebooks must define a single seed constant.

Example:

```python
SEED = 42