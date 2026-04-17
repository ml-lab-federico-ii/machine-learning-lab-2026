# Create Lesson Notebook

Generate a lesson notebook for the course  
"Machine Learning for Financial Analysis" (MSc Mathematics, Federico II).

Follow all global Copilot instructions strictly.

---

# Required Parameters

Lesson number: {{lesson_number}}  
Topic/title: {{topic_title}}

Sections to include:
{{sections_to_include}}

Sections to exclude:
{{sections_to_exclude}}

Include interactive reasoning section: {{yes/no}}

Is graded/challenge-related material: {{yes/no}}

---

# Structural Requirements

Create notebook under:

notebooks/lesson-0{{lesson_number}}/

Student notebook:

lesson_{{lesson_number}}.py

Also generate live coding version:

lesson_{{lesson_number}}_live_coding.py

Both notebooks must:

- Follow repository naming conventions
- Respect artifact-based communication
- Define SEED when stochastic components exist
- Use pathlib for paths
- Save artifacts inside outputs/

---

# Notebook File Format

The notebook must be written using **Jupytext percent format**.

Cell syntax:

Code cell

# %%

Markdown cell

# %% [markdown]

Example:

# %% [markdown]
## Section Title

Explanation of the analytical step.

# %%
# Python code

# %% [markdown]
Interpretation of the results.

Never generate `.ipynb` files.  
Notebooks must always be authored as `.py` sources.

---

# Notebook Construction Workflow (MANDATORY)

The notebook must be constructed **incrementally and outcome-driven**.

For every analytical block Copilot must follow this cycle:

1. Write a markdown cell explaining the analytical objective.
2. Write the corresponding code cell.
3. Execute the code cell.
4. Inspect the outputs (tables, metrics, plots).
5. Write a markdown interpretation referring to the real outputs.

Structure of each block:

Markdown → Code → Execution → Markdown Interpretation

Markdown interpretations must always reference:

- real computed values
- relative differences
- trends
- patterns in the outputs

Generic interpretations are not allowed.

---

# Content Rules

Markdown must be written in **Italian**.

Markdown explanations must:

- be data-driven
- refer to actual outputs
- avoid vague language
- avoid teacher-directed phrasing

Prefer phrasing like:

"Si osserva che la variabile mostra una distribuzione asimmetrica."

Avoid:

"Come possiamo vedere..."

Code must be:

- clear
- readable
- educational
- minimal but rigorous

The course syllabus located in:

syllabus/README.md

is the **source of truth for lesson scope**.

Do not anticipate future lessons beyond brief pointers.

---

# Analytical Depth Requirement (MANDATORY)

Analytical coverage must be substantial.

The syllabus defines **scope**, not depth.

Minimum expectations for analytical sections:

- At least **12–18 analytical blocks** (Markdown → Code → Markdown)
- At least **8–12 visualizations when applicable**
- At least **4 grouped statistical comparisons**
- Correlation analysis whenever numeric features exist
- Multiple bivariate views when a target variable exists
- At least **2 alternative analytical perspectives for key findings**

Whenever a dataset is loaded the notebook must include:

- dataset shape
- data types
- missing values inspection
- descriptive statistics
- target distribution (if applicable)

---

# Visualization Rules

Plots must:

- include titles
- include axis labels
- use readable figure sizes
- use matplotlib or seaborn
- be followed by a markdown interpretation

---

# Interpretation Rule

Every table, metric, or plot must be followed by a markdown interpretation.

Interpretations must reference:

- magnitudes
- differences
- relationships
- potential modeling implications

Avoid vague statements.

---

# Engagement Rule

Lesson notebooks must:

- be intellectually engaging
- guide the reader step by step
- include structured reasoning
- encourage analytical thinking

Use clear logical transitions between sections.

---

# Section Logic

Only generate sections listed in:

Sections to include

Never generate sections listed in:

Sections to exclude

If a section appears in **include**, it must always be generated.

If a section appears in **exclude**, it must never appear.

Examples of sections:

- Learning objectives
- Setup
- Data loading
- Data inspection
- Exploratory Data Analysis
- Preprocessing
- Feature engineering
- Modeling
- Evaluation
- Interpretation
- Exercises
- Summary

---

# Interactive Reasoning (if enabled)

Add the following section:

## Domande guidate

Rules:

- 4–6 questions
- Each question must refer to computed results
- Each answer must appear immediately below the question
- Answers must be data-driven

---

# Live Coding Notebook Rules

The live coding notebook must have:

- identical structure
- identical section headings
- identical order of sections
- identical imports and setup

Replace selected implementation steps with:

# TODO(LIVE): implement this step during lecture

The notebook must still remain coherent.

Do not remove sections.

Leave enough material for meaningful live development.

---

# Determinism Requirement

Whenever stochastic components are used, define:

SEED = 42

Use the seed consistently across:

- numpy
- random
- scikit-learn models

Example:

model = RandomForestClassifier(random_state=SEED)

---

# Artifact Rules

Define project root using pathlib.
Project root must be the root folder of the repository.
All outputs must be saved under:

outputs/

Structure:

outputs/
data/
config/
models/
predictions/
figures/
submissions/

Never rely on shared kernel state between notebooks.

---

# Dataset Location (MANDATORY)

The dataset is stored under:

data/archive.zip

Notebook code must not call `pd.read_csv` on non-existing paths.
It must load data from the zip archive by inspecting its content first.

---

# Challenge Rules (if applicable)

Metric:

ROC-AUC

Submission format:

id,churn_probability

These column names must never change.

---

# Final Quality Checklist (MANDATORY)

Before finishing generation Copilot must verify:

- Notebook follows Markdown → Code → Markdown structure
- All analytical outputs have interpretations
- Code runs without errors
- Markdown explanations match outputs
- Plots contain titles and labels
- SEED is defined when randomness exists
- No repository structure modifications occurred
- Artifacts are saved correctly