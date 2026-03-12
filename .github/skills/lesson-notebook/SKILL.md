---
name: lesson-notebook
description: Generates a complete student lesson notebook (.py Jupytext percent-format) for the "Machine Learning for Financial Analysis" lab. Use this skill whenever the user asks to create, write, build, generate, or scaffold any lesson notebook,   even if they only mention a topic or a lesson number without explicitly saying "notebook". Also use it when the user asks to add a new lesson, continue an existing lesson, or flesh out a lesson outline.
---

# Skill: Create Lesson Notebook

This skill defines the standard workflow to create a new lesson notebook.

The workflow is topic-agnostic and reused for every lesson.

---

## Language Contract

**All narrative content — markdown cells, comments, docstrings — must be written
in Italian.** This applies throughout the notebook without exception. Code
identifiers remain in English.

---

# Step 1 – Clarify Scope

Before generating the notebook, confirm:

- Lesson number (e.g., `02`)
- Topic / title
- Authors (default: course instructors)
- Emails for the authors (do not guess; ask if not provided)
- Last updated date to print in the header (YYYY-MM-DD; default: today)
- Sections to include
- Sections to exclude
- Whether interactive reasoning ("Domande guidate") is needed
- Whether a modeling section is in scope

The syllabus (`syllabus/README.md`) defines the canonical lesson scope.

If the user wants Copilot to derive sections from the dataset, first run the
**lesson-discovery** phase to produce:

- `outputs/lesson_plans/lesson_NN_plan.md`

Then generate the notebook using those artifacts (especially the ordered
section list) to avoid inventing structure.

---

# Step 2 – File Format

Notebooks are authored as **Jupytext percent-format `.py` files** and later
converted to `.ipynb` by `tools/build_notebooks.sh`.

Never create `.ipynb` files directly.

Target path: `notebooks/lesson-NN/lesson_NN.py`

### Cell syntax

Markdown cell:

```python
# %% [markdown]
# ## Titolo sezione
#
# Testo narrativo in italiano.
# - punto elenco
# - punto elenco
```

Code cell:

```python
# %%
import pandas as pd
```

Every `# ` prefix in a markdown cell is part of the Jupytext comment syntax.

---

# Step 3 – Required Sections

Every notebook must contain these sections in order:

1. **Title + Metadata + Learning objectives** — one combined cell
2. **Outline** — one markdown cell derived from the discovery plan
3. **Setup** — imports, constants, helpers (see Step 4)
4. **Requested analytical sections** — one or more (see Step 6)
5. **Domande guidate** — optional (see Step 8)
6. **Riepilogo** — mandatory closing section (see Step 9)

### Outline section example

The Outline must be derived from `outputs/lesson_plans/lesson_NN_plan.json`
(`sections[].title`) when available.

```python
# %% [markdown]
# ## Outline
#
# - Comprendere il target e lo sbilanciamento
# - Pulizia dati: valori mancanti e outlier
# - Encoding delle categoriche e scaling delle numeriche
# - Pipeline sklearn: ColumnTransformer + modello baseline
```

### Title cell example

```python
# %% [markdown]
# # Machine Learning per l’Analisi Finanziaria
#
# ## Lezione 02 — Preprocessing e Ingegneria delle Feature
#
# **Authors:**
# - Enrico Huber
# - Pietro Soglia
#
# **Emails:**
# - TODO: inserire email
# - TODO: inserire email
#
# **Last updated:** 2026-03-05
#
# ## Obiettivi di apprendimento
#
# - Identificare e gestire valori mancanti con strategie appropriate.
# - Applicare encoding e scaling in modo corretto e non-leaky.
# - Costruire una Pipeline sklearn riproducibile.
# - Separare train / validation / test rispettando la data boundary.
```

The **Authors / Emails / Last updated** block is mandatory. If emails are not
provided, do not invent them: keep explicit TODO placeholders and ask the user
to supply the correct addresses.

---

# Step 4 – Setup Section

The setup section standardises paths, I/O contracts, and reproducibility.
Copy this pattern verbatim for every notebook (adjust lesson number and
relevant output subdirs as needed):

```python
# %% [markdown]
# ## Setup
#
# Definiamo dipendenze, percorsi e cartelle di output.

# %%
from __future__ import annotations

import random
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.2f}".format)


def resolve_project_root() -> Path:
    """Risolve la root del repository indipendentemente dalla cwd."""
    start_points = [Path.cwd().resolve()]
    if "__file__" in globals():
        start_points.append(Path(__file__).resolve().parent)
    for start in start_points:
        for candidate in [start, *start.parents]:
            if (candidate / "data" / "archive.zip").exists() and (
                candidate / "notebooks"
            ).exists():
                return candidate
    raise FileNotFoundError(
        "Impossibile determinare la root del progetto. "
        "Atteso un percorso contenente data/archive.zip e notebooks/."
    )


ROOT = resolve_project_root()
DATA_ARCHIVE_PATH = ROOT / "data" / "archive.zip"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for _dir in [OUTPUTS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_current_figure(filename: str) -> None:
    """Salva la figura corrente in outputs/figures/."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=120, bbox_inches="tight")


def load_dataset_from_archive(
    archive_path: Path,
    filename_patterns: tuple[str, ...] = ("train", "test", "churn", "customer"),
) -> pd.DataFrame:
    """Carica il CSV principale dall'archivio ZIP validando lo schema."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archivio dati non trovato: {archive_path}")
    with ZipFile(archive_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        print("Membri archivio:")
        for m in members:
            print(f"  - {m}")
        candidates = [
            m for m in members
            if any(p in m.lower() for p in filename_patterns) and m.endswith(".csv")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Nessun CSV trovato nell'archivio con i pattern {filename_patterns}. "
                f"Membri disponibili: {members}"
            )
        chosen = candidates[0]
        print(f"\nFile selezionato: {chosen}")
        with zf.open(chosen) as f:
            return pd.read_csv(f)
```

Add additional subdirectories (e.g., `MODELS_DIR`, `PREDICTIONS_DIR`) when
the lesson involves model persistence.

---

# Step 5 – Data Loading Block

After Setup, always include an explicit data loading block that:

1. Calls `load_dataset_from_archive(DATA_ARCHIVE_PATH)`.
2. Prints shape and a `.head()`.
3. Prints `.dtypes` and `.describe()`.

This establishes the dataset contract visibly for students.

---

# Step 6 – Incremental Analytical Workflow

Every analytical block follows this three-part cycle.
Never skip to code without the intent markdown, or close a topic without an
interpretation:

```
Markdown (intent)  →  Code (computation)  →  Markdown (interpretation)
```

**Intent markdown** explains *why* this step matters for modeling or business
understanding — one short paragraph, no bullets.

**Code cell** performs one focused task. Avoid mixing unrelated operations.
One plot per cell.

**Interpretation markdown** contains 3–7 bullet points that reference the
*actual computed values* produced by running the cell. Generic commentary
("i dati mostrano una distribuzione normale") is not allowed. Concretise:
"Il 20.4% dei clienti ha abbandonato la banca (3,936 / 10,000) — la classe
positiva è minoritaria; occorrerà valutare strategie di bilanciamento."

Execute each code cell before writing its interpretation so the values are real.

---

# Step 7 – Code Standards

Write code that is readable, educational, and minimal:

- Clear variable names (`X_train`, `df_feat`, not `a`, `tmp`)
- Avoid overly long cells — one conceptual action per cell
- Always set plot titles and axis labels
- Avoid unnecessary scientific notation (`pd.set_option` handles floats)
- When preprocessing appears use `Pipeline` + `ColumnTransformer`
- No data leakage — fit only on training data

---

# Step 8 – Interactive Reasoning (Optional)

When requested, add a **Domande guidate** section with 4–6 questions.

Rules:
- Each question follows immediately from a computed result
- Each question is followed immediately by its answer in the same cell
- Answers must cite specific numbers or patterns from the outputs
- No speculation

```python
# %% [markdown]
# ## Domande guidate
#
# **1. Quale feature numerica mostra la correlazione più alta con il target?**
#
# `Balance` presenta il coefficiente φ più elevato (0.12) tra le feature
# numeriche, indicando che i clienti con saldo elevato tendono leggermente
# di più al churn. Tuttavia la correlazione è debole — serviranno interazioni.
```

---

# Step 9 – Riepilogo Section

Every notebook closes with a `## Riepilogo` section that:

- Recaps the main findings with specific numbers
- States implications for modeling
- Bridges to the next lesson

---

# Step 10 – Content Density

Prefer more analytical blocks over fewer. Instructors trim; they cannot
spontaneously generate content. If uncertain, add the extra analysis.

---

# Step 11 – Artifact Discipline

Outputs go under `outputs/` with lesson-scoped filenames:

```
outputs/data/lesson_02_clean.parquet
outputs/figures/lesson_02_correlation_heatmap.png
```

Validate file existence before reading artifacts in later cells.

Also treat `outputs/lesson_plans/lesson_NN_plan.md` and
`outputs/lesson_plans/lesson_NN_plan.json` as canonical intermediate artifacts
when the notebook is generated from a discovery phase.

---

# Step 12 – Validation Checklist

Before finishing, verify:

- [ ] All markdown is in Italian
- [ ] Header includes Authors, Emails, and Last updated (YYYY-MM-DD)
- [ ] Outline section is present and matches the discovery plan (if provided)
- [ ] Sections match the agreed scope (no extras, no omissions)
- [ ] No teacher-directed language ("come vedete", "notate che")  
- [ ] All code cells have been executed and produce the expected output
- [ ] Interpretations reference real computed values
- [ ] `Markdown → Code → Markdown` cycle is respected throughout
- [ ] No data leakage
- [ ] Repository structure has not been modified
- [ ] File saved to `notebooks/lesson-NN/lesson_NN.py`

---

# Step 13 – Live Coding Version

After the student notebook is complete, offer to generate the live coding
counterpart using the **live-coding-notebook** skill.
The live coding notebook is saved to `notebooks/lesson-NN/lesson_NN_live_coding.py`.