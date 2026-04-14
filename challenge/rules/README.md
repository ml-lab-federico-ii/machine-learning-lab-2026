# ML Challenge — Rules

## Participation

- Participation is **individual**.
- You may only use the tools, libraries, and techniques covered in the course lessons.
- Sharing predictions, trained models, or code with other students is not allowed.

## Submission Mechanism

Submissions are made through the interactive web app (`challenge/app`).  
See [challenge/app/README.md](../app/README.md) for setup instructions.

Your submission consists of:
- `model.pkl` — your full trained sklearn Pipeline
- `pipeline_config.json` — your hyperparameter configuration

These are committed to your fork and submitted as a **Pull Request** to the instructor's repository automatically by the app.

## Attempts

- You may submit **multiple times** before the deadline.
- Each re-submission **overwrites** your previous entry (same PR branch, new commit).
- Only the **latest commit** is scored.

## Deadline

- Submissions close **before Lesson 5**.
- The exact timestamp is set by the instructor via the `DEADLINE` environment variable and displayed as a countdown timer in the app.
- Submissions after the deadline are blocked.

## Constraints

To keep training times reasonable on standard laptops:

- `n_estimators ≤ 300`
- `max_depth ≤ 6`
- No GPU required

## Evaluation

See [challenge/evaluation/README.md](../evaluation/README.md) for the evaluation metric and scoring procedure.
