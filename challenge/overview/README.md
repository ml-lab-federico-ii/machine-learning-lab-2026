# Churn Prediction Challenge — Overview

## Objective

Build a binary classification model on a new churn dataset that **maximises ROC-AUC** on the instructors' hidden test set.

## How it works

1. Fork this repository and follow the setup guide in [`challenge/app/README.md`](../app/README.md).
2. Launch the interactive web app locally with a single `docker compose up`.
3. Use the guided wizard to explore the data, design your preprocessing pipeline, choose a model, and evaluate it against your own train/validation/test split.
4. When satisfied, submit from Step 5 of the app — this creates a Pull Request on this repository containing your trained model and pipeline configuration.
5. You may resubmit as many times as you like before the deadline; only the last submission counts.

## Evaluation

Final scoring is performed by the instructors on a **hidden test set** (not available to students) using the `model.pkl` file you submit.

Your trained model must be a complete `sklearn.Pipeline` that accepts raw input features and outputs class probabilities via `predict_proba`.

## Submission artifacts

The app automatically commits two files to your Pull Request:

| File | Description |
|---|---|
| `model.pkl` | Fitted `sklearn.Pipeline` (preprocessing + model) |
| `pipeline_config.json` | Full pipeline configuration used to produce the model |

## Metric

**ROC-AUC** (Area Under the ROC Curve) — see [`challenge/evaluation/`](../evaluation/README.md) for details.
