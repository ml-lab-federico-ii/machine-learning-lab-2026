# ML Challenge — Evaluation

## Metric: ROC-AUC

Submissions are ranked by **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve), computed on a **hidden test set** that students never see.

### Why ROC-AUC?

Churn datasets are typically **imbalanced** — churners are a minority. Accuracy is misleading in this setting because a model that always predicts "no churn" can still score 80%+.

ROC-AUC measures the probability that the model ranks a random positive example higher than a random negative one. It is:
- **Threshold-independent** — evaluates the full scoring ability of the model
- **Robust to class imbalance** — does not penalise a model for the base rate
- Bounded in [0, 1], with 0.5 = random classifier, 1.0 = perfect

### Computing it in scikit-learn

```python
from sklearn.metrics import roc_auc_score

# y_proba = predicted probabilities for the positive class
auc = roc_auc_score(y_test, y_proba)
```

## Submission Artifact

Students submit a **full sklearn Pipeline** (`model.pkl`) serialised with `joblib`.  
The pipeline must accept the **raw feature columns** (before any preprocessing) and expose `predict_proba`, returning class probabilities.

The instructor uploads the hidden test CSV to the scoring page at `/instructor`, which:
1. Downloads each student's `model.pkl` via the GitHub API
2. Calls `pipeline.predict_proba(X_test)[:, 1]` in a restricted sandbox
3. Computes ROC-AUC, Precision, Recall, F1, the full ROC curve, and the Confusion Matrix

## Leaderboard

The leaderboard is **hidden from students** until Lesson 5.  
After Lesson 5, the instructor reveals rankings using the export from the scoring page.
