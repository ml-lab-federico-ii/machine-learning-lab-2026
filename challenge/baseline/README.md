# ML Challenge — Baseline

The reference baseline is a **Logistic Regression** trained on a minimal preprocessed feature set.

It is provided as a starting point. Any submission that exceeds this baseline is considered a valid contribution.

## Baseline Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model.pkl")
```

> The app builds and trains an equivalent pipeline automatically when you select **Logistic Regression** in the Model step.

## Expected Performance

| Metric  | Approximate value |
|---------|-------------------|
| ROC-AUC | ~0.76 – 0.80      |

Actual values depend on the feature selection and split used.  
The app displays validation and test ROC-AUC after training so you can compare directly.
