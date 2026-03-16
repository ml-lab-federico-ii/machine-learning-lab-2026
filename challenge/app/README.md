# ML Challenge App

Interactive guided web app for the Machine Learning for Financial Analysis lab challenge.

Students work through an end-to-end ML pipeline — EDA → Preprocessing → Model selection → Scoring — and submit a trained `model.pkl` as a Pull Request to the instructor's repository.

---

## Table of Contents

- [Student Guide](#student-guide)
- [Instructor Setup Guide](#instructor-setup-guide)

---

## Student Guide

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A GitHub account

### Quick Start

1. **Fork** the instructor's repository to your GitHub account.

2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/machine-learning-lab-2026.git
   cd machine-learning-lab-2026/challenge/app
   ```

3. **Copy the example environment file** and fill in the OAuth credentials provided by the instructor:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   ```
   GITHUB_CLIENT_ID=<provided by instructor — do NOT register your own OAuth app>
   TARGET_REPO=<instructor-org>/machine-learning-lab-2026
   DEADLINE=<provided by instructor>
   ```
   Leave `GITHUB_CLIENT_SECRET`, `INSTRUCTOR_PASSWORD`, `GITHUB_INSTRUCTOR_TOKEN`, and `SESSION_SECRET` blank — these are only needed by the instructor.

4. **Start the app**:
   ```bash
   docker compose up
   ```
   Wait until both services are healthy (takes ~60 s on first run while images download).

5. **Open** [http://localhost:3000](http://localhost:3000) in your browser.

### Wizard Workflow

| Step | What you do |
|------|-------------|
| **1 – EDA** | Explore the dataset, select features, set train/val/test split |
| **2 – Preprocessing** | Configure imputation, scaling, encoding, and class imbalance handling |
| **3 – Model** | Pick a model and tune its hyperparameters |
| **4 – Train & Score** | Train your pipeline, inspect ROC-AUC, and fine-tune the decision threshold |
| **5 – Submit** | Log in with GitHub OAuth and submit your model as a Pull Request |

### Submission Policy

- Each student may submit **multiple times** before the deadline. Each re-submission overwrites the previous one in the same PR branch (`challenge-submission`).
- The instructor scores the **latest commit** in each PR.
- Submissions after the deadline are blocked by the app.

### Hardware Constraints

The training pipeline is intentionally CPU-friendly:

- `n_estimators ≤ 300`
- `max_depth ≤ 6`
- `n_jobs = -1` (uses all cores)
- No GPU required

---

## Instructor Setup Guide

### 1. Create a GitHub OAuth App

> **Students do not need to do this.** The instructor registers one OAuth App and shares only the `GITHUB_CLIENT_ID` with students (via the pre-filled `.env`). The `GITHUB_CLIENT_SECRET` stays on the instructor's machine / server only.

1. Go to **GitHub → Settings → Developer settings → OAuth Apps → New OAuth App**.

2. Fill in the form **exactly** as shown below:

   | Field | Value |
   |-------|-------|
   | **Application name** | `Machine Learning Challenge` (or any name) |
   | **Homepage URL** | `http://localhost:3000` |
   | **Authorization callback URL** | `http://localhost:3000/api/auth/callback` |

   > **Common mistake:** the form shows "Homepage URL" *above* "Authorization callback URL". Do not put the `/callback` URL in the Homepage field. The homepage is just `http://localhost:3000`; the callback is `http://localhost:3000/callback`.

3. Leave **Enable Device Flow** unchecked.

4. Click **Register application**.

5. On the next page, copy the **Client ID** and click **Generate a new client secret** to get the secret.

6. Put both values in `.env`:
   ```
   GITHUB_CLIENT_ID=Iv1.xxxxxxxxxxxxxxxx
   GITHUB_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

7. Share **only** `GITHUB_CLIENT_ID` with students in their `.env` file. Never share `GITHUB_CLIENT_SECRET`.

> If you deploy to a public URL, replace `http://localhost:3000` with your actual domain in both the OAuth App settings and in `FRONTEND_ORIGIN`.

### 2. Create a GitHub Personal Access Token

This token is used by the app to read student PR branches and download their `model.pkl` files for scoring.

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**.
2. Set **Resource owner** to your org (or your account).
3. Set **Repository access** to the `machine-learning-lab-2026` repo.
4. Grant **Contents: Read** and **Pull requests: Read** permissions.
5. Copy the token to `.env`:
   ```
   GITHUB_INSTRUCTOR_TOKEN=github_pat_xxxxxxxxxx
   ```

### 3. Author `schema.json`

The app is dataset-agnostic. You must tell it which columns are the ID and the target.

Edit `backend/data/schema.json`:
```json
{
  "id_column": "customer_id",
  "target_column": "churn",
  "target_positive_class": 1
}
```

- `id_column` — column to drop before training (row identifier)
- `target_column` — column the model predicts
- `target_positive_class` — value of the positive class (for confusion matrix orientation)

Everything else in the CSV is automatically treated as a feature.

Place the **training CSV** (without the hidden test set) at `backend/data/train.csv`.

**Example** — if your CSV header is:
```
RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,...,Exited
```
then `schema.json` should be:
```json
{
  "id_column": "CustomerId",
  "target_column": "Exited",
  "target_positive_class": 1
}
```

> `backend/data/.gitignore` already excludes `test_hidden.csv`, `*.pkl`, and `*.joblib`.

### 4. Set the Deadline

ISO 8601 UTC timestamp. Students cannot submit after this time.

```
DEADLINE=2026-04-01T23:59:59Z
```

A countdown timer is shown to students in the Submit step.

### 5. Set Security Credentials

```
INSTRUCTOR_PASSWORD=choose_a_strong_password
SESSION_SECRET=generate_a_long_random_string_here
```

Generate a random session secret with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 6. Start the App

```bash
docker compose up
```

### 7. Score Submissions (Instructor Page)

1. Open [http://localhost:3000/instructor](http://localhost:3000/instructor).
2. Enter the `INSTRUCTOR_PASSWORD`.
3. In **Upload Hidden Test Set**, upload `test_hidden.csv`. The file is held in memory only and never written to disk.
4. Click **Score All**. The app:
   - Lists all open Pull Requests in `TARGET_REPO`
   - Downloads each student's `model.pkl` via the GitHub API
   - Runs inference on the uploaded test CSV in a restricted sandbox (only sklearn, imblearn, xgboost, lightgbm, numpy, pandas modules may be unpickled)
   - Returns ROC-AUC, Precision, Recall, F1, ROC curve data, and Confusion Matrix
5. Inspect the **leaderboard table**, **ROC curve overlay**, and **side-by-side model comparison**.
6. Click **Export CSV** to download the full results.

### 8. Target Repository Setup

Set `TARGET_REPO` to the repository where student PRs should land:
```
TARGET_REPO=your-org/machine-learning-lab-2026
```

Student submission files are committed to `challenge/submissions/<github-username>/`:
- `pipeline_config.json` — hyperparameter configuration
- `model.pkl` — serialized fitted sklearn Pipeline
