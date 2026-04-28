# MLflow, end to end — a hands-on tutorial for ML Engineers

A long-form, reproducible MLflow tutorial that goes from your very first
`mlflow.log_param` call all the way to a Docker-Compose stack with a remote
tracking server, a Postgres metadata store, and an S3-compatible artifact
store.

Every concept is paired with a runnable script under [`src/`](src/), so you can
read the code, run it, then look at exactly what changed in the MLflow UI.

---

## Table of contents

- [What you'll learn](#what-youll-learn)
- [Prerequisites](#prerequisites)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Mental model: the four pillars of MLflow](#mental-model-the-four-pillars-of-mlflow)
- [Lesson 1 — Your first MLflow run](#lesson-1--your-first-mlflow-run)
- [Lesson 2 — Experiments, runs, tags, descriptions](#lesson-2--experiments-runs-tags-descriptions)
- [Lesson 3 — Comparing multiple models](#lesson-3--comparing-multiple-models)
- [Lesson 4 — Autologging](#lesson-4--autologging)
- [Lesson 5 — Signatures, input examples, and loading models](#lesson-5--signatures-input-examples-and-loading-models)
- [Lesson 6 — Hyperparameter tuning with nested runs](#lesson-6--hyperparameter-tuning-with-nested-runs)
- [Lesson 7 — The Model Registry](#lesson-7--the-model-registry)
- [Lesson 8 — Custom PyFunc models](#lesson-8--custom-pyfunc-models)
- [Lesson 9 — MLflow Projects (reproducible runs)](#lesson-9--mlflow-projects-reproducible-runs)
- [Lesson 10 — A production tracking server with Docker Compose](#lesson-10--a-production-tracking-server-with-docker-compose)
- [Bonus — Serving a registered model](#bonus--serving-a-registered-model)
- [Troubleshooting](#troubleshooting)
- [Where to go next](#where-to-go-next)

---

## What you'll learn

By the end of this tutorial you will be comfortable with:

- **Tracking** — logging parameters, metrics, tags, and artifacts of every run.
- **The MLflow UI** — comparing runs, sorting, filtering, charting.
- **Autologging** — `mlflow.sklearn.autolog()` and `mlflow.xgboost.autolog()`.
- **Signatures & input examples** — making models self-describing.
- **Hyperparameter tuning** with **nested runs** (Optuna).
- **The Model Registry** — versioning, aliases (`@champion`/`@challenger`).
- **Custom PyFunc models** — wrapping preprocessing + estimator + post-processing.
- **MLflow Projects** — turning a script into a reproducible, parameterised job.
- **Production setup** — running a remote MLflow server backed by Postgres and
  S3-compatible storage in Docker Compose.
- **Serving** — exposing a registered model behind a REST API.

The dataset is [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html),
which ships with scikit-learn. Nothing has to be downloaded by hand.

---

## Prerequisites

- Python 3.11
- `pip` (or `uv`/`poetry` — the lessons only use `pip`-installable libraries)
- Optional but recommended: **Docker Desktop** (for [Lesson 10](#lesson-10--a-production-tracking-server-with-docker-compose))
- Familiarity with scikit-learn (the tutorial doesn't teach modelling)

---

## Project layout

```
mlflow_tutorial/
├── README.md                 # ← you are here
├── requirements.txt
├── .gitignore
├── src/                      # one runnable script per lesson
│   ├── utils.py              # shared data loader + metrics
│   ├── lesson_01_first_run.py
│   ├── lesson_02_experiments_and_runs.py
│   ├── lesson_03_multiple_models.py
│   ├── lesson_04_autologging.py
│   ├── lesson_05_signatures_and_loading.py
│   ├── lesson_06_hyperparameter_tuning.py
│   ├── lesson_07_model_registry.py
│   ├── lesson_08_custom_pyfunc.py
│   ├── lesson_09_mlproject.py
│   └── lesson_10_remote_tracking.py
├── mlproject/                # a real MLflow Project (lesson 9)
│   ├── MLproject
│   ├── python_env.yaml
│   └── train.py
├── docker/                   # the production tracking-server stack (lesson 10)
│   ├── Dockerfile.mlflow
│   ├── docker-compose.yml
│   └── .env.example
└── mlruns/                   # auto-created on first run; gitignored
```

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-fork-url> mlflow_tutorial
cd mlflow_tutorial

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

That's it for lessons 1–9. Lesson 10 additionally needs Docker.

> **Tip:** every script writes to `./mlruns` by default, so the tracking data
> lives inside the project folder. To start fresh between lessons, run:
>
> ```bash
> rm -rf mlruns mlartifacts outputs *.db
> ```

---

## Mental model: the four pillars of MLflow

Before running anything, it helps to know what MLflow is actually doing.
MLflow is four loosely-coupled components — you can use any subset:

1. **Tracking** — a server (or a local folder) that stores **runs**. Each run
   has params, metrics, tags, and arbitrary file artifacts.
2. **Models** — a standard package format for trained models, with optional
   **signatures** (input/output schemas) and **input examples**. Any framework
   can be wrapped (`sklearn`, `xgboost`, `pytorch`, …, or a custom PyFunc).
3. **Model Registry** — a layer on top of tracking that gives models a **name**,
   **versions**, and **aliases** (`@champion`, `@staging`, …).
4. **Projects** — a YAML-ish file (`MLproject`) that turns a directory into a
   reproducible, parameterised job that anyone can run with `mlflow run`.

A useful one-liner: **runs are facts, registered models are decisions**. You
log everything to runs; you promote the runs you trust into the registry.

The data flow:

```
   your code
   │
   │ mlflow.log_param / log_metric / log_artifact
   ▼
┌────────────────────────────┐        ┌────────────────────────┐
│   Backend store (SQL/file) │◀──────▶│  Artifact store (S3/fs)│
│   params / metrics / tags  │        │  models, plots, files  │
└────────────────────────────┘        └────────────────────────┘
            ▲
            │  (REST + UI)
            ▼
        MLflow UI
```

In lessons 1–9 the backend store and the artifact store are both folders on
your laptop (`./mlruns/`). In lesson 10 we replace them with Postgres and
MinIO running in Docker.

---

## Lesson 1 — Your first MLflow run

**Code:** [`src/lesson_01_first_run.py`](src/lesson_01_first_run.py)

The minimal MLflow call site has just five moving parts:

```python
import mlflow

mlflow.set_experiment("lesson-01-first-run")     # bucket of related runs
with mlflow.start_run(run_name="my-run"):        # ← the run is open here
    mlflow.log_param("alpha", 0.1)               # one-shot, immutable
    mlflow.log_metric("rmse", 0.42)              # numeric, can be updated
    mlflow.log_artifact("plot.png")              # any file
    mlflow.sklearn.log_model(model, "model")     # the model itself
```

Run it:

```bash
python -m src.lesson_01_first_run
```

You'll see something like:

```
Run id: 8f0c…
Metrics: {'rmse': 0.7456, 'mae': 0.5332, 'r2': 0.5758}
```

Now launch the UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open http://127.0.0.1:5000. You should see:

- An experiment called `lesson-01-first-run` in the left sidebar.
- One run in the table.
- Click the run → you'll see **Parameters**, **Metrics**, **Artifacts** (with
  the residuals plot under `plots/`), and a serialized **Model** under `model/`.

### What just happened?

- `mlflow.set_experiment(...)` created (or selected) an experiment folder.
- `mlflow.start_run(...)` opened a run as a context manager. When the `with`
  block exits, the run is marked `FINISHED` (or `FAILED` if there was an
  exception).
- `log_param`, `log_metric`, and `log_artifact` are independent — you can call
  any subset.
- `mlflow.sklearn.log_model(...)` serialised the estimator with `joblib` and
  wrote a small `MLmodel` YAML file describing the flavour. That `model/`
  folder is what makes the run reproducible later.

---

## Lesson 2 — Experiments, runs, tags, descriptions

**Code:** [`src/lesson_02_experiments_and_runs.py`](src/lesson_02_experiments_and_runs.py)

A real project produces many experiments and many runs per experiment. Tags
and descriptions are how you keep them findable months later.

Three things to learn here:

```python
# Tags on the experiment itself — "what is this experiment about?"
client.create_experiment(
    name="lesson-02-ridge-sweep",
    tags={"owner": "george", "stage": "exploration", ...},
)

# Tags on a run — short, searchable, free-form
mlflow.set_tag("model_family", "linear")
mlflow.set_tag("estimator", "Ridge")

# A *description* is just a special tag — it renders as Markdown in the UI
mlflow.set_tag(
    "mlflow.note.content",
    f"Ridge regression with alpha={alpha}. Sweep aims to ...",
)
```

Run it:

```bash
python -m src.lesson_02_experiments_and_runs
```

This produces four Ridge runs (alphas 0.01, 0.1, 1.0, 10.0) and prints a
leaderboard:

```
Top 3 runs by RMSE (lower is better):
                              run_id params.alpha  metrics.rmse  metrics.r2
abc...                                    0.1         0.7455     0.5758
…
```

The same query can be done in the UI: open the experiment, click the
**Search** bar, type `metrics.rmse < 0.8`, and sort by `metrics.rmse ASC`.

Useful UI features to try:

- Select two runs with the checkbox column → click **Compare** → see params
  and metrics side by side.
- Click the **Chart** view to plot any metric across runs.
- Right-click a column header → **Manage columns** to control what's shown.

---

## Lesson 3 — Comparing multiple models

**Code:** [`src/lesson_03_multiple_models.py`](src/lesson_03_multiple_models.py)

Now we move past the linear baseline. We train a Ridge, a Random Forest, and
a Gradient Boosting Regressor in the same experiment so the UI can compare
them side by side.

```bash
python -m src.lesson_03_multiple_models
```

Two patterns worth highlighting:

1. **One run per estimator, all in one experiment.** This makes the UI
   leaderboard view directly useful.
2. **Logging feature importances as metrics.** For tree-based models we log
   each `feature_importances_` value as its own metric:

   ```python
   for name, importance in zip(feature_names, model.feature_importances_):
       mlflow.log_metric(f"feature_importance__{name}", float(importance))
   ```

   In the UI you can now **Compare** the two tree models and see all of these
   metrics side by side. (This is a small trick — there's also
   `mlflow.log_dict(...)` for richer payloads.)

---

## Lesson 4 — Autologging

**Code:** [`src/lesson_04_autologging.py`](src/lesson_04_autologging.py)

For supported libraries (sklearn, XGBoost, LightGBM, PyTorch, TensorFlow,
spark.ml, transformers, …) you don't need to call `log_param`/`log_metric`
yourself at all. One line at the top of the script:

```python
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
```

…and now every `model.fit(...)` call inside an active run produces a fully
populated MLflow run: hyperparameters, training-set metrics, the serialized
model, the input example, and the schema. With `mlflow.xgboost.autolog()` you
also get **per-iteration** training and validation metrics — those become a
beautiful loss curve in the UI's **Metrics → Chart** view.

```bash
python -m src.lesson_04_autologging
```

Two gotchas:

- **Autologgers persist** for the whole Python process. If you switch
  libraries (sklearn → xgboost), call `mlflow.sklearn.autolog(disable=True)`
  first to keep the logs clean.
- Autologgers don't replace your own logging. You can still call
  `mlflow.log_metric("custom_kpi", …)` to add domain metrics on top.

---

## Lesson 5 — Signatures, input examples, and loading models

**Code:** [`src/lesson_05_signatures_and_loading.py`](src/lesson_05_signatures_and_loading.py)

A **signature** is the typed input/output schema of a model. It's stored in
the `MLmodel` file and powers schema validation when the model is served. An
**input example** is a small DataFrame stored next to the model so reviewers
can see what the model expects.

```python
from mlflow.models import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature,
    input_example=X_train.head(3),
)
```

Then load the model back two ways:

```python
# 1. Native — returns the original sklearn estimator
model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# 2. Generic — returns a thin wrapper with a uniform .predict()
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
```

The `pyfunc` flavour is the one used by **serving** and by downstream code
that doesn't know which library produced the model. Get into the habit of
using it.

```bash
python -m src.lesson_05_signatures_and_loading
```

In the UI, open the run, then the **model** artifact. Notice the new
**Schema** tab and the saved input example.

---

## Lesson 6 — Hyperparameter tuning with nested runs

**Code:** [`src/lesson_06_hyperparameter_tuning.py`](src/lesson_06_hyperparameter_tuning.py)

When you run a 50-trial Optuna study, you don't want 50 sibling runs cluttering
the experiment. MLflow has **nested runs** for exactly this:

```python
with mlflow.start_run(run_name="optuna-gbr-study") as parent:
    def objective(trial):
        params = {...}
        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            model = GradientBoostingRegressor(**params).fit(X_train, y_train)
            rmse = ...
            mlflow.log_metric("rmse", rmse)
            return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)
```

The UI now collapses all 25 children under the parent. The parent itself logs
the *best* params and a *refit* model — your single, canonical artifact for
the whole study.

```bash
python -m src.lesson_06_hyperparameter_tuning
```

In the UI, expand the parent run; you'll see the trial children, and at the
top the parent has a `best_model/` artifact and `best_*` parameter entries.

> **Why this matters:** when 200 people on your team each launch tuning runs,
> the registry should point at `best_model/` artifacts on parent runs, not at
> raw trials. The nested structure makes that policy enforceable.

---

## Lesson 7 — The Model Registry

**Code:** [`src/lesson_07_model_registry.py`](src/lesson_07_model_registry.py)

So far our models have been identified by run id. That doesn't scale. The
registry adds a **stable name** and **versions**:

```
runs:/8f0c.../model      ← unstable: who has 8f0c memorised?
models:/california-housing-regressor/2  ← stable: version 2 of a named model
models:/california-housing-regressor@champion  ← even better: an alias
```

The script trains two candidates, registers the better one as version 1 with
alias `champion`, then trains an "improved" candidate, registers it as version
2 with alias `challenger`, and loads each by alias:

```python
mlflow.register_model(f"runs:/{run_id}/model", "california-housing-regressor")

client = mlflow.MlflowClient()
client.set_registered_model_alias(
    name="california-housing-regressor",
    alias="champion",
    version=1,
)

# Anywhere downstream:
champion = mlflow.pyfunc.load_model(
    "models:/california-housing-regressor@champion"
)
```

```bash
python -m src.lesson_07_model_registry
```

In the UI, open the **Models** tab in the top nav. You'll see the registered
model, its versions, and the aliases attached to each version. Promotion is
just *moving the alias*: when challenger wins in production, point
`champion` at version 2 and your serving code instantly switches.

> **Stages vs. aliases.** The old `Staging`/`Production`/`Archived` stages are
> deprecated in favour of aliases, which are arbitrary strings — `champion`,
> `eu-prod`, `experiment-week-12`, anything you want. This tutorial uses the
> modern aliases-only approach.

---

## Lesson 8 — Custom PyFunc models

**Code:** [`src/lesson_08_custom_pyfunc.py`](src/lesson_08_custom_pyfunc.py)

A real model often isn't just `estimator.predict(X)`. You may need:

- input clipping or imputation,
- a learned scaler that has to live with the model,
- post-processing into business categories (`low`/`mid`/`high`),
- or even a small ensemble of two estimators.

The `mlflow.pyfunc.PythonModel` interface lets you wrap *anything*:

```python
class PriceBandModel(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator, feature_clips):
        self.estimator = estimator
        self.feature_clips = feature_clips

    def predict(self, context, model_input, params=None):
        X = self._clip(model_input)
        raw = self.estimator.predict(X)
        return pd.DataFrame({
            "prediction": raw,
            "band": [self._band(p) for p in raw],
        })

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=PriceBandModel(estimator, clips),
    signature=signature,
    input_example=sample_input,
    pip_requirements=["mlflow", "scikit-learn", "pandas", "numpy"],
)
```

```bash
python -m src.lesson_08_custom_pyfunc
```

The reloaded model has the same uniform interface as any other:

```python
loaded = mlflow.pyfunc.load_model("runs:/<run_id>/model")
loaded.predict(df)   # → DataFrame with prediction + band
```

This is the pattern most real teams use, because it means the artifact in
the registry is *the* deployable thing — not "the estimator plus a doc page
explaining how to call it".

---

## Lesson 9 — MLflow Projects (reproducible runs)

**Code:** [`src/lesson_09_mlproject.py`](src/lesson_09_mlproject.py) and the
project itself in [`mlproject/`](mlproject/).

An MLflow Project is just a folder with three things:

- `MLproject` — declares entry points, parameters, and commands.
- `python_env.yaml` (or `conda.yaml`) — pins the environment.
- a script — the actual training code.

Here's the [`MLproject`](mlproject/MLproject) we use:

```yaml
name: mlflow_tutorial_project
python_env: python_env.yaml

entry_points:
  main:
    parameters:
      n_estimators:    {type: int,   default: 300}
      learning_rate:   {type: float, default: 0.05}
      max_depth:       {type: int,   default: 3}
      experiment_name: {type: str,   default: "lesson-09-mlproject"}
    command: >
      python train.py
      --n-estimators {n_estimators}
      --learning-rate {learning_rate}
      --max-depth {max_depth}
      --experiment-name {experiment_name}
```

Now anyone, anywhere, can run this project with one command:

```bash
# Inside the project (uses your current env)
mlflow run mlproject -P n_estimators=500 --env-manager=local

# From a Git URL — MLflow will clone, build the env, and run
mlflow run https://github.com/<you>/mlflow_tutorial.git#mlproject \
    -P n_estimators=500
```

The convenience script [`src/lesson_09_mlproject.py`](src/lesson_09_mlproject.py)
does the same thing through the Python API:

```bash
python -m src.lesson_09_mlproject
```

> **Why this matters:** `mlflow run` is the integration point with CI, with
> Airflow's `MLflowProjectOperator`, with Databricks Jobs, and with anyone you
> want to hand a "reproduce-this-experiment" button.

---

## Lesson 10 — A production tracking server with Docker Compose

**Code:** [`src/lesson_10_remote_tracking.py`](src/lesson_10_remote_tracking.py),
stack in [`docker/`](docker/).

`./mlruns` is fine for one person on one laptop. A team needs:

- A **shared backend** so everyone sees the same runs and registry.
- A **real database** for fast searches and the registry's foreign keys.
- An **object store** for large artifacts (models, plots, datasets).

The [`docker/docker-compose.yml`](docker/docker-compose.yml) brings up four
services:

| Service       | Image                | Purpose                                 | Port             |
| ------------- | -------------------- | --------------------------------------- | ---------------- |
| `postgres`    | `postgres:16-alpine` | backend store (params/metrics/registry) | internal         |
| `minio`       | `minio/minio`        | S3-compatible artifact store            | `9000` API, `9001` console |
| `minio-init`  | `minio/mc`           | one-shot job: create the bucket         | —                |
| `mlflow`      | (built locally)      | the tracking server                     | `5001` → `5000`  |

Boot it:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env.example up -d --build
```

Wait ~10 seconds for Postgres and MinIO to become healthy, then:

- MLflow UI: http://localhost:5001
- MinIO console: http://localhost:9001  (login: `minioadmin` / `minioadmin`)

Now log a run from your laptop into the remote server:

```bash
python -m src.lesson_10_remote_tracking
```

The script just sets two env vars before its first MLflow call:

```python
os.environ["MLFLOW_TRACKING_URI"]   = "http://localhost:5001"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "minioadmin"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
```

The training code itself is unchanged — that's the whole point. Visit the
MLflow UI at port 5001 to see the run, and the MinIO console at port 9001 to
see the model files in the `mlflow-artifacts` bucket.

Tear down (data is preserved in named volumes):

```bash
docker compose -f docker/docker-compose.yml down
```

### Anatomy of the tracking-server command

Looking at [`docker/docker-compose.yml`](docker/docker-compose.yml), the
server runs:

```bash
mlflow server \
    --host 0.0.0.0 --port 5000 \
    --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow \
    --artifacts-destination s3://mlflow-artifacts \
    --serve-artifacts
```

Three flags worth understanding:

- `--backend-store-uri` — where runs/params/metrics/the registry live. Any
  SQLAlchemy URL works.
- `--artifacts-destination` — where the server tells clients to put artifacts.
- `--serve-artifacts` — proxies artifact reads/writes through the MLflow
  server itself. With this flag on, **clients don't need S3 credentials at
  all** for read-only access; only write paths do. (The lesson 10 script
  still sets them so artifact uploads work.)

### Production hardening — what's missing here

This stack is a *demo*. For a real deployment you'd add:

- TLS termination (a reverse proxy like nginx or Traefik).
- Real secrets (don't bake `minioadmin` into anything).
- Authentication on MLflow itself (mlflow `>=2.5` supports basic auth).
- Persistent disks instead of named Docker volumes.
- Backups for the Postgres database.
- A real S3 (or GCS / Azure Blob) bucket instead of MinIO.

---

## Bonus — Serving a registered model

After running [Lesson 7](#lesson-7--the-model-registry) the registry has a
model named `california-housing-regressor` with the alias `champion`. MLflow
can serve it as a REST API with one command:

```bash
mlflow models serve \
    -m "models:/california-housing-regressor@champion" \
    --host 127.0.0.1 --port 5002 --env-manager local
```

Then send a prediction request:

```bash
curl -X POST http://127.0.0.1:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["MedInc","HouseAge","AveRooms","AveBedrms","Population",
                  "AveOccup","Latitude","Longitude"],
      "data":    [[8.3252, 41.0, 6.984, 1.024, 322.0, 2.555, 37.88, -122.23]]
    }
  }'
```

Response:

```json
{"predictions": [4.123]}
```

The exact same `mlflow models serve` command works against the remote
tracking server too — just `export MLFLOW_TRACKING_URI=http://localhost:5001`
first.

For containerised serving, MLflow can build a self-contained Docker image
from any model URI:

```bash
mlflow models build-docker \
    -m "models:/california-housing-regressor@champion" \
    -n housing-regressor:latest
docker run -p 5002:8080 housing-regressor:latest
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'` when running a lesson.**
Run scripts with `python -m src.lesson_XX_*` (this is what the Makefile does)
or set `PYTHONPATH=.` so the `src` package is importable.

**The MLflow UI is empty.**
Make sure you point the UI at the same backend store the scripts wrote to:
`mlflow ui --backend-store-uri ./mlruns`.

**Lesson 10 script can't reach the server.**
Confirm `docker compose ps` shows all four services up and that
`curl http://localhost:5001/health` returns `OK`. Logs:
`docker compose -f docker/docker-compose.yml logs -f mlflow`.

**`No such bucket: mlflow-artifacts`.**
The `minio-init` one-shot may have failed. Run it again manually:
`docker compose -f docker/docker-compose.yml run --rm minio-init`.

**XGBoost autologging warns about scikit-learn API.**
The `xgb.train` low-level API is the one autolog supports best on linux/macOS;
if you see warnings, they're cosmetic — the run is still logged correctly.

**A run shows status `RUNNING` forever.**
The Python process probably crashed *outside* the `with mlflow.start_run()`
block. Either restart the lesson, or open the run in the UI and click
**Delete**.

---

## Where to go next

- **Datasets API** — `mlflow.data.from_pandas(df).log()` lets you log dataset
  fingerprints (hash, schema, source) as part of a run, so reruns can detect
  data drift.
- **Evaluations** — `mlflow.evaluate(...)` produces a standardised set of
  metrics + diagnostic plots for classification/regression/LLM tasks.
- **LLM tracing** — `mlflow.openai.autolog()`, `mlflow.langchain.autolog()`,
  and the new tracing UI for prompt-engineering workflows.
- **Pipelines/Recipes** — opinionated end-to-end project templates.
- **Deployment plugins** — `mlflow deployments` ships with adapters for
  SageMaker, Azure ML, Databricks, and more.

The official docs live at https://mlflow.org/docs/latest/index.html — they're
genuinely good. Use this tutorial to build the muscle memory, then dive in.

---

If you spot a bug or want to suggest an extra lesson, open an issue or PR.
Happy tracking.
