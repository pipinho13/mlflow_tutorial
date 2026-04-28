"""Lesson 2 — Experiments, runs, tags, and descriptions.

In Lesson 1 we logged everything into a single experiment. In real projects you
will:
    - have many experiments (one per problem / iteration / dataset),
    - have many runs per experiment (one per trained variant),
    - tag runs with searchable metadata (git sha, dataset version, owner, ...),
    - and write a short description per run so future-you understands it.

This script creates two experiments and produces several runs in each, showing
the small but useful API surface around organisation.

Run (from the project root, with the venv active):
    python -m src.lesson_02_experiments_and_runs
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import mlflow
from sklearn.linear_model import Ridge

from src.utils import PROJECT_ROOT, load_data, regression_metrics


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    data = load_data()

    # An "experiment" is a folder of related runs. Tags on the experiment itself
    # are great for "what is this experiment about?" metadata.
    exp_id = _ensure_experiment(
        name="lesson-02-ridge-sweep",
        tags={
            "owner": "george",
            "problem": "regression",
            "dataset": "california-housing",
            "stage": "exploration",
        },
    )

    # A small alpha sweep — three runs, one per alpha. Each run gets its own
    # name, tags, and description.
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=f"ridge-alpha-{alpha}",
        ) as run:
            # set_tag is for short, searchable strings (e.g. git sha, model family).
            mlflow.set_tag("model_family", "linear")
            mlflow.set_tag("estimator", "Ridge")
            mlflow.set_tag("python_version", platform.python_version())

            # The special tag mlflow.note.content shows up as a Markdown
            # description on the run page in the UI.
            mlflow.set_tag(
                "mlflow.note.content",
                f"Ridge regression with `alpha={alpha}`. "
                "Sweep aims to find the best regularisation strength on the held-out test set.",
            )

            mlflow.log_params({"alpha": alpha, "fit_intercept": True})

            model = Ridge(alpha=alpha, random_state=0)
            model.fit(data.X_train, data.y_train)
            preds = model.predict(data.X_test)
            mlflow.log_metrics(regression_metrics(data.y_test, preds))

            print(f"alpha={alpha:>5}  run_id={run.info.run_id}")

    # Searching runs programmatically — equivalent to using the UI's search bar.
    print("\nTop 3 runs by RMSE (lower is better):")
    top = mlflow.search_runs(
        experiment_ids=[exp_id],
        order_by=["metrics.rmse ASC"],
        max_results=3,
    )
    print(top[["run_id", "params.alpha", "metrics.rmse", "metrics.r2"]].to_string(index=False))


def _ensure_experiment(name: str, tags: dict[str, str]) -> str:
    """Create the experiment if it doesn't exist, and return its id."""
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is None:
        return client.create_experiment(
            name=name,
            tags={
                **tags,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
    return existing.experiment_id


if __name__ == "__main__":
    main()
