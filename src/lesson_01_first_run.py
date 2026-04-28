"""Lesson 1 — Your first MLflow run.

Goal: train a baseline model and log a few things to MLflow so you can see them
in the UI. We deliberately use only the most fundamental MLflow API here:
    - mlflow.set_experiment
    - mlflow.start_run
    - mlflow.log_param / mlflow.log_metric / mlflow.log_artifact
    - mlflow.sklearn.log_model

Run:
    python src/lesson_01_first_run.py
Then:
    mlflow ui
and open http://127.0.0.1:5000
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

from src.utils import PROJECT_ROOT, ensure_dir, load_data, regression_metrics

EXPERIMENT_NAME = "lesson-01-first-run"


def main() -> None:
    # By default MLflow logs to ./mlruns in the current working directory.
    # We pin it explicitly so this lesson is independent of where you run from.
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_data()
    artifacts_dir = ensure_dir(PROJECT_ROOT / "outputs" / "lesson_01")

    with mlflow.start_run(run_name="linear-regression-baseline") as run:
        # 1. Log a couple of hand-picked parameters. These are searchable in the UI.
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("n_features", data.X_train.shape[1])
        mlflow.log_param("n_train", len(data.X_train))
        mlflow.log_param("n_test", len(data.X_test))

        # 2. Train.
        model = LinearRegression()
        model.fit(data.X_train, data.y_train)

        # 3. Evaluate and log metrics. Metrics are numeric and can be compared across runs.
        preds = model.predict(data.X_test)
        metrics = regression_metrics(data.y_test, preds)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # 4. Log an artifact (a plot file). Artifacts can be any file: figures,
        # CSVs, HTML reports, configs, etc.
        plot_path = _residuals_plot(data.y_test.to_numpy(), preds, artifacts_dir)
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        # 5. Log the model itself. This stores the serialized estimator under the
        # run's artifacts so it can later be loaded with mlflow.sklearn.load_model.
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        print(f"Run id: {run.info.run_id}")
        print(f"Metrics: {metrics}")
        print(f"Open the UI with `mlflow ui` and find experiment '{EXPERIMENT_NAME}'.")


def _residuals_plot(y_true, y_pred, out_dir: Path) -> Path:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Residuals — Linear Regression")
    fig.tight_layout()
    out = out_dir / "residuals.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
