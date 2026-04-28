"""Lesson 10 — Logging to a remote tracking server (the Docker stack).

Up to here, every run has been written to a local ``mlruns/`` folder. In a
team setting you instead point at a *tracking server* — a long-running MLflow
process backed by:
    - a SQL database (for run metadata, params, metrics, the registry),
    - an object store (S3-compatible, for artifacts).

This script assumes you have launched the docker-compose stack in ``docker/``:

    docker compose -f docker/docker-compose.yml up -d

That brings up:
    - postgres   (run metadata + registry)
    - minio      (S3-compatible artifact store)
    - mlflow     (the tracking server itself, on http://localhost:5001)

We point MLflow at it, set the S3 credentials so artifact uploads work, and
log a single run. The exact same Python code as the previous lessons, just
with two extra environment variables.

Run:
    python src/lesson_10_remote_tracking.py
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import load_data, regression_metrics

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# Credentials for the local MinIO instance defined in docker-compose.yml. In a
# real deployment these would come from your secrets manager.
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("lesson-10-remote-tracking")
    data = load_data()

    with mlflow.start_run(run_name="gbr-on-remote-server") as run:
        mlflow.set_tag("environment", "docker-compose")

        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0
        )
        model.fit(data.X_train, data.y_train)
        preds = model.predict(data.X_test)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(regression_metrics(data.y_test, preds))

        signature = infer_signature(data.X_train, model.predict(data.X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=data.X_train.head(3),
        )

        print(f"Run id: {run.info.run_id}")
        print(f"Tracking URI: {TRACKING_URI}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")
        print("Open http://localhost:5001 to see the run, and")
        print("http://localhost:9001 (MinIO console) to see the artifact bucket.")


if __name__ == "__main__":
    main()
