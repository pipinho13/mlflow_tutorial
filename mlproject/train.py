"""Entry point for the MLflow Project in Lesson 9.

This script is intentionally self-contained — when MLflow runs it via
``mlflow run``, the working directory is this folder, so it cannot import from
the parent ``src/`` package. We therefore reimplement the small helpers locally.
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--experiment-name", type=str, default="lesson-09-mlproject")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)

    raw = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        raw.data, raw.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="mlproject-run"):
        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
            }
        )

        model = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )
        print(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")


if __name__ == "__main__":
    main()
