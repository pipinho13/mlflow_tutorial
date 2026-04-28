"""Lesson 3 — Comparing several model families in one experiment.

Now we move beyond the linear baseline and train three different estimators in
the same experiment so they can be compared side-by-side in the MLflow UI.

The key MLflow trick here: each model family is its own run, but they all live
in the same experiment, so the UI's run table can sort/filter them together.

Run:
    python src/lesson_03_multiple_models.py
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from src.utils import PROJECT_ROOT, load_data, regression_metrics

EXPERIMENT_NAME = "lesson-03-model-comparison"


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = load_data()

    candidates: list[tuple[str, Any, dict[str, Any]]] = [
        ("ridge", Ridge(alpha=1.0, random_state=0), {"alpha": 1.0}),
        (
            "random-forest",
            RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=0),
            {"n_estimators": 200, "max_depth": 12},
        ),
        (
            "gradient-boosting",
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0
            ),
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
        ),
    ]

    for run_name, estimator, params in candidates:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_family", _family_for(run_name))
            mlflow.log_param("estimator", estimator.__class__.__name__)
            mlflow.log_params(params)

            estimator.fit(data.X_train, data.y_train)
            preds = estimator.predict(data.X_test)
            mlflow.log_metrics(regression_metrics(data.y_test, preds))

            # Tree-based models expose feature_importances_ — log them as a
            # separate metric per feature so they can be compared in the UI.
            if hasattr(estimator, "feature_importances_"):
                for name, importance in zip(data.feature_names, estimator.feature_importances_):
                    mlflow.log_metric(f"feature_importance__{name}", float(importance))

            mlflow.sklearn.log_model(sk_model=estimator, artifact_path="model")
            print(f"{run_name:<20} rmse={regression_metrics(data.y_test, preds)['rmse']:.4f}")

    # Programmatic comparison of the runs we just produced.
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.rmse ASC"],
    )
    print("\nLeaderboard:")
    cols = ["tags.mlflow.runName", "params.estimator", "metrics.rmse", "metrics.mae", "metrics.r2"]
    print(runs[cols].to_string(index=False))


def _family_for(name: str) -> str:
    return "linear" if "ridge" in name else "tree"


if __name__ == "__main__":
    main()
