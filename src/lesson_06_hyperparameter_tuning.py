"""Lesson 6 — Hyperparameter tuning with Optuna and *nested runs*.

When you tune a model, you typically train dozens of variants. MLflow handles
this via **nested runs**: a parent run for the whole study, and one child run
per trial. The UI then collapses children under the parent, making it easy to
see "the best trial in study X".

We use Optuna for the search itself; MLflow only owns logging.

Run:
    python src/lesson_06_hyperparameter_tuning.py
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import PROJECT_ROOT, load_data, regression_metrics

EXPERIMENT_NAME = "lesson-06-tuning"
N_TRIALS = 25

# Quiet Optuna's per-trial INFO logs — MLflow gives us a richer picture.
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = load_data()

    with mlflow.start_run(run_name="optuna-gbr-study") as parent_run:
        mlflow.set_tag("search_method", "optuna-tpe")
        mlflow.log_param("n_trials", N_TRIALS)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }

            # nested=True attaches this run as a child of the parent above.
            with mlflow.start_run(nested=True, run_name=f"trial-{trial.number:03d}"):
                mlflow.log_params(params)

                model = GradientBoostingRegressor(random_state=0, **params)
                model.fit(data.X_train, data.y_train)
                preds = model.predict(data.X_test)
                metrics = regression_metrics(data.y_test, preds)
                mlflow.log_metrics(metrics)

                return metrics["rmse"]

        study = optuna.create_study(direction="minimize", study_name="gbr-rmse")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_trial
        mlflow.log_metric("best_rmse", best.value)
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

        # Refit a final model with the best params and log it on the *parent*
        # run, so there's a single canonical model for this study.
        final = GradientBoostingRegressor(random_state=0, **best.params)
        final.fit(data.X_train, data.y_train)
        signature = infer_signature(data.X_train, final.predict(data.X_train))
        mlflow.sklearn.log_model(
            sk_model=final,
            artifact_path="best_model",
            signature=signature,
            input_example=data.X_train.head(3),
        )

        print(f"\nBest trial: #{best.number}  RMSE={best.value:.4f}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"\nParent run id: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
