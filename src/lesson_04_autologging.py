"""Lesson 4 — Autologging.

Manually calling log_param/log_metric/log_model for every estimator quickly
becomes tedious. MLflow ships with library-specific autologgers that hook into
fit/predict and log the right things automatically.

This script demonstrates two flavours:
    1. mlflow.sklearn.autolog() — params, metrics on training/eval sets, model.
    2. mlflow.xgboost.autolog() — same idea, plus per-iteration training metrics.

Run (from the project root, with the venv active):
    python -m src.lesson_04_autologging
"""

from __future__ import annotations

import mlflow
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.utils import PROJECT_ROOT, load_data

EXPERIMENT_NAME = "lesson-04-autologging"


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = load_data()

    # --- 1. sklearn autolog ----------------------------------------------------
    # log_input_examples=True asks MLflow to also store a small slice of the
    # training data alongside the model — useful for the UI's "Schema" tab and
    # for downstream serving.
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="rf-autolog"):
        rf = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=0)
        rf.fit(data.X_train, data.y_train)
        rf_rmse = mean_squared_error(data.y_test, rf.predict(data.X_test)) ** 0.5
        # We can still add our own metrics on top of what autolog produces.
        mlflow.log_metric("test_rmse", rf_rmse)

    # Disable sklearn autolog before switching libraries — autologgers stay on
    # for the rest of the process otherwise.
    mlflow.sklearn.autolog(disable=True)

    # --- 2. xgboost autolog ----------------------------------------------------
    mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

    dtrain = xgb.DMatrix(data.X_train, label=data.y_train)
    dtest = xgb.DMatrix(data.X_test, label=data.y_test)
    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
    }

    with mlflow.start_run(run_name="xgb-autolog"):
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=400,
            evals=[(dtrain, "train"), (dtest, "valid")],
            verbose_eval=False,
        )
        # autolog already captured per-iteration RMSE on train and valid; this
        # last metric is the final one, useful for UI sorting.
        final_rmse = mean_squared_error(data.y_test, booster.predict(dtest)) ** 0.5
        mlflow.log_metric("final_test_rmse", final_rmse)

    mlflow.xgboost.autolog(disable=True)

    print("Autologging demos complete. Open the UI to inspect what was captured.")
    print("In the run page you'll find: parameters, metrics, training curves,")
    print("the model artifact, and (for xgboost) feature importance plots.")


if __name__ == "__main__":
    main()
