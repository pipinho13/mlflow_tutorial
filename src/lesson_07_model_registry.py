"""Lesson 7 — The MLflow Model Registry.

The registry sits on top of run artifacts and adds *versioning* and *aliases*:
    - A registered model is just a name (e.g. "california-housing-regressor").
    - Every time you register a run's model under that name you get a new
      monotonically-increasing version number.
    - Aliases (e.g. "champion", "challenger") are lightweight, mutable pointers
      to a specific version — they replaced the old "Stages" API.

This script:
    1. Trains two candidates and logs them as runs.
    2. Registers the better one as version 1, marks it "champion".
    3. Trains a slightly better candidate, registers it as version 2, marks it
       "challenger".
    4. Loads each by alias to show how downstream code stays version-agnostic.

Run (from the project root, with the venv active):
    python -m src.lesson_07_model_registry
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.utils import PROJECT_ROOT, load_data, regression_metrics

EXPERIMENT_NAME = "lesson-07-registry"
REGISTERED_MODEL = "california-housing-regressor"


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = load_data()
    client = MlflowClient()

    # 1. Two candidates.
    rf_run_id, rf_rmse = _train_and_log(
        run_name="rf-candidate",
        estimator=RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=0),
        data=data,
    )
    gbr_run_id, gbr_rmse = _train_and_log(
        run_name="gbr-candidate",
        estimator=GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0
        ),
        data=data,
    )
    winner_run_id, winner_rmse = (
        (gbr_run_id, gbr_rmse) if gbr_rmse < rf_rmse else (rf_run_id, rf_rmse)
    )
    print(f"Picked winner run {winner_run_id}  (rmse={winner_rmse:.4f})")

    # 2. Register version 1 from the winner.
    _ensure_registered_model(client, REGISTERED_MODEL)
    v1 = mlflow.register_model(f"runs:/{winner_run_id}/model", REGISTERED_MODEL)
    client.update_model_version(
        name=REGISTERED_MODEL,
        version=v1.version,
        description="Initial production candidate from lesson 7.",
    )
    client.set_registered_model_alias(REGISTERED_MODEL, alias="champion", version=v1.version)
    print(f"Registered as {REGISTERED_MODEL} v{v1.version}, alias=champion")

    # 3. Train an "improved" candidate and register as v2 -> challenger.
    challenger_run_id, _ = _train_and_log(
        run_name="gbr-challenger",
        estimator=GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, random_state=0
        ),
        data=data,
    )
    v2 = mlflow.register_model(f"runs:/{challenger_run_id}/model", REGISTERED_MODEL)
    client.update_model_version(
        name=REGISTERED_MODEL,
        version=v2.version,
        description="Deeper GBR with smaller learning rate.",
    )
    client.set_registered_model_alias(REGISTERED_MODEL, alias="challenger", version=v2.version)
    print(f"Registered as {REGISTERED_MODEL} v{v2.version}, alias=challenger")

    # 4. Load by alias from downstream code.
    champion = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")
    challenger = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@challenger")

    sample = data.X_test.head(5)
    print("\nFirst 5 predictions:")
    print(f"  champion  : {champion.predict(sample).round(3).tolist()}")
    print(f"  challenger: {challenger.predict(sample).round(3).tolist()}")

    print("\nIn the UI: open the 'Models' tab to see versions and aliases.")
    print("If the challenger wins in production, point the 'champion' alias at v2 to promote.")


def _train_and_log(run_name: str, estimator, data) -> tuple[str, float]:
    with mlflow.start_run(run_name=run_name) as run:
        estimator.fit(data.X_train, data.y_train)
        preds = estimator.predict(data.X_test)
        metrics = regression_metrics(data.y_test, preds)
        mlflow.log_metrics(metrics)
        mlflow.log_params(estimator.get_params())

        signature = infer_signature(data.X_train, estimator.predict(data.X_train))
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            signature=signature,
            input_example=data.X_train.head(3),
        )
        return run.info.run_id, metrics["rmse"]


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name=name, description="Models trained on California Housing.")


if __name__ == "__main__":
    main()
