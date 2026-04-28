"""Lesson 5 — Signatures, input examples, and loading models back.

A *model signature* describes the schema of inputs and outputs for a model. The
MLflow UI shows it on the model artifact page, and serving uses it to validate
incoming requests. An *input example* is a tiny sample of inputs stored next to
the model so reviewers (and you, six months later) can see what the model
expects.

We also demonstrate the two main ways to load a logged model:
    - mlflow.sklearn.load_model — gets back the original sklearn estimator.
    - mlflow.pyfunc.load_model — gets back a generic .predict() interface that
      works regardless of the library, perfect for serving.

Run (from the project root, with the venv active):
    python -m src.lesson_05_signatures_and_loading
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import PROJECT_ROOT, load_data, regression_metrics

EXPERIMENT_NAME = "lesson-05-signatures"


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = load_data()

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0
    )
    model.fit(data.X_train, data.y_train)
    preds = model.predict(data.X_test)

    # Infer the signature from real inputs and outputs. This produces a typed
    # schema describing every column.
    signature = infer_signature(data.X_train, model.predict(data.X_train))

    # An input example is a small DataFrame stored as JSON next to the model.
    input_example = data.X_train.head(3)

    with mlflow.start_run(run_name="gbr-with-signature") as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(regression_metrics(data.y_test, preds))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        run_id = run.info.run_id

    # ------------------------------------------------------------------
    # Load the model back two different ways and verify they agree.
    # ------------------------------------------------------------------
    model_uri = f"runs:/{run_id}/model"

    sk_model = mlflow.sklearn.load_model(model_uri)
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    sample: pd.DataFrame = data.X_test.head(5)
    sk_preds = sk_model.predict(sample)
    pyfunc_preds = pyfunc_model.predict(sample)

    print("Loaded with mlflow.sklearn.load_model :", sk_preds.round(3).tolist())
    print("Loaded with mlflow.pyfunc.load_model  :", pyfunc_preds.round(3).tolist())
    print()
    print(f"Model URI you would use anywhere else: {model_uri}")
    print("Inspect the run in the UI — note the Schema tab and the saved input_example.")


if __name__ == "__main__":
    main()
