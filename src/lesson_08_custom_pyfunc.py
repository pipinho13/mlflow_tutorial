"""Lesson 8 — Custom PyFunc models.

Sometimes a single sklearn estimator isn't enough: you might want to bundle
preprocessing, business rules, or post-processing with the estimator so that
serving code only has to call ``model.predict(X)``.

The `mlflow.pyfunc.PythonModel` interface lets you wrap *anything* — multiple
estimators, ensembles, post-processing logic — into a model that MLflow can log,
load, register, and serve like any other.

Here we build a "PriceBandModel" that:
    - takes raw California-Housing-shaped inputs,
    - clips outliers to a safe range (preprocessing),
    - predicts with a GradientBoostingRegressor,
    - returns both the numeric prediction and a discrete "band" (low/mid/high).

Run:
    python src/lesson_08_custom_pyfunc.py
"""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import PROJECT_ROOT, load_data

EXPERIMENT_NAME = "lesson-08-custom-pyfunc"


class PriceBandModel(mlflow.pyfunc.PythonModel):
    """A custom model that bundles preprocessing + estimator + post-processing."""

    BAND_EDGES = (1.5, 3.0)  # split predictions into low/mid/high (units of $100k)

    def __init__(self, estimator, feature_clips: dict[str, tuple[float, float]]):
        self.estimator = estimator
        self.feature_clips = feature_clips

    # MLflow calls this for every prediction request. ``model_input`` arrives
    # as a DataFrame when loaded via mlflow.pyfunc.
    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        X = self._preprocess(model_input)
        raw = self.estimator.predict(X)
        return pd.DataFrame(
            {
                "prediction": raw,
                "band": [self._band(p) for p in raw],
            }
        )

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (low, high) in self.feature_clips.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=low, upper=high)
        return X

    @classmethod
    def _band(cls, value: float) -> str:
        low, high = cls.BAND_EDGES
        if value < low:
            return "low"
        if value < high:
            return "mid"
        return "high"


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = load_data()

    # Train the underlying estimator separately — the wrapper is library-agnostic.
    estimator = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0
    )
    estimator.fit(data.X_train, data.y_train)

    # Pick clip ranges from the training distribution to harden the model
    # against outliers at inference time.
    feature_clips = {
        col: (float(np.quantile(data.X_train[col], 0.001)),
              float(np.quantile(data.X_train[col], 0.999)))
        for col in data.feature_names
    }

    pyfunc_model = PriceBandModel(estimator=estimator, feature_clips=feature_clips)

    # Build the signature from real input/output shapes.
    sample_input = data.X_train.head(5)
    sample_output = pyfunc_model.predict(context=None, model_input=sample_input)
    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run(run_name="price-band-pyfunc") as run:
        mlflow.log_params(
            {
                "wrapped_estimator": estimator.__class__.__name__,
                "band_edges": str(PriceBandModel.BAND_EDGES),
            }
        )

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_model,
            signature=signature,
            input_example=sample_input,
            # pip_requirements pins what serving will install.
            pip_requirements=[
                "mlflow",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
        )
        run_id = run.info.run_id

    # Reload as pyfunc and call it like any other MLflow model.
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    out = loaded.predict(data.X_test.head(5))
    print("Custom PyFunc predictions on 5 rows:")
    print(out.to_string(index=False))

    print(
        "\nThe key idea: downstream code only sees `model.predict(df)`.\n"
        "Preprocessing and post-processing are baked into the artifact, so\n"
        "serving and offline scoring stay identical."
    )


if __name__ == "__main__":
    main()
