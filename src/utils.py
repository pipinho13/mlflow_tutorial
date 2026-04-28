"""Shared helpers used across the tutorial lessons.

The dataset is California Housing (regression). It ships with scikit-learn so
nothing has to be downloaded manually, which keeps the tutorial reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def load_data(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> Dataset:
    """Load the California Housing dataset as a typed train/test split."""
    raw = fetch_california_housing(as_frame=True)
    X = raw.data
    y = raw.target.rename("median_house_value")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a small standard set of regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
