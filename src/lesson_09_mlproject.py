"""Lesson 9 — MLflow Projects: reproducible runs you can fire from anywhere.

An MLflow Project is just a folder with an ``MLproject`` file describing:
    - the Python (or conda) environment to use,
    - the entry point(s) and their parameters,
    - the command to run.

Once that exists, anyone (CI, a teammate, future-you) can reproduce the run
identically with a single command. No "what version of pandas did you have?".

This script is just a convenience wrapper that calls ``mlflow.projects.run``
on the project at ``./mlproject``. You can equivalently run it from a shell:

    mlflow run mlproject -P n_estimators=500 --env-manager=local

Run (from the project root, with the venv active):
    python -m src.lesson_09_mlproject
"""

from __future__ import annotations

import mlflow

from src.utils import PROJECT_ROOT


def main() -> None:
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")

    project_path = PROJECT_ROOT / "mlproject"

    # env_manager="local" reuses the current Python environment instead of
    # creating a fresh one. For a true reproducibility demo, drop this flag and
    # MLflow will build a virtualenv from python_env.yaml.
    submitted = mlflow.projects.run(
        uri=str(project_path),
        entry_point="main",
        parameters={
            "n_estimators": 400,
            "learning_rate": 0.04,
            "max_depth": 4,
        },
        env_manager="local",
        experiment_name="lesson-09-mlproject",
    )
    print(f"Project run finished with status: {submitted.get_status()}")
    print(f"Run id: {submitted.run_id}")


if __name__ == "__main__":
    main()
