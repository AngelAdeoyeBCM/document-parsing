"""Log a scored run to MLflow: params, metric means, the evaluation files, and the run's settings and log."""

from __future__ import annotations

import json
from pathlib import Path

from .score import EVALUATION_DIR


def log_run(run_dir: Path, experiment: str) -> str:
    """One MLflow run named after the parse run; returns the MLflow run id."""
    import mlflow  # optional dependency: only logging needs it

    evaluation = run_dir / EVALUATION_DIR
    if not (evaluation / 'metrics.json').is_file():
        raise FileNotFoundError(f'{evaluation}: not scored yet; run `evaluation score {run_dir}` first')
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_dir.name) as run:
        mlflow.log_params(json.loads((evaluation / 'params.json').read_text(encoding='utf-8')))
        mlflow.log_metrics(json.loads((evaluation / 'metrics.json').read_text(encoding='utf-8')))
        mlflow.log_artifacts(str(evaluation), artifact_path=EVALUATION_DIR)
        for name in ('settings.json', 'run.jsonl'):  # what was parsed, and how
            if (run_dir / name).is_file():
                mlflow.log_artifact(str(run_dir / name))
    return run.info.run_id
