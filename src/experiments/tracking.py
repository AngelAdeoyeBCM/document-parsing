"""Log a stage's experiments to MLflow: one MLflow experiment per experiment, one MLflow run per row.

The description of each experiment and run (``mlflow.note.content``) is the markdown from ``describe.py``.
Re-logging a run directory replaces its earlier MLflow run, so the table never shows duplicates.
"""

from __future__ import annotations

from collections.abc import Sequence

from evaluation import EVALUATION_DIR

from .describe import experiment_description, experiment_name, overview_name, run_description, stage_overview
from .results import ArmResult
from .spec import Experiment, Spec

NOTE = 'mlflow.note.content'  # rendered as the description in the MLflow UI
KIND = ('mlflow.experimentKind', 'custom_model_development')  # open in the UI's run table, not the GenAI view


def log_overview(spec: Spec) -> str:
    import mlflow  # optional dependency: only logging needs it

    experiment_id = mlflow.set_experiment(overview_name(spec)).experiment_id
    client = mlflow.MlflowClient()
    client.set_experiment_tag(experiment_id, *KIND)
    client.set_experiment_tag(experiment_id, 'stage', spec.stage)
    client.set_experiment_tag(experiment_id, NOTE, stage_overview(spec))
    return experiment_id


def log_experiment(spec: Spec, experiment: Experiment, results: Sequence[ArmResult]) -> str:
    """Log every parsed row and the experiment's description; returns the MLflow experiment id."""
    import mlflow

    client = mlflow.MlflowClient()
    experiment_id = mlflow.set_experiment(experiment_name(spec, experiment)).experiment_id
    changed = spec.changed(experiment)
    for result in results:
        if result.status != 'ok':
            continue
        arm, run_dir = result.arm, result.arm.run_dir.as_posix()
        for old in client.search_runs([experiment_id], f"tags.run_dir = '{run_dir}'"):
            client.delete_run(old.info.run_id)
        options = arm.resolved_options()
        missing = 'n/a' if arm.role == 'reference' else 'default'
        tags = {
            'stage': spec.stage,
            'experiment': experiment.id,
            'role': arm.role,
            'row': arm.name,
            'run_dir': run_dir,
            NOTE: run_description(spec, experiment, result),
        }
        with mlflow.start_run(experiment_id=experiment_id, run_name=arm.name):
            mlflow.set_tags(tags)
            mlflow.log_params({**result.params, **{f'iv.{key}': options.get(key, missing) for key in changed}})
            mlflow.log_metrics(_metrics(result))
            mlflow.log_artifacts(str(arm.run_dir / EVALUATION_DIR), artifact_path=EVALUATION_DIR)
            if (arm.run_dir / 'settings.json').is_file():
                mlflow.log_artifact(str(arm.run_dir / 'settings.json'))
            mlflow.log_dict(dict(arm.raw), 'config.yaml')
            if result.comparison:
                columns = list(result.comparison[0])
                mlflow.log_table({c: [row[c] for row in result.comparison] for c in columns}, 'comparison.json')
    client.set_experiment_tag(experiment_id, *KIND)
    client.set_experiment_tag(experiment_id, 'stage', spec.stage)
    client.set_experiment_tag(experiment_id, 'experiment', experiment.id)
    client.set_experiment_tag(experiment_id, NOTE, experiment_description(spec, experiment, results))
    return experiment_id


def _metrics(result: ArmResult) -> dict[str, float]:
    """Means, then the overall difference from the control: ``diff/<m>``, ``diff_low/<m>``, ``diff_high/<m>`` and
    page counts. No key nests under another (``diff/<m>/low``), which a file store cannot hold."""
    metrics = dict(result.means)
    for row in result.comparison:
        if row['stratum'] == 'all':
            m = row['metric']
            metrics |= {f'diff/{m}': row['diff'], f'diff_low/{m}': row['low'], f'diff_high/{m}': row['high']}
            metrics |= {f'{count}/{m}': row[count] for count in ('better', 'worse', 'tie')}
    return metrics
