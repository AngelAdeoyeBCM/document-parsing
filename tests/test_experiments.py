from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from evaluation.score import EVALUATION_DIR, write_csv
from experiments import Spec
from experiments.describe import experiment_description, stage_overview
from experiments.results import _orient, evaluate


def _write_yaml(path: Path, raw: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


@pytest.fixture
def spec_path(tmp_path: Path) -> Path:
    def run_config(name: str, parser: dict) -> Path:
        raw = {
            'name': name,
            'input': {'directory': 'storage'},
            'output': (tmp_path / name).as_posix(),
            'parser': parser,
        }
        return _write_yaml(tmp_path / f'{name}.yaml', raw)

    base = run_config('b0', {'name': 'docling', 'options': {'do_ocr': False, 'layout_model': 'layout_heron_default'}})
    reference = run_config('p0', {'name': 'pypdf'})
    return _write_yaml(
        tmp_path / 'spec.yaml',
        {
            'stage': '1.1',
            'title': 'Text-only extraction',
            'base_config': base.as_posix(),
            'references': [reference.as_posix()],
            'output': (tmp_path / 'stage').as_posix(),
            'configs': (tmp_path / 'configs').as_posix(),
            'metrics': {'body.loss': {'label': 'Loss rate', 'target': 'C', 'gold': True}, 'layout.f1': {'target': 'B', 'better': 'higher'}},
            'experiments': [
                {'id': 'E1', 'title': 'Backend', 'variants': {'pypdfium2': {'backend': 'pypdfium2'}}, 'metrics': ['body.loss']},
                {
                    'id': 'E2',
                    'title': 'Assemble and tables',
                    'variants': {
                        'separators': {'use_reading_order_separators': False},
                        'no_tables': {'overrides': {'do_table_structure': False}},
                    },
                    'metrics': ['body.loss'],
                },
                {'id': 'E8', 'title': 'Validate', 'split': 'holdout', 'variants': {'best': {'from': 'E1.pypdfium2'}}},
            ],
        },
    )  # fmt: skip


def test_variants_change_options_on_the_current_best(spec_path: Path) -> None:
    spec = Spec.load(spec_path)
    e1, e2, e8 = spec.experiments
    assert spec.control(e1).name == 'b0' and spec.best_from(e1) is None
    pypdfium2 = spec.variant(e1, 'pypdfium2')
    assert pypdfium2.options == {'do_ocr': False, 'layout_model': 'layout_heron_default', 'backend': 'pypdfium2'}
    assert pypdfium2.run_dir.parent == spec.output and pypdfium2.run_dir.name.startswith('e1_pypdfium2_')
    assert spec.changed(e1) == ['backend']
    assert spec.split_of(e8) == 'holdout' and spec.split_of(e1) == 'screen'
    assert spec.variant(e8, 'best').run_dir == pypdfium2.run_dir  # reuses E1's run
    assert [arm.role for arm in spec.arms(e2)] == ['reference', 'control', 'variant', 'variant']

    raw = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    raw['experiments'][0]['winner'] = 'pypdfium2'
    spec_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    spec = Spec.load(spec_path)
    e1, e2, _ = spec.experiments
    assert spec.control(e2).run_dir == pypdfium2.run_dir and spec.best_from(e2) == e1  # E1's winner is E2's control
    no_tables = spec.variant(e2, 'no_tables')
    assert no_tables.options['backend'] == 'pypdfium2' and no_tables.options['overrides.do_table_structure'] is False
    assert set(spec.changed(e2)) == {'use_reading_order_separators', 'overrides.do_table_structure'}
    other = spec.variant(e2, 'separators')
    assert other.run_dir != no_tables.run_dir and 'overrides.do_table_structure' not in other.options


def test_control_base_ignores_winners_and_a_named_control_waits_for_its_winner(spec_path: Path) -> None:
    from experiments import NoWinnerYet
    from experiments.cli import _ready

    raw = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    raw['experiments'][1]['control'] = 'base'
    raw['experiments'][2]['control'] = 'E1'
    spec_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    spec = Spec.load(spec_path)
    e1, e2, e8 = spec.experiments
    with pytest.raises(NoWinnerYet):
        spec.control(e8)
    assert _ready(spec, []) == [e1, e2]  # E8 waits for E1's winner
    before = spec.variant(e2, 'separators').run_dir

    raw['experiments'][0]['winner'] = 'pypdfium2'
    spec_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    spec = Spec.load(spec_path)
    e1, e2, e8 = spec.experiments
    assert spec.control(e2).name == 'b0' and spec.variant(e2, 'separators').run_dir == before  # runs stay valid
    assert spec.control(e8).run_dir == spec.variant(e1, 'pypdfium2').run_dir and spec.best_from(e8) == e1


def _row(metric: str, diff: float, low: float, high: float, stratum: str = 'all') -> dict[str, str]:
    counts = {'higher': '1', 'equal': '0', 'lower': '2'}
    return {'metric': metric, 'stratum': stratum, 'documents': '2', 'pages': '3', 'diff': diff, 'low': low, 'high': high} | counts  # fmt: skip


def test_differences_are_oriented_by_the_metrics_direction() -> None:
    loss = _orient(_row('body.loss', -0.02, -0.03, -0.01), 'lower')
    assert (loss['significant'], loss['better'], loss['worse'], loss['tie']) == ('better', 2, 1, 0)
    assert _orient(_row('body.loss', 0.0, -0.01, 0.01), 'lower')['significant'] == ''
    assert _orient(_row('body.separation', 0.1, 0.05, 0.2), 'higher')['significant'] == 'better'


def _fake_run(run_dir: Path, loss: list[float], config: dict) -> None:
    """A parsed and scored run: run.jsonl, then evaluation files written after it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {'status': 'ok', 'parser': {'name': 'docling', 'config': config}}
    (run_dir / 'run.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')
    rows = [{'doc_id': d, 'page': 1, 'split': 'screen', 'body.loss': v} for d, v in zip('abc', loss, strict=True)]
    write_csv(rows, run_dir / EVALUATION_DIR / 'pages.csv')
    mean = sum(loss) / len(loss)
    metrics = {'body.loss': mean, 'screen/body.loss': mean, 'screen/body.leak': 0.05}
    (run_dir / EVALUATION_DIR / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (run_dir / EVALUATION_DIR / 'params.json').write_text(json.dumps({'run': run_dir.name}), encoding='utf-8')


def test_runs_scored_before_the_gold_changed_are_scored_again(spec_path: Path) -> None:
    spec = Spec.load(spec_path)
    e1 = spec.experiments[0]
    arms = [spec.reference_arms()[0], spec.control(e1), spec.variant(e1, 'pypdfium2')]
    for arm in arms:
        _fake_run(arm.run_dir, [0.3, 0.3, 0.3], {})
        os.utime(arm.run_dir / 'run.jsonl', (1000, 1000))
        os.utime(arm.run_dir / EVALUATION_DIR / 'metrics.json', (2000, 2000))  # scored after parsing

    scored: list[Path] = []
    evaluate(spec, e1, score=scored.append, gold_time=1500)  # gold older than every score
    assert scored == []
    evaluate(spec, e1, score=scored.append, gold_time=3000)  # gold collected since
    assert scored == [arm.run_dir for arm in arms]


def test_log_experiment_to_mlflow(spec_path: Path, tmp_path: Path, monkeypatch) -> None:
    mlflow = pytest.importorskip('mlflow')
    from experiments.tracking import log_experiment, log_overview

    spec = Spec.load(spec_path)
    e1 = spec.experiments[0]
    control, variant = spec.control(e1), spec.variant(e1, 'pypdfium2')
    _fake_run(spec.reference_arms()[0].run_dir, [0.5, 0.5, 0.5], {})
    _fake_run(control.run_dir, [0.3, 0.3, 0.3], {'backend': 'docling_parse', 'do_ocr': False})
    _fake_run(variant.run_dir, [0.2, 0.2, 0.25], {'backend': 'pypdfium2', 'do_ocr': False})
    results = evaluate(spec, e1, score=lambda run: pytest.fail(f'{run} is already scored'))
    assert [r.status for r in results] == ['ok', 'ok', 'ok'] and results[1].comparison == []  # the control

    description = experiment_description(spec, e1, results)
    assert '## Controls held fixed' in description and '| `do_ocr` | `false` |' in description
    assert '`backend`: `default` → `pypdfium2`' in description  # set on the variant only
    assert '## Experiments' in stage_overview(spec)
    assert stage_overview(spec).index('`layout.f1`') < stage_overview(spec).index('`body.loss`')  # B before C
    assert '| C | `body.loss` | Loss rate |' in description

    monkeypatch.setenv('MLFLOW_TRACKING_URI', (tmp_path / 'mlruns').as_uri())
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE', 'true')
    for _ in range(2):  # logging again replaces the runs
        experiment_id = log_experiment(spec, e1, results)
    log_overview(spec)

    client = mlflow.MlflowClient()
    assert client.get_experiment(experiment_id).name == 'stage-1.1/E1 Backend'
    assert '# E1 · Backend' in client.get_experiment(experiment_id).tags['mlflow.note.content']
    runs = {run.info.run_name: run for run in client.search_runs([experiment_id])}
    assert set(runs) == {'p0', 'b0', 'pypdfium2'}
    logged = runs['pypdfium2'].data
    assert logged.tags['role'] == 'variant' and 'verdict' not in logged.tags
    assert logged.params['iv.backend'] == 'pypdfium2' and runs['b0'].data.params['iv.backend'] == 'docling_parse'
    assert runs['p0'].data.params['iv.backend'] == 'n/a'
    assert logged.metrics['diff/body.loss'] == pytest.approx(-0.25 / 3) and logged.metrics['better/body.loss'] == 3
    artifacts = {item.path for item in client.list_artifacts(runs['pypdfium2'].info.run_id)}
    assert {'evaluation', 'config.yaml', 'comparison.json'} <= artifacts
    assert mlflow.get_experiment_by_name('stage-1.1/00 overview') is not None


def test_free_gpus_are_those_with_room_most_free_first(monkeypatch) -> None:
    from experiments import device

    monkeypatch.setattr(device, 'gpu_memory', lambda: {0: (100, 400), 1: (300, 400), 2: (250, 400)})
    assert device.free_gpus(min_free=0.5) == [1, 2]  # GPU 0 is three-quarters full
    assert device.free_gpus(min_free=0.2) == [1, 2, 0]
    monkeypatch.setattr(device, 'gpu_memory', dict)  # no nvidia-smi
    assert device.free_gpus() == []
