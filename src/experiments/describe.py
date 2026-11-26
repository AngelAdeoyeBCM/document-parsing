"""Markdown for the MLflow UI: the stage overview, each experiment's description and each run's description."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime

from .results import ArmResult
from .spec import Experiment, Spec

STATISTICS = (
    'Pages are the unit. Each row is paired with the control on the same pages, and the difference is the mean of '
    '(row − control) over pages. The 95% interval comes from 2000 bootstrap resamples of documents, because '
    "pages of one document are not independent. Better / worse / tie count pages, using each metric's direction. "
    '✓ marks an interval entirely on the better side, ✗ one entirely on the worse side.'
)


def overview_name(spec: Spec) -> str:
    return f'stage-{spec.stage}/00 overview'


def experiment_name(spec: Spec, experiment: Experiment) -> str:
    return f'stage-{spec.stage}/{experiment.id} {experiment.title}'


def stage_overview(spec: Spec) -> str:
    experiments = [
        [
            e.id, e.title, e.pipeline_stage, e.target, e.question, len(e.variants),
            e.control or 'latest winner', _code(e.winner or 'not chosen'),
        ]
        for e in spec.experiments
    ]  # fmt: skip
    return _join([
        f'# Stage {spec.stage}: {spec.title}',
        spec.aim,
        _pairs([
            ('Spec', _code(spec.path.as_posix())),
            ('Base config', _code(spec.base_config.as_posix())),
            ('Reference rows, in every table', ', '.join(_code(p.as_posix()) for p in spec.references) or 'none'),
            ('Default split', spec.split),
            ('Strata', _code(spec.strata.as_posix()) if spec.strata else 'none'),
        ]),
        '## Experiments',
        (
            'Each one is its own MLflow experiment and changes parser options on its control: the base config '
            '(`base`), a named experiment\'s winner, or the winner of the latest earlier experiment that has one.'
        ),
        _table(['id', 'title', 'pipeline stage', 'target', 'question', 'variants', 'control', 'winner'], experiments),
        '## Metrics',
        _metrics(spec.metrics),
        '## Statistics',
        STATISTICS,
    ])  # fmt: skip


def experiment_description(spec: Spec, experiment: Experiment, results: Sequence[ArmResult]) -> str:
    control = spec.control(experiment)
    changed = spec.changed(experiment)
    best = spec.best_from(experiment)
    origin = f'winner of {best.id}' if best else 'base config'
    fixed = [[_code(k), _code(v)] for k, v in sorted(control.resolved_options().items()) if k not in changed]
    arms = [
        [_code(r.arm.name), r.arm.role, _changes(spec, experiment, r), _code(r.arm.run_dir.as_posix()), r.status]
        for r in results
    ]
    metrics = experiment.metrics
    return _join([
        f'# {experiment.id} · {experiment.title}',
        f'Stage {spec.stage} · pipeline stage: {experiment.pipeline_stage} · target {experiment.target}',
        f'**Question.** {experiment.question}',
        experiment.notes,
        _pairs([
            ('Independent variables', ', '.join(map(_code, changed)) or 'none'),
            ('Control', f'{_code(control.name)} ({origin})'),
            ('Split', spec.split_of(experiment)),
            ('Winner', _code(experiment.winner) if experiment.winner else 'not chosen'),
        ]),
        '## Rows',
        _table(['row', 'role', 'changes from control', 'run directory', 'status'], arms),
        '## Controls held fixed',
        'Every parser option of the control that no variant changes. All rows also share the input manifest.',
        _table(['option', 'value'], fixed),
        '## Metrics',
        _metrics({m: spec.info(m) for m in metrics}),
        '## Statistics',
        STATISTICS,
        f'## Results ({spec.split_of(experiment)} split, updated {datetime.now():%Y-%m-%d %H:%M})',
        '### Means',
        _table(
            ['row', 'role', *metrics],
            [[_code(r.arm.name), r.arm.role, *(_num(r.means.get(m)) for m in metrics)] for r in results],
        ),
        '### Difference from the control',
        _table(
            ['row', *metrics],
            [[_code(r.arm.name), *(_interval(r.overall(m)) for m in metrics)] for r in results if r.comparison],
        ),
    ])  # fmt: skip


def run_description(spec: Spec, experiment: Experiment, result: ArmResult) -> str:
    rows = [
        [r['metric'], r['stratum'], _interval(r), r['better'], r['worse'], r['tie'], r['pages'], r['documents']]
        for r in result.comparison
    ]
    parts = [
        f'**{result.arm.role}** {_code(result.arm.name)} in {experiment.id} · {experiment.title}',
        f'Run directory: {_code(result.arm.run_dir.as_posix())}',
        f'Changes from the control: {_changes(spec, experiment, result)}',
    ]
    if rows:
        parts += [
            f'### Difference from the control ({spec.split_of(experiment)} split)',
            _table(['metric', 'stratum', 'diff [95% CI]', 'better', 'worse', 'tie', 'pages', 'documents'], rows),
        ]
    return _join(parts)


def _changes(spec: Spec, experiment: Experiment, result: ArmResult) -> str:
    arm = result.arm
    if arm.role == 'control':
        return 'none (control)'
    if arm.role == 'reference':
        return f'reference: parser {_code(arm.raw["parser"]["name"])}'
    control = spec.control(experiment).options
    changes = [
        f'{_code(k)}: {_code(control.get(k, "default"))} → {_code(v)}'
        for k, v in arm.options.items()
        if k not in control or control[k] != v
    ]
    return ', '.join(changes) or 'none'


def _metrics(metrics: dict) -> str:
    """Metric glossary rows, grouped by target (A text quality, B layout, C serializer)."""
    ordered = sorted(metrics.values(), key=lambda i: i.target or '~')  # stable: spec order within a target
    rows = [[i.target, _code(i.column), i.label, i.better, 'yes' if i.gold else 'no', i.description] for i in ordered]
    return _table(['target', 'column', 'metric', 'better', 'needs gold', 'definition'], rows)


def _interval(row: dict | None) -> str:
    if row is None:
        return 'n/a'
    mark = {'better': ' ✓', 'worse': ' ✗'}.get(row['significant'], '')
    return f'{row["diff"]:+.4f} [{row["low"]:+.4f}, {row["high"]:+.4f}]{mark}'


def _num(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.4f}'


def _code(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return f'`{text}`'


def _join(parts: Iterable[str]) -> str:
    return '\n\n'.join(part for part in parts if part)


def _pairs(pairs: Iterable[tuple[str, object]]) -> str:
    return _table(['', ''], [[f'**{k}**', v] for k, v in pairs])


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rows = list(rows)
    if not rows:
        return '_none_'
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += [
        '| ' + ' | '.join(str(cell).replace('|', '\\|').replace('\n', ' ') for cell in row) + ' |' for row in rows
    ]
    return '\n'.join(lines)
