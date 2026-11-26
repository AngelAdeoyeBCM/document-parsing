"""Each run of an experiment against its control, on the experiment's split.

Comparisons reuse ``evaluation.compare_runs``: paired pages, 95% bootstrap interval resampling documents. Counts
are turned from higher/lower into better/worse with the metric's direction.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation import EVALUATION_DIR, compare_runs
from evaluation.score import read_csv

from .spec import Arm, Experiment, Spec


@dataclass
class ArmResult:
    arm: Arm
    status: str  # 'ok' or 'not parsed'
    params: dict[str, Any] = field(default_factory=dict)  # evaluation/params.json
    means: dict[str, float] = field(default_factory=dict)  # metric means on the experiment's split
    comparison: list[dict[str, Any]] = field(default_factory=list)  # against the control, per metric and stratum

    def overall(self, metric: str) -> dict[str, Any] | None:
        return next((r for r in self.comparison if r['metric'] == metric and r['stratum'] == 'all'), None)


def stale(run_dir: Path) -> bool:
    """Parsed, but not scored since its last parse."""
    metrics, manifest = run_dir / EVALUATION_DIR / 'metrics.json', run_dir / 'run.jsonl'
    return manifest.is_file() and (not metrics.is_file() or metrics.stat().st_mtime < manifest.stat().st_mtime)


def evaluate(spec: Spec, experiment: Experiment, score: Callable[[Path], object], seed: int = 0) -> list[ArmResult]:
    """Score stale runs with ``score``, then compare every parsed run of the experiment with its control."""
    split = spec.split_of(experiment)
    metrics = experiment.metrics
    strata = spec.strata if spec.strata and spec.strata.is_file() else None
    control, arms = spec.control(experiment), spec.arms(experiment)
    for arm in arms:  # all first: every run is compared with the control
        if stale(arm.run_dir):
            score(arm.run_dir)
    results = []
    for arm in arms:
        if not arm.parsed:
            results.append(ArmResult(arm, 'not parsed'))
            continue
        evaluation = arm.run_dir / EVALUATION_DIR
        means = json.loads((evaluation / 'metrics.json').read_text(encoding='utf-8'))
        prefix = '' if split == 'all' else f'{split}/'
        result = ArmResult(
            arm,
            'ok',
            params=json.loads((evaluation / 'params.json').read_text(encoding='utf-8')),
            means={m: means[prefix + m] for m in metrics if prefix + m in means},
        )
        if arm.role != 'control' and control.parsed and arm.run_dir != control.run_dir:
            path = compare_runs(control.run_dir, arm.run_dir, metrics, None if split == 'all' else split, strata, seed)
            result.comparison = [_orient(row, spec.info(row['metric']).better) for row in read_csv(path)]
        results.append(result)
    return results


def _orient(row: dict[str, str], better: str) -> dict[str, Any]:
    """Compare-CSV row with numbers; ``significant`` says whether the interval excludes zero, and on which side."""
    sign = 1 if better == 'higher' else -1
    diff, low, high = (float(row[key]) for key in ('diff', 'low', 'high'))
    gain_low, gain_high = sorted((sign * low, sign * high))
    return {
        'metric': row['metric'],
        'stratum': row['stratum'],
        'documents': int(row['documents']),
        'pages': int(row['pages']),
        'diff': diff,
        'low': low,
        'high': high,
        'significant': 'better' if gain_low > 0 else 'worse' if gain_high < 0 else '',
        'better': int(row['higher' if sign > 0 else 'lower']),
        'worse': int(row['lower' if sign > 0 else 'higher']),
        'tie': int(row['equal']),
    }
