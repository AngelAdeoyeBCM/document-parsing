"""Paired comparison of two scored runs on the same pages, with a bootstrap interval that resamples documents.

Pages are the unit; documents are resampled because pages of one document are not independent. The result is
written to ``<other run>/evaluation/compare_<base run>.csv``, so it is logged with that run.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .score import EVALUATION_DIR, as_number, read_csv, write_csv

Key = tuple[str, str]  # (doc_id, page); a blank page in a strata file tags the whole document


def compare_runs(
    base: Path,
    other: Path,
    metrics: Sequence[str],
    split: str | None = 'screen',
    strata: Path | None = None,
    seed: int = 0,
) -> Path:
    """Compare ``other`` with ``base`` on each metric, over all pages and per stratum tag; returns the CSV path."""
    base_rows, other_rows = (read_csv(run / EVALUATION_DIR / 'pages.csv') for run in (base, other))
    groups: dict[str, set[Key] | None] = {'all': None} | (load_strata(strata) if strata else {})
    results = []
    for metric in metrics:
        for name, keys in groups.items():
            chosen = [_select(rows, split, keys) for rows in (base_rows, other_rows)]
            result = compare(*chosen, metric, seed=seed)
            if result is not None:
                results.append({'metric': metric, 'stratum': name, 'split': split or 'all', **result})
    path = other / EVALUATION_DIR / f'compare_{base.name}.csv'
    write_csv(results, path)
    return path


def compare(
    base: Iterable[Mapping[str, object]],
    other: Iterable[Mapping[str, object]],
    metric: str,
    n: int = 2000,
    seed: int = 0,
) -> dict[str, object] | None:
    """Mean of ``other - base`` over shared pages with a 95% interval; None if no page has the metric in both."""
    before = {_key(row): value for row in base if (value := as_number(row.get(metric))) is not None}
    diffs: dict[str, list[float]] = defaultdict(list)
    for row in other:
        value, key = as_number(row.get(metric)), _key(row)
        if value is not None and key in before:
            diffs[key[0]].append(value - before[key])
    if not diffs:
        return None

    documents, rng = sorted(diffs), random.Random(seed)

    def mean(sample: Iterable[str]) -> float:
        return statistics.fmean(d for document in sample for d in diffs[document])

    boot = sorted(mean(rng.choices(documents, k=len(documents))) for _ in range(n))
    flat = [d for document in documents for d in diffs[document]]
    return {
        'documents': len(documents),
        'pages': len(flat),
        'diff': mean(documents),
        'low': boot[int(0.025 * n)],
        'high': boot[int(0.975 * n)],
        'higher': sum(d > 0 for d in flat),  # other above base; whether that is better depends on the metric
        'equal': sum(d == 0 for d in flat),
        'lower': sum(d < 0 for d in flat),
    }


def load_strata(path: Path) -> dict[str, set[Key]]:
    """``doc_id,page,tags`` rows with space-separated tags, as tag -> pages."""
    strata: dict[str, set[Key]] = defaultdict(set)
    for row in read_csv(path):
        for tag in row['tags'].split():
            strata[tag].add((row['doc_id'], row.get('page') or ''))
    return dict(strata)


def _select(rows: list[dict[str, str]], split: str | None, keys: set[Key] | None) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if (split is None or row['split'] == split)
        and (keys is None or _key(row) in keys or (row['doc_id'], '') in keys)
    ]


def _key(row: Mapping[str, object]) -> Key:
    return str(row['doc_id']), str(row['page'])
