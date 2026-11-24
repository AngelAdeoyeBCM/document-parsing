from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from parsing.core import ParsedDocument

from .metrics import DEFAULT_METRICS, Metric
from .reference import Reference

Row = dict[str, object]


class Evaluator:
    """Scores parsed documents with every metric; references are cached per source PDF."""

    def __init__(self, metrics: Sequence[Metric] = DEFAULT_METRICS) -> None:
        self.metrics = tuple(metrics)
        self._references: dict[str, Reference] = {}

    def evaluate(self, document: ParsedDocument, run: str) -> Row:
        reference = self._reference(document)
        row: Row = {
            'run': run,
            'doc_id': document.source.doc_id,
            'publisher': document.source.metadata.get('publisher', ''),
            'parser': document.parser.name,
            'pages_parsed': len(document.document.pages),
            'pages_skipped': len(document.skipped_pages),
        }
        for metric in self.metrics:
            row.update({f'{metric.name}.{key}': value for key, value in metric.score(document, reference).items()})
        return row

    def _reference(self, document: ParsedDocument) -> Reference:
        key = document.source.hash_id
        if key not in self._references:
            self._references[key] = Reference.from_source(document.source)
        return self._references[key]


def summarise(rows: Iterable[Row]) -> list[Row]:
    """Mean of every numeric column per run, ignoring NaN (e.g. coverage on scanned PDFs)."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[str(row['run'])].append(row)
    summary: list[Row] = []
    for run, members in sorted(groups.items()):
        means: Row = {'run': run, 'documents': len(members)}
        for key in members[0]:
            values = [
                value for row in members if isinstance(value := row.get(key), (int, float)) and not math.isnan(value)
            ]
            if values and key not in means:
                means[key] = sum(values) / len(values)
        summary.append(means)
    return summary


def write_csv(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
