"""Score one parse run and write its evaluation directory, laid out for MLflow.

``<run dir>/evaluation/`` holds ``pages.csv`` (one row per parsed page), ``metrics.json`` (means, overall and per
split) and ``params.json`` (the parser, its settings and the serializations scored). One parse run is one MLflow run.

A. Text quality is measured on the 'extracted' serialization (everything the parser read), overall and per element
kind (``extracted.garble_rate.caption``, ``.page_footer``, ``.picture_text``, ...) to show where garbling comes from;
on gold pages ``extracted.loss`` is the word loss against the gold full-page text.
B. Layout, on gold pages: ``layout.f1`` (and ``layout.f1.<label>``) from the labelled boxes, and the targeted
confusions ``layout.leak_from.<group>`` and ``layout.loss_to.<group>``.
C. Serializer: each serialization scored against gold (``parsing.core.SERIALIZATIONS``) gets columns prefixed with
its name, e.g. ``body.loss``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from annotation import Gold
from inventory import InventoryStore
from parsing.core import serialize, serialize_elements
from pipeline import RunManifest

from .metrics import CONFUSION_GROUPS, dropped_text_rate, garble_rate, gold_metrics, label_confusions, layout_f1

logger = logging.getLogger(__name__)

EVALUATION_DIR = 'evaluation'
HOLDOUT_SHARE = 0.2  # documents held out of screening (E1-E7) until validation (E8)
DEFAULT_SERIALIZATIONS = ('body', 'body_tables')
_KEYS = ('doc_id', 'page', 'split')

Row = dict[str, object]


def split(hash_id: str) -> str:
    """'holdout' or 'screen', fixed by the document's content hash so every run agrees."""
    return 'holdout' if int(hash_id[:8], 16) < HOLDOUT_SHARE * 16**8 else 'screen'


def evaluate_run(
    run_dir: Path,
    store: InventoryStore,
    gold: Mapping[str, Gold],
    gold_name: str = '',
    serializations: Sequence[str] = DEFAULT_SERIALIZATIONS,
) -> Path:
    """Score every parsed page of the run and write ``<run dir>/evaluation``; returns that directory.

    ``gold`` is ``annotation.load_gold``'s result; the pages a gold document holds are scored under each of
    ``serializations``, run and gold serialized alike.
    """
    names = tuple(dict.fromkeys(('extracted', 'body', *serializations)))  # 'body' for the layout confusions
    rows: list[Row] = []
    documents = gold_pages = 0
    parser: dict[str, object] = {}
    for parsed in RunManifest(run_dir).documents():
        source = parsed.source
        reference = store.page_texts(source.hash_id)
        if reference is None:
            raise ValueError(f'{source.doc_id}: no stored text layer; is this the inventory the run used?')
        text = {name: serialize(parsed.document, name) for name in names}
        elements = serialize_elements(parsed.document)
        truth = _gold_text(gold, source.doc_id, source.hash_id, names)
        seconds = parsed.seconds / max(1, len(parsed.page_numbers))
        pypdf = {page.page_number: page.pypdf_seconds for page in store.pages(source.hash_id)}
        numbers = parsed.page_numbers
        pypdf_seconds = statistics.fmean(pypdf.get(n, math.nan) for n in numbers) if numbers else math.nan
        for number in parsed.page_numbers:
            output = text['extracted'].get(number, '')
            row: Row = {
                'doc_id': source.doc_id,
                'page': number,
                'split': split(source.hash_id),
                'seconds_per_page': seconds,
                'pypdf_seconds_per_page': pypdf_seconds,  # the inventory's pypdf pass over the same pages
                'extracted.garble_rate': garble_rate(output),
                'extracted.dropped_text_rate': dropped_text_rate(output, reference[number - 1]),
            }
            for kind, pages in elements.items():
                if number in pages:
                    row[f'extracted.garble_rate.{kind}'] = garble_rate(pages[number])
            if truth and number in truth['pages']:
                gold_pages += 1
                full = gold_metrics(truth['extracted'].get(number, ''), '', output)
                if full:
                    row['extracted.loss'] = full['loss']
                layout = layout_f1(truth['document'], parsed.document, number) | label_confusions(
                    truth['body'].get(number, ''),
                    _groups(truth['elements'], number),
                    text['body'].get(number, ''),
                    _groups(elements, number),
                )
                row |= {f'layout.{metric}': value for metric, value in layout.items()}
                for name in serializations:
                    expected, everything = truth[name].get(number, ''), truth['extracted'].get(number, '')
                    scores = gold_metrics(expected, everything, text[name].get(number, ''))
                    row |= {f'{name}.{metric}': value for metric, value in scores.items()}
            rows.append(row)
        documents += 1
        parser = _parser_params(parsed.parser.name, parsed.parser.version, parsed.parser.config)

    out = run_dir / EVALUATION_DIR
    write_csv(rows, out / 'pages.csv')
    params = {
        'run': run_dir.name,
        'gold': gold_name,
        'documents': documents,
        'pages': len(rows),
        'gold_pages': gold_pages,
        'serializations': ','.join(serializations),
        **parser,
    }
    (out / 'params.json').write_text(json.dumps(params, indent=1), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(means(rows), indent=1), encoding='utf-8')
    return out


def means(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Mean of every metric over pages, ignoring NaN: ``<metric>`` overall and ``<split>/<metric>`` per split."""
    rows = list(rows)
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            number = as_number(value)
            if key not in _KEYS and number is not None:
                for name in (key, f'{row["split"]}/{key}'):
                    values.setdefault(name, []).append(number)
    return {name: statistics.fmean(found) for name, found in sorted(values.items())}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _gold_text(gold: Mapping[str, Gold], doc_id: str, hash_id: str, names: Sequence[str]) -> dict[str, Any] | None:
    """The gold document under each serialization, with its annotated pages; None if it has no usable gold."""
    if doc_id not in gold:
        return None
    source_hash, document = gold[doc_id]
    if source_hash != hash_id:
        logger.warning('%s: gold was annotated on a different file; its pages are not scored', doc_id)
        return None
    return {
        'pages': set(document.pages),
        'document': document,
        'elements': serialize_elements(document),
        **{name: serialize(document, name) for name in names},
    }


def _groups(elements: Mapping[str, Mapping[int, str]], page: int) -> dict[str, str]:
    """The page's text in each targeted-confusion group, from ``serialize_elements``."""
    return {
        group: '\n'.join(elements.get(kind, {}).get(page, '') for kind in kinds)
        for group, kinds in CONFUSION_GROUPS.items()
    }


def _parser_params(name: str, version: str, config: Mapping[str, object]) -> dict[str, object]:
    """Flat, MLflow-ready parser settings; nested values become JSON."""
    flat = {
        f'parser.{key}': value if isinstance(value, str | int | float | bool | None) else json.dumps(value)
        for key, value in config.items()
    }
    return {'parser': name, 'parser_version': version, **flat}


def as_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number
