from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from parsing.core import ParsedDocument

from .evaluator import Row
from .metrics import _recall, numbers, page_texts, words

_FILE = re.compile(r'p(\d+)(_table)?')
_SPACED_PERCENT = re.compile(r'(\d)\s+%')


def _normalise(text: str) -> str:
    return _SPACED_PERCENT.sub(r'\1%', text)


def _key(text: str) -> str:
    return ''.join(words(text))


@dataclass(frozen=True, slots=True)
class GoldPage:
    """A hand-checked page: prose in reading order, or table rows of label then values."""

    number: int
    text: str = ''
    rows: tuple[tuple[str, ...], ...] = ()


def load_gold(directory: Path) -> list[GoldPage]:
    """Read ``p<N>.txt`` (prose) and ``p<N>_table.tsv`` (tab-separated rows) from one document's gold directory."""
    pages = []
    for path in sorted(directory.iterdir()):
        match = _FILE.fullmatch(path.stem)
        if not match:
            continue
        content = path.read_text(encoding='utf-8')
        if match[2]:
            rows = tuple(tuple(line.split('\t')) for line in content.splitlines() if line.strip())
            pages.append(GoldPage(int(match[1]), rows=rows))
        else:
            pages.append(GoldPage(int(match[1]), text=content))
    return pages


def score(parsed: ParsedDocument, gold: list[GoldPage], run: str) -> Row:
    """Accuracy of the parsed text against the gold pages; prose by word alignment, tables by value."""
    texts = {number: _normalise('\n'.join(items)) for number, items in page_texts(parsed.document).items()}
    row: Row = {'run': run, 'doc_id': parsed.source.doc_id, 'parser': parsed.parser.name}
    for page in gold:
        output = texts.get(page.number, '')
        prefix = f'p{page.number}'
        if page.text:
            row.update({f'{prefix}.{key}': value for key, value in _prose(page.text, output).items()})
        if page.rows:
            row.update({f'{prefix}.{key}': value for key, value in _table(page, output, parsed.document).items()})
    row['seconds_per_page'] = parsed.seconds / max(1, len(parsed.document.pages))
    return row


def _prose(gold: str, output: str) -> dict[str, float]:
    expected, produced = words(gold), words(output)
    aligned = sum(
        block.size for block in SequenceMatcher(None, expected, produced, autojunk=False).get_matching_blocks()
    )
    return {
        'word_recall': aligned / len(expected),
        'wer': _edit_distance(expected, produced) / len(expected),
        'number_recall': _recall(Counter(numbers(_normalise(gold))), Counter(numbers(output))),
    }


def _table(page: GoldPage, output: str, doc: DoclingDocument) -> dict[str, float]:
    labels = Counter(word for row in page.rows for word in words(row[0]))
    values = Counter(number for row in page.rows for value in row[1:] for number in numbers(value))
    return {
        'label_recall': _recall(labels, Counter(words(output))),
        'number_recall': _recall(values, Counter(numbers(output))),
        'cell_accuracy': _cell_accuracy(page, doc),
    }


def _cell_accuracy(page: GoldPage, doc: DoclingDocument) -> float:
    """Share of gold value cells found in the right table row, so it also depends on table detection."""
    found: dict[str, list[str]] = {}
    for table in doc.tables:
        if not table.prov or table.prov[0].page_no != page.number:
            continue
        for cells in table.data.grid:
            # a spanning cell repeats the same object across the columns it covers
            unique = {id(cell): cell for cell in cells if cell.text.strip()}.values()
            texts = [_normalise(cell.text).replace(' ', '') for cell in unique]
            if texts:
                found.setdefault(_key(texts[0]), texts[1:])
    total = correct = 0
    for label, *expected in page.rows:
        actual = found.get(_key(label), [])
        total += len(expected)
        correct += sum(1 for want, got in zip(expected, actual, strict=False) if want == got)
    return correct / total if total else float('nan')


def _edit_distance(first: list[str], second: list[str]) -> int:
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current = [i]
        for j, b in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]
