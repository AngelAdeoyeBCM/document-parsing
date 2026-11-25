"""Stage 1.1 metrics for one page.

Text quality (target A) needs no gold: garble rate, and dropped text against the pypdf text layer the current
pipeline reads. Serializer output (target C) is scored against the gold document serialized the same way: loss,
leak, separation and order gap. Words are compared as multisets after the same normalisation, so order only
enters order gap.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_WORD = re.compile(r'[^\W_]+')
_HYPHEN_BREAK = re.compile(r'(?<=[^\W\d_])-\r?\n\s*(?=[^\W\d_])')  # 'eco-\nnomic', not '2024-\n25'
# Replacement character, unmapped glyphs (cid:n), private-use code points, spacing or stray combining diacritics.
_GARBLE = re.compile(r'\ufffd|\(cid:\d+\)|[\ue000-\uf8ff\u00a8\u00b4\u00b8\u02c6-\u02dd\u0300-\u036f]')


def normalise(text: str) -> str:
    """NFC, with words hyphenated across a line break joined."""
    return unicodedata.normalize('NFC', _HYPHEN_BREAK.sub('', text))


def words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(normalise(text))]


def garble_rate(text: str) -> float:
    """Share of whitespace-separated tokens holding a broken character."""
    tokens = normalise(text).split()
    return sum(1 for token in tokens if _GARBLE.search(token)) / len(tokens) if tokens else math.nan


def dropped_text_rate(output: str, reference: str) -> float:
    """Share of the reference words (the pypdf text layer) missing from the output."""
    return 1 - _recall(Counter(words(reference)), Counter(words(output)))


def gold_metrics(expected: str, gold_all: str, output: str) -> dict[str, float]:
    """One page of a run's output against the gold page under the same serialization (``expected``).

    ``gold_all`` is all the gold page's text (its 'extracted' serialization): what it holds beyond ``expected`` (captions, footers, picture text,
    tables when the serialization drops them) is what the serialization excludes, tracked by ``leak_excluded``
    and ``separation``.
    """
    want_words, produced = words(expected), words(output)
    if not want_words:
        return {}
    want, out = Counter(want_words), Counter(produced)
    excluded = Counter(words(gold_all)) - want
    leaked = out - want
    leaked_excluded = sum((leaked & excluded).values())
    n = len(produced) or 1
    return {
        'loss': 1 - _recall(want, out),
        'leak': sum(leaked.values()) / n,
        'leak_excluded': leaked_excluded / n,
        'leak_other': (sum(leaked.values()) - leaked_excluded) / n,  # text the gold page does not hold at all
        'separation': 1 - _recall(excluded, leaked),
        'order_gap': _f1(want, out) - (1 - _edit_distance(want_words, produced) / len(want_words)),
    }


def _recall(expected: Counter[str], produced: Counter[str]) -> float:
    total = sum(expected.values())
    return sum((expected & produced).values()) / total if total else math.nan


def _f1(expected: Counter[str], produced: Counter[str]) -> float:
    overlap = sum((expected & produced).values())
    return 2 * overlap / (sum(expected.values()) + sum(produced.values())) if overlap else 0.0


def _edit_distance(first: list[str], second: list[str]) -> int:
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current = [i]
        for j, b in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]
