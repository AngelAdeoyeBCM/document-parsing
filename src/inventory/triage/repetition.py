from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .base import PageClassifier, PageVerdict

_DIGITS = re.compile(r'\d+')
_SPACE = re.compile(r'\s+')


class RepetitionClassifier(PageClassifier):
    """Flags pages whose lines recur across many documents, e.g. a publisher's standard disclosures."""

    def __init__(self, min_documents: int = 20, min_share: float = 0.6, min_line_chars: int = 20) -> None:
        self.min_documents = min_documents
        self.min_share = min_share
        self.min_line_chars = min_line_chars
        self._document_frequency: Counter[int] = Counter()  # keyed by line hash to bound memory

    @property
    def needs_corpus(self) -> bool:
        return True

    def observe(self, pages: Sequence[str]) -> None:
        self._document_frequency.update({hash(line) for text in pages for line in self._lines(text)})

    def classify(self, pages: Sequence[str]) -> list[PageVerdict]:
        return [self._verdict(number, text) for number, text in enumerate(pages, start=1)]

    def _verdict(self, number: int, text: str) -> PageVerdict:
        lines = self._lines(text)
        total = sum(len(line) for line in lines)
        if not total:
            return PageVerdict(page=number, boilerplate=False, score=0.0)
        repeated = sum(len(line) for line in lines if self._document_frequency[hash(line)] >= self.min_documents)
        share = repeated / total
        return PageVerdict(
            page=number,
            boilerplate=share >= self.min_share,
            score=share,
            reason=f'{share:.0%} of text repeats across documents' if repeated else '',
        )

    def _lines(self, text: str) -> list[str]:
        normalised = (_SPACE.sub(' ', _DIGITS.sub('0', line.lower())).strip() for line in text.splitlines())
        return [line for line in normalised if len(line) >= self.min_line_chars]
