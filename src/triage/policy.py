from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .base import PageVerdict


class SkipPolicy(ABC):
    """Turns page verdicts into the pages the parser skips; the first ``keep_first`` pages are always kept."""

    def __init__(self, keep_first: int = 1) -> None:
        self.keep_first = keep_first

    def skip(self, verdicts: Sequence[PageVerdict]) -> frozenset[int]:
        return frozenset(page for page in self._select(verdicts) if page > self.keep_first)

    @abstractmethod
    def _select(self, verdicts: Sequence[PageVerdict]) -> set[int]: ...


class FlaggedPolicy(SkipPolicy):
    """Every flagged page, wherever it falls."""

    def _select(self, verdicts: Sequence[PageVerdict]) -> set[int]:
        return {verdict.page for verdict in verdicts if verdict.boilerplate}


class TailPolicy(SkipPolicy):
    """One block ending at the final page: from the start of the last flagged run through to the end.

    The run may bridge up to ``max_gap`` unflagged pages, such as a ratings table inside a disclosure appendix, and
    never starts within the first ``keep_first`` pages. With ``max_gap=1``, flags on 6, 9 and 11 of 12 skip 9-12;
    flags on 6 and 11 skip 11-12.
    """

    def __init__(self, keep_first: int = 1, max_gap: int = 1) -> None:
        super().__init__(keep_first)
        self.max_gap = max_gap

    def _select(self, verdicts: Sequence[PageVerdict]) -> set[int]:
        flagged = sorted(verdict.page for verdict in verdicts if verdict.boilerplate and verdict.page > self.keep_first)
        if not flagged:
            return set()
        start = flagged[-1]
        for page in reversed(flagged[:-1]):
            if start - page - 1 > self.max_gap:
                break
            start = page
        return set(range(start, len(verdicts) + 1))
