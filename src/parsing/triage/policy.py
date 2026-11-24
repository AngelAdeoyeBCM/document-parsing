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

    Flags on pages 6, 9 and 11 of 12 skip 11-12; flags on 9-11 of 12 skip 9-12.
    """

    def _select(self, verdicts: Sequence[PageVerdict]) -> set[int]:
        flagged = {verdict.page for verdict in verdicts if verdict.boilerplate}
        if not flagged:
            return set()
        start = max(flagged)
        while start - 1 in flagged:
            start -= 1
        return set(range(start, len(verdicts) + 1))
