from __future__ import annotations

from collections.abc import Sequence

from .base import PageVerdict


class TailPolicy:
    """Skips one block ending at the final page: from the start of the last flagged run through to the end.

    The run may bridge up to ``max_gap`` unflagged pages, such as a ratings table inside a disclosure appendix, and
    never starts within the first ``keep_first`` pages. With ``max_gap=1``, flags on 6, 9 and 11 of 12 skip 9-12;
    flags on 6 and 11 skip 11-12. The pages kept are always the leading ones, so a truncated PDF keeps the source's
    page numbers.
    """

    def __init__(self, keep_first: int = 1, max_gap: int = 1) -> None:
        self.keep_first = keep_first
        self.max_gap = max_gap

    def skip(self, verdicts: Sequence[PageVerdict]) -> frozenset[int]:
        flagged = sorted(verdict.page for verdict in verdicts if verdict.boilerplate and verdict.page > self.keep_first)
        if not flagged:
            return frozenset()
        start = flagged[-1]
        for page in reversed(flagged[:-1]):
            if start - page - 1 > self.max_gap:
                break
            start = page
        return frozenset(range(start, len(verdicts) + 1))
