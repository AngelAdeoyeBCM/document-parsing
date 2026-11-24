from __future__ import annotations

import re
from collections.abc import Sequence

from .base import PageClassifier, PageVerdict

DISCLOSURE_PATTERNS = (
    r'important disclosures?',
    r'analyst certification',
    r'disclosure appendix',
    r'distribution of ratings',
    r'ratings? (?:definitions|distribution|key|system)',
    r'conflicts? of interest',
    r'investment banking (?:services|relationship|compensation)',
    r'this (?:report|document|material|publication) (?:has been|is|was) (?:prepared|issued|produced|distributed)',
    r'not (?:an|a) (?:offer|solicitation)',
    r'past performance is not',
    r'regulatory disclosures?',
    r'for (?:professional|institutional) (?:investors|clients) only',
    r'authori[sz]ed (?:and|by) .{0,40}regulated by',
    r'financial conduct authority',
    r'prudential regulation authority',
    r'securities and exchange commission',
    r'all rights reserved',
    r'general disclaimer',
    r'price target history',
    r'does not constitute (?:investment|legal|tax) advice',
)


class RuleClassifier(PageClassifier):
    """Flags pages dense in disclosure and disclaimer phrasing."""

    def __init__(self, patterns: Sequence[str] = DISCLOSURE_PATTERNS, min_hits: int = 3) -> None:
        self._patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        self.min_hits = min_hits

    def classify(self, pages: Sequence[str]) -> list[PageVerdict]:
        return [self._verdict(number, text) for number, text in enumerate(pages, start=1)]

    def _verdict(self, number: int, text: str) -> PageVerdict:
        hits = sum(1 for pattern in self._patterns if pattern.search(text))
        return PageVerdict(
            page=number,
            boilerplate=hits >= self.min_hits,
            score=min(1.0, hits / self.min_hits),
            reason=f'{hits} disclosure phrases' if hits else '',
        )
