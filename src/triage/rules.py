from __future__ import annotations

import re
from collections.abc import Sequence

from .base import PageVerdict

DISCLOSURE_PATTERNS = (
    r'important disclosures?',
    r'disclaimers?(?: and (?:important )?(?:risk )?information)?',
    r'important (?:risk )?(?:and legal )?information',
    r'analyst certification',
    r'disclosure appendix',
    r'distribution of ratings',
    r'ratings? (?:definitions|distribution|key|system)',
    r'conflicts? of interest',
    r'investment banking (?:services|relationship|compensation)',
    r'this (?:report|document|material|publication|communication) (?:has been|is|was|has not been) (?:prepared|issued|produced|distributed|reviewed)',
    r'(?:provided|issued|distributed|communicat(?:ed|ion)).{0,80}(?:professional clients?|eligible counterparties|institutional investors?)',
    r'(?:professional|institutional) (?:investors?|clients?) (?:or |only|and )',
    r'not (?:an|a) (?:offer|solicitation|recommendation)',
    r'past performance is not',
    r'past (?:investment )?performance',
    r'regulatory disclosures?',
    r'general (?:nature|disclaimer)',
    r'informational purposes? only',
    r'(?:general nature|information provided).{0,30}(?:only|not reviewed)',
    r'not (?:been )?reviewed by any regulatory authority',
    r'authori[sz]ed (?:and|by) .{0,40}regulated by',
    r'regulated (?:and supervised )?by (?:the )?',
    r'financial conduct authority',
    r'prudential regulation authority',
    r'securities and exchange commission',
    r'all rights reserved',
    r'price target history',
    r'(?:does|do) not constitute (?:investment|legal|tax) advice',
    r'(?:does|do) not constitute (?:a )?(?:recommendation|offer|solicitation)',
    r'is (?:not|for) (?:investment|legal|tax) advice',
    r'no (?:representation|warranty|guarantee)',
)


class RuleClassifier:
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
