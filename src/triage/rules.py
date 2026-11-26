from __future__ import annotations

import math
import re
from collections import Counter
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
    r'\bthis (?:document|report|publication|material|communication|presentation)\b',
    r'\bherein\b',
    r'\brecipients?\b',
    r'without (?:prior )?notice',
    r'accura(?:cy|te).{0,40}complete',
    r'\bliab(?:ility|ilities|le)\b',
    r'\bwarrant(?:y|ies)\b',
    r'\bsolicitation\b',
    r'offer to (?:buy|sell)',
    r'investment advice',
    r'(?:professional|retail) clients?',
    r'eligible counterpart',
    r'\baffiliates?\b',
    r'\bsubsidiar(?:y|ies)\b',
    r'\bdisclos(?:e|ed|ure|ures)\b',
    r'copyright',
    r'(?:reproduc|redistribut|disseminat)\w*',
    r'\bjurisdictions?\b',
    r'prior written consent',
    r'(?:independent|investment) research',
    r'investment recommendations?',
    r'\bMiFID\b',
    r'market abuse regulation',
    r'\bnot (?:intended|be construed|constitute)',
    r'(?:registered|incorporated) (?:in|with|office)',
)

_SENTENCE_END = re.compile(r'(?<=[.!?;:])\s+')
_DIGITS = re.compile(r'\d+')


class RuleClassifier:
    """Flags prose pages whose sentences are mostly disclosure and disclaimer text.

    Lines repeated on at least ``repeat_share`` of a document's pages (running headers and footers) are ignored, so a
    footer such as "see important disclosures at the end" cannot flag a page. Pages that are short or mostly numbers
    (tables, chart labels) are never flagged.
    """

    def __init__(
        self,
        patterns: Sequence[str] = DISCLOSURE_PATTERNS,
        min_share: float = 0.3,
        min_terms: int = 3,
        min_words: int = 150,
        max_numeric: float = 0.15,
        repeat_share: float = 0.5,
    ) -> None:
        self._patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        self._any = re.compile('|'.join(f'(?:{pattern})' for pattern in patterns), re.IGNORECASE)
        self.min_share = min_share
        self.min_terms = min_terms
        self.min_words = min_words
        self.max_numeric = max_numeric
        self.repeat_share = repeat_share

    def classify(self, pages: Sequence[str]) -> list[PageVerdict]:
        lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
        repeated = self._repeated(lines)
        return [
            self._verdict(number, ' '.join(line for line in page if _key(line) not in repeated))
            for number, page in enumerate(lines, start=1)
        ]

    def _repeated(self, lines: Sequence[Sequence[str]]) -> set[str]:
        if len(lines) < 3:
            return set()
        counts = Counter(key for page in lines for key in {_key(line) for line in page})
        threshold = max(2, math.ceil(len(lines) * self.repeat_share))
        return {key for key, count in counts.items() if count >= threshold}

    def _verdict(self, number: int, text: str) -> PageVerdict:
        tokens = text.split()
        if not tokens:
            return PageVerdict(page=number, boilerplate=False, score=0.0, reason='')
        numeric = sum(1 for token in tokens if any(char.isdigit() for char in token)) / len(tokens)
        if len(tokens) < self.min_words or numeric > self.max_numeric:
            return PageVerdict(
                page=number,
                boilerplate=False,
                score=0.0,
                reason=f'not prose: {len(tokens)} words, {numeric:.0%} numeric',
            )
        sentences = [sentence for sentence in _SENTENCE_END.split(' '.join(tokens)) if sentence]
        share = sum(len(sentence) for sentence in sentences if self._any.search(sentence)) / sum(map(len, sentences))
        terms = sum(1 for pattern in self._patterns if pattern.search(text))
        return PageVerdict(
            page=number,
            boilerplate=share >= self.min_share and terms >= self.min_terms,
            score=share,
            reason=f'{share:.0%} disclosure text, {terms} distinct phrases',
        )


def _key(line: str) -> str:
    """A line with whitespace collapsed and digits masked, so "Page 3" and "Page 4" repeat."""
    return _DIGITS.sub('#', ' '.join(line.split())).lower()
