from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageVerdict:
    page: int  # 1-based
    boilerplate: bool
    score: float  # 0 = content, 1 = boilerplate
    reason: str = ''
