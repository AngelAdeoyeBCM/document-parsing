from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageVerdict:
    page: int  # 1-based
    boilerplate: bool
    score: float  # 0 = content, 1 = boilerplate
    reason: str = ''


class PageClassifier(ABC):
    """Labels each page of a document as content or boilerplate from its text layer."""

    @property
    def needs_corpus(self) -> bool:
        """Whether observe() must see the corpus before classify() is meaningful."""
        return False

    def observe(self, pages: Sequence[str]) -> None:
        """Collect corpus statistics from one document's page texts."""

    @abstractmethod
    def classify(self, pages: Sequence[str]) -> list[PageVerdict]:
        """One verdict per page, where ``pages[i]`` is the text of page i + 1."""
