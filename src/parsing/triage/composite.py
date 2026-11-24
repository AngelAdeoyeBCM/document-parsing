from __future__ import annotations

from collections.abc import Sequence

from .base import PageClassifier, PageVerdict


class CompositeClassifier(PageClassifier):
    """Boilerplate when any member says so; the score is the highest member score."""

    def __init__(self, classifiers: Sequence[PageClassifier]) -> None:
        if not classifiers:
            raise ValueError('CompositeClassifier needs at least one classifier')
        self.classifiers = tuple(classifiers)

    @property
    def needs_corpus(self) -> bool:
        return any(classifier.needs_corpus for classifier in self.classifiers)

    def observe(self, pages: Sequence[str]) -> None:
        for classifier in self.classifiers:
            classifier.observe(pages)

    def classify(self, pages: Sequence[str]) -> list[PageVerdict]:
        columns = zip(*(classifier.classify(pages) for classifier in self.classifiers))
        return [self._combine(verdicts) for verdicts in columns]

    @staticmethod
    def _combine(verdicts: Sequence[PageVerdict]) -> PageVerdict:
        return PageVerdict(
            page=verdicts[0].page,
            boilerplate=any(verdict.boilerplate for verdict in verdicts),
            score=max(verdict.score for verdict in verdicts),
            reason='; '.join(verdict.reason for verdict in verdicts if verdict.reason),
        )
