from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .base import PageClassifier, PageVerdict
from .composite import CompositeClassifier
from .policy import FlaggedPolicy, SkipPolicy, TailPolicy
from .repetition import RepetitionClassifier
from .rules import RuleClassifier

CLASSIFIERS: Mapping[str, type[PageClassifier]] = {'rules': RuleClassifier, 'repetition': RepetitionClassifier}
POLICIES: Mapping[str, type[SkipPolicy]] = {'tail': TailPolicy, 'flagged': FlaggedPolicy}


@dataclass(frozen=True, slots=True)
class Triage:
    """Which classifiers and skip policy to run; builds fresh, stateless-at-start instances on demand.

    ``key`` identifies these settings, so verdicts stored under one key are never reused for another.
    """

    classifiers: tuple[tuple[str, Mapping[str, Any]], ...]
    policy: str = 'tail'
    keep_first: int = 1

    def __post_init__(self) -> None:
        if not self.classifiers:
            raise ValueError('Triage needs at least one classifier')
        for name, _ in self.classifiers:
            _lookup(CLASSIFIERS, name)
        _lookup(POLICIES, self.policy)

    @classmethod
    def of(
        cls, classifiers: Sequence[str | tuple[str, Mapping[str, Any]]], policy: str = 'tail', keep_first: int = 1
    ) -> Triage:
        specs = tuple((item, {}) if isinstance(item, str) else (item[0], dict(item[1])) for item in classifiers)
        return cls(specs, policy, keep_first)

    @property
    def key(self) -> str:
        return json.dumps(
            {
                'classifiers': [[name, dict(options)] for name, options in self.classifiers],
                'policy': self.policy,
                'keep_first': self.keep_first,
            },
            sort_keys=True,
        )

    def build_classifier(self) -> PageClassifier:
        classifiers = [_lookup(CLASSIFIERS, name)(**options) for name, options in self.classifiers]
        return classifiers[0] if len(classifiers) == 1 else CompositeClassifier(classifiers)

    def build_policy(self) -> SkipPolicy:
        return _lookup(POLICIES, self.policy)(keep_first=self.keep_first)


def run_triage(
    classifier: PageClassifier, policy: SkipPolicy, pages: Sequence[str]
) -> tuple[tuple[PageVerdict, ...], frozenset[int]]:
    verdicts = tuple(classifier.classify(pages))
    return verdicts, policy.skip(verdicts)


def _lookup(table: Mapping[str, Any], name: str) -> Any:
    try:
        return table[name]
    except KeyError:
        raise KeyError(f'unknown component {name!r}; available: {sorted(table)}') from None
