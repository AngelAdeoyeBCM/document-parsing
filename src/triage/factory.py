from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .base import PageVerdict
from .policy import FlaggedPolicy, TailPolicy
from .rules import RuleClassifier


@dataclass(frozen=True, slots=True)
class Triage:
    """Boilerplate triage settings: fixed rules classifier plus skip policy."""

    policy: str = 'tail'
    keep_first: int = 1
    min_hits: int = 3

    def __post_init__(self) -> None:
        if self.policy not in ('tail', 'flagged'):
            raise ValueError(f"unknown triage policy {self.policy!r}; available: ['flagged', 'tail']")

    @property
    def key(self) -> str:
        return json.dumps(
            {'policy': self.policy, 'keep_first': self.keep_first, 'min_hits': self.min_hits},
            sort_keys=True,
        )

    def build_classifier(self) -> RuleClassifier:
        return RuleClassifier(min_hits=self.min_hits)

    def build_policy(self) -> FlaggedPolicy | TailPolicy:
        policy = FlaggedPolicy if self.policy == 'flagged' else TailPolicy
        return policy(keep_first=self.keep_first)


def run_triage(
    classifier: RuleClassifier,
    policy: FlaggedPolicy | TailPolicy,
    pages: Sequence[str],
) -> tuple[tuple[PageVerdict, ...], frozenset[int]]:
    verdicts = tuple(classifier.classify(pages))
    return verdicts, policy.skip(verdicts)
