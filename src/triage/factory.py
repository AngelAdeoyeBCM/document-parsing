from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .base import PageVerdict
from .policy import FlaggedPolicy, TailPolicy
from .rules import RuleClassifier

RULES_VERSION = 2  # 2: prose pages scored by disclosure share; bump to invalidate stored verdicts


@dataclass(frozen=True, slots=True)
class Triage:
    """Boilerplate triage settings: fixed rules classifier plus skip policy."""

    policy: str = 'tail'
    keep_first: int = 1
    min_share: float = 0.3

    def __post_init__(self) -> None:
        if self.policy not in ('tail', 'flagged'):
            raise ValueError(f"unknown triage policy {self.policy!r}; available: ['flagged', 'tail']")

    @property
    def key(self) -> str:
        return json.dumps(
            {'policy': self.policy, 'keep_first': self.keep_first, 'min_share': self.min_share, 'rules': RULES_VERSION},
            sort_keys=True,
        )

    def build_classifier(self) -> RuleClassifier:
        return RuleClassifier(min_share=self.min_share)

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
