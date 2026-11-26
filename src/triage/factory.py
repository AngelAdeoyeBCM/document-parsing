from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .base import PageVerdict
from .policy import TailPolicy
from .rules import RuleClassifier

RULES_VERSION = 2  # 2: prose pages scored by disclosure share; bump to invalidate stored verdicts


@dataclass(frozen=True, slots=True)
class Triage:
    """Boilerplate triage settings: fixed rules classifier plus the tail skip policy."""

    keep_first: int = 1
    min_share: float = 0.3

    @property
    def key(self) -> str:
        return json.dumps(
            {'keep_first': self.keep_first, 'min_share': self.min_share, 'rules': RULES_VERSION},
            sort_keys=True,
        )

    def build_classifier(self) -> RuleClassifier:
        return RuleClassifier(min_share=self.min_share)

    def build_policy(self) -> TailPolicy:
        return TailPolicy(keep_first=self.keep_first)


def run_triage(
    classifier: RuleClassifier,
    policy: TailPolicy,
    pages: Sequence[str],
) -> tuple[tuple[PageVerdict, ...], frozenset[int]]:
    verdicts = tuple(classifier.classify(pages))
    return verdicts, policy.skip(verdicts)
