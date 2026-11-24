from __future__ import annotations

import pytest

from parsing.triage import (
    CompositeClassifier,
    FlaggedPolicy,
    PageVerdict,
    RepetitionClassifier,
    RuleClassifier,
    TailPolicy,
)

DISCLOSURE = (
    'Important disclosures. Analyst certification: the views expressed accurately reflect. '
    'This report has been prepared by the bank. Past performance is not indicative.'
)
DISCLAIMER_LINE = 'This material is for professional investors only and is not an offer to buy'


def test_rules_flag_disclosure_pages_only() -> None:
    verdicts = RuleClassifier().classify(['Rates outlook: we expect two cuts in 2025.', DISCLOSURE])
    assert [verdict.boilerplate for verdict in verdicts] == [False, True]


def test_repetition_flags_lines_shared_across_documents() -> None:
    classifier = RepetitionClassifier(min_documents=3, min_share=0.6)
    assert classifier.needs_corpus
    for index in range(3):
        classifier.observe([f'Unique analysis number {index} about the euro area outlook', DISCLAIMER_LINE])
    verdicts = classifier.classify(['Fresh content that nobody else wrote about copper', DISCLAIMER_LINE])
    assert [verdict.boilerplate for verdict in verdicts] == [False, True]


def test_composite_takes_any_member() -> None:
    composite = CompositeClassifier([RuleClassifier(), RepetitionClassifier(min_documents=1)])
    composite.observe(['shared line that appears in every single document'])
    verdicts = composite.classify([DISCLOSURE, 'shared line that appears in every single document', 'novel text here'])
    assert [verdict.boilerplate for verdict in verdicts] == [True, True, False]


def _verdicts(pages: int, flagged: set[int]) -> list[PageVerdict]:
    return [PageVerdict(page, page in flagged, float(page in flagged)) for page in range(1, pages + 1)]


@pytest.mark.parametrize(
    ('flagged', 'expected'),
    [
        ({6, 9, 11}, {11, 12}),
        ({9, 10, 11}, {9, 10, 11, 12}),
        ({10, 11, 12}, {10, 11, 12}),
        (set(), set()),
        ({1, 2}, set(range(2, 13))),
    ],
)
def test_tail_policy_skips_one_block_to_the_end(flagged: set[int], expected: set[int]) -> None:
    assert TailPolicy(keep_first=1).skip(_verdicts(12, flagged)) == expected


def test_flagged_policy_skips_each_flagged_page() -> None:
    assert FlaggedPolicy(keep_first=1).skip(_verdicts(12, {1, 6, 9})) == {6, 9}
