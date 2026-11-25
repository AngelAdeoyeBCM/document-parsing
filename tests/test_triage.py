from __future__ import annotations

import pytest
from triage import FlaggedPolicy, PageVerdict, RuleClassifier, TailPolicy

DISCLOSURE = (
    'Important disclosures. Analyst certification: the views expressed accurately reflect. '
    'This report has been prepared by the bank. Past performance is not indicative.'
)


def test_rules_flag_disclosure_pages_only() -> None:
    verdicts = RuleClassifier().classify(['Rates outlook: we expect two cuts in 2025.', DISCLOSURE])
    assert [verdict.boilerplate for verdict in verdicts] == [False, True]


def test_rules_use_flexible_disclaimer_phrasing() -> None:
    stored_style_page = (
        'Disclaimers and Important Risk Information. '
        'This communication is provided only to professional clients or eligible counterparties. '
        'The information is of a general nature only and has not been reviewed by any regulatory authority. '
        'State Street entities are authorized and regulated by the relevant authorities. '
        'All rights reserved.'
    )
    verdicts = RuleClassifier().classify([stored_style_page, 'The euro area recovery continues.'])
    assert [verdict.boilerplate for verdict in verdicts] == [True, False]


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
