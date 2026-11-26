from __future__ import annotations

import pytest
from conftest import DISCLOSURE

from triage import FlaggedPolicy, PageVerdict, RuleClassifier, TailPolicy

FOOTER = 'Please refer to important information at the end of the report and MAR disclosures'
ANALYSIS = (
    'Rates outlook: we expect the central bank to cut twice next year as inflation returns towards target.',
    'Growth has slowed across the euro area, with manufacturing weak and services holding up better than feared.',
    'Labour markets remain tight, but wage growth is easing and should allow services inflation to moderate.',
    'We see the curve steepening as front-end yields fall faster than long-end yields over the coming quarters.',
    'Fiscal policy is turning less supportive, which adds to the case for easier monetary policy into next year.',
    'Our base case keeps a soft landing, with risks skewed towards weaker growth if credit conditions tighten.',
    'Energy prices have stabilised, removing one of the main upside risks to headline inflation for now.',
    'We prefer receiving two-year rates and hold a modest steepener in the five-year to thirty-year sector.',
    'Spreads in the periphery should stay contained while the central bank keeps its backstop available.',
    'Survey data point to a gradual recovery in the second half, led by consumption as real incomes improve.',
    'Housing activity remains soft, though mortgage rates have started to come down from their peaks.',
    'Corporate investment intentions are mixed, with spending on technology offsetting weaker construction.',
)
TABLE = '\n'.join(
    ['Important disclosures: see appendix. Past performance is not indicative. All rights reserved.']
    + [f'XL{n} {n}.1% {n}.4 -0.{n} {n}.8 1{n}.2% 4{n}' for n in range(40)]
)
CHART_CAPTIONS = '\n'.join(
    f'Fig. {n}: Sector flows and positioning across regions. Source: BNP Paribas. '
    f'Past performance is not indicative of future performance.'
    for n in range(14)
)


def test_rules_flag_disclosure_pages_only() -> None:
    verdicts = RuleClassifier().classify(['\n'.join(ANALYSIS), '\n'.join(DISCLOSURE)])
    assert [verdict.boilerplate for verdict in verdicts] == [False, True]


def test_running_footer_with_disclosure_wording_does_not_flag_content() -> None:
    pages = [
        '\n'.join([FOOTER, *(f'{market}: {line}' for line in ANALYSIS)])
        + f'\n{market} past performance is not indicative of future performance.'
        for market in ('Germany', 'France', 'Italy', 'Spain')
    ]
    verdicts = RuleClassifier().classify([*pages, f'{FOOTER}\n' + '\n'.join(DISCLOSURE), FOOTER])
    assert [verdict.boilerplate for verdict in verdicts] == [False, False, False, False, True, False]


def test_tables_and_chart_captions_are_not_disclosure_prose() -> None:
    table, captions = RuleClassifier().classify([TABLE, CHART_CAPTIONS])
    assert not table.boilerplate and table.reason.startswith('not prose')
    assert not captions.boilerplate  # one caveat repeated under every chart is not a disclosure page


def _verdicts(pages: int, flagged: set[int]) -> list[PageVerdict]:
    return [PageVerdict(page, page in flagged, float(page in flagged)) for page in range(1, pages + 1)]


@pytest.mark.parametrize(
    ('flagged', 'expected'),
    [
        ({6, 9, 11}, {9, 10, 11, 12}),
        ({6, 11}, {11, 12}),
        ({9, 10, 11}, {9, 10, 11, 12}),
        ({10, 11, 12}, {10, 11, 12}),
        (set(), set()),
        ({1}, set()),  # a cover page with distribution notes does not start the tail
        ({1, 2}, set(range(2, 13))),
    ],
)
def test_tail_policy_skips_one_block_to_the_end(flagged: set[int], expected: set[int]) -> None:
    assert TailPolicy(keep_first=1).skip(_verdicts(12, flagged)) == expected


def test_tail_policy_without_gaps_needs_a_contiguous_run() -> None:
    assert TailPolicy(keep_first=1, max_gap=0).skip(_verdicts(12, {9, 11})) == {11, 12}


def test_flagged_policy_skips_each_flagged_page() -> None:
    assert FlaggedPolicy(keep_first=1).skip(_verdicts(12, {1, 6, 9})) == {6, 9}
