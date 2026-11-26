from __future__ import annotations

import random
from collections import Counter
from itertools import product

import pytest

from inventory import StratifiedSampler, Stratum
from inventory.sampling import DENSITIES, IMAGERY, LENGTHS, allocate


class FakeStore:
    """The one query the sampler makes, over hand-built rows."""

    def __init__(self, rows: list[tuple[str, int, float, float, float, bool]]) -> None:
        self.rows = rows

    def sampling_rows(self, text_quality=None, do_ocr=None, image_page_coverage=0.25):
        return [row for row in self.rows if do_ocr is None or row[5] == do_ocr]


def _row(path: str, kept: int = 6, chars: float = 2500, coverage: float = 0.0, heavy: float = 0.0, ocr: bool = False):
    return (path, kept, chars, coverage, heavy, ocr)


@pytest.mark.parametrize(
    ('row', 'expected'),
    [
        (_row('a', kept=3, chars=1999, coverage=0.019), Stratum('short', 'light', 'text')),
        (_row('a', kept=4, chars=2000, coverage=0.02), Stratum('medium', 'typical', 'illustrated')),
        (_row('a', kept=11, chars=3200, coverage=0.10), Stratum('long', 'dense', 'image_heavy')),
        (_row('a', kept=31, coverage=0.0, heavy=0.25), Stratum('very_long', 'typical', 'image_heavy')),
        (_row('a', kept=2, chars=10, coverage=0.9, ocr=True), Stratum('short', 'scanned', 'scanned')),
    ],
)
def test_strata_bands(row, expected) -> None:
    [(stratum, _)] = StratifiedSampler(FakeStore([row])).strata().items()
    assert stratum == expected


def test_fully_triaged_and_unlisted_documents_are_not_candidates() -> None:
    store = FakeStore([_row('kept'), _row('all_triaged', kept=0), _row('unlisted')])
    groups = StratifiedSampler(store).strata(within={'kept', 'all_triaged'})
    assert [member.relative_path for members in groups.values() for member in members] == ['kept']


def test_equal_shares_capped_by_stratum_size() -> None:
    groups = {Stratum('short', 'light', 'text'): [0], Stratum('long', 'dense', 'text'): [0] * 9}
    groups[Stratum('medium', 'typical', 'text')] = [0] * 9
    assert sorted(allocate(groups, 7, random.Random(0)).values()) == [1, 3, 3]
    assert sorted(allocate(groups, 100, random.Random(0)).values()) == [1, 9, 9]


def test_small_samples_spread_over_every_axis() -> None:
    groups = {Stratum(*cell): [0] * 5 for cell in product(LENGTHS, DENSITIES, IMAGERY)}
    for seed in range(5):
        quotas = allocate(groups, 12, random.Random(seed))
        assert max(quotas.values()) == 1
        for axis, levels in enumerate((LENGTHS, DENSITIES, IMAGERY)):
            taken = Counter({level: 0 for level in levels})
            taken.update(stratum[axis] for stratum, quota in quotas.items() if quota)
            assert max(taken.values()) - min(taken.values()) <= 1, (seed, axis, taken)


def test_ocr_option_and_determinism() -> None:
    rows = [_row(f'text{i}', kept=1 + i) for i in range(20)] + [
        _row(f'scan{i}', kept=1 + i, ocr=True) for i in range(6)
    ]
    sampler = StratifiedSampler(FakeStore(rows))

    sample = sampler.sample(8, seed=3)
    assert len(sample) == 8 and sample == sampler.sample(8, seed=3)
    assert any(path.startswith('scan') for path in sample)
    assert not any(path.startswith('scan') for path in sampler.sample(8, do_ocr=False))
    assert all(path.startswith('scan') for path in sampler.sample(8, do_ocr=True))
    assert len(sampler.sample(100)) == len(rows)
