from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .store import InventoryStore

LENGTHS = ('short', 'medium', 'long', 'very_long')
DENSITIES = ('light', 'typical', 'dense')
IMAGERY = ('text', 'illustrated', 'image_heavy')
SCANNED = 'scanned'  # density and imagery of OCR documents, whose text layer says nothing about either


class Stratum(NamedTuple):
    length: str  # kept (non-triaged) pages
    density: str  # text characters per kept page
    imagery: str  # drawn raster images on kept pages; vector charts are not counted


@dataclass(frozen=True, slots=True)
class Candidate:
    relative_path: str
    kept_pages: int
    chars_per_page: float
    image_coverage: float  # mean share of a kept page covered by drawn images
    image_page_share: float  # share of kept pages mostly covered by images
    do_ocr: bool


class StratifiedSampler:
    """Deterministic samples of N full PDFs, spread evenly over length, text density and imagery strata.

    Every measure is taken over the pages triage keeps. The default bands come from the corpus: kept pages 1-3, 4-10,
    11-30 and 31+; 2,000 and 3,200 characters per page (about the quartiles); and images covering 2% or 10% of the
    average page, or a quarter of the pages mostly covered, for image-heavy. Scanned documents form their own strata,
    split by length only.

    Each non-empty cell gets an equal share of N, capped by its size. When N is not a multiple of the cells, the extra
    documents go to cells whose levels are least represented so far, so each axis stays as even as the cells allow.
    """

    def __init__(
        self,
        store: InventoryStore,
        *,
        length_bands: tuple[int, int, int] = (4, 11, 31),
        density_bands: tuple[float, float] = (2000, 3200),
        imagery_bands: tuple[float, float] = (0.02, 0.10),
        image_page_coverage: float = 0.25,
        image_heavy_share: float = 0.25,
    ) -> None:
        if len(length_bands) != len(LENGTHS) - 1 or len(density_bands) != len(DENSITIES) - 1:
            raise ValueError(f'need {len(LENGTHS) - 1} length and {len(DENSITIES) - 1} density band edges')
        if len(imagery_bands) != len(IMAGERY) - 1:
            raise ValueError(f'need {len(IMAGERY) - 1} imagery band edges')
        self.store = store
        self.length_bands = length_bands
        self.density_bands = density_bands
        self.imagery_bands = imagery_bands
        self.image_page_coverage = image_page_coverage
        self.image_heavy_share = image_heavy_share

    def stratum(self, candidate: Candidate) -> Stratum:
        length = LENGTHS[_band(candidate.kept_pages, self.length_bands)]
        if candidate.do_ocr:
            return Stratum(length, SCANNED, SCANNED)
        if candidate.image_page_share >= self.image_heavy_share:
            imagery = IMAGERY[-1]
        else:
            imagery = IMAGERY[_band(candidate.image_coverage, self.imagery_bands)]
        return Stratum(length, DENSITIES[_band(candidate.chars_per_page, self.density_bands)], imagery)

    def strata(
        self,
        *,
        text_quality: Iterable[str] | None = None,
        do_ocr: bool | None = None,
        within: Collection[str] | None = None,
    ) -> dict[Stratum, list[Candidate]]:
        """Documents with at least one kept page, by stratum; ``within`` limits them to these relative paths."""
        groups: dict[Stratum, list[Candidate]] = defaultdict(list)
        for row in self.store.sampling_rows(text_quality, do_ocr, self.image_page_coverage):
            candidate = Candidate(*row)
            if candidate.kept_pages and (within is None or candidate.relative_path in within):
                groups[self.stratum(candidate)].append(candidate)
        return dict(groups)

    def sample(
        self,
        number: int,
        *,
        seed: int = 0,
        text_quality: Iterable[str] | None = None,
        do_ocr: bool | None = None,
        within: Collection[str] | None = None,
    ) -> list[str]:
        """Relative paths of ``number`` documents (all of them, if fewer qualify)."""
        if number < 0:
            raise ValueError('number must be non-negative')
        groups = self.strata(text_quality=text_quality, do_ocr=do_ocr, within=within)
        quotas = allocate(groups, number, random.Random(seed))
        chosen: list[str] = []
        for stratum in sorted(groups):
            chosen.extend(_spread(groups[stratum], quotas[stratum], random.Random(f'{seed}:{stratum}')))
        return sorted(chosen)


def allocate(
    groups: Mapping[Stratum, Collection[object]], number: int, rng: random.Random, attempts: int = 10
) -> dict[Stratum, int]:
    """How many documents each stratum gives: equal shares, capped by size, extras to the least-represented levels.

    A single greedy pass can strand an axis one level uneven, so the most even of ``attempts`` seeded passes wins.
    """
    cells = Counter(level for stratum in groups for level in enumerate(stratum))
    total = min(number, sum(map(len, groups.values())))

    def attempt() -> dict[Stratum, int]:
        quotas = dict.fromkeys(groups, 0)
        taken: Counter[tuple[int, str]] = Counter()
        order = sorted(groups)
        rng.shuffle(order)  # ties go to a seed-dependent stratum

        def need(stratum: Stratum) -> tuple[int, float]:
            return quotas[stratum], sum(taken[level] / cells[level] for level in enumerate(stratum))

        for _ in range(total):
            stratum = min((s for s in order if quotas[s] < len(groups[s])), key=need)
            quotas[stratum] += 1
            taken.update(enumerate(stratum))
        return quotas

    def unevenness(quotas: dict[Stratum, int]) -> float:
        taken: Counter[tuple[int, str]] = Counter()
        for stratum, quota in quotas.items():
            for level in enumerate(stratum):
                taken[level] += quota
        per_cell: dict[int, list[float]] = defaultdict(list)
        for (axis, level), count in cells.items():
            per_cell[axis].append(taken[axis, level] / count)
        return sum(max(loads) - min(loads) for loads in per_cell.values())

    return min((attempt() for _ in range(attempts)), key=unevenness)


def _spread(members: Collection[Candidate], quota: int, rng: random.Random) -> list[str]:
    """``quota`` members, one from each equal slice of the stratum ordered by source, length and density."""
    if quota >= len(members):
        return [member.relative_path for member in members]
    ordered = sorted(
        members,
        key=lambda m: (m.relative_path.split('/', 1)[0], m.kept_pages, m.chars_per_page, m.relative_path),
    )
    width = len(ordered) / quota
    return [rng.choice(ordered[int(index * width) : int((index + 1) * width)]).relative_path for index in range(quota)]


def _band(value: float, edges: Sequence[float]) -> int:
    """Index of the first band whose upper edge (exclusive) is above ``value``."""
    return next((index for index, edge in enumerate(edges) if value < edge), len(edges))
