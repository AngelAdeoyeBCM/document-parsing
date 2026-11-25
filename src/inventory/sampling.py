from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from .store import InventoryStore


class StratifiedSampler:
    """Deterministic, balanced sampling of full PDFs across quality, OCR and kept-page length strata."""

    def __init__(
        self,
        store: InventoryStore,
        *,
        number: int | None = None,
        per_stratum: int | None = None,
        seed: int = 0,
        page_bands: tuple[int, ...] = (5, 15, 40),
    ) -> None:
        if number is not None and number < 0:
            raise ValueError('number must be non-negative')
        if per_stratum is not None and per_stratum < 0:
            raise ValueError('per_stratum must be non-negative')
        self.store = store
        self.number = number
        self.per_stratum = per_stratum if per_stratum is not None else 10
        self.seed = seed
        self.page_bands = page_bands

    def sample(
        self,
        *,
        text_quality: Iterable[str] | None = None,
        do_ocr: bool | None = None,
    ) -> list[str]:
        groups: dict[tuple[str, int, bool], list[tuple[str, str, int]]] = defaultdict(list)
        for hash_id, relative_path, pages, triaged, quality, ocr in self.store.sampling_rows(text_quality, do_ocr):
            kept_pages = max(0, pages - (triaged or 0))
            band = next(
                (index for index, upper in enumerate(self.page_bands) if kept_pages <= upper),
                len(self.page_bands),
            )
            groups[(quality, band, bool(ocr))].append((hash_id, relative_path, kept_pages))

        if not groups:
            return []

        rng = random.Random(self.seed)
        chosen: set[str] = set()

        if self.number is not None:
            remaining = self.number
            quotas = {key: 0 for key in sorted(groups)}
            keys = sorted(groups)
            while remaining:
                advanced = False
                for key in keys:
                    if remaining == 0:
                        break
                    if quotas[key] < len(groups[key]):
                        quotas[key] += 1
                        remaining -= 1
                        advanced = True
                if not advanced:
                    break
        else:
            quotas = {
                key: min(self.per_stratum, len(members))
                for key, members in groups.items()
            }

        for key in sorted(groups):
            quota = quotas[key]
            if not quota:
                continue
            members = sorted(groups[key], key=lambda item: (item[2], item[1], item[0]))
            stratum_rng = random.Random(f'{self.seed}:{key}')

            if len(members) <= quota:
                chosen.update(relative_path for _, relative_path, _ in members)
                continue

            if quota == 1:
                chosen.add(stratum_rng.choice(members)[1])
                continue

            # Split by kept-page length and choose within each equal-width segment.
            # This covers short, medium and long documents rather than random cluster-heavy selection.
            width = len(members) / quota
            for index in range(quota):
                start = int(index * width)
                end = max(start + 1, int((index + 1) * width))
                segment = members[start:min(end, len(members))]
                chosen.add(stratum_rng.choice(segment)[1])

        return sorted(chosen)
