from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from .store import InventoryStore


class StratifiedSampler:
    """Deterministic sampling across (text quality, page band, OCR) strata; documents are unique by content hash."""

    def __init__(
        self,
        store: InventoryStore,
        *,
        per_stratum: int = 10,
        seed: int = 0,
        page_bands: tuple[int, ...] = (5, 15, 40),
    ) -> None:
        self.store = store
        self.per_stratum = per_stratum
        self.seed = seed
        self.page_bands = page_bands

    def sample(
        self,
        *,
        text_quality: Iterable[str] | None = None,
        do_ocr: bool | None = None,
    ) -> list[str]:
        groups: dict[tuple[str, int, bool], list[tuple[str, str]]] = defaultdict(list)
        for hash_id, relative_path, pages, quality, ocr in self.store.document_rows(text_quality, do_ocr):
            band = next((index for index, upper in enumerate(self.page_bands) if pages <= upper), len(self.page_bands))
            groups[(quality, band, bool(ocr))].append((hash_id, relative_path))

        rng = random.Random(self.seed)
        chosen: list[str] = []
        for key in sorted(groups):
            members = sorted(groups[key])  # fixed order, so the seed alone decides the draw
            chosen.extend(
                relative_path for _, relative_path in rng.sample(members, min(self.per_stratum, len(members)))
            )
        return sorted(chosen)
