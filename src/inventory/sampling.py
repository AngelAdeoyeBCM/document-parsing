from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


class StratifiedSampler:
    """Deterministic hash-deduplicated sampling across document text/OCR strata."""

    def __init__(
        self,
        database: Path,
        *,
        per_stratum: int = 10,
        seed: int = 0,
        page_bands: tuple[int, ...] = (5, 15, 40),
    ) -> None:
        self.database = Path(database)
        self.per_stratum = per_stratum
        self.seed = seed
        self.page_bands = page_bands

    def sample(
        self,
        *,
        text_quality: Iterable[str] | None = None,
        do_ocr: bool | None = None,
    ) -> list[str]:
        clauses: list[str] = []
        params: list[object] = []

        if text_quality is not None:
            values = tuple(text_quality)
            if values:
                clauses.append(f'text_quality IN ({", ".join("?" for _ in values)})')
                params.extend(values)
        if do_ocr is not None:
            clauses.append('do_ocr=?')
            params.append(int(do_ocr))

        query = 'SELECT hash_id, relative_path, pages, text_quality, do_ocr FROM documents'
        if clauses:
            query += ' WHERE ' + ' AND '.join(clauses)

        groups: dict[tuple[str, int, bool], list[tuple[str, str]]] = defaultdict(list)
        seen: set[str] = set()

        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(query, params).fetchall()

        for hash_id, relative_path, pages, quality, ocr in rows:
            if hash_id in seen:
                continue
            seen.add(hash_id)
            band = next((index for index, upper in enumerate(self.page_bands) if pages <= upper), len(self.page_bands))
            groups[(quality, band, bool(ocr))].append((hash_id, relative_path))

        rng = random.Random(self.seed)
        chosen: list[str] = []
        for key in sorted(groups):
            members = groups[key]
            chosen.extend(relative_path for _, relative_path in rng.sample(members, min(self.per_stratum, len(members))))
        return sorted(chosen)
