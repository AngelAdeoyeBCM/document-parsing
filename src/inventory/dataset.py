"""A dataset of tail-truncated PDFs drawn by the stratified sampler, in the folder layout docling-eval reads.

<root>/
  pdfs/<hash_id>.pdf    the sampled PDFs without their triaged tail, named by the original's content hash
                        (docling-eval's PlainFiles source: a flat folder it globs, so nothing else goes in it)
  documents.json        hash_id -> source path, kept pages (1..k), skipped pages, stratum, truncated file's hash
  manifest.txt          the originals' relative paths, one per line: the parse runs' input manifest
  settings.json         how the sample was drawn

The annotation workspace (``annotation.Workspace``) adds its docling-eval, CVAT and gold folders to the same root.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inventory import Inventory, file_hash
from .pdf import truncate_pdf
from .sampling import StratifiedSampler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    root: Path

    @property
    def pdfs(self) -> Path:
        return self.root / 'pdfs'

    @property
    def documents_file(self) -> Path:
        return self.root / 'documents.json'

    @property
    def manifest_file(self) -> Path:
        return self.root / 'manifest.txt'

    @property
    def settings_file(self) -> Path:
        return self.root / 'settings.json'

    def documents(self) -> dict[str, dict[str, Any]]:
        """hash_id -> {'source', 'pages', 'skipped', 'stratum', 'truncated_hash'}; empty before the dataset is built."""
        if not self.documents_file.is_file():
            return {}
        return json.loads(self.documents_file.read_text(encoding='utf-8'))

    def pages(self) -> dict[str, list[int]]:
        """hash_id -> the pages kept (the ones annotated); empty before the dataset is built."""
        return {hash_id: document['pages'] for hash_id, document in self.documents().items()}


def build_dataset(
    inventory: Inventory,
    root: Path,
    number: int,
    *,
    seed: int = 0,
    text_quality: Iterable[str] | None = ('embedded_text',),
    do_ocr: bool | None = False,
) -> Dataset:
    """Sample ``number`` documents over the strata and write each without the tail its stored triage skips.

    The inventory must be built (with triage, for truncation); the dataset folder must not hold one already.
    """
    dataset = Dataset(root)
    if dataset.documents_file.exists():
        raise FileExistsError(f'{root} already holds a dataset; use a new folder for a new sample')
    text_quality = tuple(text_quality) if text_quality is not None else None
    sampler = StratifiedSampler(inventory.store)
    chosen = sampler.sample(number, seed=seed, text_quality=text_quality, do_ocr=do_ocr)
    strata = sampler.strata(text_quality=text_quality, do_ocr=do_ocr, within=set(chosen))
    stratum_of = {member.relative_path: stratum for stratum, members in strata.items() for member in members}
    hash_of = dict(inventory.store.file_rows())  # every known file's relative path -> content hash

    dataset.pdfs.mkdir(parents=True, exist_ok=True)
    documents: dict[str, dict[str, Any]] = {}
    for relative_path in chosen:
        hash_id = hash_of[relative_path]
        skip = inventory.store.skip_pages(hash_id)
        target = dataset.pdfs / f'{hash_id}.pdf'
        documents[hash_id] = {
            'source': relative_path,
            'pages': truncate_pdf(inventory.directory / relative_path, skip, target),
            'skipped': sorted(skip),
            'stratum': stratum_of[relative_path]._asdict(),
            'truncated_hash': file_hash(target),  # docling-eval's doc_hash for this file
        }

    settings = {
        'number': number,
        'seed': seed,
        'text_quality': text_quality,
        'do_ocr': do_ocr,
        'inventory': inventory.directory.as_posix(),
    }
    dataset.settings_file.write_text(json.dumps(settings, indent=1), encoding='utf-8')
    dataset.manifest_file.write_text(''.join(f'{path}\n' for path in sorted(chosen)), encoding='utf-8')
    dataset.documents_file.write_text(json.dumps(dict(sorted(documents.items())), indent=1), encoding='utf-8')
    pages = sum(len(document['pages']) for document in documents.values())
    logger.info('dataset: %d documents, %d pages, in %s', len(documents), pages, root)
    return dataset
