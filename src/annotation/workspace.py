"""The annotation workspace: one folder holding every stage from the sampled PDFs to the ground truth.

<root>/
  sources/<doc_id>.pdf      sampled PDFs with their triaged pages removed
  documents.json            per doc_id: source path, content hashes, and the source page of each kept page
  dataset/gt_dataset/       docling-eval parquet of the truncated PDFs (create-gt --benchmark PlainFiles)
  dataset/eval_dataset/     the same with Docling's predictions, the pre-annotations (create-eval)
  cvat/                     CVAT project files (create-cvat); exports go in cvat/cvat_annotations/zips/
  gold/gt_dataset/          the ground truth from the corrected annotations (create-gt --benchmark CVAT)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    @property
    def sources(self) -> Path:
        return self.root / 'sources'

    @property
    def documents_file(self) -> Path:
        return self.root / 'documents.json'

    @property
    def dataset(self) -> Path:
        return self.root / 'dataset'

    @property
    def cvat(self) -> Path:
        return self.root / 'cvat'

    @property
    def zips(self) -> Path:
        return self.cvat / 'cvat_annotations' / 'zips'

    @property
    def gold(self) -> Path:
        return self.root / 'gold'

    def documents(self) -> dict[str, dict[str, Any]]:
        """doc_id -> {'source', 'hash_id', 'subset_hash', 'pages'}; ``pages[i]`` is the source page of page i + 1."""
        if not self.documents_file.is_file():
            return {}
        return json.loads(self.documents_file.read_text(encoding='utf-8'))
