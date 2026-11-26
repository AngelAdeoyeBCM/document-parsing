"""The annotation workspace: a dataset folder (``pipeline dataset``) plus every annotation stage built on it.

<root>/
  pdfs/<hash_id>.pdf        the dataset: sampled PDFs without their triaged tail, named by the original's hash
  documents.json            the dataset: per hash_id, source path, kept pages, stratum and truncated file's hash
  manifest.txt              the dataset: the originals' relative paths, the parse runs' input manifest
  docling_eval/gt_dataset/  docling-eval parquet of the PDFs (create-gt --benchmark PlainFiles)
  docling_eval/eval_dataset/  the same with Docling's predictions, the pre-annotations (create-eval)
  cvat/                     CVAT project files (create-cvat); exports go in cvat/cvat_annotations/zips/
  gold/gt_dataset/          the ground truth from the corrected annotations (create-gt --benchmark CVAT)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inventory import Dataset


@dataclass(frozen=True)
class Workspace(Dataset):
    @property
    def docling_eval(self) -> Path:
        return self.root / 'docling_eval'

    @property
    def cvat(self) -> Path:
        return self.root / 'cvat'

    @property
    def zips(self) -> Path:
        return self.cvat / 'cvat_annotations' / 'zips'

    @property
    def gold(self) -> Path:
        return self.root / 'gold'
