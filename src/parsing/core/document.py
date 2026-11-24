from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from docling_core.types.doc import DoclingDocument

from inventory import PdfSource


@dataclass(frozen=True, slots=True)
class ParserInfo:
    """Which parser and settings produced a document."""

    name: str
    version: str
    config: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """A DoclingDocument plus the run facts Docling does not track."""

    source: PdfSource
    parser: ParserInfo
    document: DoclingDocument
    skipped_pages: frozenset[int] = frozenset()  # removed before parsing, e.g. boilerplate
    seconds: float = 0.0

    def __post_init__(self) -> None:
        overlap = self.skipped_pages.intersection(self.document.pages)
        if overlap:
            raise ValueError(f'{self.source.doc_id}: pages {sorted(overlap)} are both parsed and skipped')

    @property
    def page_numbers(self) -> list[int]:
        return sorted(self.document.pages)

    @property
    def page_count(self) -> int:
        """Pages in the source PDF, parsed or skipped."""
        return len(self.document.pages) + len(self.skipped_pages)
