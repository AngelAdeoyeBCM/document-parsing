from __future__ import annotations

import tempfile
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from docling_core.types.doc import DoclingDocument
from pypdf import PdfReader, PdfWriter

from inventory import PdfSource

from .base import Parser, ParserConfig


@dataclass(frozen=True, slots=True)
class ModelConfig(ParserConfig):
    device: str = 'cuda'  # 'cpu', 'cuda' or 'cuda:N'


class ModelParser(Parser):
    """Parser backed by ML models; skipped pages are cut from the PDF so the models never see them."""

    config: ModelConfig

    def _convert(self, source: PdfSource, skip: frozenset[int]) -> DoclingDocument:
        if not skip:
            return self._convert_pdf(source.path)
        with tempfile.TemporaryDirectory() as tmp:
            subset = Path(tmp) / source.path.name
            kept = _write_subset(source.path, skip, subset)
            if not kept:
                return DoclingDocument(name=source.doc_id)
            return _renumber(self._convert_pdf(subset), kept)

    @abstractmethod
    def _convert_pdf(self, path: Path) -> DoclingDocument:
        """Convert every page of the PDF at ``path``."""


def _renumber(doc: DoclingDocument, numbers: Sequence[int]) -> DoclingDocument:
    """Map page numbers of a subset PDF back to the source, where subset page i is source page numbers[i - 1]."""
    for items in (doc.texts, doc.tables, doc.pictures, doc.key_value_items, doc.form_items):
        for item in items:
            for prov in item.prov:
                prov.page_no = numbers[prov.page_no - 1]
    doc.pages = {
        numbers[number - 1]: page.model_copy(update={'page_no': numbers[number - 1]})
        for number, page in doc.pages.items()
    }
    return doc


def _write_subset(path: Path, skip: frozenset[int], out: Path) -> list[int]:
    """Write the pages of ``path`` not in ``skip`` to ``out`` and return their original numbers."""
    reader = PdfReader(path)
    writer = PdfWriter()
    kept = [number for number in range(1, len(reader.pages) + 1) if number not in skip]
    for number in kept:
        writer.add_page(reader.pages[number - 1])
    with open(out, 'wb') as handle:
        writer.write(handle)
    return kept
