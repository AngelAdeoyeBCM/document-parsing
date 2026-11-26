from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, ProvenanceItem, Size
from pypdf import PageObject, PdfReader

from .base import Parser, ParserConfig


@dataclass(frozen=True, slots=True)
class PypdfConfig(ParserConfig):
    extraction_mode: Literal['plain', 'layout'] = 'plain'


class PypdfParser(Parser):
    """Baseline: the PDF text layer as one text item per page, with no structure."""

    name = 'pypdf'
    package = 'pypdf'
    config_cls = PypdfConfig
    config: PypdfConfig

    def _convert(self, path: Path) -> DoclingDocument:
        doc = DoclingDocument(name=path.stem)  # named after the file, as Docling names its documents
        for number, page in enumerate(PdfReader(path).pages, start=1):
            self._add_page(doc, number, page, page.extract_text(extraction_mode=self.config.extraction_mode).strip())
        return doc

    @staticmethod
    def _add_page(doc: DoclingDocument, number: int, page: PageObject, text: str) -> None:
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        doc.add_page(page_no=number, size=Size(width=width, height=height))
        if text:
            box = BoundingBox(l=0, t=height, r=width, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
            prov = ProvenanceItem(page_no=number, bbox=box, charspan=(0, len(text)))
            doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
