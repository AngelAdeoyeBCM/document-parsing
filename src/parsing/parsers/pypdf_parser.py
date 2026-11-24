from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, ProvenanceItem, Size
from pypdf import PageObject, PdfReader
from inventory import PdfSource

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

    def _convert(self, source: PdfSource, skip: frozenset[int]) -> DoclingDocument:
        doc = DoclingDocument(name=source.doc_id)
        for number, page in enumerate(PdfReader(source.path).pages, start=1):
            if number not in skip:
                self._add_page(doc, number, page)
        return doc

    def _add_page(self, doc: DoclingDocument, number: int, page: PageObject) -> None:
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        doc.add_page(page_no=number, size=Size(width=width, height=height))
        text = page.extract_text(extraction_mode=self.config.extraction_mode).strip()
        if text:
            box = BoundingBox(l=0, t=height, r=width, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
            prov = ProvenanceItem(page_no=number, bbox=box, charspan=(0, len(text)))
            doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
