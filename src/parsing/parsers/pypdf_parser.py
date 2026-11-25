from __future__ import annotations

from collections.abc import Sequence
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
    """Baseline: the PDF text layer as one text item per page, with no structure.

    Reuses the inventory's stored text (plain mode) when given it, so the baseline is exactly what triage saw.
    """

    name = 'pypdf'
    package = 'pypdf'
    config_cls = PypdfConfig
    config: PypdfConfig

    def _convert(self, source: PdfSource, skip: frozenset[int], texts: Sequence[str]) -> DoclingDocument:
        doc = DoclingDocument(name=source.doc_id)
        pages = PdfReader(source.path).pages
        stored = self.config.extraction_mode == 'plain' and len(texts) == len(pages)
        for number, page in enumerate(pages, start=1):
            if number not in skip:
                text = texts[number - 1] if stored else page.extract_text(extraction_mode=self.config.extraction_mode)
                self._add_page(doc, number, page, text.strip())
        return doc

    @staticmethod
    def _add_page(doc: DoclingDocument, number: int, page: PageObject, text: str) -> None:
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        doc.add_page(page_no=number, size=Size(width=width, height=height))
        if text:
            box = BoundingBox(l=0, t=height, r=width, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)
            prov = ProvenanceItem(page_no=number, bbox=box, charspan=(0, len(text)))
            doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
