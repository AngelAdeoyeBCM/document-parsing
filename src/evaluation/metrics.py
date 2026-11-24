from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import ClassVar

import pypdfium2 as pdfium
from docling_core.types.doc import ContentLayer, DocItem, DocItemLabel, DoclingDocument, TableItem, TextItem

from parsing.core import (
    CHART_CLASSES,
    ParsedDocument,
    pdf_text_in_picture,
    picture_class,
    picture_texts,
    texts_in_picture_box,
)

from .reference import Reference

_WORD = re.compile(r'[^\W_]+')
_NUMBER = re.compile(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?')
_DIGITS = re.compile(r'\d+')
_HEADINGS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}


def words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


def numbers(text: str) -> list[str]:
    return [number.replace(',', '') for number in _NUMBER.findall(text)]


def page_texts(doc: DoclingDocument) -> dict[int, list[str]]:
    """Body text per page that should come from the PDF; generated picture descriptions and chart data are left out."""
    pages: dict[int, list[str]] = defaultdict(list)
    for item, _ in doc.iterate_items(included_content_layers={ContentLayer.BODY}, traverse_pictures=True):
        if not isinstance(item, DocItem) or not item.prov:
            continue
        page = pages[item.prov[0].page_no]
        if isinstance(item, TableItem):
            page.extend(cell.text for cell in item.data.table_cells)
        elif isinstance(item, TextItem):
            page.append(item.text)
    return pages


def extracted_text(doc: DoclingDocument) -> str:
    return '\n'.join(text for texts in page_texts(doc).values() for text in texts)


def _recall(reference: Counter[str], output: Counter[str]) -> float:
    total = sum(reference.values())
    return sum((reference & output).values()) / total if total else math.nan


class Metric(ABC):
    name: ClassVar[str]

    @abstractmethod
    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]: ...

    @staticmethod
    def _reference_text(parsed: ParsedDocument, reference: Reference) -> str:
        return reference.text(parsed.page_numbers)


class TextCoverage(Metric):
    """Share of text-layer words that survive parsing, counted with multiplicity."""

    name = 'coverage'

    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]:
        expected = Counter(words(self._reference_text(parsed, reference)))
        return {'word_recall': _recall(expected, Counter(words(extracted_text(parsed.document))))}


class NumericFidelity(Metric):
    """Share of text-layer numbers reproduced exactly, the figures a financial reader relies on."""

    name = 'numbers'

    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]:
        expected = Counter(numbers(self._reference_text(parsed, reference)))
        return {'number_recall': _recall(expected, Counter(numbers(extracted_text(parsed.document))))}


class Hallucination(Metric):
    """Share of output words absent from the text layer; high for OCR or VLM output that invents text."""

    name = 'hallucination'

    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]:
        known = set(words(self._reference_text(parsed, reference)))
        produced = words(extracted_text(parsed.document))
        unsupported = sum(1 for word in produced if word not in known)
        return {'unsupported_word_rate': unsupported / len(produced) if produced else math.nan}


class FurnitureLeakage(Metric):
    """Share of pages whose body still carries a line repeated on most pages, i.e. a running header or footer."""

    name = 'leakage'

    def __init__(self, min_share: float = 0.5, min_pages: int = 3) -> None:
        self.min_share = min_share
        self.min_pages = min_pages

    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]:
        texts = page_texts(parsed.document)
        pages = [self._lines(texts.get(number, [])) for number in parsed.page_numbers]
        if len(pages) < self.min_pages:
            return {'repeated_line_page_rate': math.nan}
        counts = Counter(line for lines in pages for line in lines)
        threshold = max(self.min_pages, self.min_share * len(pages))
        leaked = sum(1 for lines in pages if any(counts[line] >= threshold for line in lines))
        return {'repeated_line_page_rate': leaked / len(pages)}

    @staticmethod
    def _lines(texts: list[str]) -> set[str]:
        return {_DIGITS.sub('0', line.strip().lower()) for text in texts for line in text.splitlines() if line.strip()}


class Structure(Metric):
    """Docling structure found per parsed page."""

    name = 'structure'

    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]:
        doc = parsed.document
        pages = max(1, len(doc.pages))
        return {
            'headings_per_page': sum(1 for item in doc.texts if item.label in _HEADINGS) / pages,
            'tables_per_page': len(doc.tables) / pages,
            'pictures_per_page': len(doc.pictures) / pages,
            'charts_per_page': sum(1 for p in doc.pictures if (top := picture_class(p)) and top[0] in CHART_CLASSES)
            / pages,
            'furniture_per_page': sum(1 for item in doc.texts if item.content_layer == ContentLayer.FURNITURE) / pages,
        }


class PictureTextRetention(Metric):
    """PDF words inside picture boxes (axes, legends) kept nested in the picture (inner) or anywhere in the box (region)."""

    name = 'pictures'

    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]:
        doc = parsed.document
        pictures = [p for p in doc.pictures if p.prov]
        nested = picture_texts(doc)
        inner = Counter(words(' '.join(t.text for p in pictures for t in nested[p.self_ref])))
        region = Counter(words(' '.join(t.text for p in pictures for t in texts_in_picture_box(doc, p))))
        pdf = pdfium.PdfDocument(parsed.source.path)
        try:
            expected = Counter(words(' '.join(pdf_text_in_picture(pdf, doc, p) for p in pictures)))
        finally:
            pdf.close()
        return {
            'inner_text_recall': _recall(expected, inner),
            'region_text_recall': _recall(expected, region),
            'inner_pdf_words': float(sum(expected.values())),
        }


class Throughput(Metric):
    name = 'speed'

    def score(self, parsed: ParsedDocument, reference: Reference) -> dict[str, float]:
        return {'seconds_per_page': parsed.seconds / max(1, len(parsed.document.pages))}


DEFAULT_METRICS: tuple[Metric, ...] = (
    TextCoverage(),
    NumericFidelity(),
    Hallucination(),
    FurnitureLeakage(),
    PictureTextRetention(),
    Structure(),
    Throughput(),
)
