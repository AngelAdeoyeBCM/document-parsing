from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader, apply_configuration
from pypdf.generic import DictionaryObject

# Bump whenever profiling output changes, so stored profiles are recomputed rather than reused.
PROFILER_VERSION = 2

_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class PageProfile:
    page_number: int  # 1-based
    text_chars: int
    word_count: int
    image_count: int = 0  # images actually drawn on the page, inline or XObject, including inside forms
    image_coverage: float = 0.0  # summed drawn image area over the page area, capped at 1
    sparse: bool = False  # under the profiler's ``sparse_chars``
    needs_ocr: bool = False  # sparse text over an image covering at least ``scan_coverage``: a scanned page

    @property
    def blank(self) -> bool:
        return self.text_chars == 0


@dataclass(frozen=True, slots=True)
class PDFProfile:
    relative_path: str
    hash_id: str
    pages: int
    text_chars: int
    char_min: int
    char_max: int
    char_mean: float
    char_median: float
    word_count: int
    word_min: int
    word_max: int
    word_mean: float
    word_median: float
    sparse_pages: int
    blank_pages: int
    image_count: int
    image_pages: int
    scanned_pages: int
    max_image_coverage: float
    size_bytes: int
    text_quality: str  # 'embedded_text', 'mixed' (some sparse pages) or 'image_only' (every page sparse)
    do_ocr: bool  # at least ``ocr_share`` of the pages are scanned

    @property
    def chars_per_page(self) -> float:
        return self.text_chars / self.pages if self.pages else 0.0

    @property
    def image_page_share(self) -> float:
        return self.image_pages / self.pages if self.pages else 0.0


@dataclass(frozen=True, slots=True)
class ProfileResult:
    profile: PDFProfile
    pages: tuple[PageProfile, ...]
    texts: tuple[str, ...]  # pypdf text layer, ``texts[i]`` is page i + 1


class Profiler:
    """One pypdf pass per PDF: the text layer, its statistics, and how much of each page is drawn as images."""

    def __init__(
        self,
        sparse_chars: int = 50,
        scan_coverage: float = 0.5,
        ocr_share: float = 0.5,
        *,
        zlib_maximum_output_length: int = 300_000_000,
        array_based_stream_maximum_output_length: int = 300_000_000,
        image_maximum_buffer_size: int = 300_000_000,
        stream_decoding_work_maximum_length: int = 1_000_000_000,
        stream_parsing_work_maximum_count: int = 20_000_000,
        maximum_declared_stream_length: int = 300_000_000,
    ) -> None:
        self.sparse_chars = sparse_chars
        self.scan_coverage = scan_coverage
        self.ocr_share = ocr_share
        self.pdf_configuration = {
            'zlib_maximum_output_length': zlib_maximum_output_length,
            'array_based_stream_maximum_output_length': array_based_stream_maximum_output_length,
            'image_maximum_buffer_size': image_maximum_buffer_size,
            'stream_decoding_work_maximum_length': stream_decoding_work_maximum_length,
            'stream_parsing_work_maximum_count': stream_parsing_work_maximum_count,
            'maximum_declared_stream_length': maximum_declared_stream_length,
        }

    def profile(self, path: Path, relative_path: str, hash_id: str) -> ProfileResult:
        path = Path(path)
        with apply_configuration(**self.pdf_configuration):
            return self._profile(path, relative_path, hash_id)

    def _profile(self, path: Path, relative_path: str, hash_id: str) -> ProfileResult:
        path = Path(path)
        reader = PdfReader(path)
        texts: list[str] = []
        pages: list[PageProfile] = []
        for number, page in enumerate(reader.pages, start=1):
            images = _ImageCollector(page)
            text = page.extract_text(visitor_operand_before=images.before, visitor_operand_after=images.after) or ''
            chars, words = len(text.strip()), len(text.split())
            coverage = images.coverage(_page_area(page))
            sparse = chars < self.sparse_chars
            texts.append(text)
            pages.append(
                PageProfile(
                    page_number=number,
                    text_chars=chars,
                    word_count=words,
                    image_count=images.count,
                    image_coverage=coverage,
                    sparse=sparse,
                    needs_ocr=sparse and coverage >= self.scan_coverage,
                )
            )
        return ProfileResult(
            self._summarise(pages, relative_path, hash_id, path.stat().st_size), tuple(pages), tuple(texts)
        )

    def _summarise(self, pages: Sequence[PageProfile], relative_path: str, hash_id: str, size: int) -> PDFProfile:
        chars = [page.text_chars for page in pages]
        words = [page.word_count for page in pages]
        sparse = sum(page.sparse for page in pages)
        scanned = sum(page.needs_ocr for page in pages)
        if pages and sparse == len(pages):
            text_quality = 'image_only'
        elif sparse:
            text_quality = 'mixed'
        else:
            text_quality = 'embedded_text'
        return PDFProfile(
            relative_path=relative_path,
            hash_id=hash_id,
            pages=len(pages),
            text_chars=sum(chars),
            char_min=min(chars, default=0),
            char_max=max(chars, default=0),
            char_mean=statistics.fmean(chars) if chars else 0.0,
            char_median=statistics.median(chars) if chars else 0.0,
            word_count=sum(words),
            word_min=min(words, default=0),
            word_max=max(words, default=0),
            word_mean=statistics.fmean(words) if words else 0.0,
            word_median=statistics.median(words) if words else 0.0,
            sparse_pages=sparse,
            blank_pages=sum(page.blank for page in pages),
            image_count=sum(page.image_count for page in pages),
            image_pages=sum(page.image_count > 0 for page in pages),
            scanned_pages=scanned,
            max_image_coverage=max((page.image_coverage for page in pages), default=0.0),
            size_bytes=size,
            text_quality=text_quality,
            do_ocr=bool(pages) and scanned / len(pages) >= self.ocr_share,
        )


class _ImageCollector:
    """Measures drawn images from the operators pypdf's text extraction already walks, so the PDF is read once.

    An image fills the unit square mapped by the current transformation matrix, so its area on the page is
    ``|det(cm)|``. pypdf restarts the matrix at identity inside a form XObject, so the scale of each enclosing
    form (the matrix at its ``Do`` times its ``/Matrix``) is kept on a stack, along with the resources that names
    inside it resolve against.
    """

    def __init__(self, page: Any) -> None:
        self.count = 0
        self.area = 0.0
        self._failed = False
        self._frames: list[tuple[float, Any]] = [(1.0, _resources(page))]
        self._forms: list[bool] = []  # one entry per ``Do`` in progress: whether it opened a frame

    def before(self, operator: bytes, operands: Any, cm: Sequence[float], tm: Any) -> None:
        if operator == b'INLINE IMAGE':
            self._draw(cm)
        elif operator == b'Do':
            opened = False
            try:
                scale, resources = self._frames[-1]
                xobject = _xobject(resources, operands[0]) if operands else None
                subtype = xobject.get('/Subtype') if xobject is not None else None
                if subtype == '/Image':
                    self._draw(cm)
                elif subtype == '/Form':
                    matrix = [float(value) for value in xobject.get('/Matrix', _IDENTITY)]
                    inner = _resources(xobject) or resources
                    self._frames.append((scale * abs(_det(cm)) * abs(_det(matrix)), inner))
                    opened = True
            except Exception:
                self._failed = True
            self._forms.append(opened)

    def after(self, operator: bytes, operands: Any, cm: Any, tm: Any) -> None:
        if operator == b'Do' and self._forms and self._forms.pop():
            self._frames.pop()

    def coverage(self, page_area: float) -> float:
        if self._failed or page_area <= 0:
            return 0.0
        return min(1.0, self.area / page_area)

    def _draw(self, cm: Sequence[float]) -> None:
        self.count += 1
        try:
            self.area += abs(_det(cm)) * self._frames[-1][0]
        except Exception:
            self._failed = True


def _det(matrix: Sequence[float]) -> float:
    a, b, c, d = (float(value) for value in matrix[:4])
    return a * d - b * c


def _resources(obj: Any) -> DictionaryObject | None:
    getter = getattr(obj, 'get_inherited', None)
    resources = getter('/Resources', None) if getter else obj.get('/Resources')
    resources = resources.get_object() if resources is not None else None
    return resources if isinstance(resources, DictionaryObject) else None


def _xobject(resources: DictionaryObject | None, name: str) -> DictionaryObject | None:
    if resources is None or '/XObject' not in resources:
        return None
    xobjects = resources['/XObject'].get_object()
    if not isinstance(xobjects, DictionaryObject) or name not in xobjects:
        return None
    xobject = xobjects[name].get_object()
    return xobject if isinstance(xobject, DictionaryObject) else None


def _page_area(page: Any) -> float:
    box = page.cropbox
    return abs(float(box.width) * float(box.height))
