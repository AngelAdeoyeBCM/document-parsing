from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class PageProfile:
    page_number: int
    text_chars: int
    word_count: int
    image_xobjects: int = 0

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
    image_xobjects: int
    image_pages: int
    image_heavy_pages: int
    size_bytes: int

    @property
    def chars_per_page(self) -> float:
        return self.text_chars / self.pages if self.pages else 0.0

    @property
    def image_page_share(self) -> float:
        return self.image_pages / self.pages if self.pages else 0.0


class Profiler:
    """Cheap pypdf profile; returns bounded statistics and transient text for triage."""

    def __init__(self, sparse_chars: int = 50) -> None:
        self.sparse_chars = sparse_chars

    def profile(self, path: Path, relative_path: str) -> tuple[PDFProfile, tuple[PageProfile, ...], tuple[str, ...]]:
        path = Path(path)
        with path.open('rb') as handle:
            hash_id = hashlib.file_digest(handle, 'sha256').hexdigest()

        reader = PdfReader(path)
        texts = tuple(page.extract_text() or '' for page in reader.pages)
        char_counts = tuple(len(text.strip()) for text in texts)
        word_counts = tuple(len(text.split()) for text in texts)

        image_xobjects = 0
        image_pages = 0
        image_heavy_pages = 0
        page_profiles: list[PageProfile] = []
        for page, char_count, word_count, number in zip(
            reader.pages,
            char_counts,
            word_counts,
            range(1, len(reader.pages) + 1),
        ):
            count = self._image_xobject_count(page)
            page_profiles.append(PageProfile(number, char_count, word_count, count))
            if count:
                image_pages += 1
                image_xobjects += count
                if char_count < self.sparse_chars:
                    image_heavy_pages += 1

        pages = tuple(page_profiles)

        pdf_profile = PDFProfile(
            relative_path=relative_path,
            hash_id=hash_id,
            pages=len(reader.pages),
            text_chars=sum(char_counts),
            char_min=min(char_counts, default=0),
            char_max=max(char_counts, default=0),
            char_mean=statistics.fmean(char_counts) if char_counts else 0.0,
            char_median=statistics.median(char_counts) if char_counts else 0.0,
            word_count=sum(word_counts),
            word_min=min(word_counts, default=0),
            word_max=max(word_counts, default=0),
            word_mean=statistics.fmean(word_counts) if word_counts else 0.0,
            word_median=statistics.median(word_counts) if word_counts else 0.0,
            image_xobjects=image_xobjects,
            image_pages=image_pages,
            image_heavy_pages=image_heavy_pages,
            size_bytes=path.stat().st_size,
        )
        return pdf_profile, pages, texts

    @staticmethod
    def _image_xobject_count(page: object) -> int:
        try:
            resources = page.get('/Resources')
            if resources is None:
                return 0
            xobjects = resources.get('/XObject')
            if xobjects is None:
                return 0
            return sum(
                1
                for _, reference in xobjects.items()
                if reference.get_object().get('/Subtype') == '/Image'
            )
        except Exception:
            return 0
