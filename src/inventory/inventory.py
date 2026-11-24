from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .profiler import PageProfile, PDFProfile, Profiler
from .store import InventoryStore


@dataclass(frozen=True, slots=True)
class PdfSource:
    """One inventory-selected PDF ready for parsing."""

    path: Path
    hash_id: str
    metadata: dict[str, str]

    @classmethod
    def from_path(cls, path: Path | str, **metadata: str) -> 'PdfSource':
        path = Path(path)
        with path.open('rb') as handle:
            hash_id = hashlib.file_digest(handle, 'sha256').hexdigest()
        return cls(path=path, hash_id=hash_id, metadata=dict(metadata))

    @property
    def doc_id(self) -> str:
        return self.path.stem

    @property
    def relative_path(self) -> str:
        return self.metadata.get('relative_path', self.path.name)


@dataclass(frozen=True, slots=True)
class DocumentProfile:
    profile: PDFProfile
    text_quality: str
    do_ocr: bool
    sparse_pages: int
    blank_pages: int
    pages_profile: tuple[PageProfile, ...]


class Inventory:
    """Discover, cheaply profile, persist, and select PDFs from a directory."""

    def __init__(
        self,
        directory: Path | str,
        database: Path | str,
        *,
        sparse_chars: int = 50,
    ) -> None:
        self.directory = Path(directory)
        self.store = InventoryStore(database)
        self.profiler = Profiler(sparse_chars=sparse_chars)
        self.sparse_chars = sparse_chars

    def pdf_paths(self, manifest: Path | None = None) -> list[Path]:
        if manifest is not None:
            return [self.directory / line.strip() for line in manifest.read_text().splitlines() if line.strip()] 
        return sorted(path for path in self.directory.rglob('*') if path.suffix.lower() == '.pdf')

    def build(self, manifest: Path | None = None) -> list[DocumentProfile]:
        self.store.create_schema()
        profiles: list[DocumentProfile] = []

        for path in self.pdf_paths(manifest):
            try:
                relative_path = path.relative_to(self.directory).as_posix()
                pdf_profile, page_profiles, _page_texts = self.profiler.profile(path, relative_path)
            except Exception:
                continue

            blank = sum(page.blank for page in page_profiles)
            sparse = sum(page.text_chars < self.sparse_chars for page in page_profiles)
            text_share = (pdf_profile.pages - sparse) / pdf_profile.pages if pdf_profile.pages else 0.0

            if pdf_profile.pages > 0 and blank == pdf_profile.pages:
                text_quality = 'image_only'
                do_ocr = True
            elif text_share < 0.5:
                text_quality = 'mixed'
                do_ocr = True
            elif sparse:
                text_quality = 'mixed'
                do_ocr = False
            else:
                text_quality = 'embedded_text'
                do_ocr = False

            document = DocumentProfile(pdf_profile, text_quality, do_ocr, sparse, blank, page_profiles)
            profiles.append(document)
            self._upsert(document)

        return profiles

    def _upsert(self, document: DocumentProfile) -> None:
        profile = document.profile
        with self.store.connect() as connection:
            connection.execute(
                '''INSERT INTO documents (
                    hash_id, relative_path, pages, text_chars, char_min, char_max,
                    char_mean, char_median, word_count, word_min, word_max, word_mean,
                    word_median, chars_per_page, size_bytes, text_quality, do_ocr,
                    sparse_pages, blank_pages, image_xobjects, image_pages,
                    image_heavy_pages, image_page_share, n_triage_pages,
                    tail_triage_start, triage_policy, keep_first
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'tail', 1)
                ON CONFLICT(hash_id) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    pages=excluded.pages,
                    text_chars=excluded.text_chars,
                    char_min=excluded.char_min,
                    char_max=excluded.char_max,
                    char_mean=excluded.char_mean,
                    char_median=excluded.char_median,
                    word_count=excluded.word_count,
                    word_min=excluded.word_min,
                    word_max=excluded.word_max,
                    word_mean=excluded.word_mean,
                    word_median=excluded.word_median,
                    chars_per_page=excluded.chars_per_page,
                    size_bytes=excluded.size_bytes,
                    text_quality=excluded.text_quality,
                    do_ocr=excluded.do_ocr,
                    sparse_pages=excluded.sparse_pages,
                    blank_pages=excluded.blank_pages,
                    image_xobjects=excluded.image_xobjects,
                    image_pages=excluded.image_pages,
                    image_heavy_pages=excluded.image_heavy_pages,
                    image_page_share=excluded.image_page_share,
                    n_triage_pages=NULL,
                    tail_triage_start=NULL,
                    triage_policy='tail',
                    keep_first=1''',
                (
                    profile.hash_id,
                    profile.relative_path,
                    profile.pages,
                    profile.text_chars,
                    profile.char_min,
                    profile.char_max,
                    profile.char_mean,
                    profile.char_median,
                    profile.word_count,
                    profile.word_min,
                    profile.word_max,
                    profile.word_mean,
                    profile.word_median,
                    profile.chars_per_page,
                    profile.size_bytes,
                    document.text_quality,
                    int(document.do_ocr),
                    document.sparse_pages,
                    document.blank_pages,
                    profile.image_xobjects,
                    profile.image_pages,
                    profile.image_heavy_pages,
                    profile.image_page_share,
                ),
            )

            connection.execute('DELETE FROM pages WHERE hash_id=?', (profile.hash_id,))
            connection.executemany(
                '''INSERT INTO pages
                   (hash_id, page_number, text_chars, word_count, blank, sparse, image_xobjects, do_ocr)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    (
                        profile.hash_id,
                        page.page_number,
                        page.text_chars,
                        page.word_count,
                        int(page.blank),
                        int(page.text_chars < self.sparse_chars),
                        page.image_xobjects,
                        int(page.text_chars < self.sparse_chars),
                    )
                    for page in document.pages_profile
                ),
            )

    def sources(
        self,
        *,
        include_text_quality: tuple[str, ...] | None = None,
        include_ocr: bool | None = None,
    ) -> list[PdfSource]:
        if include_text_quality is None and include_ocr is None:
            return [
                PdfSource.from_path(path, relative_path=path.relative_to(self.directory).as_posix())
                for path in self.pdf_paths()
            ]

        with self.store.connect() as connection:
            if include_text_quality is not None:
                placeholders = ', '.join('?' for _ in include_text_quality)
                clause = f'text_quality IN ({placeholders})'
                params: list[object] = list(include_text_quality)
            else:
                clause = '1=1'
                params = []
            if include_ocr is not None:
                clause += ' AND do_ocr=?'
                params.append(int(include_ocr))
            rows = connection.execute(
                f'SELECT relative_path, hash_id FROM documents WHERE {clause} ORDER BY relative_path',
                params,
            ).fetchall()

        sources: list[PdfSource] = []
        for relative_path, hash_id in rows:
            path = self.directory / relative_path
            if path.exists():
                sources.append(PdfSource(path=path, hash_id=hash_id, metadata={'relative_path': relative_path}))
        return sources

    def relative_paths(
        self,
        *,
        include_text_quality: tuple[str, ...] | None = None,
        include_ocr: bool | None = None,
    ) -> list[str]:
        with self.store.connect() as connection:
            if include_text_quality is not None:
                placeholders = ', '.join('?' for _ in include_text_quality)
                clause = f'text_quality IN ({placeholders})'
                params: list[object] = list(include_text_quality)
            else:
                clause = '1=1'
                params = []
            if include_ocr is not None:
                clause += ' AND do_ocr=?'
                params.append(int(include_ocr))
            rows = connection.execute(
                f'SELECT relative_path FROM documents WHERE {clause} ORDER BY relative_path',
                params,
            ).fetchall()
        return [row[0] for row in rows]

    def sample(
        self,
        *,
        per_stratum: int = 10,
        seed: int = 0,
        text_quality: tuple[str, ...] | None = None,
        include_ocr: bool | None = None,
        page_bands: tuple[int, ...] = (5, 15, 40),
    ) -> list[str]:
        from .sampling import StratifiedSampler

        return StratifiedSampler(
            self.store.path,
            per_stratum=per_stratum,
            seed=seed,
            page_bands=page_bands,
        ).sample(text_quality=text_quality, do_ocr=include_ocr)
