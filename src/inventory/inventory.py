from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .profiler import PROFILER_VERSION, PageProfile, PDFProfile, Profiler
from .store import InventoryStore
from triage import Triage, run_triage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PdfSource:
    """One inventory-selected PDF ready for parsing."""

    path: Path
    hash_id: str
    metadata: dict[str, str]

    @classmethod
    def from_path(cls, path: Path | str, **metadata: str) -> PdfSource:
        path = Path(path)
        return cls(path=path, hash_id=file_hash(path), metadata=dict(metadata))

    @property
    def doc_id(self) -> str:
        return self.path.stem

    @property
    def relative_path(self) -> str:
        return self.metadata.get('relative_path', self.path.name)


@dataclass(frozen=True, slots=True)
class BuildReport:
    profiled: int = 0  # read with pypdf in this build
    cached: int = 0  # content already profiled by this profiler version; pypdf not called
    failed: int = 0  # pypdf could not read it; recorded in profile_errors
    missing: int = 0  # listed in the manifest but not on disk
    triaged: int = 0  # documents given new triage verdicts

    @property
    def documents(self) -> int:
        return self.profiled + self.cached


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class Inventory:
    """Discover PDFs, profile each unique file once with pypdf, persist profiles, text and triage, and select."""

    def __init__(
        self,
        directory: Path | str,
        database: Path | str,
        *,
        sparse_chars: int = 50,
        scan_coverage: float = 0.5,
        ocr_share: float = 0.5,
    ) -> None:
        self.directory = Path(directory)
        self.store = InventoryStore(database)
        self.profiler = Profiler(sparse_chars=sparse_chars, scan_coverage=scan_coverage, ocr_share=ocr_share)

    def pdf_paths(self, manifest: Path | None = None) -> list[Path]:
        if manifest is not None:
            return [self.directory / relative for relative in _manifest_paths(manifest)]
        return sorted(path for path in self.directory.rglob('*') if path.suffix.lower() == '.pdf')

    def build(self, manifest: Path | None = None, triage: Triage | None = None) -> BuildReport:
        """Profile new or changed PDFs, then, given ``triage``, classify the pages of every document built here."""
        self.store.create_schema()
        counts = {'profiled': 0, 'cached': 0, 'failed': 0, 'missing': 0}
        built: dict[str, None] = {}

        for path in self.pdf_paths(manifest):
            relative_path = path.relative_to(self.directory).as_posix()
            try:
                stat = path.stat()
            except OSError:
                logger.warning('inventory: %s is missing', relative_path)
                counts['missing'] += 1
                continue
            hash_id = self.store.file_hash(relative_path, stat.st_size, stat.st_mtime_ns) or file_hash(path)
            self.store.set_file(relative_path, hash_id, stat.st_size, stat.st_mtime_ns)

            status = self.store.status(hash_id, PROFILER_VERSION)
            if status is not None:
                self.store.set_path(hash_id, relative_path)
                counts['cached' if status == 'profiled' else 'failed'] += 1
            else:
                try:
                    result = self.profiler.profile(path, relative_path, hash_id)
                except Exception as error:
                    logger.warning('inventory: cannot profile %s: %s', relative_path, error)
                    self.store.record_error(
                        hash_id, relative_path, PROFILER_VERSION, f'{type(error).__name__}: {error}'
                    )
                    counts['failed'] += 1
                    continue
                self.store.save_profile(result, PROFILER_VERSION)
                counts['profiled'] += 1
            if status != 'failed':
                built[hash_id] = None

        triaged = self._triage(built, triage) if triage is not None else 0
        return BuildReport(**counts, triaged=triaged)

    def _triage(self, hash_ids: Iterable[str], triage: Triage) -> int:
        """Classify from the stored text using the fixed page-local rules."""
        classifier, policy = triage.build_classifier(), triage.build_policy()
        pending = [hash_id for hash_id in hash_ids if self.store.triage_state(hash_id) != (triage.key, '')]
        if not pending:
            return 0
        for hash_id, texts in self.store.iter_page_texts(pending):
            verdicts, skip = run_triage(classifier, policy, texts)
            self.store.save_triage(
                hash_id,
                key=triage.key,
                corpus='',
                policy=triage.policy,
                keep_first=triage.keep_first,
                verdicts=verdicts,
                skip=skip,
            )
        return len(pending)

    def profiles(
        self, *, include_text_quality: tuple[str, ...] | None = None, include_ocr: bool | None = None
    ) -> list[PDFProfile]:
        return self.store.profiles(include_text_quality, include_ocr)

    def pages(self, hash_id: str) -> tuple[PageProfile, ...]:
        return self.store.pages(hash_id)

    def sources(
        self,
        manifest: Path | None = None,
        *,
        include_text_quality: tuple[str, ...] | None = None,
        include_ocr: bool | None = None,
    ) -> list[PdfSource]:
        """Built files still on disk, one per unique content, optionally limited to ``manifest``.

        Files pypdf could not read are included when no filter is given, so the pipeline records their failure.
        """
        wanted = set(_manifest_paths(manifest)) if manifest is not None else None
        sources: list[PdfSource] = []
        seen: set[str] = set()
        for relative_path, hash_id in self.store.file_rows(include_text_quality, include_ocr):
            if hash_id in seen or (wanted is not None and relative_path not in wanted):
                continue
            path = self.directory / relative_path
            if path.exists():
                seen.add(hash_id)
                sources.append(PdfSource(path=path, hash_id=hash_id, metadata={'relative_path': relative_path}))
        return sources

    def relative_paths(
        self,
        *,
        include_text_quality: tuple[str, ...] | None = None,
        include_ocr: bool | None = None,
    ) -> list[str]:
        return [
            source.relative_path
            for source in self.sources(include_text_quality=include_text_quality, include_ocr=include_ocr)
        ]

    def sample(
        self,
        *,
        number: int | None = None,
        per_stratum: int | None = None,
        seed: int = 0,
        text_quality: tuple[str, ...] | None = None,
        include_ocr: bool | None = None,
        page_bands: tuple[int, ...] = (5, 15, 40),
    ) -> list[str]:
        from .sampling import StratifiedSampler

        return StratifiedSampler(
            self.store,
            number=number,
            per_stratum=per_stratum,
            seed=seed,
            page_bands=page_bands,
        ).sample(text_quality=text_quality, do_ocr=include_ocr)


def _manifest_paths(manifest: Path) -> list[str]:
    return [line.strip() for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
