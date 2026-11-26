from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from inventory import PdfSource, truncate_pdf
from parsing.core import ParsedDocument
from parsing.parsers import Parser
from triage import PageVerdict

from .manifest import save_document
from .routing import DEFAULT_ROUTE

# Disabled stages (Stage 1.1 runs are Parse -> Write); restore these imports with them:
# from pypdf import PdfReader
# from inventory import InventoryStore
# from parsing.postprocess import DocumentProcessor
# from triage import Triage, run_triage
# from .routing import Router


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One document's state as it moves through the stages."""

    source: PdfSource
    page_texts: tuple[str, ...] = ()  # pypdf text layer, one entry per source page
    skip: frozenset[int] = frozenset()
    verdicts: tuple[PageVerdict, ...] = ()
    route: str = DEFAULT_ROUTE
    document: ParsedDocument | None = None
    notes: Mapping[str, object] = field(default_factory=dict)  # post-processing outcomes for run.jsonl
    outputs: tuple[Path, ...] = ()

    def require_document(self) -> ParsedDocument:
        if self.document is None:
            raise RuntimeError(f'{self.source.doc_id}: no parsed document; a ParseStage must run first')
        return self.document


class Stage(ABC):
    def prepare(self, corpus: Iterable[PdfSource]) -> None:
        """Corpus-level pass before any document runs; most stages need none."""

    @abstractmethod
    def run(self, item: WorkItem) -> WorkItem: ...


# Disabled: TextLayerStage (its text only fed triage and routing), TriageStage and RouteStage.
#
# def read_text_layer(source: PdfSource, store: InventoryStore | None = None) -> tuple[str, ...]:
#     """The pypdf text layer, from the inventory when it holds this content, else read from the PDF."""
#     texts = store.page_texts(source.hash_id) if store is not None else None
#     if texts is None:
#         texts = tuple(page.extract_text() or '' for page in PdfReader(source.path).pages)
#     return texts
#
#
# class TextLayerStage(Stage):
#     """Each page's embedded text, for triage and routing; reused from the inventory rather than re-read."""
#
#     def __init__(self, store: InventoryStore | None = None) -> None:
#         self.store = store
#
#     def run(self, item: WorkItem) -> WorkItem:
#         return replace(item, page_texts=read_text_layer(item.source, self.store))
#
#
#
# class TriageStage(Stage):
#     """Marks boilerplate pages so the parser skips them, reusing verdicts the inventory stored for these settings."""
#
#     def __init__(self, triage: Triage, store: InventoryStore | None = None) -> None:
#         self.triage = triage
#         self.store = store
#         self.classifier = triage.build_classifier()
#         self.policy = triage.build_policy()
#
#     def run(self, item: WorkItem) -> WorkItem:
#         stored = self._stored(item.source)
#         verdicts, skip = stored if stored is not None else run_triage(self.classifier, self.policy, item.page_texts)
#         return replace(item, skip=skip, verdicts=verdicts)
#
#     def _stored(self, source: PdfSource) -> tuple[tuple[PageVerdict, ...], frozenset[int]] | None:
#         return self.store.triage(source.hash_id, self.triage.key) if self.store is not None else None
#
#
# class RouteStage(Stage):
#     """Chooses which parser configuration handles the document, e.g. OCR for image-only PDFs."""
#
#     def __init__(self, router: Router) -> None:
#         self.router = router
#
#     def run(self, item: WorkItem) -> WorkItem:
#         return replace(item, route=self.router.route(item.page_texts))


class ParseStage(Stage):
    """Parses the source PDF, or, when triage skipped its tail, a copy truncated to the kept leading pages."""

    def __init__(self, parsers: Mapping[str, Parser]) -> None:
        if DEFAULT_ROUTE not in parsers:
            raise ValueError(f'ParseStage needs a {DEFAULT_ROUTE!r} parser')
        self.parsers = dict(parsers)

    def run(self, item: WorkItem) -> WorkItem:
        parser = self.parsers[item.route]
        if not item.skip:
            return replace(item, document=parser.parse(item.source))
        with tempfile.TemporaryDirectory() as tmp:
            truncated = Path(tmp) / item.source.path.name  # same file name, so the document is named alike
            truncate_pdf(item.source.path, item.skip, truncated)
            parsed = parser.parse(item.source, truncated)
        return replace(item, document=replace(parsed, skipped_pages=item.skip))


# Disabled: PostprocessStage.
#
# class PostprocessStage(Stage):
#     """Applies document processors in order, e.g. dropping non-chart pictures."""
#
#     def __init__(self, processors: Sequence[DocumentProcessor]) -> None:
#         self.processors = tuple(processors)
#
#     def run(self, item: WorkItem) -> WorkItem:
#         document, notes = item.require_document(), dict(item.notes)
#         for processor in self.processors:
#             result = processor.process(document)
#             document = result.document
#             notes[processor.name] = result.notes
#         return replace(item, document=document, notes=notes)


class WriteStage(Stage):
    """Saves the lossless DoclingDocument JSON; serializations and visualizations are derived from it on demand."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def run(self, item: WorkItem) -> WorkItem:
        return replace(item, outputs=(save_document(item.require_document(), self.out_dir),))
