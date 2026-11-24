from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from inventory import InventoryStore, PdfSource
from inventory.triage import PageVerdict, Triage, run_triage
from pypdf import PdfReader

from ..core import ParsedDocument
from ..parsers import Parser
from ..postprocess import DocumentProcessor
from ..render import Renderer
from .routing import DEFAULT_ROUTE, Router


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


def read_text_layer(source: PdfSource, store: InventoryStore | None = None) -> tuple[str, ...]:
    """The pypdf text layer, from the inventory when it holds this content, else read from the PDF."""
    texts = store.page_texts(source.hash_id) if store is not None else None
    if texts is None:
        texts = tuple(page.extract_text() or '' for page in PdfReader(source.path).pages)
    return texts


class TextLayerStage(Stage):
    """Each page's embedded text, for triage and routing; reused from the inventory rather than re-read."""

    def __init__(self, store: InventoryStore | None = None) -> None:
        self.store = store

    def run(self, item: WorkItem) -> WorkItem:
        return replace(item, page_texts=read_text_layer(item.source, self.store))


class TriageStage(Stage):
    """Marks boilerplate pages so the parser skips them, reusing verdicts the inventory stored for these settings."""

    def __init__(self, triage: Triage, store: InventoryStore | None = None) -> None:
        self.triage = triage
        self.store = store
        self.classifier = triage.build_classifier()
        self.policy = triage.build_policy()

    def prepare(self, corpus: Iterable[PdfSource]) -> None:
        if not self.classifier.needs_corpus:
            return
        corpus = list(corpus)
        if all(self._stored(source) is not None for source in corpus):
            return
        for source in corpus:
            self.classifier.observe(read_text_layer(source, self.store))

    def run(self, item: WorkItem) -> WorkItem:
        stored = self._stored(item.source)
        verdicts, skip = stored if stored is not None else run_triage(self.classifier, self.policy, item.page_texts)
        return replace(item, skip=skip, verdicts=verdicts)

    def _stored(self, source: PdfSource) -> tuple[tuple[PageVerdict, ...], frozenset[int]] | None:
        return self.store.triage(source.hash_id, self.triage.key) if self.store is not None else None


class RouteStage(Stage):
    """Chooses which parser configuration handles the document, e.g. OCR for image-only PDFs."""

    def __init__(self, router: Router) -> None:
        self.router = router

    def run(self, item: WorkItem) -> WorkItem:
        return replace(item, route=self.router.route(item.page_texts))


class ParseStage(Stage):
    def __init__(self, parsers: Mapping[str, Parser]) -> None:
        if DEFAULT_ROUTE not in parsers:
            raise ValueError(f'ParseStage needs a {DEFAULT_ROUTE!r} parser')
        self.parsers = dict(parsers)

    def run(self, item: WorkItem) -> WorkItem:
        return replace(item, document=self.parsers[item.route].parse(item.source, item.skip))


class PostprocessStage(Stage):
    """Applies document processors in order, e.g. dropping non-chart pictures."""

    def __init__(self, processors: Sequence[DocumentProcessor]) -> None:
        self.processors = tuple(processors)

    def run(self, item: WorkItem) -> WorkItem:
        document, notes = item.require_document(), dict(item.notes)
        for processor in self.processors:
            result = processor.process(document)
            document = result.document
            notes[processor.name] = result.notes
        return replace(item, document=document, notes=notes)


class WriteStage(Stage):
    def __init__(self, renderers: Sequence[Renderer], out_dir: Path) -> None:
        self.renderers = tuple(renderers)
        self.out_dir = out_dir

    def run(self, item: WorkItem) -> WorkItem:
        document = item.require_document()
        return replace(item, outputs=tuple(renderer.write(document, self.out_dir) for renderer in self.renderers))
