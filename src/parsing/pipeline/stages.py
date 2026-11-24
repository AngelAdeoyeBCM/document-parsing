from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from inventory import PdfSource
from pypdf import PdfReader

from ..core import ParsedDocument
from ..parsers import Parser
from ..postprocess import DocumentProcessor
from ..render import Renderer
from ..triage import PageClassifier, PageVerdict, SkipPolicy
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


def read_text_layer(source: PdfSource) -> tuple[str, ...]:
    return tuple(page.extract_text() or '' for page in PdfReader(source.path).pages)


class TextLayerStage(Stage):
    """Reads each page's embedded text once with pypdf, for triage and routing."""

    def run(self, item: WorkItem) -> WorkItem:
        return replace(item, page_texts=read_text_layer(item.source))


class TriageStage(Stage):
    """Marks boilerplate pages so the parser skips them."""

    def __init__(self, classifier: PageClassifier, policy: SkipPolicy) -> None:
        self.classifier = classifier
        self.policy = policy

    def prepare(self, corpus: Iterable[PdfSource]) -> None:
        if self.classifier.needs_corpus:
            for source in corpus:
                self.classifier.observe(read_text_layer(source))

    def run(self, item: WorkItem) -> WorkItem:
        verdicts = self.classifier.classify(item.page_texts)
        return replace(item, skip=self.policy.skip(verdicts), verdicts=tuple(verdicts))


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
