from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace

from docling_core.types.doc import DoclingDocument, PictureItem, RefItem

from ..core import ParsedDocument
from .base import DocumentProcessor, ProcessResult
from .pictures import CHART_CLASSES, picture_class


class ChartFilter(DocumentProcessor):
    """Drops every picture whose top class is not a chart; pictures without a classification are kept."""

    name = 'chart_filter'

    def __init__(self, keep: Iterable[str] = CHART_CLASSES) -> None:
        self.keep = frozenset(keep)

    def process(self, parsed: ParsedDocument) -> ProcessResult:
        doc = parsed.document.model_copy(deep=True)
        drop, dropped, unclassified = [], Counter(), 0
        for picture in doc.pictures:
            top = picture_class(picture)
            if top is None:
                unclassified += 1
            elif top[0] not in self.keep:
                drop.append(picture)
                dropped[top[0]] += 1
        if drop:
            for picture in drop:
                _detach_attached_text(doc, picture)
            doc.delete_items(node_items=drop)
        kept = Counter(top[0] for picture in doc.pictures if (top := picture_class(picture)))
        notes = {'pictures_dropped': dict(dropped), 'pictures_kept': dict(kept), 'pictures_unclassified': unclassified}
        return ProcessResult(replace(parsed, document=doc), notes)


def _detach_attached_text(doc: DoclingDocument, picture: PictureItem) -> None:
    """Move the picture's captions and footnotes next to it, so deleting the picture keeps their text."""
    parent = picture.parent.resolve(doc)
    position = next(i for i, ref in enumerate(parent.children) if ref.cref == picture.self_ref)
    for ref in (*picture.captions, *picture.footnotes, *picture.references):
        child = ref.resolve(doc)
        if child.parent is not None and child.parent.cref == picture.self_ref:
            picture.children = [c for c in picture.children if c.cref != child.self_ref]
            child.parent = picture.parent
            position += 1
            parent.children.insert(position, RefItem(cref=child.self_ref))
    picture.captions, picture.footnotes, picture.references = [], [], []
