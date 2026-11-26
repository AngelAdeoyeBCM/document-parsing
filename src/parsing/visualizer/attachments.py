"""Captions, footnotes and references Docling attached to tables and pictures, and those it attached to nothing."""

from __future__ import annotations

from docling_core.types.doc import DocItem, DocItemLabel, DoclingDocument
from PIL import ImageDraw
from PIL.Image import Image
from typing_extensions import override

from .base import OverlayVisualizer, pixel_box

RELATIONS = ('captions', 'footnotes', 'references')  # how Docling attaches text to a table or picture
ATTACHABLE = {DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE}  # labels expected to belong to a table or picture
COLOURS = {'captions': (219, 39, 119), 'footnotes': (75, 85, 99), 'references': (124, 58, 237)}  # line per relation


def attachments(doc: DoclingDocument) -> list[tuple[DocItem, str, DocItem]]:
    """Every (table or picture, relation, attached item), e.g. (picture, 'captions', caption)."""
    return [
        (owner, relation, ref.resolve(doc))
        for owner in (*doc.tables, *doc.pictures)
        for relation in RELATIONS
        for ref in getattr(owner, relation)
    ]


def unattached(doc: DoclingDocument) -> list[DocItem]:
    """Captions and footnotes attached to no table or picture."""
    attached = {child.self_ref for _, _, child in attachments(doc)}
    return [text for text in doc.texts if text.label in ATTACHABLE and text.self_ref not in attached]


class AttachmentVisualizer(OverlayVisualizer):
    """Lines from each table or picture to what Docling attached to it; unattached captions and footnotes in red."""

    unattached_colour: tuple[int, int, int] = (220, 38, 38)

    @override
    def _draw(self, doc: DoclingDocument, images: dict[int | None, Image]) -> None:
        for owner, relation, child in attachments(doc):
            for prov in owner.prov:
                target = next((p for p in child.prov if p.page_no == prov.page_no), None)
                if target is None or (image := images.get(prov.page_no)) is None:
                    continue
                start, end = pixel_box(doc, prov, image), pixel_box(doc, target, image)
                (x0, y0), (x1, y1) = _centre(start), _centre(end)
                draw = ImageDraw.Draw(image)
                draw.line([(x0, y0), (x1, y1)], fill=COLOURS[relation], width=3)
                draw.ellipse([x1 - 5, y1 - 5, x1 + 5, y1 + 5], fill=COLOURS[relation])  # the attached end
        for item in unattached(doc):
            for prov in item.prov:
                if (image := images.get(prov.page_no)) is not None:
                    box = pixel_box(doc, prov, image)
                    rectangle = [box.l - 3, box.t - 3, box.r + 3, box.b + 3]
                    ImageDraw.Draw(image).rectangle(rectangle, outline=self.unattached_colour, width=3)


def _centre(box) -> tuple[float, float]:
    return (box.l + box.r) / 2, (box.t + box.b) / 2
