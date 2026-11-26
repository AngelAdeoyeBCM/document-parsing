"""Text inside pictures (chart axes, legends, labels): what Docling kept there, drawn and measured."""

from __future__ import annotations

from docling_core.types.doc import ContentLayer, DoclingDocument, PictureItem, TextItem
from PIL import ImageDraw
from PIL.Image import Image
from pypdfium2 import PdfDocument
from typing_extensions import override

from .base import OverlayVisualizer, pixel_box

_ALL_LAYERS = set(ContentLayer)


def picture_texts(doc: DoclingDocument) -> dict[str, list[TextItem]]:
    """Text Docling nested inside each picture, by picture ref; attached captions and footnotes excluded."""
    attached = {ref.cref for p in doc.pictures for ref in (*p.captions, *p.footnotes, *p.references)}
    return {
        picture.self_ref: [
            item
            for item, _ in doc.iterate_items(root=picture, traverse_pictures=True, included_content_layers=_ALL_LAYERS)
            if isinstance(item, TextItem) and item.self_ref not in attached
        ]
        for picture in doc.pictures
    }


def texts_in_picture_box(doc: DoclingDocument, picture: PictureItem) -> list[TextItem]:
    """Every text item centred inside the picture's box, whether Docling nested it in the picture or not."""
    prov = picture.prov[0]
    height = doc.pages[prov.page_no].size.height
    box = prov.bbox.to_top_left_origin(page_height=height)
    found = []
    for text in doc.texts:
        for p in text.prov:
            if p.page_no == prov.page_no:
                b = p.bbox.to_top_left_origin(page_height=height)
                if box.l <= (b.l + b.r) / 2 <= box.r and box.t <= (b.t + b.b) / 2 <= box.b:
                    found.append(text)
                    break
    return found


def pdf_text_in_picture(pdf: PdfDocument, doc: DoclingDocument, picture: PictureItem) -> str:
    """What the PDF text layer holds inside the picture's box: the yardstick for the text Docling kept."""
    prov = picture.prov[0]
    box = prov.bbox.to_bottom_left_origin(page_height=doc.pages[prov.page_no].size.height)
    textpage = pdf[prov.page_no - 1].get_textpage()
    return textpage.get_text_bounded(left=box.l, bottom=box.b, right=box.r, top=box.t)


class PictureTextVisualizer(OverlayVisualizer):
    """Outlines the text Docling kept inside each picture.

    docling's layout visualizer draws nested text like any other text; this marks which of it sits in a picture.
    """

    colour: tuple[int, int, int] = (8, 145, 178)

    @override
    def _draw(self, doc: DoclingDocument, images: dict[int | None, Image]) -> None:
        for texts in picture_texts(doc).values():
            for text in texts:
                for prov in text.prov:
                    if (image := images.get(prov.page_no)) is not None:
                        box = pixel_box(doc, prov, image)
                        ImageDraw.Draw(image).rectangle([box.l, box.t, box.r, box.b], outline=self.colour, width=2)
