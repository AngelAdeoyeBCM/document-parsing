from __future__ import annotations

from docling_core.types.doc import ContentLayer, DoclingDocument, PictureItem, TextItem
from pypdfium2 import PdfDocument

_ALL_LAYERS = set(ContentLayer)

# Docling picture classes treated as charts; 'table' is a table drawn as an image.
CHART_CLASSES = frozenset(
    {
        'bar_chart',
        'stacked_bar_chart',
        'line_chart',
        'pie_chart',
        'scatter_plot',
        'scatter_chart',
        'box_plot',
        'heatmap',
        'flow_chart',
        'other_chart',
        'table',
    }
)


def picture_class(picture: PictureItem) -> tuple[str, float] | None:
    """Top class Docling's picture classifier gave this picture, with its confidence; None if not classified."""
    classification = picture.meta.classification if picture.meta else None
    if classification is None or not classification.predictions:
        return None
    best = max(classification.predictions, key=lambda p: p.confidence or 0.0)
    return best.class_name, best.confidence or 0.0


def picture_texts(doc: DoclingDocument) -> dict[str, list[TextItem]]:
    """Text Docling kept inside each picture, e.g. chart axis labels; attached captions and footnotes excluded."""
    attached = {ref.cref for p in doc.pictures for ref in (*p.captions, *p.footnotes, *p.references)}
    return {
        picture.self_ref: [
            item
            for item, _ in doc.iterate_items(root=picture, traverse_pictures=True, included_content_layers=_ALL_LAYERS)
            if isinstance(item, TextItem) and item.self_ref not in attached
        ]
        for picture in doc.pictures
    }


def pdf_text_in_picture(pdf: PdfDocument, doc: DoclingDocument, picture: PictureItem) -> str:
    """What the PDF text layer holds inside the picture's box, the yardstick for retained chart text."""
    prov = picture.prov[0]
    box = prov.bbox.to_bottom_left_origin(page_height=doc.pages[prov.page_no].size.height)
    textpage = pdf[prov.page_no - 1].get_textpage()
    return textpage.get_text_bounded(left=box.l, bottom=box.b, right=box.r, top=box.t)


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
