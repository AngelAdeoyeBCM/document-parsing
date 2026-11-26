from __future__ import annotations

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    Size,
)
from PIL import Image

from inventory import PdfSource
from parsing.parsers import get_parser
from parsing.visualizer import (
    TOGGLES,
    AttachmentVisualizer,
    InteractivePage,
    PictureTextVisualizer,
    visualize,
    visualizer,
)


def _prov(left: float, top: float, right: float, bottom: float) -> ProvenanceItem:
    """A box in PDF coordinates (origin bottom-left), as Docling records them."""
    box = BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.BOTTOMLEFT)
    return ProvenanceItem(page_no=1, bbox=box, charspan=(0, 1))


def _document() -> DoclingDocument:
    """One white 100 pt page (200 px image): a paragraph, a captioned chart with an axis label Docling nested inside
    it, and a footnote attached to nothing."""
    doc = DoclingDocument(name='d')
    image = ImageRef.from_pil(Image.new('RGB', (200, 200), 'white'), dpi=144)
    doc.add_page(page_no=1, size=Size(width=100, height=100), image=image)
    doc.add_text(DocItemLabel.TEXT, 'Prices rose.', prov=_prov(5, 95, 95, 85))
    caption = doc.add_text(DocItemLabel.CAPTION, 'Fig 1: Copper', prov=_prov(10, 80, 90, 72))
    chart = doc.add_picture(prov=_prov(10, 70, 90, 10), caption=caption)
    doc.add_text(DocItemLabel.TEXT, '2024', prov=_prov(20, 30, 40, 20), parent=chart)
    doc.add_text(DocItemLabel.FOOTNOTE, 'Source: Bloomberg', prov=_prov(5, 8, 60, 2))
    return doc


def test_picture_text_visualizer_outlines_text_inside_pictures() -> None:
    doc = _document()
    alone = PictureTextVisualizer().get_visualization(doc=doc)[1]
    # Only the nested label is outlined (cyan); the page image itself is left untouched.
    assert alone.getpixel((40, 140)) == (8, 145, 178) and alone.getpixel((20, 20)) == (255, 255, 255)
    assert doc.pages[1].image.pil_image.getpixel((40, 140)) == (255, 255, 255)
    stacked = PictureTextVisualizer(base_visualizer=visualizer()).get_visualization(doc=doc)[1]  # on docling's boxes
    assert stacked.getpixel((40, 140)) == (8, 145, 178) and stacked.getpixel((10, 10)) != (255, 255, 255)
    layer = PictureTextVisualizer(layer=True).get_visualization(doc=doc)[1]  # only the outlines, to stack over a page
    assert layer.mode == 'RGBA' and layer.size == (200, 200)
    assert layer.getpixel((40, 140)) == (8, 145, 178, 255) and layer.getpixel((20, 20))[3] == 0


def test_attachment_visualizer_links_captions_and_flags_unattached_footnotes() -> None:
    layer = AttachmentVisualizer(layer=True).get_visualization(doc=_document())[1]
    assert layer.getpixel((100, 90)) == (219, 39, 119, 255)  # on the line from the chart's centre to its caption's
    assert layer.getpixel((7, 190)) == (220, 38, 38, 255)  # around the footnote attached to nothing
    assert layer.getpixel((150, 150))[3] == 0  # nothing else drawn


def test_interactive_page_links_boxes_items_and_what_is_attached() -> None:
    doc = _document()
    [text, caption, nested, footnote] = doc.texts
    chart = doc.pictures[0]
    layers = {name: drawer.get_visualization(doc=doc) for name, (_, drawer) in TOGGLES.items()}
    page = InteractivePage(doc, visualizer().get_visualization(doc=doc), {chart.self_ref: 4}, layers).render('d')

    # Nested text and both layers sit behind their header toggles.
    assert 'data-toggle="picture-text"' in page and 'data-toggle="attachments"' in page
    assert page.count('class="box"') == 4 and page.count('class="box toggled picture-text"') == 1
    assert page.count('class="layer toggled picture-text"') == page.count('class="layer toggled attachments"') == 1
    assert page.count('class="item"') == 4  # nested text is listed under its chart, not on its own

    # Hovering the chart highlights the text inside it and its caption; each of those links back to the chart.
    assert f'data-ref="{chart.self_ref}" data-links="{nested.self_ref} {caption.self_ref}"' in page
    assert f'data-ref="{nested.self_ref}" data-links="{chart.self_ref}"' in page
    assert f'data-ref="{caption.self_ref}" data-links="{chart.self_ref}"' in page
    assert f'data-ref="{text.self_ref}" data-links=""' in page

    # The notes the attachments toggle shows.
    assert '↳ caption: Fig 1: Copper' in page and f'↳ caption of picture {chart.self_ref}' in page
    assert 'class="assoc orphan toggled attachments">⚠ not attached to any table or picture' in page
    assert f'data-ref="{footnote.self_ref}"' in page

    assert f'<span class="cell" data-ref="{nested.self_ref}">2024</span>' in page
    assert '<p>Prices rose.</p>' in page  # items render as docling's HTML
    assert 'PDF words nested 1/4 (25%)' in page and 'class="inner-text low"' in page  # most chart text was lost


def test_pages_skipped_by_triage_are_shown_as_skipped(make_pdf) -> None:
    path = make_pdf('note.pdf', [['Copper outlook 2025'], ['Disclosures'], ['More analysis']])
    parsed = get_parser('pypdf').parse(PdfSource.from_path(path), skip=frozenset({2}))
    page = visualize(parsed)

    assert page.count('<section class="page">') == 3 and page.count('data:image/png') == 3  # every source page
    assert page.count('class="skipped"') == 1 and 'Page 2</div><div class="sheet"><img' in page
    assert 'Disclosures' not in page  # not parsed, so nothing listed for it
    assert page.count('class="item"') == 2 and 'skipped pages: 2' in page
    assert 'type="checkbox"' not in page  # no pictures or captions, so nothing to toggle
