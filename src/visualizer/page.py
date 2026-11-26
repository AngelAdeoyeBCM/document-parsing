"""An interactive review page built from docling's own parts.

docling's visualizers draw each page (layout boxes, optionally reading order) and docling's HTML serializer renders
each item. This page links the two: hovering a box or an item highlights its partner and what is attached to it, and
clicking a box scrolls to its item. Header toggles show more, each as a transparent layer over the page images:
the text inside pictures (``PictureTextVisualizer``), and what Docling attached to tables and pictures
(``AttachmentVisualizer``). Pictures also show how much of the PDF's text inside them was kept.
"""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Mapping

import pypdfium2 as pdfium
from docling_core.transforms.serializer.html import HTMLDocSerializer, HTMLParams
from docling_core.transforms.visualizer.base import BaseVisualizer
from docling_core.transforms.visualizer.layout_visualizer import LayoutVisualizer
from docling_core.transforms.visualizer.reading_order_visualizer import ReadingOrderVisualizer
from docling_core.types.doc import ContentLayer, DocItem, DocItemLabel, DoclingDocument, ImageRef, PictureItem, TextItem
from PIL.Image import Image

from parsing.core import ParsedDocument

from .attachments import AttachmentVisualizer, attachments, unattached
from .base import OverlayVisualizer
from .pictures import PictureTextVisualizer, pdf_text_in_picture, picture_texts, texts_in_picture_box

_LAYERS = {ContentLayer.BODY, ContentLayer.FURNITURE}
_INNER_COLOUR = '#0891b2'  # PictureTextVisualizer's outline
_LOW_RETENTION = 0.8  # flag pictures whose box text was mostly lost from the parse

# Header toggles: name -> (label, the layer drawn for it). Each shows its layer and the page elements of its class.
TOGGLES: dict[str, tuple[str, OverlayVisualizer]] = {
    'picture-text': ('text inside pictures (axes, labels)', PictureTextVisualizer(layer=True)),
    'attachments': ('captions and footnotes of tables and pictures', AttachmentVisualizer(layer=True)),
}

_STYLE = """
body{margin:0;background:#e5e7eb;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;padding:10px 20px;background:#111827;color:#f9fafb;font-size:14px}
header label{margin-left:18px;cursor:pointer}
.page{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px;margin:16px}
.pageno{grid-column:1/-1;font-weight:600}
.sheet{position:relative;align-self:start;background:#fff;box-shadow:0 1px 4px #0004}
.sheet img{display:block;width:100%;height:auto}
.sheet img.layer{position:absolute;inset:0;height:100%;pointer-events:none}
.box{position:absolute;cursor:pointer}
.box.hl{outline:3px solid var(--c);background:color-mix(in srgb,var(--c) 25%,transparent)}
.items{position:relative;background:#fff;box-shadow:0 1px 4px #0004}
.items>div{position:absolute;inset:0;overflow-y:auto;padding:8px 12px}
.item{border-left:4px solid var(--c);padding:4px 8px;margin:6px 0;font-size:13px}
.item.hl{background:#fef3c7}
.item.furniture{opacity:.65;font-style:italic}
.item>:is(p,h1,h2,h3,h4,h5,h6,ul,ol){margin:2px 0}
.tag{font:11px monospace;color:#fff;background:var(--c);padding:1px 5px;border-radius:3px}
table{border-collapse:collapse;font-size:12px;margin-top:4px}
td,th{border:1px solid #d1d5db;padding:2px 5px}
caption{text-align:left;font-size:12px;color:#6b7280}
.inner-text{font-size:12px;margin-top:3px}
.inner-text summary{cursor:pointer;color:#0e7490}
.inner-text.low summary{color:#dc2626;font-weight:600}
.cell{display:inline-block;margin:2px;padding:0 4px;border:1px dotted #0891b2;border-radius:3px;font-size:11px}
.cell.hl{background:#cffafe}
.assoc{font-size:12px;color:#374151;margin-top:3px}
.assoc.missing{color:#9ca3af}
.assoc.orphan{color:#dc2626;font-weight:600}
.skipped{position:absolute;inset:0;background:#0008;color:#fff;display:grid;place-items:center;font-size:24px}
.toggled{display:none!important}
"""
# Each toggle reveals the elements carrying its name as a class.
_STYLE += ''.join(f'body.show-{name} .toggled.{name}{{display:block!important}}\n' for name in TOGGLES)

# Hovering a box, item or cell highlights everything with its ref or linked to it; clicking a box scrolls to its entry.
_SCRIPT = """
document.querySelectorAll('.box,.item,.cell').forEach(el => {
  const refs = [el.dataset.ref, ...(el.dataset.links || '').split(' ').filter(Boolean)];
  const group = () => refs.flatMap(r => [...document.querySelectorAll(`[data-ref="${r}"]`)]);
  el.addEventListener('mouseenter', () => group().forEach(p => p.classList.add('hl')));
  el.addEventListener('mouseleave', () => group().forEach(p => p.classList.remove('hl')));
  if (el.classList.contains('box')) el.addEventListener('click', () => {
    const entry = document.querySelector(`.item[data-ref="${el.dataset.ref}"], .cell[data-ref="${el.dataset.ref}"]`);
    entry?.closest('details')?.setAttribute('open', '');
    entry?.scrollIntoView({block: 'center'});
  });
});
document.querySelectorAll('header input[data-toggle]').forEach(box => box.addEventListener('change', () =>
  document.body.classList.toggle(`show-${box.dataset.toggle}`, box.checked)));
"""


def visualizer(reading_order: bool = False, labels: bool = False) -> BaseVisualizer:
    """docling's layout boxes, and optionally its reading order on top."""
    layout = LayoutVisualizer(params=LayoutVisualizer.Params(show_label=labels))
    return ReadingOrderVisualizer(base_visualizer=layout) if reading_order else layout


def visualize(parsed: ParsedDocument, scale: float = 1.5, reading_order: bool = False, labels: bool = False) -> str:
    """The parsed document as an interactive HTML page, with every source page rendered from the PDF.

    Pages skipped by triage are shown too, as they look in the PDF, marked as skipped.
    """
    doc = parsed.document.model_copy(deep=True)
    pdf = pdfium.PdfDocument(parsed.source.path)
    try:
        pages = {number: pdf[number - 1].render(scale=scale).to_pil() for number in range(1, len(pdf) + 1)}
        for number, page in doc.pages.items():
            page.image = ImageRef.from_pil(pages[number], dpi=round(72 * scale))
        words = {p.self_ref: len(pdf_text_in_picture(pdf, doc, p).split()) for p in doc.pictures if p.prov}
    finally:
        pdf.close()
    images = pages | visualizer(reading_order, labels).get_visualization(doc=doc)  # drawn over the parsed pages
    layers = {name: drawer.get_visualization(doc=doc) for name, (_, drawer) in TOGGLES.items()}
    skipped = ', '.join(map(str, sorted(parsed.skipped_pages))) or 'none'
    subtitle = f'{parsed.parser.name} {parsed.parser.version} · skipped pages: {skipped}'
    return InteractivePage(doc, images, words, layers).render(parsed.source.doc_id, subtitle)


class InteractivePage:
    """Each page's visualization beside the items on that page, linked on hover.

    ``images`` may hold pages the document does not (skipped by triage); they are shown, marked as skipped.
    ``layers`` maps a ``TOGGLES`` name to transparent page layers (an ``OverlayVisualizer`` with ``layer=True``);
    its header toggle shows them, with the page elements that go with them.
    """

    def __init__(
        self,
        doc: DoclingDocument,
        images: Mapping[int | None, Image],
        picture_words: Mapping[str, int] | None = None,
        layers: Mapping[str, Mapping[int | None, Image]] | None = None,
    ) -> None:
        self.doc = doc
        self.images = images
        self.picture_words = picture_words or {}  # PDF text-layer word count inside each picture
        # Only layers with something drawn on them, so blank pages carry no extra image.
        self.layers = {
            name: {n: image for n, image in pages.items() if image.getchannel('A').getbbox()}
            for name, pages in (layers or {}).items()
        }
        self.inner = picture_texts(doc)  # picture ref -> the text Docling kept inside it
        self.picture_of = {text.self_ref: ref for ref, texts in self.inner.items() for text in texts}
        self.attached: dict[str, list[tuple[str, DocItem]]] = {}  # table or picture ref -> (relation, item)
        self.owner: dict[str, tuple[str, DocItem]] = {}  # attached item ref -> (relation, table or picture)
        for owner, relation, child in attachments(doc):
            self.attached.setdefault(owner.self_ref, []).append((relation, child))
            self.owner[child.self_ref] = (relation, owner)
        self.unattached = {item.self_ref for item in unattached(doc)}
        self.serializer = HTMLDocSerializer(doc=doc, params=HTMLParams(layers=_LAYERS))

    def render(self, title: str, subtitle: str = '') -> str:
        pages = '\n'.join(self._page(number) for number in sorted(n for n in self.images if n is not None))
        toggles = ''.join(
            f'<label><input type="checkbox" data-toggle="{name}"> show {label}</label>'
            for name, (label, _) in TOGGLES.items()
            if self.layers.get(name)  # only toggles with something to show
        )
        title = html.escape(title)
        return (
            f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title><style>{_STYLE}</style></head>'
            f'<body><header><b>{title}</b> · {html.escape(subtitle)}{toggles}</header>{pages}'
            f'<script>{_SCRIPT}</script></body></html>'
        )

    def _page(self, number: int) -> str:
        if number in self.doc.pages:
            overlay = ''.join(
                f'<img class="layer toggled {name}" src="{_data_uri(pages[number])}">'
                for name, pages in self.layers.items()
                if number in pages
            )
            overlay += ''.join(self._boxes(number))
            items = ''.join(self._item(item) for item in self._items(number, traverse_pictures=False))
        else:  # not parsed: removed before parsing
            overlay, items = '<div class="skipped">skipped by triage</div>', ''
        return (
            f'<section class="page"><div class="pageno">Page {number}</div>'
            f'<div class="sheet"><img src="{_data_uri(self.images[number])}">{overlay}</div>'
            f'<div class="items"><div>{items}</div></div></section>'
        )

    def _items(self, number: int, traverse_pictures: bool) -> list[DocItem]:
        found = self.doc.iterate_items(
            page_no=number, included_content_layers=_LAYERS, traverse_pictures=traverse_pictures
        )
        return [item for item, _ in found if isinstance(item, DocItem)]

    def _links(self, item: DocItem) -> str:
        """Refs to highlight with the item: its picture, the text inside it, what is attached to it or what it is
        attached to."""
        refs = [self.picture_of[item.self_ref]] if item.self_ref in self.picture_of else []
        refs += [text.self_ref for text in self.inner.get(item.self_ref, [])]
        refs += [child.self_ref for _, child in self.attached.get(item.self_ref, [])]
        refs += [self.owner[item.self_ref][1].self_ref] if item.self_ref in self.owner else []
        return html.escape(' '.join(refs))

    def _boxes(self, number: int) -> list[str]:
        """An invisible hit-box over each drawn box, carrying its item's ref; those of text inside pictures are
        only there while their toggle is on, so a chart's labels don't take its hover."""
        size = self.doc.pages[number].size
        boxes = []
        for item in self._items(number, traverse_pictures=True):
            inner = item.self_ref in self.picture_of
            kind, colour = ('box toggled picture-text', _INNER_COLOUR) if inner else ('box', _colour(item))
            for prov in item.prov:
                if prov.page_no != number:
                    continue
                box = prov.bbox.to_top_left_origin(page_height=size.height)
                style = (
                    f'left:{box.l / size.width:.3%};top:{box.t / size.height:.3%};'
                    f'width:{(box.r - box.l) / size.width:.3%};height:{(box.b - box.t) / size.height:.3%};--c:{colour}'
                )
                boxes.append(
                    f'<div class="{kind}" data-ref="{html.escape(item.self_ref)}" data-links="{self._links(item)}"'
                    f' style="{style}" title="{html.escape(item.label.value)}"></div>'
                )
        return boxes

    def _item(self, item: DocItem) -> str:
        """The item as docling's HTML renders it, or its text where docling renders it elsewhere (e.g. captions)."""
        body = self.serializer.serialize(item=item).text
        if not body:
            body = html.escape(item.text) if isinstance(item, TextItem) else ''
        body += self._associations(item)
        if isinstance(item, PictureItem):
            body += self._retention(item)
        furniture = ' furniture' if item.content_layer == ContentLayer.FURNITURE else ''
        return (
            f'<div class="item{furniture}" data-ref="{html.escape(item.self_ref)}" data-links="{self._links(item)}"'
            f' style="--c:{_colour(item)}"><span class="tag">{html.escape(item.label.value)}</span> {body}</div>'
        )

    def _associations(self, item: DocItem) -> str:
        """What Docling attached to this table or picture, or what this caption or footnote is attached to."""
        notes: list[tuple[str, str]] = []  # (class, text)
        if item.label in (DocItemLabel.TABLE, DocItemLabel.PICTURE):
            attached = self.attached.get(item.self_ref, [])
            notes = [('', f'↳ {relation[:-1]}: {_snippet(child)}') for relation, child in attached]
            notes = notes or [('missing', 'no caption, footnote or reference attached')]
        elif item.self_ref in self.owner:
            relation, owner = self.owner[item.self_ref]
            notes = [('', f'↳ {relation[:-1]} of {owner.label.value} {owner.self_ref}')]
        elif item.self_ref in self.unattached:
            notes = [('orphan', '⚠ not attached to any table or picture')]
        return ''.join(
            f'<div class="assoc {kind} toggled attachments">{html.escape(text)}</div>' for kind, text in notes
        )

    def _retention(self, picture: PictureItem) -> str:
        """How much of the PDF's text inside this picture Docling kept, with the kept text listed."""
        texts = self.inner.get(picture.self_ref, [])
        nested = sum(len(text.text.split()) for text in texts)
        in_box = sum(len(t.text.split()) for t in texts_in_picture_box(self.doc, picture)) if picture.prov else 0
        source = self.picture_words.get(picture.self_ref)
        low = False
        if source is None:
            summary = f'{len(texts)} text cells kept inside'
        elif source == 0:
            summary = f'{len(texts)} text cells kept inside · no text in the PDF here (image only)'
        else:
            share, found = min(nested / source, 1.0), min(in_box / source, 1.0)
            summary = (
                f'{len(texts)} text cells kept inside · PDF words nested {nested}/{source} ({share:.0%}),'
                f' found anywhere in the box {in_box}/{source} ({found:.0%})'
            )
            low = found < _LOW_RETENTION
        cells = ''.join(
            f'<span class="cell" data-ref="{html.escape(t.self_ref)}">{html.escape(t.text)}</span>' for t in texts
        )
        return f'<details class="inner-text{" low" if low else ""}"><summary>{summary}</summary>{cells}</details>'


def _colour(item: DocItem) -> str:
    """The colour docling's layout visualizer gives the item's label, so boxes and entries match."""
    red, green, blue = DocItemLabel.get_color(item.label)
    return f'rgb({red},{green},{blue})'


def _snippet(item: DocItem, length: int = 80) -> str:
    text = getattr(item, 'text', '') or item.label.value
    return text if len(text) <= length else text[: length - 1] + '…'


def _data_uri(image: Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
