from __future__ import annotations

import base64
import html
import io
import re
from collections.abc import Mapping

from docling_core.types.doc import (
    ContentLayer,
    DocItem,
    PictureItem,
    ProvenanceItem,
    SectionHeaderItem,
    TableData,
    TableItem,
    TextItem,
)
from PIL.Image import Image

from ..core import ParsedDocument, picture_class, picture_texts, texts_in_picture_box
from .chart_svg import chart_svg, number

_COLOURS = {
    'title': '#dc2626',
    'section_header': '#dc2626',
    'text': '#2563eb',
    'list_item': '#7c3aed',
    'table': '#059669',
    'picture': '#d97706',
    'chart': '#d97706',
    'caption': '#db2777',
    'footnote': '#4b5563',
    'page_header': '#94a3b8',
    'page_footer': '#94a3b8',
}
_LAYERS = {ContentLayer.BODY, ContentLayer.FURNITURE}
_RELATIONS = ('captions', 'footnotes', 'references')  # how Docling attaches text to a table or picture
_ATTACHABLE = {'caption', 'footnote'}
_TICK = re.compile(r'[-−]?[\d,]*\.?\d+\s*%?')

_STYLE = """
body{margin:0;background:#e5e7eb;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;padding:10px 20px;background:#111827;color:#f9fafb;font-size:14px}
.page{display:grid;grid-template-columns:auto minmax(0,1fr);gap:16px;margin:16px}
.pageno{grid-column:1/-1;font-weight:600}
.sheet{position:relative;background:#fff;box-shadow:0 1px 4px #0004}
.sheet img{display:block;width:100%;height:auto}
.box{position:absolute;border:2px solid var(--c);cursor:pointer}
.box.furniture{border-style:dashed}
.box.hl{background:color-mix(in srgb,var(--c) 30%,transparent)}
.box.orphan{outline:3px dotted #dc2626;outline-offset:2px}
.links{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.links line{stroke-width:2;vector-effect:non-scaling-stroke;opacity:.7}
.links line.captions{stroke:#db2777}
.links line.footnotes{stroke:#4b5563;stroke-dasharray:5 3}
.links line.references{stroke:#0891b2;stroke-dasharray:2 3}
.links line.hl{stroke-width:4;opacity:1}
.assoc{font-size:12px;color:#374151;margin-top:3px}
.assoc.missing{color:#9ca3af}
.assoc.orphan{color:#dc2626;font-weight:600}
.legend span{margin-left:14px}
.box.inner{border-width:1px;border-style:dotted;display:none}
body.show-inner .box.inner{display:block}
.inner-text{font-size:12px;margin-top:3px}
.inner-text summary{cursor:pointer;color:#0e7490}
.inner-text.low summary{color:#dc2626;font-weight:600}
.cell{display:inline-block;margin:2px;padding:0 4px;border:1px dotted #0891b2;border-radius:3px;font-size:11px}
.chart-compare{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:4px 0;white-space:normal}
.chart-compare figure{margin:0;border:1px solid #e5e7eb;padding:4px;background:#fff}
.chart-compare img,.chart-compare svg{width:100%;height:auto;display:block}
.chart-compare figcaption{font-size:11px;color:#6b7280;text-align:center}
.cell.hl{background:#cffafe}
header label{margin-left:18px;cursor:pointer}
.skipped{position:absolute;inset:0;background:#0008;color:#fff;display:grid;place-items:center;font-size:24px}
.items{position:relative;background:#fff;box-shadow:0 1px 4px #0004}
.items>div{position:absolute;inset:0;overflow-y:auto;padding:8px 12px}
.item{border-left:4px solid var(--c);padding:4px 8px;margin:6px 0;font-size:13px;white-space:pre-wrap}
.item.hl{background:#fef3c7}
.item.furniture{opacity:.65;font-style:italic}
.tag{font:11px monospace;color:#fff;background:var(--c);padding:1px 5px;border-radius:3px;margin-right:6px}
table{border-collapse:collapse;font-size:12px;white-space:normal}
td,th{border:1px solid #d1d5db;padding:2px 5px}
th{background:#f3f4f6}
"""

# Hovering a box or entry highlights it, its partner on the other side, and anything Docling linked to it.
_SCRIPT = """
document.querySelectorAll('.box,.item,.cell').forEach(el => {
  const refs = [el.dataset.ref, ...(el.dataset.links || '').split(' ').filter(Boolean)];
  const group = () => refs.flatMap(r => [...document.querySelectorAll(`[data-ref="${r}"]`)]);
  el.addEventListener('mouseenter', () => group().forEach(p => p.classList.add('hl')));
  el.addEventListener('mouseleave', () => group().forEach(p => p.classList.remove('hl')));
  if (el.classList.contains('box')) el.addEventListener('click', () =>
    document.querySelector(`.item[data-ref="${el.dataset.ref}"]`)?.scrollIntoView({block: 'center'}));
});
document.getElementById('show-inner').addEventListener('change', e =>
  document.body.classList.toggle('show-inner', e.target.checked));
"""

_LEGEND = (
    '<span class="legend">links: <span style="color:#db2777">── caption</span>'
    '<span style="color:#9ca3af">- - footnote</span><span style="color:#22d3ee">··· reference</span>'
    '<span style="color:#f87171">⋯ unattached caption/footnote</span></span>'
    '<label><input type="checkbox" id="show-inner"> show text inside pictures (axes, labels)</label>'
)
_LOW_RETENTION = 0.8  # flag pictures whose box text was mostly lost from the parse


class ProofHtml:
    """Page image with Docling's boxes on the left, the parsed items of that page on the right, linked on hover."""

    def __init__(
        self,
        parsed: ParsedDocument,
        images: Mapping[int, Image],
        scale: float,
        zoom: float = 0.5,
        picture_words: Mapping[str, int] | None = None,
    ) -> None:
        self.parsed = parsed
        self.doc = parsed.document
        self.images = images
        self.scale = scale  # pixels per PDF point in ``images``
        self.zoom = zoom  # 1.0 shows a page at the size a browser PDF viewer uses at 100%
        self.picture_words = picture_words or {}  # PDF text-layer word count inside each picture
        self._inner = picture_texts(self.doc)  # picture ref -> text Docling kept inside it
        self._links: dict[str, list[tuple[str, DocItem]]] = {}  # owner ref -> (relation, attached item)
        self._owners: dict[str, tuple[str, DocItem]] = {}  # attached ref -> (relation, owner item)
        for owner in (*self.doc.tables, *self.doc.pictures):
            for relation in _RELATIONS:
                for ref in getattr(owner, relation, []):
                    child = ref.resolve(self.doc)
                    self._links.setdefault(owner.self_ref, []).append((relation, child))
                    self._owners[child.self_ref] = (relation, owner)

    def render(self) -> str:
        source, parser = self.parsed.source, self.parsed.parser
        title = html.escape(source.doc_id)
        skipped = ', '.join(map(str, sorted(self.parsed.skipped_pages))) or 'none'
        pages = '\n'.join(self._page(number) for number in sorted(self.images))
        return (
            f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title><style>{_STYLE}</style></head>'
            f'<body><header><b>{title}</b> · {html.escape(parser.name)} {html.escape(parser.version)}'
            f' · skipped pages: {skipped}{_LEGEND}</header>{pages}<script>{_SCRIPT}</script></body></html>'
        )

    def _page(self, number: int) -> str:
        picture = self.images[number]
        width = picture.width / self.scale * 96 / 72 * self.zoom
        image = f'<img src="{_data_uri(picture)}">'
        if number not in self.doc.pages:
            overlay, items = '<div class="skipped">skipped by triage</div>', ''
        else:
            entries = [(item, self._geometry(prov, number)) for item, prov in self._entries(number)]
            first = {}
            for item, geometry in entries:
                first.setdefault(item.self_ref, (item, geometry))
            overlay = ''.join(self._box(item, geometry) for item, geometry in entries)
            overlay += ''.join(self._inner_boxes(number))
            overlay += self._connectors({ref: geometry for ref, (_, geometry) in first.items()})
            items = ''.join(self._item(item) for item, _ in first.values())
        return (
            f'<section class="page"><div class="pageno">Page {number}</div>'
            f'<div class="sheet" style="width:{width:.0f}px">{image}{overlay}</div>'
            f'<div class="items"><div>{items}</div></div></section>'
        )

    def _entries(self, number: int) -> list[tuple[DocItem, ProvenanceItem]]:
        entries = []
        for item, _ in self.doc.iterate_items(page_no=number, included_content_layers=_LAYERS):
            if isinstance(item, DocItem):
                entries.extend((item, prov) for prov in item.prov if prov.page_no == number)
        return entries

    def _geometry(self, prov: ProvenanceItem, number: int) -> tuple[float, float, float, float]:
        """Left, top, width and height as percentages of the page."""
        size = self.doc.pages[number].size
        box = prov.bbox.to_top_left_origin(page_height=size.height)
        return (
            box.l / size.width * 100,
            box.t / size.height * 100,
            (box.r - box.l) / size.width * 100,
            (box.b - box.t) / size.height * 100,
        )

    def _box(self, item: DocItem, geometry: tuple[float, float, float, float]) -> str:
        left, top, width, height = geometry
        style = f'left:{left:.3f}%;top:{top:.3f}%;width:{width:.3f}%;height:{height:.3f}%'
        return f'<div {self._attrs(item, "box")} style="{style};--c:{_colour(item)}" title="{item.label.value}"></div>'

    def _inner_boxes(self, number: int) -> list[str]:
        """Dotted boxes for text kept inside pictures; hidden until the header toggle is on."""
        boxes = []
        for picture_ref, texts in self._inner.items():
            for text in texts:
                for prov in text.prov:
                    if prov.page_no == number:
                        left, top, width, height = self._geometry(prov, number)
                        style = f'left:{left:.3f}%;top:{top:.3f}%;width:{width:.3f}%;height:{height:.3f}%'
                        boxes.append(
                            f'<div class="box inner" data-ref="{html.escape(text.self_ref)}" '
                            f'data-links="{html.escape(picture_ref)}" style="{style};--c:#0891b2" '
                            f'title="{html.escape(text.text)}"></div>'
                        )
        return boxes

    def _connectors(self, geometry: Mapping[str, tuple[float, float, float, float]]) -> str:
        """Lines from each table or picture to the captions, footnotes and references Docling attached on this page."""
        lines = []
        for owner in geometry:
            for relation, child in self._links.get(owner, []):
                if child.self_ref in geometry:
                    (x1, y1), (x2, y2) = _centre(geometry[owner]), _centre(geometry[child.self_ref])
                    lines.append(
                        f'<line class="{relation}" data-ref="{html.escape(owner)}" '
                        f'x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}"/>'
                    )
        if not lines:
            return ''
        return f'<svg class="links" viewBox="0 0 100 100" preserveAspectRatio="none">{"".join(lines)}</svg>'

    def _item(self, item: DocItem) -> str:
        label = item.label.value
        if isinstance(item, SectionHeaderItem):
            label += f' h{item.level}'
        if isinstance(item, PictureItem) and (top := picture_class(item)):
            label += f' · {top[0]} {top[1]:.2f}'
        tag = f'<span class="tag">{html.escape(label)}</span>'
        body = f'{tag}{self._content(item)}{self._associations(item)}'
        if isinstance(item, PictureItem):
            body += self._retention(item)
        return f'<div {self._attrs(item, "item")} style="--c:{_colour(item)}">{body}</div>'

    def _retention(self, picture: PictureItem) -> str:
        """How much of the PDF's text inside this picture Docling kept, with the kept text listed."""
        texts = self._inner.get(picture.self_ref, [])
        nested = sum(len(text.text.split()) for text in texts)
        in_box = sum(len(text.text.split()) for text in texts_in_picture_box(self.doc, picture)) if picture.prov else 0
        source = self.picture_words.get(picture.self_ref)
        if source is None:
            summary, low = f'{len(texts)} text cells kept inside', False
        elif source == 0:
            summary, low = f'{len(texts)} text cells kept inside · no text in the PDF here (image only)', False
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

    def _associations(self, item: DocItem) -> str:
        """What Docling attached to this table or picture, or what this caption or footnote is attached to."""
        if isinstance(item, (TableItem, PictureItem)):
            links = self._links.get(item.self_ref, [])
            if not links:
                return '<div class="assoc missing">no caption, footnote or reference attached</div>'
            return ''.join(
                f'<div class="assoc">↳ {relation[:-1]}: {html.escape(_snippet(child))}</div>'
                for relation, child in links
            )
        if item.self_ref in self._owners:
            relation, owner = self._owners[item.self_ref]
            return f'<div class="assoc">↳ {relation[:-1]} of {owner.label.value} {html.escape(owner.self_ref)}</div>'
        if item.label.value in _ATTACHABLE:
            return '<div class="assoc orphan">⚠ not attached to any table or picture</div>'
        return ''

    def _content(self, item: DocItem) -> str:
        if isinstance(item, TableItem):
            return _grid(item.data)
        if isinstance(item, PictureItem):
            parts = []
            if item.meta and item.meta.description:
                parts.append(html.escape(item.meta.description.text))
            if item.meta and item.meta.tabular_chart:
                parts.append(self._chart_comparison(item))
            return ''.join(parts) or '<i>picture</i>'
        if isinstance(item, TextItem):
            return html.escape(item.text)
        return ''

    def _chart_comparison(self, picture: PictureItem) -> str:
        """The chart as printed next to a re-plot of the data the chart model extracted, to judge it by eye."""
        data = picture.meta.tabular_chart.chart_data
        top = picture_class(picture)
        original, aspect = '', None
        prov = picture.prov[0] if picture.prov else None
        if prov and prov.page_no in self.images:
            page = self.images[prov.page_no]
            left, top_, width, height = self._geometry(prov, prov.page_no)
            box = (
                page.width * left / 100,
                page.height * top_ / 100,
                page.width * (left + width) / 100,
                page.height * (top_ + height) / 100,
            )
            aspect = (box[2] - box[0]) / max(box[3] - box[1], 1)
            original = f'<img src="{_data_uri(page.crop(tuple(round(v) for v in box)))}">'
        kind = top[0] if top else 'line_chart'
        replot = chart_svg(data, kind, aspect, self._y_axis_ticks(picture)) or '<i>no numeric data to plot</i>'
        return (
            f'<div class="chart-compare"><figure>{original}<figcaption>original</figcaption></figure>'
            f'<figure>{replot}<figcaption>re-plotted from extracted data</figcaption></figure></div>'
            f'<details><summary>extracted data ({data.num_rows} × {data.num_cols})</summary>{_grid(data)}</details>'
        )

    def _y_axis_ticks(self, picture: PictureItem) -> list[float]:
        """Numbers stacked down the left quarter of the printed chart, read top to bottom: its y-axis labels."""
        if not picture.prov:
            return []
        prov = picture.prov[0]
        height = self.doc.pages[prov.page_no].size.height
        box = prov.bbox.to_top_left_origin(page_height=height)
        ticks = []
        for text in self._inner.get(picture.self_ref, []):
            if not _TICK.fullmatch(text.text.strip()) or not text.prov:
                continue
            b = text.prov[0].bbox.to_top_left_origin(page_height=height)
            if (b.l + b.r) / 2 < box.l + (box.r - box.l) * 0.25:
                ticks.append(((b.t + b.b) / 2, number(text.text)))
        values = [value for _, value in sorted(ticks)]
        # Real y-axis labels fall steadily from top to bottom; anything else is not an axis.
        return values if len(values) >= 3 and values == sorted(values, reverse=True) else []

    def _attrs(self, item: DocItem, kind: str) -> str:
        classes = [kind]
        if item.content_layer == ContentLayer.FURNITURE:
            classes.append('furniture')
        if item.label.value in _ATTACHABLE and item.self_ref not in self._owners:
            classes.append('orphan')
        linked = [child.self_ref for _, child in self._links.get(item.self_ref, [])]
        linked += [text.self_ref for text in self._inner.get(item.self_ref, [])]
        if item.self_ref in self._owners:
            linked.append(self._owners[item.self_ref][1].self_ref)
        return (
            f'class="{" ".join(classes)}" data-ref="{html.escape(item.self_ref)}" '
            f'data-links="{html.escape(" ".join(linked))}"'
        )


def _colour(item: DocItem) -> str:
    return _COLOURS.get(item.label.value, '#64748b')


def _centre(geometry: tuple[float, float, float, float]) -> tuple[float, float]:
    left, top, width, height = geometry
    return left + width / 2, top + height / 2


def _snippet(item: DocItem, length: int = 80) -> str:
    text = getattr(item, 'text', '') or item.label.value
    return text if len(text) <= length else text[: length - 1] + '…'


def _data_uri(image: Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def _grid(data: TableData) -> str:
    """Table with merged cells spanned once, from Docling's cell grid."""
    rows = []
    for r, row in enumerate(data.grid):
        cells = []
        for c, cell in enumerate(row):
            if cell.start_row_offset_idx != r or cell.start_col_offset_idx != c:
                continue
            tag = 'th' if cell.column_header or cell.row_header else 'td'
            spans = f' rowspan="{cell.row_span}"' if cell.row_span > 1 else ''
            spans += f' colspan="{cell.col_span}"' if cell.col_span > 1 else ''
            cells.append(f'<{tag}{spans}>{html.escape(cell.text)}</{tag}>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    return f'<table>{"".join(rows)}</table>'
