"""Named serializations of a DoclingDocument, built on docling-core's own serializers.

Parse runs and gold documents are serialized by the same name, so every comparison is like for like. A new
serialization is one more entry in ``SERIALIZATIONS``: a docling serializer class and its parameters.
``serialize_elements`` splits the 'extracted' serialization by element kind, to trace problems such as garbling.

A run saves only the lossless DoclingDocument JSON; every serialization is derived from it on demand
(``DoclingDocument.load_from_json``), so serializations can be added or changed without parsing again.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import CommonParams, DocSerializer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.transforms.serializer.plain_text import PlainTextDocSerializer, PlainTextParams
from docling_core.types.doc import ContentLayer, DocItem, DocItemLabel, DoclingDocument, PictureItem

# The Stage 1.1 keep set: body prose; tables, pictures, captions, footnotes and furniture are left out.
BODY_LABELS = frozenset(
    {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE, DocItemLabel.LIST_ITEM}
)
PICTURE_TEXT = 'picture_text'  # element kind for text found inside pictures, e.g. chart axes and legends
# Only text the parser read from the page: no chart data, picture classes or descriptions that models generated.
_EXTRACTED_ONLY = {'enable_chart_tables': False, 'include_picture_classification': False, 'allowed_meta_names': set()}


@dataclass(frozen=True, slots=True)
class Serialization:
    serializer: type[DocSerializer]
    params: CommonParams

    @property
    def suffix(self) -> str:
        # docling's plain-text serializer subclasses its Markdown one, so test for plain text.
        return '.txt' if issubclass(self.serializer, PlainTextDocSerializer) else '.md'

    def pages(self, doc: DoclingDocument) -> dict[int, str]:
        """The serialized text of each page that has any.

        One walk of the document yields its top-level parts, grouped by the page of the items in each. A page
        touched by a part spanning pages (e.g. a list continued overleaf) is serialized with docling's page filter
        instead, which splits that part by item; filtering every page that way walks the whole document per page.
        """
        serializer = self.serializer(doc=doc, params=self.params)
        by_page: dict[int, list[SerializationResult]] = defaultdict(list)
        split: set[int] = set()
        for part in serializer.get_parts():
            pages = {span.item.prov[0].page_no if span.item.prov else None for span in part.spans}
            if len(pages) == 1 and None not in pages:
                by_page[pages.pop()].append(part)
            else:
                split |= pages - {None}  # items without a location are on no page, as under the filter
        texts = {}
        for page in sorted(doc.pages):
            if page in split:
                params = self.params.merge_with_patch({'pages': {page}})
                text = self.serializer(doc=doc, params=params).serialize().text
            else:
                text = serializer.serialize_doc(parts=by_page.get(page, [])).text
            if text.strip():
                texts[page] = text
        return texts


SERIALIZATIONS: dict[str, Serialization] = {
    # Everything the parser extracted, on every content layer and inside pictures: for text-quality metrics.
    'extracted': Serialization(
        PlainTextDocSerializer, PlainTextParams(layers=set(ContentLayer), traverse_pictures=True, **_EXTRACTED_ONLY)
    ),
    # The Stage 1.1 text-only output.
    'body': Serialization(PlainTextDocSerializer, PlainTextParams(labels=set(BODY_LABELS), **_EXTRACTED_ONLY)),
    # Body with each table in place, as a plain-text table.
    'body_tables': Serialization(
        PlainTextDocSerializer, PlainTextParams(labels=set(BODY_LABELS | {DocItemLabel.TABLE}), **_EXTRACTED_ONLY)
    ),
    # Docling's standard Markdown export, as a RAG pipeline would ingest it.
    'markdown': Serialization(MarkdownDocSerializer, MarkdownParams()),
}


def get_serialization(name: str) -> Serialization:
    try:
        return SERIALIZATIONS[name]
    except KeyError:
        raise KeyError(f'unknown serialization {name!r}; available: {sorted(SERIALIZATIONS)}') from None


def serialize(doc: DoclingDocument, name: str = 'body') -> dict[int, str]:
    """Text per page under the named serialization."""
    return get_serialization(name).pages(doc)


def serialize_document(doc: DoclingDocument, name: str = 'body') -> str:
    """The whole document under the named serialization, each page after a ``<!-- page N -->`` marker.

    The markers let a chunker recover page numbers; between them is exactly the text that is scored.
    """
    return '\n\n'.join(f'<!-- page {number} -->\n\n{text}' for number, text in serialize(doc, name).items())


def serialize_elements(doc: DoclingDocument) -> dict[str, dict[int, str]]:
    """The 'extracted' text split by element kind: kind -> page -> text.

    Kinds are the docling labels of the items on the page (text, table, caption, page_footer, ...), each serialized
    on its own, plus 'picture_text' for the text found inside pictures.
    """
    extracted = SERIALIZATIONS['extracted']
    layers = set(ContentLayer)
    labels = {item.label for item, _ in doc.iterate_items(included_content_layers=layers) if isinstance(item, DocItem)}
    kinds = {
        label.value: Serialization(
            extracted.serializer, extracted.params.merge_with_patch({'labels': {label}, 'traverse_pictures': False})
        ).pages(doc)
        for label in labels - {DocItemLabel.PICTURE}
    }
    kinds[PICTURE_TEXT] = _picture_text(doc, extracted)
    return {kind: pages for kind, pages in sorted(kinds.items()) if pages}


def _picture_text(doc: DoclingDocument, extracted: Serialization) -> dict[int, str]:
    """Per page, the text inside each top-level picture, without its caption or footnotes."""
    params = extracted.params.merge_with_patch(
        {'labels': set(DocItemLabel) - {DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE}, 'traverse_pictures': True}
    )
    serializer = extracted.serializer(doc=doc, params=params)
    pages: dict[int, list[str]] = defaultdict(list)
    for picture in doc.pictures:
        if not picture.prov or (picture.parent and isinstance(picture.parent.resolve(doc), PictureItem)):
            continue  # a picture nested in another is walked with its parent
        text = '\n'.join(part.text for part in serializer.get_parts(item=picture) if part.text)
        if text.strip():
            pages[picture.prov[0].page_no].append(text)
    return {page: '\n\n'.join(texts) for page, texts in sorted(pages.items())}
