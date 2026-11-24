from __future__ import annotations

from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    PictureClassificationMetaField,
    PictureMeta,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.items.picture.meta import PictureClassificationPrediction

from inventory import PdfSource
from parsing.core import ParsedDocument, ParserInfo
from parsing.postprocess import ChartFilter


def _picture(doc: DoclingDocument, caption: str, class_name: str, confidence: float) -> None:
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=10, t=100, r=200, b=10), charspan=(0, 0))
    text = doc.add_text(label=DocItemLabel.CAPTION, text=caption, prov=prov)
    picture = doc.add_picture(caption=text, prov=prov)
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[PictureClassificationPrediction(class_name=class_name, confidence=confidence)]
        )
    )


def test_chart_filter_drops_every_non_chart_and_keeps_their_captions(make_pdf) -> None:
    doc = DoclingDocument(name='x')
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    _picture(doc, 'Fig 1: Copper balance', 'line_chart', 0.9)
    _picture(doc, 'Fig 2: Mislinked caption', 'logo', 0.95)
    _picture(doc, 'Unsure', 'photograph', 0.3)
    parsed = ParsedDocument(PdfSource.from_path(make_pdf('a.pdf', [['x']])), ParserInfo('docling', '0'), doc)

    result = ChartFilter().process(parsed)

    pictures = result.document.document.pictures
    assert len(pictures) == 1
    assert result.notes['pictures_dropped'] == {'logo': 1, 'photograph': 1}
    assert {t.text for t in result.document.document.texts} >= {
        'Fig 1: Copper balance',
        'Fig 2: Mislinked caption',
        'Unsure',
    }
    assert len(parsed.document.pictures) == 3
