from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, FloatObject, NameObject, NumberObject

PAGE_WIDTH, PAGE_HEIGHT = 612, 792

# A disclosure page as the triage rules see one: over 150 words of prose, mostly disclaimer wording.
DISCLOSURE = [
    'Important disclosures and analyst certification',
    'This report has been prepared by the research desk for professional clients and eligible counterparties only.',
    'It does not constitute investment advice and is not an offer or solicitation to buy or sell any security.',
    'Past performance is not indicative of future results and the value of investments may go down as well as up.',
    'The information herein is believed to be reliable but no representation or warranty is given as to accuracy.',
    'Opinions may change without notice and the bank accepts no liability for any loss arising from its use.',
    'The bank and its affiliates may have positions in the securities mentioned and may have conflicts of interest.',
    'This document may not be reproduced or redistributed to any recipient without prior written consent.',
    'It is not intended for distribution in any jurisdiction where such distribution would be contrary to law.',
    'The bank is authorised by the regulator and regulated by the Financial Conduct Authority in the United Kingdom.',
    'Recipients should seek their own advice before making any decision on the basis of this material.',
    'All rights reserved, and the research is non-independent for the purposes of the applicable rules.',
    'Further regulatory disclosures, including the distribution of ratings, are available on request from the desk.',
    'Investors should consider whether the products described are suitable for them in light of their objectives.',
]


@dataclass(frozen=True)
class Image:
    """A 1x1 image drawn at the page origin, scaled to ``width`` x ``height`` points.

    ``how`` is 'xobject' (``Do`` an image XObject), 'inline' (``BI ... EI``) or 'form' (an image XObject drawn
    inside a form XObject, the scale split between the page and the form so both must be accounted for).
    """

    width: float = PAGE_WIDTH
    height: float = PAGE_HEIGHT
    how: str = 'xobject'


MakePdf = Callable[..., Path]


def _escape(text: str) -> str:
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Write a PDF where ``pages[i]`` holds the text lines of page i + 1 and ``images[i]`` what page i + 1 draws."""

    def build(name: str, pages: Sequence[Sequence[str]], images: Mapping[int, Sequence[Image]] | None = None) -> Path:
        writer = PdfWriter()
        font = writer._add_object(
            DictionaryObject(
                {
                    NameObject('/Type'): NameObject('/Font'),
                    NameObject('/Subtype'): NameObject('/Type1'),
                    NameObject('/BaseFont'): NameObject('/Helvetica'),
                }
            )
        )
        pixel = _stream(
            writer,
            b'\x00',
            {
                '/Type': '/XObject',
                '/Subtype': '/Image',
                '/Width': 1,
                '/Height': 1,
                '/ColorSpace': '/DeviceGray',
                '/BitsPerComponent': 8,
            },
        )
        for index, lines in enumerate(pages):
            page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            commands: list[str] = []
            xobjects = DictionaryObject()
            for number, image in enumerate((images or {}).get(index, ())):
                commands.extend(_draw(writer, image, number, pixel, xobjects))
            commands.extend(['BT', '/F1 11 Tf', '14 TL', '72 720 Td'])
            commands.extend(f'({_escape(line)}) Tj T*' for line in lines)
            commands.append('ET')
            page[NameObject('/Contents')] = _stream(writer, '\n'.join(commands).encode('latin-1'), {})
            resources = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
            if xobjects:
                resources[NameObject('/XObject')] = xobjects
            page[NameObject('/Resources')] = resources
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as handle:
            writer.write(handle)
        return path

    return build


def _draw(writer: PdfWriter, image: Image, number: int, pixel: object, xobjects: DictionaryObject) -> list[str]:
    if image.how == 'inline':
        return [f'q {image.width} 0 0 {image.height} 0 0 cm', 'BI /W 1 /H 1 /CS /G /BPC 8 ID \x00 EI', 'Q']
    if image.how == 'xobject':
        xobjects[NameObject(f'/Im{number}')] = pixel
        return [f'q {image.width} 0 0 {image.height} 0 0 cm /Im{number} Do Q']
    if image.how == 'form':
        # Page scales by 2, the form's /Matrix by 0.5 x 1, its content by the rest: the area is still width x height.
        form = _stream(
            writer,
            f'q {image.width} 0 0 {image.height / 2} 0 0 cm /Px Do Q'.encode('latin-1'),
            {
                '/Type': '/XObject',
                '/Subtype': '/Form',
                '/BBox': [0, 0, PAGE_WIDTH, PAGE_HEIGHT],
                '/Matrix': [0.5, 0, 0, 1, 0, 0],
                '/Resources': {'/XObject': {'/Px': pixel}},
            },
        )
        xobjects[NameObject(f'/Fm{number}')] = form
        return [f'q 2 0 0 2 0 0 cm /Fm{number} Do Q']
    raise ValueError(image.how)


def _stream(writer: PdfWriter, data: bytes, entries: Mapping[str, object]) -> object:
    stream = DecodedStreamObject()
    stream.set_data(data)
    for key, value in entries.items():
        stream[NameObject(key)] = _pdf(value)
    return writer._add_object(stream)


def _pdf(value: object) -> object:
    if isinstance(value, str):
        return NameObject(value)
    if isinstance(value, bool | int):
        return NumberObject(value)
    if isinstance(value, float):
        return FloatObject(value)
    if isinstance(value, list):
        return ArrayObject(_pdf(item) for item in value)
    if isinstance(value, dict):
        return DictionaryObject({NameObject(key): _pdf(item) for key, item in value.items()})
    return value  # already a PDF object, e.g. an indirect reference
