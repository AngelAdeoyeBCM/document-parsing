from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

MakePdf = Callable[[str, Sequence[Sequence[str]]], Path]


def _escape(text: str) -> str:
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Write a PDF where ``pages[i]`` holds the text lines of page i + 1."""

    def build(name: str, pages: Sequence[Sequence[str]]) -> Path:
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
        for lines in pages:
            page = writer.add_blank_page(width=612, height=792)
            commands = ['BT', '/F1 11 Tf', '14 TL', '72 720 Td']
            commands.extend(f'({_escape(line)}) Tj T*' for line in lines)
            commands.append('ET')
            stream = DecodedStreamObject()
            stream.set_data('\n'.join(commands).encode('latin-1'))
            page[NameObject('/Contents')] = writer._add_object(stream)
            page[NameObject('/Resources')] = DictionaryObject(
                {NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})}
            )
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as handle:
            writer.write(handle)
        return path

    return build
