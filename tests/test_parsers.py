from __future__ import annotations

from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from inventory import PdfSource

from parsing.core import ParsedDocument, ParserInfo
from parsing.parsers import get_parser
from parsing.parsers.model import ModelConfig, ModelParser
from parsing.parsers.pypdf_parser import PypdfParser


class EchoModelParser(ModelParser):
    """Stands in for a model parser: returns the pypdf text of whatever PDF it is given."""

    name = 'echo'
    package = 'pypdf'
    config_cls = ModelConfig

    def _convert_pdf(self, path: Path) -> DoclingDocument:
        return PypdfParser().parse(PdfSource.from_path(path)).document


def test_pypdf_extracts_text_and_honours_skip(make_pdf) -> None:
    source = PdfSource.from_path(make_pdf('a.pdf', [['first page'], ['second page'], ['third page']]))
    parsed = PypdfParser().parse(source, skip=frozenset({2}))
    assert parsed.page_numbers == [1, 3]
    assert 'third page' in parsed.document.export_to_text(page_no=3)
    assert parsed.page_count == 3
    assert parsed.parser.name == 'pypdf'


def test_model_parser_sees_only_kept_pages_and_renumbers(make_pdf) -> None:
    source = PdfSource.from_path(make_pdf('a.pdf', [['one'], ['two'], ['three'], ['four']]))
    parsed = EchoModelParser().parse(source, skip=frozenset({1, 3}))
    assert parsed.page_numbers == [2, 4]
    assert parsed.document.pages[4].page_no == 4
    assert [(item.text, item.prov[0].page_no) for item in parsed.document.texts] == [('two', 2), ('four', 4)]


def test_parsed_document_rejects_page_both_parsed_and_skipped(make_pdf) -> None:
    source = PdfSource.from_path(make_pdf('a.pdf', [['one']]))
    document = PypdfParser().parse(source).document
    with pytest.raises(ValueError):
        ParsedDocument(source, ParserInfo('x', '0'), document, skipped_pages=frozenset({1}))


def test_source_hash_tracks_content(tmp_path: Path) -> None:
    first, second = tmp_path / 'a.pdf', tmp_path / 'b.pdf'
    first.write_bytes(b'same')
    second.write_bytes(b'same')
    assert PdfSource.from_path(first).hash_id == PdfSource.from_path(second).hash_id
    second.write_bytes(b'different')
    assert PdfSource.from_path(first).hash_id != PdfSource.from_path(second).hash_id


def test_registry_builds_configured_parser() -> None:
    parser = get_parser('pypdf', {'extraction_mode': 'layout'})
    assert parser.info.config['extraction_mode'] == 'layout'
    with pytest.raises(KeyError):
        get_parser('nope')
    with pytest.raises(TypeError):
        get_parser('pypdf', {'unknown_option': 1})
