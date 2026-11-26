from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from inventory import PdfSource, truncate_pdf
from parsing import gpus
from parsing.core import ParsedDocument, ParserInfo
from parsing.parsers import get_parser
from parsing.parsers.docling_standard import set_options
from parsing.parsers.pypdf_parser import PypdfParser


def test_pypdf_extracts_every_page(make_pdf) -> None:
    source = PdfSource.from_path(make_pdf('a.pdf', [['first page'], ['second page'], ['third page']]))
    parsed = PypdfParser().parse(source)
    assert parsed.page_numbers == [1, 2, 3] and parsed.page_count == 3
    assert 'third page' in parsed.document.export_to_text(page_no=3)
    assert parsed.parser.name == 'pypdf' and parsed.document.name == 'a'
    assert parsed.seconds > 0


def test_a_truncated_copy_is_parsed_as_its_source(make_pdf, tmp_path: Path) -> None:
    source = PdfSource.from_path(make_pdf('a.pdf', [['first page'], ['second page'], ['disclosures']]))
    truncated = tmp_path / 'copy' / 'a.pdf'
    truncated.parent.mkdir()
    truncate_pdf(source.path, {3}, truncated)
    parsed = PypdfParser().parse(source, truncated)
    assert parsed.source == source  # the source's identity and hash, not the copy's
    assert parsed.page_numbers == [1, 2]  # the leading pages keep their numbers
    assert 'disclosures' not in parsed.document.export_to_text()


def test_set_options_sets_dotted_paths_and_rejects_unknown_names() -> None:
    options = SimpleNamespace(do_table_structure=True, layout_options=SimpleNamespace(create_orphan_clusters=True))
    set_options(options, {'do_table_structure': False, 'layout_options.create_orphan_clusters': False})
    assert options.do_table_structure is False and options.layout_options.create_orphan_clusters is False
    with pytest.raises(AttributeError):
        set_options(options, {'layout_options.typo': 1})


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


def test_auto_device_takes_the_visible_gpu_with_most_free_memory(monkeypatch) -> None:
    monkeypatch.setattr(gpus, 'gpu_memory', lambda: {0: (100, 400), 1: (300, 400), 2: (200, 400)})
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert gpus.auto_device() == 'cuda:1'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2')
    assert gpus.auto_device() == 'cuda:1'  # physical GPU 2, second of those visible
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert gpus.auto_device() == 'auto'
