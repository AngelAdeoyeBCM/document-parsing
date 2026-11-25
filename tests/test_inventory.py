from __future__ import annotations

import sqlite3

import pytest
from conftest import Image

import inventory.profiler as profiler_module
from inventory import Inventory
from triage import Triage

TEXT = 'Enough meaningful text on this page to exceed the configured sparse threshold for embedded text'
DISCLOSURE = [
    'Important disclosures and analyst certification',
    'This report has been prepared by the research desk',
    'Past performance is not a guide to future returns',
]


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every path the profiler opens with pypdf."""
    opened: list[str] = []
    real = profiler_module.PdfReader

    def counting(path, *args, **kwargs):
        opened.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(profiler_module, 'PdfReader', counting)
    return opened


def _by_name(inventory: Inventory) -> dict[str, object]:
    return {profile.relative_path.rsplit('/', 1)[-1].removesuffix('.pdf'): profile for profile in inventory.profiles()}


def test_inventory_classifies_text_scans_and_blank_pages(tmp_path, make_pdf):
    make_pdf('corpus/text.pdf', [[TEXT], [TEXT]])
    make_pdf('corpus/scan.pdf', [[], []], images={0: [Image()], 1: [Image()]})
    make_pdf('corpus/blank.pdf', [[], []])

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    report = inventory.build()
    assert (report.profiled, report.failed) == (3, 0)

    by_id = _by_name(inventory)
    assert (by_id['text'].text_quality, by_id['text'].do_ocr) == ('embedded_text', False)
    assert (by_id['scan'].text_quality, by_id['scan'].do_ocr) == ('image_only', True)
    assert by_id['scan'].scanned_pages == 2 and by_id['scan'].max_image_coverage == pytest.approx(1.0)
    # Empty pages with nothing drawn have no text to recover, so they are not sent to OCR.
    assert (by_id['blank'].text_quality, by_id['blank'].do_ocr) == ('image_only', False)
    assert by_id['blank'].blank_pages == 2

    assert [source.doc_id for source in inventory.sources(include_ocr=False)] == ['blank', 'text']
    assert [source.doc_id for source in inventory.sources(include_ocr=True)] == ['scan']


@pytest.mark.parametrize('how', ['xobject', 'inline', 'form'])
def test_image_coverage_counts_drawn_area(tmp_path, make_pdf, how):
    half = Image(width=612, height=396, how=how)
    make_pdf('corpus/doc.pdf', [[], [TEXT]], images={0: [half], 1: [Image(width=61.2, height=79.2, how=how)]})

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    inventory.build()
    [profile] = inventory.profiles()
    first, second = inventory.pages(profile.hash_id)

    assert first.image_count == 1 and first.image_coverage == pytest.approx(0.5)
    assert first.sparse and first.needs_ocr  # sparse text over a half-page image
    assert second.image_count == 1 and second.image_coverage == pytest.approx(0.01)
    assert not second.needs_ocr  # a logo on a text page


def test_small_image_on_an_empty_page_is_not_a_scan(tmp_path, make_pdf):
    make_pdf('corpus/logo.pdf', [[]], images={0: [Image(width=100, height=50)]})
    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    inventory.build()
    [profile] = inventory.profiles()
    [page] = inventory.pages(profile.hash_id)
    assert page.blank and page.image_count == 1 and not page.needs_ocr


def test_pypdf_reads_each_file_once_ever(tmp_path, make_pdf, reads):
    make_pdf('corpus/a.pdf', [[TEXT]])
    make_pdf('corpus/b.pdf', [[TEXT, 'different']])
    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')

    assert inventory.build().profiled == 2
    assert len(reads) == 2

    report = inventory.build(triage=Triage())
    assert (report.profiled, report.cached, report.triaged) == (0, 2, 2)
    assert len(reads) == 2  # cached profiles; triage ran on the stored text


def test_inventory_hash_deduplicates_duplicate_content(tmp_path, make_pdf, reads):
    make_pdf('corpus/a.pdf', [['Same text']])
    make_pdf('corpus/b.pdf', [['Same text']])

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    report = inventory.build()

    assert (report.profiled, report.cached) == (1, 1)
    assert len(reads) == 1
    assert len(inventory.profiles()) == 1
    assert [source.relative_path for source in inventory.sources()] == ['a.pdf']


def test_inventory_keeps_invalid_pdf_for_pipeline_failure(tmp_path):
    (tmp_path / 'corpus').mkdir()
    (tmp_path / 'corpus/broken.pdf').write_bytes(b'not a pdf')

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    assert inventory.build().failed == 1
    assert inventory.profiles() == []
    assert [source.doc_id for source in inventory.sources()] == ['broken']
    assert inventory.sources(include_ocr=False) == []
    assert inventory.build().failed == 1  # recorded, not retried


def test_sources_respect_the_manifest(tmp_path, make_pdf, reads):
    for name in ('a', 'b', 'c'):
        make_pdf(f'corpus/{name}.pdf', [[f'{TEXT} {name}']])
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text('a.pdf\nc.pdf\nmissing.pdf\n')

    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    report = inventory.build(manifest)

    assert (report.profiled, report.missing) == (2, 1)
    assert sorted(reads) == sorted(str(tmp_path / 'corpus' / f'{name}.pdf') for name in 'ac')
    assert [source.doc_id for source in inventory.sources(manifest)] == ['a', 'c']


def test_build_records_triage_verdicts(tmp_path, make_pdf):
    make_pdf('corpus/note.pdf', [[TEXT], [TEXT], DISCLOSURE])
    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    triage = Triage(policy='tail', keep_first=1)
    assert inventory.build(triage=triage).triaged == 1

    [profile] = inventory.profiles()
    verdicts, skip = inventory.store.triage(profile.hash_id, triage.key)
    assert skip == frozenset({3})
    assert [verdict.boilerplate for verdict in verdicts] == [False, False, True]
    assert inventory.store.triage(profile.hash_id, Triage(policy='flagged').key) is None

    with sqlite3.connect(inventory.store.path) as connection:
        row = connection.execute('SELECT triage_policy, skip_pages, n_triage_pages, tail_triage_start FROM documents')
        assert row.fetchone() == ('tail', '[3]', 1, 3)

    assert inventory.build(triage=triage).triaged == 0  # same settings and corpus: nothing to redo


def test_rules_triage_does_not_depend_on_corpus_changes(tmp_path, make_pdf):
    make_pdf('corpus/a.pdf', [[TEXT], DISCLOSURE])
    make_pdf('corpus/b.pdf', [[TEXT + ' b'], DISCLOSURE])
    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    triage = Triage(policy='tail', keep_first=1)

    assert inventory.build(triage=triage).triaged == 2
    assert {inventory.store.triage(p.hash_id, triage.key)[1] for p in inventory.profiles()} == {frozenset({2})}

    make_pdf('corpus/c.pdf', [[TEXT + ' c'], DISCLOSURE])
    assert inventory.build(triage=triage).triaged == 1


def test_stratified_sample_is_deterministic(tmp_path, make_pdf):
    for index in range(6):
        make_pdf(f'corpus/{index}.pdf', [[f'{TEXT} {index}']])
    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    inventory.build()
    first = inventory.sample(per_stratum=2, seed=7)
    assert len(first) == 2
    assert inventory.sample(per_stratum=2, seed=7) == first


def test_numbered_stratified_sample_is_deterministic(tmp_path, make_pdf):
    for index in range(12):
        make_pdf(f'corpus/{index}.pdf', [[f'{TEXT} {index}']])
    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    inventory.build()

    first = inventory.sample(number=5, seed=29)
    assert len(first) == 5
    assert inventory.sample(number=5, seed=29) == first


def test_sample_uses_kept_page_count_for_length_bands(tmp_path, make_pdf):
    make_pdf('corpus/short.pdf', [[TEXT], [TEXT], DISCLOSURE])
    make_pdf('corpus/long.pdf', [[TEXT] for _ in range(20)])
    inventory = Inventory(tmp_path / 'corpus', tmp_path / 'inventory.db')
    triage = Triage(policy='tail', keep_first=1)
    inventory.build(triage=triage)

    chosen = inventory.sample(number=2, seed=3)
    assert len(chosen) == 2
