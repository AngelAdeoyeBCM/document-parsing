from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DISCLOSURE

from evaluation import evaluate_run
from evaluation.score import read_csv
from inventory import Inventory, PdfSource
from parsing.core import serialize_document
from parsing.parsers import get_parser
from pipeline import DEFAULT_ROUTE, ParseStage, Pipeline, RunConfig, RunManifest, WorkItem
from pipeline.cli import main
from triage import Triage


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            'name': 'test',
            'input': {'directory': str(tmp_path / 'raw')},
            'output': str(tmp_path / 'out'),
            'parser': {'name': 'pypdf'},
        }
    )


def test_pipeline_writes_outputs_and_scores(tmp_path: Path, make_pdf) -> None:
    make_pdf('raw/bnp/note.pdf', [['Copper outlook 2025', 'Prices rise 4.5% on supply'], ['More analysis'], DISCLOSURE])
    config = _config(tmp_path)
    inventory = Inventory(config.input.directory, tmp_path / 'inventory.db')
    inventory.build()
    sources = inventory.sources()
    assert sources[0].relative_path == 'bnp/note.pdf'

    items = list(Pipeline.from_config(config).run(sources))
    assert items[0].skip == frozenset()  # no triage: every page is parsed
    assert [path.name for path in items[0].outputs] == ['note.json']  # a run saves only the lossless JSON

    log = [json.loads(line) for line in (config.output / 'run.jsonl').read_text().splitlines()]
    assert log[0]['status'] == 'ok' and log[0]['skipped_pages'] == []

    [document] = RunManifest(config.output).documents()
    assert document.skipped_pages == frozenset()
    assert document.source.relative_path == 'bnp/note.pdf'

    truth = document.document.model_copy(deep=True)
    truth.pages = {1: truth.pages[1]}  # ground truth holds only the annotated pages
    evaluation = evaluate_run(config.output, inventory.store, {document.source.hash_id: truth})
    pages = {row['page']: row for row in read_csv(evaluation / 'pages.csv')}
    assert sorted(pages) == ['1', '2', '3']
    assert float(pages['1']['extracted.dropped_text_rate']) == 0.0  # pypdf against its own stored text
    assert float(pages['1']['extracted.garble_rate.text']) == 0.0  # pypdf output is one text item per page
    assert float(pages['1']['extracted.garbled_doc_pages']) == 0.0
    assert float(pages['1']['extracted.garble_rate_clean_docs']) == 0.0
    assert float(pages['1']['pypdf_seconds_per_page']) > 0  # the inventory's pypdf time, for comparison
    assert float(pages['1']['body.loss']) == 0.0 and pages['2']['body.loss'] == ''  # page 2 not annotated
    params = json.loads((evaluation / 'params.json').read_text())
    assert (params['parser'], params['documents'], params['gold_pages']) == ('pypdf', 1, 1)
    assert json.loads((evaluation / 'metrics.json').read_text())['body.loss'] == 0.0

    markdown = evaluate_run(
        config.output, inventory.store, {document.source.hash_id: truth}, serializations=['markdown']
    )
    assert float(read_csv(markdown / 'pages.csv')[0]['markdown.loss']) == 0.0  # any named serialization, gold alike

    other_file = evaluate_run(config.output, inventory.store, {'another file': truth})
    assert json.loads((other_file / 'params.json').read_text())['gold_pages'] == 0  # gold from another file


def test_serializations_and_visualizations_come_from_a_saved_run(tmp_path: Path, make_pdf) -> None:
    make_pdf('raw/bnp/note.pdf', [['Copper outlook 2025'], ['More analysis']])
    config = _config(tmp_path)
    Inventory(config.input.directory, tmp_path / 'inventory.db').build()
    list(Pipeline.from_config(config).run(Inventory(config.input.directory, tmp_path / 'inventory.db').sources()))
    [document] = RunManifest(config.output).documents()

    assert main(['serialize', str(config.output), '--serialization', 'body']) == 0
    body = (config.output / 'serializations' / 'body' / 'note.txt').read_text(encoding='utf-8')
    assert body == serialize_document(document.document, 'body')  # exactly the scored text, with page markers
    assert body.startswith('<!-- page 1 -->') and '<!-- page 2 -->' in body
    assert main(['serialize', str(config.output), '--serialization', 'markdown']) == 0
    assert (config.output / 'serializations' / 'markdown' / 'note.md').is_file()

    assert main(['visualize', str(config.output), '--reading-order']) == 0
    page = (config.output / 'visualizations' / 'note.html').read_text(encoding='utf-8')
    assert page.count('<section class="page">') == 2 and page.count('data:image/png') == 2
    assert page.count('class="box"') == page.count('class="item"') == 2  # one text item per pypdf page
    assert 'Copper outlook 2025' in page


def test_parse_stage_parses_only_the_kept_leading_pages(make_pdf) -> None:
    source = PdfSource.from_path(make_pdf('raw/note.pdf', [['Copper outlook 2025'], ['More analysis'], DISCLOSURE]))
    stage = ParseStage({DEFAULT_ROUTE: get_parser('pypdf')})

    whole = stage.run(WorkItem(source)).require_document()
    assert whole.page_numbers == [1, 2, 3] and whole.skipped_pages == frozenset()

    parsed = stage.run(WorkItem(source, skip=frozenset({3}))).require_document()  # triage skipped the tail
    assert parsed.page_numbers == [1, 2] and parsed.skipped_pages == {3} and parsed.page_count == 3
    assert parsed.source == source and parsed.document.name == 'note'  # the copy is parsed as its source
    assert 'Copper outlook 2025' in parsed.document.export_to_text()


# Disabled with TextLayerStage (see pipeline/stages.py).
#
# def test_pipeline_reuses_inventory_text(tmp_path: Path, make_pdf, monkeypatch) -> None:
#     make_pdf('raw/bnp/note.pdf', [['Copper outlook 2025'], ['More analysis'], DISCLOSURE])
#     make_pdf('raw/bnp/other.pdf', [['Not in the manifest']])
#     manifest = tmp_path / 'manifest.txt'
#     manifest.write_text('bnp/note.pdf\n')
#     config = _config(tmp_path)
#     inventory = Inventory(config.input.directory, tmp_path / 'inventory.db')
#     inventory.build(manifest)
#
#     def no_pypdf(*_args, **_kwargs):
#         raise AssertionError('the stages must reuse the inventory instead of reading the PDF')
#
#     monkeypatch.setattr(stages, 'PdfReader', no_pypdf)
#     items = list(Pipeline.from_config(config, store=inventory.store).run(inventory.sources(manifest)))
#
#     assert [item.source.doc_id for item in items] == ['note']
#     assert items[0].page_texts[0].startswith('Copper outlook 2025')


@pytest.mark.parametrize(
    'stage',
    [
        {'triage': {'keep_first': 1}},
        {'routes': [{'name': 'image_only'}]},
        {'postprocess': ['chart_filter']},
    ],
)
def test_disabled_stages_are_refused(tmp_path: Path, stage: dict) -> None:
    raw = {'name': 'test', 'input': {'directory': str(tmp_path)}, 'output': str(tmp_path / 'out')}
    config = RunConfig.from_dict(raw | {'parser': {'name': 'pypdf'}} | stage)
    with pytest.raises(ValueError, match='disabled'):
        Pipeline.from_config(config)


def test_pipeline_logs_failures_and_continues(tmp_path: Path, make_pdf) -> None:
    make_pdf('raw/bnp/good.pdf', [['Fine text']])
    (tmp_path / 'raw/bnp/broken.pdf').write_bytes(b'not a pdf')
    config = _config(tmp_path)
    inventory = Inventory(config.input.directory, tmp_path / 'inventory.db')
    inventory.build()
    items = list(Pipeline.from_config(config).run(inventory.sources()))
    assert [item.source.doc_id for item in items] == ['good']
    statuses = [json.loads(line)['status'] for line in (config.output / 'run.jsonl').read_text().splitlines()]
    assert sorted(statuses) == ['failed', 'ok']


# Disabled with RouteStage (see pipeline/stages.py).
#
# def test_image_only_documents_take_the_override_route(tmp_path: Path, make_pdf) -> None:
#     make_pdf('raw/ss/text.pdf', [['A page with plenty of real text in its text layer for routing.']])
#     make_pdf('raw/ss/scan.pdf', [[], []])
#     config = RunConfig.from_dict(
#         {
#             'name': 'test',
#             'input': {'directory': str(tmp_path / 'raw')},
#             'output': str(tmp_path / 'out'),
#             'parser': {'name': 'pypdf'},
#             'routes': [{'name': 'image_only', 'max_chars_per_page': 20, 'options': {'extraction_mode': 'layout'}}],
#         }
#     )
#     inventory = Inventory(tmp_path / 'raw', tmp_path / 'inventory.db')
#     inventory.build()
#     items = {item.source.doc_id: item for item in Pipeline.from_config(config).run(inventory.sources())}
#     assert items['text'].route == 'default'
#     assert items['scan'].route == 'image_only'
#     assert items['scan'].require_document().parser.config['extraction_mode'] == 'layout'
#     routes = {json.loads(line)['doc_id']: json.loads(line)['route'] for line in (config.output / 'run.jsonl').open()}
#     assert routes == {'text': 'default', 'scan': 'image_only'}


def test_run_resumes_and_refuses_changed_settings_in_the_same_directory(tmp_path: Path, make_pdf) -> None:
    make_pdf('raw/bnp/note.pdf', [['Copper outlook 2025']])
    make_pdf('raw/bnp/other.pdf', [['Gold outlook 2026']])
    config = tmp_path / 'run.yaml'
    settings = {'name': 'p0', 'input': {'directory': str(tmp_path / 'raw')}, 'output': str(tmp_path / 'out')}
    config.write_text(json.dumps(settings | {'parser': {'name': 'pypdf'}}))  # JSON is valid YAML
    log = tmp_path / 'out' / 'run.jsonl'

    assert main(['run', str(config)]) == 0
    assert len(log.read_text().splitlines()) == 2
    assert main(['run', str(config)]) == 0  # resumes: both documents are already parsed
    assert len(log.read_text().splitlines()) == 2

    changed = settings | {'parser': {'name': 'pypdf', 'options': {'extraction_mode': 'layout'}}}
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='other settings'):
        main(['run', str(config)])


def test_device_is_not_a_setting() -> None:
    raw = {'name': 'b0', 'input': {'directory': 'storage'}, 'output': 'out', 'parser': {'name': 'docling'}}
    pinned = raw | {'parser': {'name': 'docling', 'options': {'device': 'cuda:2'}}}
    assert RunConfig.from_dict(pinned).settings == RunConfig.from_dict(raw).settings


def test_dataset_command_writes_the_annotation_dataset(tmp_path: Path, make_pdf, caplog) -> None:
    caplog.set_level('INFO', logger='pipeline')
    make_pdf('raw/bnp/note.pdf', [['Copper outlook 2025 ' * 5]] * 3 + [DISCLOSURE, DISCLOSURE])
    make_pdf('raw/bnp/other.pdf', [['Gold outlook 2026 ' * 5]])
    Inventory(tmp_path / 'raw', tmp_path / 'inventory.db').build(triage=Triage(keep_first=1))

    out = tmp_path / 'annotation'
    args = ['dataset', '--directory', str(tmp_path / 'raw'), '--database', str(tmp_path / 'inventory.db')]
    assert main([*args, '--number', '5', '--out', str(out)]) == 0
    assert sorted(path.suffix for path in (out / 'pdfs').iterdir()) == ['.pdf', '.pdf']
    assert (out / 'manifest.txt').read_text().splitlines() == ['bnp/note.pdf', 'bnp/other.pdf']
    assert 'dataset of 2 documents' in caplog.text and 'length' in caplog.text  # the strata it covers
