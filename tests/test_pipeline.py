from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DISCLOSURE

from evaluation import evaluate_run
from evaluation.score import read_csv
from inventory import Inventory
from parsing.cli import main
from parsing.pipeline import Pipeline, RunConfig, RunManifest, stages
from parsing.pipeline.components import build_triage


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            'name': 'test',
            'input': {'directory': str(tmp_path / 'raw')},
            'output': str(tmp_path / 'out'),
            'parser': {'name': 'pypdf'},
            'triage': {'policy': 'tail', 'keep_first': 1, 'min_share': 0.3},
        }
    )


def test_pipeline_skips_boilerplate_and_writes_outputs(tmp_path: Path, make_pdf) -> None:
    make_pdf('raw/bnp/note.pdf', [['Copper outlook 2025', 'Prices rise 4.5% on supply'], ['More analysis'], DISCLOSURE])
    config = _config(tmp_path)
    inventory = Inventory(config.input.directory, tmp_path / 'inventory.db')
    inventory.build()
    sources = inventory.sources()
    assert sources[0].relative_path == 'bnp/note.pdf'

    items = list(Pipeline.from_config(config).run(sources))
    assert items[0].skip == frozenset({3})
    assert {path.suffix for path in items[0].outputs} == {'.json', '.html'}

    log = [json.loads(line) for line in (config.output / 'run.jsonl').read_text().splitlines()]
    assert log[0]['status'] == 'ok' and log[0]['skipped_pages'] == [3]

    [document] = RunManifest(config.output).documents()
    assert document.skipped_pages == frozenset({3})
    assert document.source.relative_path == 'bnp/note.pdf'
    proof = (config.output / 'note.html').read_text()
    assert proof.count('class="page"') == 3 and 'skipped by triage' in proof

    truth = document.document.model_copy(deep=True)
    truth.pages = {1: truth.pages[1]}  # ground truth holds only the annotated pages
    evaluation = evaluate_run(config.output, inventory.store, {'note': (document.source.hash_id, truth)})
    pages = {row['page']: row for row in read_csv(evaluation / 'pages.csv')}
    assert sorted(pages) == ['1', '2']  # page 3 was triaged away
    assert float(pages['1']['extracted.dropped_text_rate']) == 0.0  # pypdf against its own stored text
    assert float(pages['1']['extracted.garble_rate.text']) == 0.0  # pypdf output is one text item per page
    assert float(pages['1']['pypdf_seconds_per_page']) > 0  # the inventory's pypdf time, for comparison
    assert float(pages['1']['body.loss']) == 0.0 and pages['2']['body.loss'] == ''  # page 2 not annotated
    params = json.loads((evaluation / 'params.json').read_text())
    assert (params['parser'], params['documents'], params['gold_pages']) == ('pypdf', 1, 1)
    assert json.loads((evaluation / 'metrics.json').read_text())['body.loss'] == 0.0

    markdown = evaluate_run(
        config.output, inventory.store, {'note': (document.source.hash_id, truth)}, serializations=['markdown']
    )
    assert float(read_csv(markdown / 'pages.csv')[0]['markdown.loss']) == 0.0  # any named serialization, gold alike

    other_file = evaluate_run(config.output, inventory.store, {'note': ('not this file', truth)})
    assert json.loads((other_file / 'params.json').read_text())['gold_pages'] == 0  # gold from another file


def test_pipeline_reuses_inventory_text_and_triage(tmp_path: Path, make_pdf, monkeypatch) -> None:
    make_pdf('raw/bnp/note.pdf', [['Copper outlook 2025'], ['More analysis'], DISCLOSURE])
    make_pdf('raw/bnp/other.pdf', [['Not in the manifest']])
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text('bnp/note.pdf\n')
    config = _config(tmp_path)
    inventory = Inventory(config.input.directory, tmp_path / 'inventory.db')
    inventory.build(manifest, triage=build_triage(config.triage))

    def no_pypdf(*_args, **_kwargs):
        raise AssertionError('the stages must reuse the inventory instead of reading the PDF')

    monkeypatch.setattr(stages, 'PdfReader', no_pypdf)
    pipeline = Pipeline.from_config(config, store=inventory.store)
    monkeypatch.setattr(pipeline.stages[1].classifier, 'classify', no_pypdf)  # verdicts come from the store
    items = list(pipeline.run(inventory.sources(manifest)))

    assert [item.source.doc_id for item in items] == ['note']
    assert items[0].skip == frozenset({3})
    assert items[0].page_texts[0].startswith('Copper outlook 2025')


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


def test_image_only_documents_take_the_override_route(tmp_path: Path, make_pdf) -> None:
    make_pdf('raw/ss/text.pdf', [['A page with plenty of real text in its text layer for routing.']])
    make_pdf('raw/ss/scan.pdf', [[], []])
    config = RunConfig.from_dict(
        {
            'name': 'test',
            'input': {'directory': str(tmp_path / 'raw')},
            'output': str(tmp_path / 'out'),
            'parser': {'name': 'pypdf'},
            'routes': [{'name': 'image_only', 'max_chars_per_page': 20, 'options': {'extraction_mode': 'layout'}}],
        }
    )
    inventory = Inventory(tmp_path / 'raw', tmp_path / 'inventory.db')
    inventory.build()
    items = {item.source.doc_id: item for item in Pipeline.from_config(config).run(inventory.sources())}
    assert items['text'].route == 'default'
    assert items['scan'].route == 'image_only'
    assert items['scan'].require_document().parser.config['extraction_mode'] == 'layout'
    routes = {json.loads(line)['doc_id']: json.loads(line)['route'] for line in (config.output / 'run.jsonl').open()}
    assert routes == {'text': 'default', 'scan': 'image_only'}


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
