from __future__ import annotations

import json
from pathlib import Path

from evaluation import Evaluator, summarise
from inventory import Inventory
from parsing.pipeline import Pipeline, RunConfig, RunManifest

DISCLOSURE = [
    'Important disclosures and analyst certification',
    'This report has been prepared by the research desk',
    'Past performance is not a guide to future returns',
]


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            'name': 'test',
            'input': {'directory': str(tmp_path / 'raw')},
            'output': str(tmp_path / 'out'),
            'parser': {'name': 'pypdf'},
            'triage': {'classifiers': ['rules']},
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
    row = Evaluator().evaluate(document, 'test')
    assert row['coverage.word_recall'] == 1.0
    assert row['numbers.number_recall'] == 1.0
    assert summarise([row])[0]['documents'] == 1


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
    items = {
        item.source.doc_id: item
        for item in Pipeline.from_config(config).run(inventory.sources())
    }
    assert items['text'].route == 'default'
    assert items['scan'].route == 'image_only'
    assert items['scan'].require_document().parser.config['extraction_mode'] == 'layout'
    routes = {json.loads(line)['doc_id']: json.loads(line)['route'] for line in (config.output / 'run.jsonl').open()}
    assert routes == {'text': 'default', 'scan': 'image_only'}
