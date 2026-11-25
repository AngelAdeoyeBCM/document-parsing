from __future__ import annotations

import json
from pathlib import Path

from evaluation import Evaluator, summarise
from inventory import Inventory
from parsing.pipeline import Pipeline, RunConfig, RunManifest, stages
from parsing.pipeline.components import build_triage

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
            'triage': {'policy': 'tail', 'keep_first': 1, 'min_hits': 3},
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
    items = {
        item.source.doc_id: item
        for item in Pipeline.from_config(config).run(inventory.sources())
    }
    assert items['text'].route == 'default'
    assert items['scan'].route == 'image_only'
    assert items['scan'].require_document().parser.config['extraction_mode'] == 'layout'
    routes = {json.loads(line)['doc_id']: json.loads(line)['route'] for line in (config.output / 'run.jsonl').open()}
    assert routes == {'text': 'default', 'scan': 'image_only'}
