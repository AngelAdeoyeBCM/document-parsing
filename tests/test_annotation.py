from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DISCLOSURE
from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size
from pypdf import PdfReader

import annotation.cvat as cvat_module
from annotation import Workspace, collect, export_documents, load_gold, prepare
from inventory.inventory import file_hash
from parsing.core import serialize
from parsing.pipeline import RunConfig

TEXT = 'Enough meaningful text on this page to count as an embedded text layer, well past fifty characters'


def _config(tmp_path: Path, make_pdf) -> RunConfig:
    make_pdf('raw/bnp/note.pdf', [[TEXT], [TEXT], [TEXT], DISCLOSURE, DISCLOSURE])
    make_pdf('raw/bnp/short.pdf', [[TEXT]])
    make_pdf('raw/ss/scan.pdf', [[], []])  # no text layer: never sampled
    make_pdf('raw/bnp/unlisted.pdf', [[TEXT]])  # not in the manifest: never sampled
    manifest = tmp_path / 'sample.txt'
    manifest.write_text('bnp/note.pdf\nbnp/short.pdf\nss/scan.pdf\n')
    return RunConfig.from_dict(
        {
            'name': 'b0',
            'input': {'directory': str(tmp_path / 'raw'), 'manifest': str(manifest)},
            'output': str(tmp_path / 'out'),
            'parser': {'name': 'pypdf'},
            'triage': {'policy': 'tail', 'keep_first': 1, 'min_share': 0.3},
        }
    )


def test_export_documents_samples_and_truncates(tmp_path: Path, make_pdf) -> None:
    workspace = Workspace(tmp_path / 'ws')
    documents = export_documents(_config(tmp_path, make_pdf), workspace, number=10)

    assert sorted(documents) == ['note', 'short']
    note = documents['note']
    assert note['pages'] == [1, 2, 3]  # the disclosure tail is gone
    assert len(PdfReader(workspace.sources / 'note.pdf').pages) == 3
    assert note['hash_id'] == file_hash(tmp_path / 'raw/bnp/note.pdf')
    assert note['subset_hash'] == file_hash(workspace.sources / 'note.pdf')
    assert workspace.documents() == documents
    with pytest.raises(FileExistsError):
        export_documents(_config(tmp_path, make_pdf), workspace, number=10)  # never resample a workspace in use


def test_prepare_and_collect_run_docling_eval_in_order(tmp_path: Path, make_pdf, monkeypatch) -> None:
    workspace = Workspace(tmp_path / 'ws')
    outputs = {
        'PlainFiles': workspace.dataset / 'gt_dataset/test',
        'Docling': workspace.dataset / 'eval_dataset/test',
        'CVAT': workspace.gold / 'gt_dataset/test',
    }
    calls: list[list[str]] = []

    def fake_docling_eval(*args: object) -> None:
        calls.append([str(arg) for arg in args])
        for key, directory in outputs.items():
            if key in calls[-1] and not (key == 'PlainFiles' and calls[-1][0] == 'create-eval'):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'shard_000000_000000.parquet').touch()

    monkeypatch.setattr(cvat_module, '_docling_eval', fake_docling_eval)
    prepare(_config(tmp_path, make_pdf), workspace, number=10)
    assert [call[0] for call in calls] == ['create-gt', 'create-eval', 'create-cvat']
    assert calls[0][calls[0].index('--dataset-source') + 1] == str(workspace.sources)
    assert calls[2][calls[2].index('--gt-dir') + 1] == str(outputs['Docling'])  # the folder holding the parquet
    assert calls[2][-2:] == ['--sliding-window', '1']

    with pytest.raises(FileNotFoundError, match='zips'):
        collect(workspace)
    workspace.zips.mkdir(parents=True)
    (workspace.zips / 'task_00.zip').touch()
    assert collect(workspace) == outputs['CVAT']
    assert calls[-1][:3] == ['create-gt', '--benchmark', 'CVAT']


def test_load_gold_renumbers_to_source_pages_and_matches_by_hash(tmp_path: Path) -> None:
    pa, pq = pytest.importorskip('pyarrow'), pytest.importorskip('pyarrow.parquet')
    workspace = Workspace(tmp_path / 'ws')
    workspace.root.mkdir()
    documents = {
        'note': {'source': 'bnp/note.pdf', 'hash_id': 'source-hash', 'subset_hash': 'subset-hash', 'pages': [1, 3, 4]}
    }
    workspace.documents_file.write_text(json.dumps(documents))

    truth = DoclingDocument(name='subset-hash')
    truth.add_page(page_no=2, size=Size(width=100, height=100))  # truncated page 2 is source page 3
    truth.add_text(DocItemLabel.TEXT, 'Prices rose.', prov=ProvenanceItem(
        page_no=2, bbox=BoundingBox(l=0, t=10, r=10, b=0), charspan=(0, 1)
    ))  # fmt: skip

    def shard(name: str, doc_ids: list[str]) -> None:
        path = workspace.gold / 'gt_dataset/test' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        records = {
            'document_id': doc_ids,
            'GroundTruthDocument': [json.dumps(truth.export_to_dict())] * len(doc_ids),
            'BinaryDocument': [b'%PDF'] * len(doc_ids),
        }
        pq.write_table(pa.table(records), path)

    shard('shard_000000_000000.parquet', ['subset-hash', 'unknown'])  # named by hash; one record matches nothing
    [(doc_id, (source_hash, document))] = load_gold(workspace).items()
    assert (doc_id, source_hash, list(document.pages)) == ('note', 'source-hash', [3])
    assert serialize(document, 'body') == {3: 'Prices rose.'}

    shard('shard_000000_000001.parquet', ['note'])
    with pytest.raises(ValueError, match='twice'):
        load_gold(workspace)
    assert load_gold(Workspace(tmp_path / 'empty')) == {}
