from __future__ import annotations

import json
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size

import annotation.cvat as cvat_module
from annotation import Workspace, collect, load_gold, prepare
from parsing.core import serialize


def test_tasks_lists_each_cvat_tasks_split(tmp_path: Path, capsys) -> None:
    from annotation.cli import main
    from evaluation import split

    source, truncated = 'ab' * 32, 'cd' * 32  # SHA-256 of the original and of the truncated copy
    workspace = Workspace(tmp_path)
    workspace.documents_file.write_text(
        json.dumps({source: _documents()['source-hash'] | {'truncated_hash': truncated}})
    )
    task = workspace.cvat / 'cvat_tasks' / 'task_00'
    task.mkdir(parents=True)
    for page in (1, 2):
        (task / f'doc_{truncated}_page_{page:06}.png').write_bytes(b'')
    assert main(['tasks', str(tmp_path)]) == 0
    assert capsys.readouterr().out.split() == ['task_00', split(source), '2', 'pages', 'from', '1', 'documents']


def _documents(pages: list[int] | None = None) -> dict[str, dict]:
    """A dataset's documents.json with one document, as `pipeline dataset` writes it."""
    entry = {
        'source': 'bnp/note.pdf',
        'pages': pages or [1, 2],
        'skipped': [3],
        'stratum': {'length': 'short', 'density': 'typical', 'imagery': 'text'},
        'truncated_hash': 'truncated-hash',
    }
    return {'source-hash': entry}


def test_prepare_needs_a_dataset_and_runs_docling_eval_in_order(tmp_path: Path, monkeypatch) -> None:
    workspace = Workspace(tmp_path / 'ws')
    outputs = {
        'PlainFiles': workspace.docling_eval / 'gt_dataset/test',
        'Docling': workspace.docling_eval / 'eval_dataset/test',
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
    with pytest.raises(FileNotFoundError, match='pipeline dataset'):
        prepare(workspace)  # no dataset built in this folder yet
    assert calls == []

    workspace.root.mkdir()
    workspace.documents_file.write_text(json.dumps(_documents()))
    prepare(workspace)
    assert [call[0] for call in calls] == ['create-gt', 'create-eval', 'create-cvat']
    assert calls[0][calls[0].index('--dataset-source') + 1] == str(workspace.pdfs)  # the flat folder of PDFs
    assert calls[2][calls[2].index('--gt-dir') + 1] == str(outputs['Docling'])  # the folder holding the parquet
    assert calls[2][-2:] == ['--sliding-window', '1']

    with pytest.raises(FileNotFoundError, match='zips'):
        collect(workspace)
    workspace.zips.mkdir(parents=True)
    (workspace.zips / 'task_00.zip').touch()
    assert collect(workspace) == outputs['CVAT']
    assert calls[-1][:3] == ['create-gt', '--benchmark', 'CVAT']


def test_load_gold_is_keyed_by_the_source_hash_and_keeps_its_page_numbers(tmp_path: Path) -> None:
    pa, pq = pytest.importorskip('pyarrow'), pytest.importorskip('pyarrow.parquet')
    workspace = Workspace(tmp_path / 'ws')
    workspace.root.mkdir()
    workspace.documents_file.write_text(json.dumps(_documents()))

    truth = DoclingDocument(name='source-hash')
    truth.add_page(page_no=2, size=Size(width=100, height=100))  # a tail-truncated copy keeps the source's numbers
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

    shard('shard_000000_000000.parquet', ['source-hash', 'unknown'])  # named by file stem; one matches nothing
    [(hash_id, document)] = load_gold(workspace).items()
    assert (hash_id, list(document.pages)) == ('source-hash', [2])
    assert serialize(document, 'body') == {2: 'Prices rose.'}

    shard('shard_000000_000001.parquet', ['truncated-hash'])  # named by the truncated file's content hash
    with pytest.raises(ValueError, match='twice'):  # ... which is the same document
        load_gold(workspace)
    (workspace.gold / 'gt_dataset/test/shard_000000_000000.parquet').unlink()
    assert list(load_gold(workspace)) == ['source-hash']
    assert load_gold(Workspace(tmp_path / 'empty')) == {}

    workspace.documents_file.write_text(json.dumps(_documents(pages=[1, 3, 4])))  # not the leading pages
    with pytest.raises(ValueError, match='not the leading pages'):
        load_gold(workspace)
