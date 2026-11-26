from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DISCLOSURE
from pypdf import PdfReader

from inventory import Dataset, Inventory, build_dataset
from inventory.inventory import file_hash
from triage import Triage

TEXT = 'Enough meaningful text on this page to count as an embedded text layer, well past fifty characters'


def _inventory(tmp_path: Path, make_pdf) -> Inventory:
    make_pdf('raw/bnp/note.pdf', [[TEXT], [TEXT], [TEXT], DISCLOSURE, DISCLOSURE])
    make_pdf('raw/bnp/short.pdf', [[TEXT]])
    make_pdf('raw/ss/scan.pdf', [[], []])  # no text layer: never sampled for annotation
    inventory = Inventory(tmp_path / 'raw', tmp_path / 'inventory.db')
    inventory.build(triage=Triage(keep_first=1))
    return inventory


def test_dataset_holds_the_sampled_pdfs_truncated_and_named_by_source_hash(tmp_path: Path, make_pdf) -> None:
    inventory = _inventory(tmp_path, make_pdf)
    dataset = build_dataset(inventory, tmp_path / 'ds', 10)
    note, short = (file_hash(tmp_path / 'raw' / path) for path in ('bnp/note.pdf', 'bnp/short.pdf'))

    # The flat folder docling-eval's PlainFiles reads: only PDFs, each named by the original's content hash.
    assert sorted(path.name for path in dataset.pdfs.iterdir()) == sorted([f'{note}.pdf', f'{short}.pdf'])
    assert len(PdfReader(dataset.pdfs / f'{note}.pdf').pages) == 3  # the disclosure tail is gone

    documents = dataset.documents()
    assert sorted(documents) == sorted([note, short])
    assert documents[note]['source'] == 'bnp/note.pdf'
    assert (documents[note]['pages'], documents[note]['skipped']) == ([1, 2, 3], [4, 5])
    assert documents[note]['truncated_hash'] == file_hash(dataset.pdfs / f'{note}.pdf')  # docling-eval's doc_hash
    assert set(documents[note]['stratum']) == {'length', 'density', 'imagery'}
    assert dataset.manifest_file.read_text().splitlines() == ['bnp/note.pdf', 'bnp/short.pdf']  # for the parse runs
    assert json.loads(dataset.settings_file.read_text())['number'] == 10

    with pytest.raises(FileExistsError):
        build_dataset(inventory, dataset.root, 10)  # never resample a dataset in use
    assert Dataset(tmp_path / 'other').documents() == {}


def test_dataset_samples_deterministically(tmp_path: Path, make_pdf) -> None:
    inventory = _inventory(tmp_path, make_pdf)
    first = build_dataset(inventory, tmp_path / 'a', 1, seed=3).documents()
    assert len(first) == 1
    assert build_dataset(inventory, tmp_path / 'b', 1, seed=3).documents() == first
