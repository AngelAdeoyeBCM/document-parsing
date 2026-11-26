"""From the inventory to CVAT: sample, truncate, then build the docling-eval datasets and the CVAT tasks."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from inventory import Inventory, truncate_pdf
from inventory.inventory import file_hash
from pipeline import RunConfig
from pipeline.components import build_triage

from .workspace import Workspace

logger = logging.getLogger(__name__)


def prepare(config: RunConfig, workspace: Workspace, number: int, seed: int = 0) -> None:
    """Sample and truncate the documents, then make the parquet, Docling's pre-annotations and the CVAT tasks."""
    export_documents(config, workspace, number, seed)
    _docling_eval(
        'create-gt', '--benchmark', 'PlainFiles', '--dataset-source', workspace.sources, '--output-dir', workspace.dataset
    )  # fmt: skip
    _expect_parquet(workspace.dataset / 'gt_dataset' / 'test')
    _docling_eval(
        'create-eval', '--benchmark', 'PlainFiles', '--output-dir', workspace.dataset, '--prediction-provider', 'Docling'
    )  # fmt: skip
    predictions = workspace.dataset / 'eval_dataset' / 'test'
    _expect_parquet(predictions)
    _docling_eval(
        'create-cvat', '--output-dir', workspace.cvat, '--gt-dir', predictions,
        '--use-predictions', '--sliding-window', '1',  # one page per CVAT image
    )  # fmt: skip


def collect(workspace: Workspace) -> Path:
    """Turn the CVAT exports in ``cvat/cvat_annotations/zips`` into the ground-truth parquet; returns its folder."""
    if not any(workspace.zips.glob('*.zip')):
        raise FileNotFoundError(f'no CVAT export in {workspace.zips}; export the annotations as a zip and put it there')
    _docling_eval(
        'create-gt', '--benchmark', 'CVAT', '--dataset-source', workspace.cvat, '--output-dir', workspace.gold
    )
    return _expect_parquet(workspace.gold / 'gt_dataset' / 'test')


def export_documents(config: RunConfig, workspace: Workspace, number: int, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Sample ``number`` documents with a fully embedded text layer and write each without its triaged pages.

    Documents come from the config's manifest when it has one, so they are among those the parse runs cover.
    """
    if workspace.documents_file.exists():
        raise FileExistsError(f'{workspace.root} is already prepared; use a new workspace for a new sample')
    inventory = Inventory(config.input.directory, config.input.directory / 'inventory.db')
    triage = build_triage(config.triage) if config.triage else None
    inventory.build(config.input.manifest, triage=triage)
    candidates = inventory.sources(config.input.manifest)
    within = {source.relative_path for source in candidates} if config.input.manifest else None
    chosen = set(
        inventory.sample(number=number, seed=seed, text_quality=('embedded_text',), include_ocr=False, within=within)
    )

    workspace.sources.mkdir(parents=True, exist_ok=True)
    documents: dict[str, dict[str, Any]] = {}
    for source in candidates:
        if source.relative_path not in chosen:
            continue
        if source.doc_id in documents:
            raise ValueError(f'{source.relative_path}: another sampled document is also named {source.doc_id!r}')
        stored = inventory.store.triage(source.hash_id, triage.key) if triage else None
        target = workspace.sources / f'{source.doc_id}.pdf'
        kept = truncate_pdf(source.path, stored[1] if stored else frozenset(), target)
        if not kept:
            target.unlink()
            continue
        documents[source.doc_id] = {
            'source': source.relative_path,
            'hash_id': source.hash_id,
            'subset_hash': file_hash(target),
            'pages': kept,
        }
    workspace.documents_file.write_text(json.dumps(documents, indent=1), encoding='utf-8')
    pages = sum(len(document['pages']) for document in documents.values())
    logger.info('%d documents, %d pages to annotate, in %s', len(documents), pages, workspace.sources)
    return documents


def _docling_eval(*args: object) -> None:
    executable = shutil.which('docling-eval')
    if executable is None:
        raise SystemExit('docling-eval is not installed; install the annotation extra: uv sync --extra annotation')
    logger.info('docling-eval %s', ' '.join(map(str, args)))
    subprocess.run([executable, *map(str, args)], check=True)


def _expect_parquet(directory: Path) -> Path:
    """docling-eval logs some failures without failing, so check that each step wrote its parquet."""
    if not any(directory.glob('*.parquet')):
        raise RuntimeError(f'docling-eval wrote no parquet to {directory}; see its log above')
    return directory
