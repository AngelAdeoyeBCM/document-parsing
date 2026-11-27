"""From a dataset to CVAT and back: docling-eval's datasets, Docling's pre-annotations, the CVAT tasks, the gold."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

from .workspace import Workspace

logger = logging.getLogger(__name__)


def prepare(workspace: Workspace, gpu: str | None = None) -> None:
    """Make the dataset's parquet, Docling's pre-annotations and the CVAT tasks; the dataset must be built.

    ``gpu`` (an nvidia-smi index) is the GPU Docling pre-annotates on; None leaves the choice to Docling.
    """
    if not workspace.documents_file.is_file():
        raise FileNotFoundError(f'no dataset in {workspace.root}; build one first with `pipeline dataset`')
    _docling_eval(
        'create-gt', '--benchmark', 'PlainFiles', '--dataset-source', workspace.pdfs,
        '--output-dir', workspace.docling_eval,
    )  # fmt: skip
    _expect_parquet(workspace.docling_eval / 'gt_dataset' / 'test')
    _docling_eval(
        'create-eval', '--benchmark', 'PlainFiles', '--output-dir', workspace.docling_eval,
        '--prediction-provider', 'Docling', gpu=gpu,
    )  # fmt: skip
    predictions = workspace.docling_eval / 'eval_dataset' / 'test'
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


def _docling_eval(*args: object, gpu: str | None = None) -> None:
    """docling-eval's command line, in this environment, through ``docling_eval_cli`` (see there for why)."""
    if sys.platform == 'win32':
        raise SystemExit(
            'docling-eval does not work on Windows: it stores image references as paths, which Windows writes with '
            "backslashes, then splits them on '/'. Run annotation prepare and collect on the Linux machine."
        )
    if importlib.util.find_spec('docling_eval') is None:
        raise SystemExit('docling-eval is not installed; install the annotation extra: uv sync --all-extras')
    logger.info('docling-eval %s%s', ' '.join(map(str, args)), f' on GPU {gpu}' if gpu is not None else '')
    env = None if gpu is None else {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': gpu}
    subprocess.run([sys.executable, '-m', 'annotation.docling_eval_cli', *map(str, args)], check=True, env=env)


def _expect_parquet(directory: Path) -> Path:
    """docling-eval logs some failures without failing, so check that each step wrote its parquet."""
    if not any(directory.glob('*.parquet')):
        raise RuntimeError(f'docling-eval wrote no parquet to {directory}; see its log above')
    return directory
