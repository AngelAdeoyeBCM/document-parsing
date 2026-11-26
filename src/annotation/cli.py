"""Gold-standard annotation of a dataset: CVAT tasks with Docling pre-annotations, then the ground truth.

0. pipeline dataset --directory <corpus> --database <inventory> --number N   sample and truncate the PDFs into
   data/annotation (pdfs/<hash_id>.pdf, documents.json, manifest.txt for the parse runs)
1. annotation prepare [dataset]   docling-eval parquet, Docling's pre-annotations, CVAT tasks
2. In CVAT: create a project with <dataset>/cvat/cvat_description.json as raw labels, create one task per
   <dataset>/cvat/cvat_tasks/task_XX/ image folder, upload task_XX_preannotate.xml as its annotations,
   correct boxes, labels and the reading-order line, then export each task's annotations as a zip into
   <dataset>/cvat/cvat_annotations/zips/
3. annotation collect [dataset]   the ground-truth parquet that `evaluation score` reads
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .cvat import collect, prepare
from .workspace import Workspace

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path('data/annotation')


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    return args.handler(args)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='annotation', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)

    preparing = commands.add_parser('prepare', help="build the dataset's CVAT tasks with Docling's pre-annotations")
    preparing.add_argument(
        'dataset', type=Path, nargs='?', default=DEFAULT_WORKSPACE, help='built by `pipeline dataset`'
    )
    preparing.set_defaults(handler=_prepare)

    collecting = commands.add_parser('collect', help='turn the CVAT exports into the ground-truth parquet')
    collecting.add_argument('dataset', type=Path, nargs='?', default=DEFAULT_WORKSPACE)
    collecting.set_defaults(handler=_collect)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    workspace = Workspace(args.dataset)
    prepare(workspace)
    logger.info('CVAT tasks are in %s; see `annotation --help` for the steps in CVAT', workspace.cvat)
    return 0


def _collect(args: argparse.Namespace) -> int:
    logger.info('ground truth written to %s', collect(Workspace(args.dataset)))
    return 0
