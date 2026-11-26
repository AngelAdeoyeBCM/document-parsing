"""Gold-standard annotation: sample PDFs into CVAT with Docling pre-annotations, then collect the ground truth.

1. annotation prepare <run config> --number N   sample, truncate, parquet, pre-annotations, CVAT tasks
2. In CVAT: create a project with <workspace>/cvat/cvat_description.json as raw labels, create one task per
   <workspace>/cvat/cvat_tasks/task_XX/ image folder, upload task_XX_preannotate.xml as its annotations,
   correct boxes, labels and the reading-order line, then export each task's annotations as a zip into
   <workspace>/cvat/cvat_annotations/zips/
3. annotation collect                           the ground-truth parquet that `evaluation score` reads
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pipeline import RunConfig

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

    preparing = commands.add_parser('prepare', help='sample and truncate PDFs, then build the CVAT tasks')
    preparing.add_argument('config', type=Path, help="run config: its input documents and 'triage:' block are used")
    preparing.add_argument('--number', type=int, required=True, help='documents to sample')
    preparing.add_argument('--seed', type=int, default=0)
    preparing.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    preparing.set_defaults(handler=_prepare)

    collecting = commands.add_parser('collect', help='turn the CVAT exports into the ground-truth parquet')
    collecting.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    collecting.set_defaults(handler=_collect)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    workspace = Workspace(args.workspace)
    prepare(RunConfig.load(args.config), workspace, args.number, args.seed)
    logger.info('CVAT tasks are in %s; see `annotation --help` for the steps in CVAT', workspace.cvat)
    return 0


def _collect(args: argparse.Namespace) -> int:
    logger.info('ground truth written to %s', collect(Workspace(args.workspace)))
    return 0
