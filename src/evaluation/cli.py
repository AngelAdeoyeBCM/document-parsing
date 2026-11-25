"""Stage 1.1 evaluation: score parse runs against the annotated gold, compare them, and log them to MLflow."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from annotation import Workspace, load_gold
from inventory import InventoryStore
from parsing.core import SERIALIZATIONS

from .compare import compare_runs
from .score import DEFAULT_SERIALIZATIONS, evaluate_run, read_csv

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    return args.handler(args)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='evaluation', description=__doc__)
    commands = parser.add_subparsers(required=True)

    score = commands.add_parser('score', help='score parse runs; writes <run>/evaluation/')
    score.add_argument('runs', type=Path, nargs='+', help='run directories, e.g. outputs/docling_b0')
    score.add_argument('--inventory', type=Path, default=Path('storage/inventory.db'), help="the runs' inventory")
    score.add_argument('--gold', type=Path, default=Path('data/annotation'), help='annotation workspace')
    score.add_argument(
        '--serialization',
        action='append',
        choices=sorted(set(SERIALIZATIONS) - {'extracted'}),
        help=f'scored against gold; repeatable (default: {", ".join(DEFAULT_SERIALIZATIONS)})',
    )
    score.set_defaults(handler=_score)

    paired = commands.add_parser('compare', help='paired difference of a run against a base run')
    paired.add_argument('base', type=Path, help='run directory of the current best, e.g. outputs/docling_b0')
    paired.add_argument('other', type=Path)
    paired.add_argument('--metric', action='append', required=True, help='pages.csv column, e.g. body.loss')
    paired.add_argument('--split', choices=('screen', 'holdout', 'all'), default='screen')
    paired.add_argument('--strata', type=Path, help='doc_id,page,tags CSV; adds one comparison per tag')
    paired.add_argument('--seed', type=int, default=0)
    paired.set_defaults(handler=_compare)

    log = commands.add_parser('log', help='log scored runs to MLflow, one MLflow run each')
    log.add_argument('runs', type=Path, nargs='+')
    log.add_argument('--experiment', default='stage-1.1')
    log.set_defaults(handler=_log)
    return parser


def _score(args: argparse.Namespace) -> int:
    if not args.inventory.is_file():
        raise SystemExit(f'no inventory at {args.inventory}; pass --inventory <input directory>/inventory.db')
    store = InventoryStore(args.inventory)
    gold = load_gold(Workspace(args.gold))
    logger.info('gold: %d documents from %s', len(gold), args.gold)
    for run in args.runs:
        serializations = args.serialization or DEFAULT_SERIALIZATIONS
        evaluation = evaluate_run(run, store, gold, args.gold.as_posix(), serializations)
        logger.info('%s: scored into %s', run.name, evaluation)
    return 0


def _compare(args: argparse.Namespace) -> int:
    split = None if args.split == 'all' else args.split
    path = compare_runs(args.base, args.other, args.metric, split, args.strata, args.seed)
    for r in read_csv(path):
        logger.info(
            '%s [%s]: %s - %s = %+.4f (%+.4f, %+.4f) over %s documents',
            r['metric'], r['stratum'], args.other.name, args.base.name,
            float(r['diff']), float(r['low']), float(r['high']), r['documents'],
        )  # fmt: skip
    logger.info('written to %s', path)
    return 0


def _log(args: argparse.Namespace) -> int:
    from .tracking import log_run

    for run in args.runs:
        logger.info('%s: logged as MLflow run %s', run.name, log_run(run, args.experiment))
    return 0
