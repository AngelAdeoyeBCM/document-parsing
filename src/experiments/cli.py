"""Experiment suite for one stage, reviewed in MLflow.

1. ``experiments configs <spec> [E1 ...]`` writes a run config for every variant and prints the parse commands.
2. Run them with ``parsing run <config> [--shard i/n]``, on as many devices as you like.
3. ``experiments log <spec> [E1 ...]`` scores new runs, compares every row with its control and logs each
   experiment to its own MLflow experiment (``stage-<stage>/<id> <title>``), with a stage overview alongside.
4. Review, set ``winner:`` in the spec, and repeat for the next experiment, which builds on that winner.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import yaml

from .spec import Spec

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    return args.handler(args)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='experiments', description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(required=True)

    configs = commands.add_parser('configs', help="write each variant's run config; prints the parse commands")
    configs.add_argument('spec', type=Path, help='e.g. configs/experiments/stage_1_1.yaml')
    configs.add_argument('ids', nargs='*', help='experiment ids (default: all)')
    configs.set_defaults(handler=_configs)

    log = commands.add_parser('log', help='score, compare and log experiments to MLflow')
    log.add_argument('spec', type=Path)
    log.add_argument('ids', nargs='*', help='experiment ids (default: all)')
    log.add_argument('--inventory', type=Path, default=Path('storage/inventory.db'), help="the runs' inventory")
    log.add_argument('--gold', type=Path, default=Path('data/annotation'), help='annotation workspace')
    log.add_argument('--seed', type=int, default=0, help='bootstrap seed')
    log.set_defaults(handler=_log)
    return parser


def _configs(args: argparse.Namespace) -> int:
    spec = Spec.load(args.spec)
    for experiment in _chosen(spec, args.ids):
        for variant in experiment.variants:
            if variant.source:
                continue  # reuses another experiment's run
            arm = spec.variant(experiment, variant.name)
            path = spec.config_path(arm)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(dict(arm.raw), sort_keys=False), encoding='utf-8')
            state = 'parsed' if arm.parsed else 'to parse'
            print(f'{experiment.id} {variant.name} ({state}): parsing run {path.as_posix()}')
    return 0


def _log(args: argparse.Namespace) -> int:
    from .results import evaluate
    from .tracking import log_experiment, log_overview

    spec = Spec.load(args.spec)
    if spec.strata and not spec.strata.is_file():
        logger.warning('no strata file at %s; comparisons are over all pages only', spec.strata)
    score = _Scorer(args.inventory, args.gold)
    for experiment in _chosen(spec, args.ids):
        results = evaluate(spec, experiment, score, seed=args.seed)
        for result in results:
            if result.status != 'ok':
                logger.warning('%s %s: %s (%s)', experiment.id, result.arm.name, result.status, result.arm.run_dir)
            missing = [m for m in experiment.metrics if m not in result.means]
            if result.status == 'ok' and missing:
                logger.warning('%s %s: no %s', experiment.id, result.arm.name, ', '.join(missing))
        log_experiment(spec, experiment, results)
        logger.info('%s: logged %d of %d rows', experiment.id, sum(r.status == 'ok' for r in results), len(results))
    log_overview(spec)
    return 0


class _Scorer:
    """``evaluation.evaluate_run`` with the inventory and gold loaded on first use."""

    def __init__(self, inventory: Path, gold: Path) -> None:
        self.inventory, self.gold = inventory, gold
        self._loaded: tuple | None = None

    def __call__(self, run_dir: Path) -> None:
        from annotation import Workspace, load_gold
        from evaluation import evaluate_run
        from inventory import InventoryStore

        if self._loaded is None:
            if not self.inventory.is_file():
                raise SystemExit(f'no inventory at {self.inventory}; pass --inventory <input directory>/inventory.db')
            self._loaded = (InventoryStore(self.inventory), load_gold(Workspace(self.gold)))
        store, gold = self._loaded
        logger.info('%s: scoring', run_dir)
        evaluate_run(run_dir, store, gold, self.gold.as_posix())


def _chosen(spec: Spec, ids: Sequence[str]):
    return [spec.experiment(i) for i in ids] if ids else list(spec.experiments)
