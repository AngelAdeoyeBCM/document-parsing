"""Experiment suite for one stage, reviewed in MLflow.

1. ``experiments configs <spec> [E1 ...]`` writes a run config for every variant and prints the parse commands.
2. ``experiments run <spec> [E1 ...]`` parses the base config, references and variants, one run per free GPU at a
   time (or run ``pipeline run <config> [--shard i/n]`` yourself, choosing the GPU with ``CUDA_VISIBLE_DEVICES``;
   without it Docling takes the first GPU, busy or not).
3. ``experiments log <spec> [E1 ...]`` scores new runs, compares every row with its control and logs each
   experiment to its own MLflow experiment (``stage-<stage>/<id> <title>``), with a stage overview alongside.
   Before the gold is collected only the gold-free metrics are filled in; once ``annotation collect`` writes or
   updates it, the next ``log`` scores every run again.
4. Review and set ``winner:`` in the spec. An experiment with ``control: base`` never waits for a winner; one with
   ``control: <id>`` is skipped until that experiment's winner is set.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, SimpleQueue

import yaml

from .device import free_gpus
from .spec import NoWinnerYet, Spec

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

    run = commands.add_parser('run', help='parse every run the experiments compare, one per free GPU at a time')
    run.add_argument('spec', type=Path)
    run.add_argument('ids', nargs='*', help='experiment ids (default: all)')
    run.add_argument('--gpus', help='comma-separated GPU indices as nvidia-smi numbers them (default: every free one)')
    run.add_argument('--min-free', type=float, default=0.5, help='share of its memory a GPU needs free to be used')
    run.set_defaults(handler=_run)

    log = commands.add_parser('log', help='score, compare and log experiments to MLflow')
    log.add_argument('spec', type=Path)
    log.add_argument('ids', nargs='*', help='experiment ids (default: all)')
    log.add_argument('--inventory', type=Path, default=Path('storage/inventory.db'), help="the runs' inventory")
    log.add_argument('--gold', type=Path, default=Path('data/annotation'), help='annotation workspace')
    log.add_argument('--seed', type=int, default=0, help='bootstrap seed')
    log.set_defaults(handler=_log)
    return parser


def _configs(args: argparse.Namespace) -> int:
    _write_configs(Spec.load(args.spec), args.ids)
    return 0


def _write_configs(spec: Spec, ids: Sequence[str]) -> list[Path]:
    """Write each variant's run config, print its parse command and return the configs."""
    paths = []
    for experiment in _ready(spec, ids):
        for variant in experiment.variants:
            if variant.source:
                continue  # reuses another experiment's run
            arm = spec.variant(experiment, variant.name)
            path = spec.config_path(arm)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(dict(arm.raw), sort_keys=False), encoding='utf-8')
            state = 'parsed' if arm.parsed else 'to parse'
            print(f'{experiment.id} {variant.name} ({state}): pipeline run {path.as_posix()}')
            paths.append(path)
    return paths


def _run(args: argparse.Namespace) -> int:
    spec = Spec.load(args.spec)
    # Finished runs return at once: `pipeline run` only parses the documents a run directory lacks.
    configs = list(dict.fromkeys([spec.base_config, *spec.references, *_write_configs(spec, args.ids)]))
    gpus = args.gpus.split(',') if args.gpus else [str(index) for index in free_gpus(args.min_free)]
    if not gpus:
        raise SystemExit(f'no GPU has {args.min_free:.0%} of its memory free; pass --gpus or a lower --min-free')
    logs = spec.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    queue: SimpleQueue[Path] = SimpleQueue()
    for config in configs:
        queue.put(config)
    failed: list[Path] = []

    def work(gpu: str) -> None:
        env = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': gpu}
        while True:
            try:
                config = queue.get_nowait()
            except Empty:
                return
            log = logs / f'{config.stem}.log'
            logger.info('GPU %s: %s (log %s)', gpu, config, log)
            with open(log, 'a', encoding='utf-8') as handle:
                command = [sys.executable, '-m', 'pipeline', 'run', str(config)]
                run = subprocess.run(command, check=False, env=env, stdout=handle, stderr=subprocess.STDOUT)
            code = run.returncode
            if code:
                failed.append(config)
                logger.error('GPU %s: %s exited with %d, see %s', gpu, config, code, log)

    logger.info('%d runs over GPUs %s', len(configs), ', '.join(gpus))
    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(work, gpus))
    logger.info('%d of %d runs finished', len(configs) - len(failed), len(configs))
    return 1 if failed else 0


def _log(args: argparse.Namespace) -> int:
    from .results import evaluate
    from .tracking import log_experiment, log_overview

    spec = Spec.load(args.spec)
    if spec.strata and not spec.strata.is_file():
        logger.warning('no strata file at %s; comparisons are over all pages only', spec.strata)
    score = _Scorer(args.inventory, args.gold)
    gold_time = _gold_time(args.gold)
    if not gold_time:
        logger.info('no gold in %s yet: logging the gold-free metrics only', args.gold)
    for experiment in _ready(spec, args.ids):
        results = evaluate(spec, experiment, score, seed=args.seed, gold_time=gold_time)
        for result in results:
            if result.status != 'ok':
                logger.warning('%s %s: %s (%s)', experiment.id, result.arm.name, result.status, result.arm.run_dir)
            expected = [m for m in experiment.metrics if gold_time or not spec.info(m).gold]
            missing = [m for m in expected if m not in result.means]
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
            workspace = Workspace(self.gold)
            self._loaded = (InventoryStore(self.inventory), load_gold(workspace), workspace.pages())
        store, gold, pages = self._loaded
        logger.info('%s: scoring', run_dir)
        evaluate_run(run_dir, store, gold, self.gold.as_posix(), pages=pages)


def _gold_time(workspace: Path) -> float:
    """When the workspace's gold last changed (`annotation collect`); 0 when there is none yet."""
    from annotation import Workspace

    return max((path.stat().st_mtime for path in Workspace(workspace).gold.rglob('*.parquet')), default=0.0)


def _ready(spec: Spec, ids: Sequence[str]):
    """The chosen experiments (default: all) with variants and a settled control; the rest are logged as waiting."""
    ready = []
    for experiment in [spec.experiment(i) for i in ids] if ids else spec.experiments:
        if not experiment.variants:
            logger.info('skipped: %s has no variants yet', experiment.id)
            continue
        try:
            spec.control(experiment)
        except NoWinnerYet as waiting:
            logger.info('skipped: %s', waiting)
            continue
        ready.append(experiment)
    return ready
