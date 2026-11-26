from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from inventory import Inventory, PdfSource
from parsing.parsers import available

from .components import build_triage
from .config import RunConfig
from .manifest import RunManifest
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    return args.handler(args)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='pipeline', description='Parse research PDFs for RAG ingestion.')
    commands = parser.add_subparsers(required=True)

    run = commands.add_parser('run', help='parse documents with one experiment config')
    run.add_argument('config', type=Path)
    run.add_argument('--shard', default='0/1', help='i/n: process every n-th document starting at i')
    run.set_defaults(handler=_run)

    inventory = commands.add_parser('inventory', help='profile every PDF (pages, text-layer size)')
    inventory.add_argument('--directory', type=Path, required=True)
    inventory.add_argument('--out', type=Path, required=True)
    inventory.add_argument('--config', type=Path, help="run config whose 'triage:' block to apply while profiling")
    inventory.set_defaults(handler=_inventory)

    sample = commands.add_parser('sample', help='draw a manifest from a SQLite inventory')
    sample.add_argument('--directory', type=Path, required=True)
    sample.add_argument('--database', type=Path, required=True)
    sample.add_argument(
        '--per-stratum', type=int, default=None, help='legacy per-stratum limit; ignored when --number is set'
    )
    sample.add_argument('--seed', type=int, default=0)
    sample.add_argument('--out', type=Path, required=True)
    sample.add_argument(
        '--number', '--count', dest='number', type=int, default=50, help='total number of full PDFs to sample'
    )
    sample.set_defaults(handler=_sample)

    parsers = commands.add_parser('parsers', help='list available parsers')
    parsers.set_defaults(handler=_parsers)
    return parser


def _parsers(_: argparse.Namespace) -> int:
    print('\n'.join(available()))
    return 0


def _run(args: argparse.Namespace) -> int:
    config = RunConfig.load(args.config)
    index, count = (int(part) for part in args.shard.split('/'))
    inventory = Inventory(config.input.directory, config.input.directory / 'inventory.db')
    corpus = _sources(inventory, config)
    manifest = RunManifest(config.output)
    manifest.claim(config.settings)  # refuses a directory that holds another config's outputs
    done = manifest.completed()
    todo = [source for source in corpus[index::count] if source.hash_id not in done]
    pipeline = Pipeline.from_config(config, store=inventory.store)
    parsed = sum(1 for _ in pipeline.run(todo, corpus=corpus))
    logger.info(
        '%s: %d parsed, %d already done, %d failed, into %s',
        config.name, parsed, len(corpus[index::count]) - len(todo), len(todo) - parsed, config.output,
    )  # fmt: skip
    return 0


def _inventory(args: argparse.Namespace) -> int:
    config = RunConfig.load(args.config) if args.config else None
    triage = build_triage(config.triage) if config and config.triage else None
    report = Inventory(args.directory, args.out).build(triage=triage)
    logger.info('inventory: %s into %s', report, args.out)
    return 0


def _sample(args: argparse.Namespace) -> int:
    inventory = Inventory(args.directory, args.database)
    chosen = inventory.sample(
        number=args.number if args.per_stratum is None else None,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(f'{relative_path}\n' for relative_path in chosen), encoding='utf-8')
    logger.info('wrote %d documents to %s', len(chosen), args.out)
    return 0


def _sources(inventory: Inventory, config: RunConfig) -> list[PdfSource]:
    spec = config.input
    report = inventory.build(spec.manifest, triage=build_triage(config.triage) if config.triage else None)
    logger.info('inventory: %s', report)
    sources = inventory.sources(spec.manifest)
    return sources[: spec.limit] if spec.limit else sources
