from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from inventory import Inventory, PdfSource

from .parsers import available
from .pipeline import InputSpec, Pipeline, RunConfig

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    return args.handler(args)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='parsing', description='Parse research PDFs for RAG ingestion.')
    commands = parser.add_subparsers(required=True)

    run = commands.add_parser('run', help='parse documents with one experiment config')
    run.add_argument('config', type=Path)
    run.add_argument('--shard', default='0/1', help='i/n: process every n-th document starting at i')
    run.set_defaults(handler=_run)

    inventory = commands.add_parser('inventory', help='profile every PDF (pages, text-layer size)')
    inventory.add_argument('--directory', type=Path, required=True)
    inventory.add_argument('--out', type=Path, required=True)
    inventory.set_defaults(handler=_inventory)

    sample = commands.add_parser('sample', help='draw a manifest from a SQLite inventory')
    sample.add_argument('--directory', type=Path, required=True)
    sample.add_argument('--database', type=Path, required=True)
    sample.add_argument('--per-stratum', type=int, default=10)
    sample.add_argument('--seed', type=int, default=0)
    sample.add_argument('--out', type=Path, required=True)
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
    corpus = _sources(config.input)
    pipeline = Pipeline.from_config(config)
    done = sum(1 for _ in pipeline.run(corpus[index::count], corpus=corpus))
    logger.info('%s: %d/%d documents parsed into %s', config.name, done, len(corpus[index::count]), config.output)
    return 0


def _inventory(args: argparse.Namespace) -> int:
    inventory = Inventory(args.directory, args.out)
    profiles = inventory.build()
    logger.info('inventory: profiled %d PDFs into %s', len(profiles), args.out)
    return 0


def _sample(args: argparse.Namespace) -> int:
    inventory = Inventory(args.directory, args.database)
    chosen = inventory.relative_paths()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(f'{relative_path}\n' for relative_path in chosen), encoding='utf-8')
    logger.info('wrote %d documents to %s', len(chosen), args.out)
    return 0


def _sources(spec: InputSpec) -> list[PdfSource]:
    inventory = Inventory(spec.directory, spec.directory / 'inventory.db')
    inventory.build(spec.manifest)
    sources = inventory.sources()
    return sources[: spec.limit] if spec.limit else sources
