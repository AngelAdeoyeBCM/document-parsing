from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

from parsing.core import ParsedDocument
from parsing.pipeline import RunManifest
from parsing.render import ProofRenderer

from .client import VlmClient
from .evaluator import Evaluator, summarise, write_csv
from .gold import load_gold
from .gold import score as score_gold
from .judge import PairwiseJudge

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    return args.handler(args)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='evaluation', description='Score and compare parser runs.')
    commands = parser.add_subparsers(required=True)

    score = commands.add_parser('score', help='automatic metrics for one or more run directories')
    score.add_argument('runs', type=Path, nargs='+')
    score.add_argument('--out', type=Path, required=True, help='per-document CSV; a _summary CSV is written beside it')
    score.set_defaults(handler=_score)

    gold = commands.add_parser('gold', help='accuracy against hand-checked transcriptions, e.g. for scanned PDFs')
    gold.add_argument('runs', type=Path, nargs='+')
    gold.add_argument('--gold', type=Path, default=Path('data/gold'), help='<doc_id>/p<N>.txt and p<N>_table.tsv')
    gold.add_argument('--out', type=Path, required=True)
    gold.set_defaults(handler=_gold)

    judge = commands.add_parser('judge', help='pairwise VLM judgement of two runs on sampled pages')
    judge.add_argument('first', type=Path)
    judge.add_argument('second', type=Path)
    judge.add_argument('--base-url', required=True, help='OpenAI-compatible endpoint, e.g. http://localhost:8000/v1')
    judge.add_argument('--model', required=True)
    judge.add_argument('--documents', type=int, default=50)
    judge.add_argument('--pages-per-document', type=int, default=2)
    judge.add_argument('--seed', type=int, default=0)
    judge.add_argument('--out', type=Path, required=True)
    judge.set_defaults(handler=_judge)

    proof = commands.add_parser('proof', help='(re)build the side-by-side proof HTML for an existing run')
    proof.add_argument('run', type=Path)
    proof.add_argument('--doc', action='append', help='doc_id to render; repeatable; default all')
    proof.add_argument('--out', type=Path, help='default: the run directory')
    proof.add_argument('--scale', type=float, default=1.5)
    proof.add_argument('--zoom', type=float, default=0.5, help='page display size; 1.0 = browser PDF viewer at 100%%')
    proof.add_argument('--overlays', action='store_true', help='also save Docling layout overlays as PNGs')
    proof.add_argument('--reading-order', action='store_true', help='draw reading-order arrows on the overlays')
    proof.set_defaults(handler=_proof)
    return parser


def _documents(run: Path) -> Iterator[ParsedDocument]:
    return RunManifest(run).documents()


def _score(args: argparse.Namespace) -> int:
    evaluator = Evaluator()
    rows = [evaluator.evaluate(document, run.name) for run in args.runs for document in _documents(run)]
    write_csv(rows, args.out)
    write_csv(summarise(rows), args.out.with_name(f'{args.out.stem}_summary.csv'))
    logger.info('scored %d documents across %d runs', len(rows), len(args.runs))
    return 0


def _gold(args: argparse.Namespace) -> int:
    rows = [
        score_gold(document, load_gold(args.gold / document.source.doc_id), run.name)
        for run in args.runs
        for document in _documents(run)
        if (args.gold / document.source.doc_id).is_dir()
    ]
    write_csv(rows, args.out)
    logger.info('scored %d documents with gold pages across %d runs', len(rows), len(args.runs))
    return 0


def _judge(args: argparse.Namespace) -> int:
    first = {document.source.hash_id: document for document in _documents(args.first)}
    second = {document.source.hash_id: document for document in _documents(args.second)}
    rng = random.Random(args.seed)
    shared = sorted(first.keys() & second.keys())
    judge = PairwiseJudge(VlmClient(args.base_url, args.model))
    wins: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        for key in rng.sample(shared, min(args.documents, len(shared))):
            pages = sorted(first[key].document.pages.keys() | second[key].document.pages.keys())
            for page in rng.sample(pages, min(args.pages_per_document, len(pages))):
                verdict = judge.judge((args.first.name, first[key]), (args.second.name, second[key]), page)
                wins[verdict.winner] += 1
                handle.write(json.dumps(asdict(verdict)) + '\n')
    logger.info('wins: %s', dict(wins))
    return 0


def _proof(args: argparse.Namespace) -> int:
    renderer = ProofRenderer(scale=args.scale, zoom=args.zoom, overlays=args.overlays, reading_order=args.reading_order)
    out_dir = args.out or args.run
    wanted = set(args.doc or ())
    for parsed in _documents(args.run):
        if not wanted or parsed.source.doc_id in wanted:
            logger.info('%s: %s', parsed.source.doc_id, renderer.write(parsed, out_dir))
    return 0
