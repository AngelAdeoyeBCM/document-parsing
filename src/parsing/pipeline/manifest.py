from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType

from docling_core.types.doc import DoclingDocument
from inventory import PdfSource

from ..core import ParsedDocument, ParserInfo
from .stages import WorkItem


class RunManifest:
    """``run.jsonl`` in a run directory: one line per document with what Docling's JSON does not hold."""

    filename = 'run.jsonl'

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / self.filename

    def record_ok(self, item: WorkItem) -> None:
        document = item.require_document()
        parser = document.parser
        self._append(
            item.source,
            status='ok',
            route=item.route,
            parser={'name': parser.name, 'version': parser.version, 'config': dict(parser.config)},
            seconds=document.seconds,
            skipped_pages=sorted(document.skipped_pages),
            triage={verdict.page: verdict.reason for verdict in item.verdicts if verdict.boilerplate},
            postprocess=dict(item.notes),
            outputs=[path.name for path in item.outputs],
        )

    def record_failure(self, source: PdfSource, error: BaseException) -> None:
        self._append(source, status='failed', error=f'{type(error).__name__}: {error}')

    def documents(self) -> Iterator[ParsedDocument]:
        """Reload every successfully parsed document; the latest record wins when a document was re-run."""
        records: dict[str, dict] = {}
        with open(self.path, encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record['status'] == 'ok':
                    records[record['doc_id']] = record
        for record in records.values():
            yield self._load(record)

    def _load(self, record: dict) -> ParsedDocument:
        source = PdfSource.from_path(record['path'], **record['metadata'])
        if source.hash_id != record['hash_id']:
            raise ValueError(f'{source.doc_id}: {source.path} has changed since it was parsed')
        json_name = next(name for name in record['outputs'] if name.endswith('.json'))
        parser = record['parser']
        return ParsedDocument(
            source=source,
            parser=ParserInfo(parser['name'], parser['version'], MappingProxyType(parser['config'])),
            # Our own run outputs, so their referenced figure files are trusted.
            document=DoclingDocument.load_from_json(self.run_dir / json_name, allow_local_images=True),
            skipped_pages=frozenset(record['skipped_pages']),
            seconds=record['seconds'],
        )

    def _append(self, source: PdfSource, **fields: object) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            'doc_id': source.doc_id,
            'hash_id': source.hash_id,
            'path': source.path.as_posix(),
            'metadata': dict(source.metadata),
            **fields,
        }
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
