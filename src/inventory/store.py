from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from collections.abc import Collection, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

from .profiler import PageProfile, PDFProfile, ProfileResult
from triage import PageVerdict

# Bump on any schema change; an inventory with another version is dropped and rebuilt, since it is only a cache.
SCHEMA_VERSION = 3  # 3: pypdf_seconds on documents and pages

_TABLES = (
    'experiment_documents',
    'experiments',
    'triage_verdicts',
    'page_text',
    'pages',
    'profile_errors',
    'files',
    'documents',
)
_SQL_TYPES = {'int': 'INTEGER', 'float': 'REAL', 'str': 'TEXT', 'bool': 'INTEGER'}  # annotations are strings here
_PROFILE_FIELDS = tuple(field.name for field in fields(PDFProfile))
_PAGE_FIELDS = tuple(field.name for field in fields(PageProfile))


def _columns(cls: type) -> str:
    return ',\n'.join(f'{field.name} {_SQL_TYPES[field.type]} NOT NULL' for field in fields(cls))


def _from_row(cls: type, row: Iterable[object]) -> Any:
    """Rebuild a profile dataclass from its columns, turning SQLite's 0/1 back into bools."""
    return cls(
        **{
            field.name: bool(value) if field.type == 'bool' else value
            for field, value in zip(fields(cls), row, strict=True)
        }
    )


def where(text_quality: Iterable[str] | None = None, do_ocr: bool | None = None) -> tuple[str, list[object]]:
    """SQL condition (and its parameters) over ``documents`` for the inventory's selection filters."""
    clauses: list[str] = []
    params: list[object] = []
    if text_quality is not None:
        values = tuple(text_quality)
        clauses.append(f'text_quality IN ({", ".join("?" for _ in values)})' if values else '0')
        params.extend(values)
    if do_ocr is not None:
        clauses.append('do_ocr=?')
        params.append(int(do_ocr))
    return (' AND '.join(clauses) or '1'), params


class InventoryStore:
    """SQLite persistence for profiles, page text and triage verdicts, keyed by content hash."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute('PRAGMA foreign_keys = ON')
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self.connect()) as connection, connection:
            yield connection

    def create_schema(self) -> None:
        with self.transaction() as connection:
            if connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
                for table in _TABLES:
                    connection.execute(f'DROP TABLE IF EXISTS {table}')
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    {_columns(PDFProfile)},
                    profiler_version INTEGER NOT NULL,
                    triage_key TEXT,
                    triage_corpus TEXT,
                    triage_policy TEXT,
                    keep_first INTEGER,
                    skip_pages TEXT,
                    n_triage_pages INTEGER,
                    tail_triage_start INTEGER,
                    PRIMARY KEY (hash_id)
                );

                CREATE TABLE IF NOT EXISTS files (
                    relative_path TEXT PRIMARY KEY,
                    hash_id TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS profile_errors (
                    hash_id TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    profiler_version INTEGER NOT NULL,
                    error TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pages (
                    hash_id TEXT NOT NULL,
                    {_columns(PageProfile)},
                    blank INTEGER NOT NULL,
                    PRIMARY KEY (hash_id, page_number),
                    FOREIGN KEY (hash_id) REFERENCES documents(hash_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS page_text (
                    hash_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text BLOB NOT NULL,  -- zlib-compressed UTF-8 pypdf text layer
                    PRIMARY KEY (hash_id, page_number),
                    FOREIGN KEY (hash_id) REFERENCES documents(hash_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS triage_verdicts (
                    hash_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    boilerplate INTEGER NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY (hash_id, page_number),
                    FOREIGN KEY (hash_id) REFERENCES documents(hash_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    name TEXT PRIMARY KEY,
                    seed INTEGER NOT NULL,
                    filters TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiment_documents (
                    experiment_name TEXT NOT NULL,
                    hash_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    included INTEGER NOT NULL,
                    PRIMARY KEY (experiment_name, hash_id),
                    FOREIGN KEY (experiment_name) REFERENCES experiments(name) ON DELETE CASCADE,
                    FOREIGN KEY (hash_id) REFERENCES documents(hash_id) ON DELETE CASCADE
                );

                PRAGMA user_version = {SCHEMA_VERSION};
                """
            )

    # -- files and profiles -------------------------------------------------------------------------------------

    def file_hash(self, relative_path: str, size_bytes: int, mtime_ns: int) -> str | None:
        """The stored hash of a file, if its size and modification time are unchanged since it was hashed."""
        with self.transaction() as connection:
            row = connection.execute(
                'SELECT hash_id FROM files WHERE relative_path=? AND size_bytes=? AND mtime_ns=?',
                (relative_path, size_bytes, mtime_ns),
            ).fetchone()
        return row[0] if row else None

    def set_file(self, relative_path: str, hash_id: str, size_bytes: int, mtime_ns: int) -> None:
        with self.transaction() as connection:
            connection.execute(
                'INSERT OR REPLACE INTO files (relative_path, hash_id, size_bytes, mtime_ns) VALUES (?, ?, ?, ?)',
                (relative_path, hash_id, size_bytes, mtime_ns),
            )

    def status(self, hash_id: str, profiler_version: int) -> str | None:
        """'profiled' or 'failed' when this profiler version already handled the content, else None."""
        with self.transaction() as connection:
            if connection.execute(
                'SELECT 1 FROM documents WHERE hash_id=? AND profiler_version=?', (hash_id, profiler_version)
            ).fetchone():
                return 'profiled'
            if connection.execute(
                'SELECT 1 FROM profile_errors WHERE hash_id=? AND profiler_version=?', (hash_id, profiler_version)
            ).fetchone():
                return 'failed'
        return None

    def set_path(self, hash_id: str, relative_path: str) -> None:
        with self.transaction() as connection:
            connection.execute('UPDATE documents SET relative_path=? WHERE hash_id=?', (relative_path, hash_id))
            connection.execute('UPDATE profile_errors SET relative_path=? WHERE hash_id=?', (relative_path, hash_id))

    def save_profile(self, result: ProfileResult, profiler_version: int) -> None:
        """Replace everything stored for this content; any earlier triage is dropped with the old text."""
        profile, hash_id = result.profile, result.profile.hash_id
        with self.transaction() as connection:
            connection.execute('DELETE FROM documents WHERE hash_id=?', (hash_id,))  # cascades to pages, text, verdicts
            connection.execute('DELETE FROM profile_errors WHERE hash_id=?', (hash_id,))
            columns = (*_PROFILE_FIELDS, 'profiler_version')
            connection.execute(
                f'INSERT INTO documents ({", ".join(columns)}) VALUES ({", ".join("?" for _ in columns)})',
                (*(getattr(profile, name) for name in _PROFILE_FIELDS), profiler_version),
            )
            page_columns = ('hash_id', *_PAGE_FIELDS, 'blank')
            connection.executemany(
                f'INSERT INTO pages ({", ".join(page_columns)}) VALUES ({", ".join("?" for _ in page_columns)})',
                ((hash_id, *(getattr(page, name) for name in _PAGE_FIELDS), page.blank) for page in result.pages),
            )
            connection.executemany(
                'INSERT INTO page_text (hash_id, page_number, text) VALUES (?, ?, ?)',
                ((hash_id, number, zlib.compress(text.encode('utf-8'))) for number, text in enumerate(result.texts, 1)),
            )

    def record_error(self, hash_id: str, relative_path: str, profiler_version: int, error: str) -> None:
        with self.transaction() as connection:
            connection.execute('DELETE FROM documents WHERE hash_id=?', (hash_id,))
            connection.execute(
                'INSERT OR REPLACE INTO profile_errors (hash_id, relative_path, profiler_version, error) '
                'VALUES (?, ?, ?, ?)',
                (hash_id, relative_path, profiler_version, error),
            )

    def profiles(self, text_quality: Iterable[str] | None = None, do_ocr: bool | None = None) -> list[PDFProfile]:
        clause, params = where(text_quality, do_ocr)
        with self.transaction() as connection:
            rows = connection.execute(
                f'SELECT {", ".join(_PROFILE_FIELDS)} FROM documents WHERE {clause} ORDER BY relative_path', params
            ).fetchall()
        return [_from_row(PDFProfile, row) for row in rows]

    def pages(self, hash_id: str) -> tuple[PageProfile, ...]:
        with self.transaction() as connection:
            rows = connection.execute(
                f'SELECT {", ".join(_PAGE_FIELDS)} FROM pages WHERE hash_id=? ORDER BY page_number', (hash_id,)
            ).fetchall()
        return tuple(_from_row(PageProfile, row) for row in rows)

    def file_rows(self, text_quality: Iterable[str] | None = None, do_ocr: bool | None = None) -> list[tuple[str, str]]:
        """``(relative_path, hash_id)`` of every known file; failed files only when no filter is given."""
        filtered = text_quality is not None or do_ocr is not None
        clause, params = where(text_quality, do_ocr)
        with self.transaction() as connection:
            return connection.execute(
                f"""SELECT files.relative_path, files.hash_id FROM files
                    {'JOIN' if filtered else 'LEFT JOIN'} documents ON documents.hash_id = files.hash_id
                    WHERE {clause} AND (documents.hash_id IS NOT NULL
                                        OR files.hash_id IN (SELECT hash_id FROM profile_errors))
                    ORDER BY files.relative_path""",
                params,
            ).fetchall()

    def document_rows(
        self, text_quality: Iterable[str] | None = None, do_ocr: bool | None = None
    ) -> list[tuple[str, str, int, str, int]]:
        """``(hash_id, relative_path, pages, text_quality, do_ocr)`` per profiled document."""
        clause, params = where(text_quality, do_ocr)
        with self.transaction() as connection:
            return connection.execute(
                f'SELECT hash_id, relative_path, pages, text_quality, do_ocr FROM documents WHERE {clause}', params
            ).fetchall()

    def sampling_rows(
        self, text_quality: Iterable[str] | None = None, do_ocr: bool | None = None
    ) -> list[tuple[str, str, int, int, str, int]]:
        """Full-document strata rows with original and triage-adjusted page counts."""
        clause, params = where(text_quality, do_ocr)
        with self.transaction() as connection:
            return connection.execute(
                f'SELECT hash_id, relative_path, pages, n_triage_pages, text_quality, do_ocr '
                f'FROM documents WHERE {clause}',
                params,
            ).fetchall()

    # -- page text -----------------------------------------------------------------------------------------------

    def page_texts(self, hash_id: str) -> tuple[str, ...] | None:
        """The stored pypdf text layer, or None when this content has not been profiled."""
        with self.transaction() as connection:
            if not connection.execute('SELECT 1 FROM documents WHERE hash_id=?', (hash_id,)).fetchone():
                return None
            rows = connection.execute(
                'SELECT text FROM page_text WHERE hash_id=? ORDER BY page_number', (hash_id,)
            ).fetchall()
        return tuple(zlib.decompress(row[0]).decode('utf-8') for row in rows)

    def iter_page_texts(self, hash_ids: Collection[str] | None = None) -> Iterator[tuple[str, tuple[str, ...]]]:
        """``(hash_id, page texts)`` per document, for corpus-level passes; every document when ``hash_ids`` is None."""
        ids = self.corpus() if hash_ids is None else sorted(hash_ids)
        for hash_id in ids:
            texts = self.page_texts(hash_id)
            if texts is not None:
                yield hash_id, texts

    def corpus(self) -> list[str]:
        with self.transaction() as connection:
            return [row[0] for row in connection.execute('SELECT hash_id FROM documents ORDER BY hash_id')]

    def corpus_fingerprint(self) -> str:
        """Changes whenever a document joins or leaves the inventory, invalidating corpus-level triage."""
        return hashlib.sha256('\n'.join(self.corpus()).encode()).hexdigest()

    # -- triage --------------------------------------------------------------------------------------------------

    def triage_state(self, hash_id: str) -> tuple[str | None, str | None] | None:
        with self.transaction() as connection:
            row = connection.execute(
                'SELECT triage_key, triage_corpus FROM documents WHERE hash_id=?', (hash_id,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def save_triage(
        self,
        hash_id: str,
        *,
        key: str,
        corpus: str,
        policy: str,
        keep_first: int,
        verdicts: Iterable[PageVerdict],
        skip: Collection[int],
    ) -> None:
        skip_pages = sorted(skip)
        tail_start = skip_pages[0] if policy == 'tail' and skip_pages else None
        with self.transaction() as connection:
            connection.execute(
                """UPDATE documents SET triage_key=?, triage_corpus=?, triage_policy=?, keep_first=?,
                       skip_pages=?, n_triage_pages=?, tail_triage_start=? WHERE hash_id=?""",
                (key, corpus, policy, keep_first, json.dumps(skip_pages), len(skip_pages), tail_start, hash_id),
            )
            connection.execute('DELETE FROM triage_verdicts WHERE hash_id=?', (hash_id,))
            connection.executemany(
                'INSERT INTO triage_verdicts (hash_id, page_number, boilerplate, score, reason) VALUES (?, ?, ?, ?, ?)',
                ((hash_id, v.page, int(v.boilerplate), v.score, v.reason) for v in verdicts),
            )

    def triage(self, hash_id: str, key: str) -> tuple[tuple[PageVerdict, ...], frozenset[int]] | None:
        """Stored verdicts and skipped pages, if this document was triaged with the settings ``key`` names."""
        with self.transaction() as connection:
            row = connection.execute(
                'SELECT skip_pages FROM documents WHERE hash_id=? AND triage_key=?', (hash_id, key)
            ).fetchone()
            if row is None:
                return None
            verdicts = connection.execute(
                'SELECT page_number, boilerplate, score, reason FROM triage_verdicts WHERE hash_id=? '
                'ORDER BY page_number',
                (hash_id,),
            ).fetchall()
        return (
            tuple(PageVerdict(page, bool(boilerplate), score, reason) for page, boilerplate, score, reason in verdicts),
            frozenset(json.loads(row[0])),
        )
