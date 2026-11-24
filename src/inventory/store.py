from __future__ import annotations

import sqlite3
from pathlib import Path


class InventoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute('PRAGMA foreign_keys = ON')
        return connection

    def create_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                '''
                CREATE TABLE IF NOT EXISTS documents (
                    hash_id TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    text_chars INTEGER NOT NULL,
                    char_min INTEGER NOT NULL,
                    char_max INTEGER NOT NULL,
                    char_mean REAL NOT NULL,
                    char_median REAL NOT NULL,
                    word_count INTEGER NOT NULL,
                    word_min INTEGER NOT NULL,
                    word_max INTEGER NOT NULL,
                    word_mean REAL NOT NULL,
                    word_median REAL NOT NULL,
                    chars_per_page REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    text_quality TEXT NOT NULL,
                    do_ocr INTEGER NOT NULL,
                    sparse_pages INTEGER NOT NULL,
                    blank_pages INTEGER NOT NULL,
                    image_xobjects INTEGER NOT NULL,
                    image_pages INTEGER NOT NULL,
                    image_heavy_pages INTEGER NOT NULL,
                    image_page_share REAL NOT NULL,
                    n_triage_pages INTEGER,
                    tail_triage_start INTEGER,
                    triage_policy TEXT,
                    keep_first INTEGER
                );

                CREATE TABLE IF NOT EXISTS pages (
                    hash_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text_chars INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    blank INTEGER NOT NULL,
                    sparse INTEGER NOT NULL,
                    image_xobjects INTEGER NOT NULL,
                    do_ocr INTEGER NOT NULL,
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
                '''
            )
