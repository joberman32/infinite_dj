"""
SQLite persistence layer.

Stores and retrieves TrackMeta objects. Hashes file paths so re-analyzing
the same file is a no-op unless the file has been modified.
"""

import os
import sqlite3
import json
import hashlib
import time
from typing import Optional, List
from .models import TrackMeta


DB_VERSION = 2

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT UNIQUE NOT NULL,
    file_hash     TEXT NOT NULL,
    title         TEXT,
    duration      REAL,
    bpm           REAL,
    bpm_confidence REAL,
    beats         TEXT,     -- JSON
    downbeats     TEXT,     -- JSON
    phrases       TEXT,     -- JSON
    key           TEXT,
    key_name      TEXT,
    key_confidence REAL,
    energy_curve  TEXT,     -- JSON
    sections      TEXT,     -- JSON
    cue_points    TEXT,     -- JSON
    analyzed_at   REAL,
    loudness      REAL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _file_hash(path: str) -> str:
    """Quick hash using file size + mtime — fast enough for large libraries."""
    stat = os.stat(path)
    raw = f"{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


class TrackDB:
    def __init__(self, db_path: str = "infinite_dj.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        self._conn.executescript(CREATE_SQL)
        self._migrate()
        self._conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('version', ?)", (str(DB_VERSION),)
        )
        self._conn.commit()

    def _migrate(self):
        """Additive migrations for DBs created by an older schema."""
        cols = {r["name"] for r in
                self._conn.execute("PRAGMA table_info(tracks)").fetchall()}
        if "loudness" not in cols:
            self._conn.execute("ALTER TABLE tracks ADD COLUMN loudness REAL")
        self._conn.commit()

    def needs_analysis(self, file_path: str) -> bool:
        """Return True if the file hasn't been analyzed or has changed."""
        file_path = os.path.abspath(file_path)
        row = self._conn.execute(
            "SELECT file_hash FROM tracks WHERE file_path = ?", (file_path,)
        ).fetchone()

        if row is None:
            return True

        current_hash = _file_hash(file_path)
        return row["file_hash"] != current_hash

    def save(self, meta: TrackMeta):
        """Insert or replace a TrackMeta record."""
        file_hash = _file_hash(meta.file_path)
        d = meta.to_dict()

        # Serialize list/dict fields to JSON strings
        json_fields = ["beats", "downbeats", "phrases", "energy_curve",
                       "sections", "cue_points"]
        for f in json_fields:
            d[f] = json.dumps(d[f])

        self._conn.execute("""
            INSERT OR REPLACE INTO tracks
            (file_path, file_hash, title, duration, bpm, bpm_confidence,
             beats, downbeats, phrases, key, key_name, key_confidence,
             energy_curve, sections, cue_points, analyzed_at, loudness)
            VALUES
            (:file_path, :file_hash, :title, :duration, :bpm, :bpm_confidence,
             :beats, :downbeats, :phrases, :key, :key_name, :key_confidence,
             :energy_curve, :sections, :cue_points, :analyzed_at, :loudness)
        """, {**d, "file_hash": file_hash})
        self._conn.commit()

    def load(self, file_path: str) -> Optional[TrackMeta]:
        """Load a TrackMeta by file path. Returns None if not found."""
        file_path = os.path.abspath(file_path)
        row = self._conn.execute(
            "SELECT * FROM tracks WHERE file_path = ?", (file_path,)
        ).fetchone()

        if row is None:
            return None

        return TrackMeta.from_dict(dict(row))

    def load_all(self) -> List[TrackMeta]:
        """Load all analyzed tracks."""
        rows = self._conn.execute("SELECT * FROM tracks ORDER BY title").fetchall()
        return [TrackMeta.from_dict(dict(r)) for r in rows]

    def stats(self) -> dict:
        row = self._conn.execute("""
            SELECT COUNT(*) as n,
                   AVG(bpm) as avg_bpm,
                   AVG(duration) as avg_dur
            FROM tracks
        """).fetchone()
        return dict(row)

    def close(self):
        self._conn.close()
