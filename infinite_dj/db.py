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


DB_VERSION = 3

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

-- ── Mined DJ-mix corpus (see mix_corpus.py) ─────────────────────────────────
-- Appending the DDL here *is* the migration: _init runs executescript on every
-- connect and everything is IF NOT EXISTS, so existing DBs pick these up.

CREATE TABLE IF NOT EXISTS mixes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path      TEXT UNIQUE NOT NULL,
    file_hash      TEXT NOT NULL,
    title          TEXT,
    artist         TEXT,
    duration       REAL,
    n_tracks       INTEGER,
    tracklist      TEXT,      -- JSON [{"t":..,"artist":..,"title":..,"genre":..}]
    tempo_segments TEXT,      -- JSON, from mix_grid.track_tempo_segments
    energy_curve   TEXT,      -- JSON, 1 value/sec (same convention as tracks)
    source         TEXT,      -- "user" | "synthetic"
    mined_at       REAL,
    miner_version  INTEGER
);

-- One row per announced boundary, INCLUDING rejected ones. Keeping rejects with
-- their sub-scores is what allows re-filtering at query time when the thresholds
-- turn out wrong on the first real corpus, without re-running audio analysis.
CREATE TABLE IF NOT EXISTS transitions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    mix_id         INTEGER NOT NULL,
    idx            INTEGER NOT NULL,
    announced_t    REAL NOT NULL,
    status         TEXT NOT NULL,          -- "ok" | "rejected"
    reject_reason  TEXT,
    track_a        TEXT,
    track_b        TEXT,
    t_start        REAL, t_end   REAL,
    t_center       REAL, t_bass  REAL,
    duration_sec   REAL, duration_beats REAL, duration_bars REAL,
    is_cut         INTEGER,
    bpm_before     REAL, bpm_after REAL, tempo_step_pct REAL,
    band_low_cp    REAL, band_mid_cp REAL, band_high_cp REAL,
    band_low_w     REAL, band_mid_w  REAL, band_high_w  REAL,
    key_before     TEXT, key_after  TEXT,
    key_conf_before REAL, key_conf_after REAL,
    camelot_score  REAL,
    dwell_bars     REAL, solo_bars REAL,
    genre          TEXT,
    confidence     REAL,
    sub_scores     TEXT,                   -- JSON
    truth          TEXT,                   -- JSON, synthetic rows only
    UNIQUE(mix_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_transitions_mix    ON transitions(mix_id);
CREATE INDEX IF NOT EXISTS idx_transitions_status ON transitions(status);
"""

# Columns of `transitions` that mine_mix fills in, in order. Kept next to the
# DDL so the two can't drift.
TRANSITION_COLUMNS = (
    "mix_id", "idx", "announced_t", "status", "reject_reason",
    "track_a", "track_b",
    "t_start", "t_end", "t_center", "t_bass",
    "duration_sec", "duration_beats", "duration_bars", "is_cut",
    "bpm_before", "bpm_after", "tempo_step_pct",
    "band_low_cp", "band_mid_cp", "band_high_cp",
    "band_low_w", "band_mid_w", "band_high_w",
    "key_before", "key_after", "key_conf_before", "key_conf_after",
    "camelot_score", "dwell_bars", "solo_bars", "genre",
    "confidence", "sub_scores", "truth",
)


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

    # ── Mined mix corpus ─────────────────────────────────────────────────────

    def mix_needs_mining(self, file_path: str, miner_version: int) -> bool:
        """
        True if this mix is unmined, changed on disk, or was mined by an older
        miner. The version check matters: a measurement change invalidates
        stored rows, and silently mixing old and new rows would corrupt the
        distributions the calibration is drawn from.
        """
        file_path = os.path.abspath(file_path)
        row = self._conn.execute(
            "SELECT file_hash, miner_version FROM mixes WHERE file_path = ?",
            (file_path,)
        ).fetchone()
        if row is None:
            return True
        if (row["miner_version"] or 0) < miner_version:
            return True
        return row["file_hash"] != _file_hash(file_path)

    def save_mix(self, *, file_path: str, title: Optional[str] = None,
                 artist: Optional[str] = None, duration: float = 0.0,
                 tracklist: Optional[list] = None,
                 tempo_segments: Optional[list] = None,
                 energy_curve: Optional[list] = None,
                 source: str = "user", miner_version: int = 1) -> int:
        """Insert or replace a mix row and return its id."""
        file_path = os.path.abspath(file_path)
        try:
            file_hash = _file_hash(file_path)
        except OSError:
            file_hash = "missing"
        self._conn.execute("""
            INSERT OR REPLACE INTO mixes
            (id, file_path, file_hash, title, artist, duration, n_tracks,
             tracklist, tempo_segments, energy_curve, source, mined_at,
             miner_version)
            VALUES
            ((SELECT id FROM mixes WHERE file_path = :file_path),
             :file_path, :file_hash, :title, :artist, :duration, :n_tracks,
             :tracklist, :tempo_segments, :energy_curve, :source, :mined_at,
             :miner_version)
        """, {
            "file_path": file_path, "file_hash": file_hash,
            "title": title, "artist": artist, "duration": duration,
            "n_tracks": len(tracklist or []),
            "tracklist": json.dumps(tracklist or []),
            "tempo_segments": json.dumps(tempo_segments or []),
            "energy_curve": json.dumps(energy_curve or []),
            "source": source, "mined_at": time.time(),
            "miner_version": miner_version,
        })
        self._conn.commit()
        row = self._conn.execute("SELECT id FROM mixes WHERE file_path = ?",
                                (file_path,)).fetchone()
        return int(row["id"])

    def delete_mix_transitions(self, mix_id: int):
        """
        Drop a mix's transition rows.

        Done explicitly rather than via ON DELETE CASCADE: this codebase never
        sets `PRAGMA foreign_keys = ON`, so a cascade would be silently inert.
        """
        self._conn.execute("DELETE FROM transitions WHERE mix_id = ?", (mix_id,))
        self._conn.commit()

    def save_transitions(self, rows: list):
        """Insert transition rows (dicts keyed by TRANSITION_COLUMNS)."""
        if not rows:
            return
        cols = ", ".join(TRANSITION_COLUMNS)
        binds = ", ".join(f":{c}" for c in TRANSITION_COLUMNS)
        self._conn.executemany(
            f"INSERT OR REPLACE INTO transitions ({cols}) VALUES ({binds})",
            [{c: r.get(c) for c in TRANSITION_COLUMNS} for r in rows],
        )
        self._conn.commit()

    def load_mixes(self) -> List[dict]:
        rows = self._conn.execute("SELECT * FROM mixes ORDER BY id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for f in ("tracklist", "tempo_segments", "energy_curve"):
                d[f] = json.loads(d[f]) if d.get(f) else []
            out.append(d)
        return out

    def load_transitions(self, min_confidence: Optional[float] = None,
                         status: Optional[str] = "ok") -> List[dict]:
        """
        Load transition rows, filtering at query time.

        Filtering here rather than at mine time is deliberate — see the note on
        the `transitions` table.
        """
        sql = "SELECT * FROM transitions"
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY mix_id, idx"
        out = []
        for r in self._conn.execute(sql, params).fetchall():
            d = dict(r)
            for f in ("sub_scores", "truth"):
                d[f] = json.loads(d[f]) if d.get(f) else None
            out.append(d)
        return out

    def corpus_counts(self) -> dict:
        mixes = self._conn.execute("SELECT COUNT(*) n FROM mixes").fetchone()["n"]
        total = self._conn.execute(
            "SELECT COUNT(*) n FROM transitions").fetchone()["n"]
        ok = self._conn.execute(
            "SELECT COUNT(*) n FROM transitions WHERE status='ok'").fetchone()["n"]
        return {"n_mixes": mixes, "n_transitions": total, "n_accepted": ok}

    def reject_reasons(self) -> dict:
        rows = self._conn.execute("""
            SELECT reject_reason r, COUNT(*) n FROM transitions
            WHERE status = 'rejected' GROUP BY reject_reason ORDER BY n DESC
        """).fetchall()
        return {(r["r"] or "unknown"): r["n"] for r in rows}

    def close(self):
        self._conn.close()
