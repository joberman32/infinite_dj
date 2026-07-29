"""
Tracklist parsing, corpus schema round-trip and reference-window placement.

All pure — no audio, no librosa. The audio path is covered by
test_transition_probe.py and test_mine_render_set.py.
"""
import json

import pytest

from infinite_dj.db import TrackDB
from infinite_dj.mix_corpus import (
    MINER_VERSION,
    corpus_stats,
    find_pairs,
    find_sidecar,
    parse_tracklist,
    phrase_multiple_histogram,
    reference_windows,
)


# ── .txt parsing ─────────────────────────────────────────────────────────────

def test_parses_a_typical_pasted_tracklist(tmp_path):
    p = tmp_path / "mix.txt"
    p.write_text(
        "00:00 Aphex Twin - Polygon Window\n"
        "01. 05:30 - Robot Junkyard - Duck Acid\n"
        "[12:04] Tom Toms – Language\n"
        "2) 1:02:15 Vlad Antonenko — Deep Groove\n"
    )
    entries, problems = parse_tracklist(str(p))

    assert [e["t"] for e in entries] == [0.0, 330.0, 724.0, 3735.0]
    assert entries[0]["artist"] == "Aphex Twin"
    assert entries[0]["title"] == "Polygon Window"
    assert entries[2]["artist"] == "Tom Toms"       # en dash
    assert entries[3]["artist"] == "Vlad Antonenko"  # em dash + hours
    assert problems == []


def test_collects_unparsed_lines_rather_than_dropping_them(tmp_path):
    p = tmp_path / "mix.txt"
    p.write_text("00:00 A - One\n"
                 "this line has no timestamp\n"
                 "# a comment\n"
                 "03:00 B - Two\n")
    entries, problems = parse_tracklist(str(p))

    assert len(entries) == 2
    assert any("no timestamp" in s for s in problems)
    # Comments are skipped silently; only genuinely unparsed content is reported.
    assert not any("comment" in s for s in problems)


def test_rejects_non_monotone_timestamps(tmp_path):
    p = tmp_path / "mix.txt"
    p.write_text("00:00 A - One\n05:00 B - Two\n05:00 C - Dup\n")
    entries, problems = parse_tracklist(str(p))

    assert [e["t"] for e in entries] == [0.0, 300.0]
    assert any("non-monotone" in s for s in problems)


def test_warns_when_the_tracklist_looks_offset(tmp_path):
    """
    A tracklist offset relative to a longer stream corrupts every boundary, so it
    has to be loud rather than silent.
    """
    p = tmp_path / "mix.txt"
    p.write_text("14:30 A - One\n20:00 B - Two\n")
    _, problems = parse_tracklist(str(p))
    assert any("offset" in s for s in problems)


def test_extracts_a_trailing_genre_tag(tmp_path):
    p = tmp_path / "mix.txt"
    p.write_text("00:00 A - One [deep house]\n"
                 "05:00 B - Two (acid techno)\n"
                 "10:00 C - Three (2019)\n")
    entries, _ = parse_tracklist(str(p))

    assert entries[0]["genre"] == "deep house"
    assert entries[0]["title"] == "One"
    assert entries[1]["genre"] == "acid techno"
    # A year is not a genre.
    assert entries[2]["genre"] is None


def test_unsplittable_label_becomes_the_title(tmp_path):
    p = tmp_path / "mix.txt"
    p.write_text("00:00 ID\n03:00 Another Unknown Track\n")
    entries, _ = parse_tracklist(str(p))

    assert entries[0]["artist"] is None
    assert entries[0]["title"] == "ID"


# ── .cue parsing ─────────────────────────────────────────────────────────────

def test_parses_a_cue_sheet_with_frames(tmp_path):
    p = tmp_path / "mix.cue"
    p.write_text(
        'PERFORMER "Some DJ"\n'
        'TITLE "A Mix"\n'
        'FILE "mix.mp3" MP3\n'
        '  TRACK 01 AUDIO\n'
        '    TITLE "Polygon Window"\n'
        '    PERFORMER "Aphex Twin"\n'
        '    INDEX 01 00:00:00\n'
        '  TRACK 02 AUDIO\n'
        '    TITLE "Duck Acid"\n'
        '    PERFORMER "Robot Junkyard"\n'
        '    INDEX 01 05:30:37\n'
    )
    entries, problems = parse_tracklist(str(p))

    assert len(entries) == 2
    assert entries[0]["title"] == "Polygon Window"
    assert entries[1]["artist"] == "Robot Junkyard"
    # 5:30 plus 37 frames at 75/sec.
    assert abs(entries[1]["t"] - (330 + 37 / 75)) < 1e-6
    assert problems == []


# ── .json parsing ────────────────────────────────────────────────────────────

def test_parses_json_escape_hatch(tmp_path):
    p = tmp_path / "mix.json"
    p.write_text(json.dumps([
        {"t": 0, "artist": "A", "title": "One", "genre": "house"},
        {"t": 300, "artist": "B", "title": "Two"},
    ]))
    entries, problems = parse_tracklist(str(p))

    assert [e["t"] for e in entries] == [0.0, 300.0]
    assert entries[0]["genre"] == "house"
    assert problems == []


def test_reports_invalid_json(tmp_path):
    p = tmp_path / "mix.json"
    p.write_text("{not json")
    entries, problems = parse_tracklist(str(p))
    assert entries == []
    assert any("invalid JSON" in s for s in problems)


# ── Sidecar discovery ────────────────────────────────────────────────────────

def test_finds_sidecar_and_pairs(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("00:00 A - One\n")
    (tmp_path / "b.wav").write_bytes(b"x")          # no sidecar
    (tmp_path / "notes.txt").write_text("ignore me")

    assert find_sidecar(str(tmp_path / "a.mp3")) == str(tmp_path / "a.txt")
    pairs, problems = find_pairs(str(tmp_path))

    assert len(pairs) == 1
    assert pairs[0][0].endswith("a.mp3")
    assert any("no tracklist sidecar for b.wav" in s for s in problems)


def test_json_sidecar_wins_over_txt(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("00:00 A - One\n")
    (tmp_path / "a.json").write_text("[]")
    assert find_sidecar(str(tmp_path / "a.mp3")).endswith(".json")


# ── Reference windows ────────────────────────────────────────────────────────

def test_reference_windows_sit_between_neighbouring_boundaries():
    entries = [{"t": 0.0}, {"t": 300.0}, {"t": 600.0}]
    (a_lo, a_hi), (b_lo, b_hi) = reference_windows(entries, 1, 900.0)

    assert 0.0 <= a_lo < a_hi <= 300.0
    assert 300.0 <= b_lo < b_hi <= 600.0
    assert a_hi - a_lo >= 15.0
    assert b_hi - b_lo >= 15.0


def test_reference_windows_rejected_when_tracks_are_too_close():
    """Back-to-back short tracks leave no clean audio; that's a rejection."""
    entries = [{"t": 0.0}, {"t": 12.0}, {"t": 24.0}]
    assert reference_windows(entries, 1, 40.0) is None


def test_reference_windows_handle_the_last_boundary():
    entries = [{"t": 0.0}, {"t": 300.0}]
    refs = reference_windows(entries, 1, 600.0)
    assert refs is not None
    (_, _), (b_lo, b_hi) = refs
    assert b_hi <= 600.0


# ── Phrase histogram ─────────────────────────────────────────────────────────

def test_phrase_histogram_buckets_to_nearest_multiple():
    counts, n = phrase_multiple_histogram([7.8, 8.1, 15.6, 16.4, 31.0, 3.9])
    assert n == 6
    assert counts[8] == 2
    assert counts[16] == 2
    assert counts[32] == 1
    assert counts[4] == 1


def test_phrase_histogram_ignores_zero_length_cuts():
    counts, n = phrase_multiple_histogram([0.0, None, 16.0])
    assert n == 1
    assert counts[16] == 1


# ── Schema round-trip ────────────────────────────────────────────────────────

def _mix_row(db, path, tracklist):
    return db.save_mix(file_path=path, title="m", duration=1800.0,
                       tracklist=tracklist,
                       tempo_segments=[{"start": 0.0, "end": 1800.0, "bpm": 124.0,
                                        "phase": 0.0, "confidence": 1.0}],
                       energy_curve=[0.5, 0.6], miner_version=MINER_VERSION)


def test_mix_and_transition_round_trip(tmp_path):
    db = TrackDB(str(tmp_path / "c.db"))
    audio = tmp_path / "m.mp3"
    audio.write_bytes(b"x")

    mix_id = _mix_row(db, str(audio), [{"t": 0.0, "title": "One"},
                                       {"t": 300.0, "title": "Two"}])
    db.save_transitions([
        {"mix_id": mix_id, "idx": 1, "announced_t": 300.0, "status": "ok",
         "duration_bars": 16.0, "duration_beats": 64.0, "is_cut": 0,
         "confidence": 0.8, "camelot_score": 0.8, "dwell_bars": 60.0,
         "sub_scores": json.dumps({"fit": 0.9})},
        {"mix_id": mix_id, "idx": 2, "announced_t": 600.0, "status": "rejected",
         "reject_reason": "low_separability", "confidence": 0.1},
    ])

    loaded = db.load_mixes()
    assert len(loaded) == 1
    assert loaded[0]["n_tracks"] == 2
    assert loaded[0]["tracklist"][1]["title"] == "Two"
    assert loaded[0]["tempo_segments"][0]["bpm"] == 124.0

    ok = db.load_transitions()
    assert len(ok) == 1
    assert ok[0]["duration_bars"] == 16.0
    assert ok[0]["sub_scores"] == {"fit": 0.9}

    assert db.corpus_counts() == {"n_mixes": 1, "n_transitions": 2,
                                  "n_accepted": 1}
    assert db.reject_reasons() == {"low_separability": 1}
    db.close()


def test_rejected_rows_are_kept_for_requerying(tmp_path):
    """
    Query-time filtering is the point: thresholds will be wrong on the first real
    corpus, and re-filtering must not mean re-running audio analysis.
    """
    db = TrackDB(str(tmp_path / "c.db"))
    audio = tmp_path / "m.mp3"
    audio.write_bytes(b"x")
    mix_id = _mix_row(db, str(audio), [{"t": 0.0, "title": "One"}])
    db.save_transitions([
        {"mix_id": mix_id, "idx": i, "announced_t": 100.0 * i, "status": "ok",
         "duration_bars": 8.0 + i, "is_cut": 0, "confidence": c}
        for i, c in enumerate([0.2, 0.5, 0.9], start=1)
    ])

    assert len(db.load_transitions(min_confidence=None)) == 3
    assert len(db.load_transitions(min_confidence=0.5)) == 2
    assert len(db.load_transitions(min_confidence=0.95)) == 0
    db.close()


def test_mix_needs_mining_tracks_version_and_hash(tmp_path):
    db = TrackDB(str(tmp_path / "c.db"))
    audio = tmp_path / "m.mp3"
    audio.write_bytes(b"x")

    assert db.mix_needs_mining(str(audio), MINER_VERSION) is True
    _mix_row(db, str(audio), [{"t": 0.0, "title": "One"}])
    assert db.mix_needs_mining(str(audio), MINER_VERSION) is False
    # A newer miner invalidates stored rows so old and new never pool.
    assert db.mix_needs_mining(str(audio), MINER_VERSION + 1) is True
    db.close()


def test_saving_a_mix_twice_reuses_its_id(tmp_path):
    """Re-mining must update in place, not orphan the previous transitions."""
    db = TrackDB(str(tmp_path / "c.db"))
    audio = tmp_path / "m.mp3"
    audio.write_bytes(b"x")

    first = _mix_row(db, str(audio), [{"t": 0.0, "title": "One"}])
    second = _mix_row(db, str(audio), [{"t": 0.0, "title": "One"}])
    assert first == second
    assert db.corpus_counts()["n_mixes"] == 1
    db.close()


def test_corpus_stats_on_an_empty_corpus(tmp_path):
    db = TrackDB(str(tmp_path / "c.db"))
    stats = corpus_stats(db)

    assert stats["counts"] == {"n_mixes": 0, "n_transitions": 0, "n_accepted": 0}
    assert stats["duration_bars"] is None
    assert stats["cut_rate"] is None
    db.close()


def test_corpus_stats_summarises_distributions(tmp_path):
    db = TrackDB(str(tmp_path / "c.db"))
    audio = tmp_path / "m.mp3"
    audio.write_bytes(b"x")
    mix_id = _mix_row(db, str(audio), [{"t": 0.0, "title": "One"}])

    rows = []
    for i, bars in enumerate([8.0, 8.0, 16.0, 16.0, 16.0, 32.0], start=1):
        rows.append({"mix_id": mix_id, "idx": i, "announced_t": 100.0 * i,
                     "status": "ok", "duration_bars": bars,
                     "duration_beats": bars * 4, "is_cut": 0, "confidence": 0.8,
                     "camelot_score": 0.8, "dwell_bars": 64.0,
                     "genre": "house" if i % 2 else "techno"})
    rows.append({"mix_id": mix_id, "idx": 99, "announced_t": 9000.0,
                 "status": "ok", "duration_bars": 0.0, "is_cut": 1,
                 "confidence": 0.8, "camelot_score": 0.0})
    db.save_transitions(rows)

    stats = corpus_stats(db, min_confidence=0.5)
    assert stats["duration_bars"]["n"] == 6
    assert stats["duration_bars"]["p50"] == 16.0
    assert stats["n_blends"] == 6
    assert stats["cut_rate"] == pytest.approx(1 / 7, abs=1e-3)
    assert stats["camelot_zero_rate"] == pytest.approx(1 / 7, abs=1e-3)
    # Genres are broken out rather than pooled.
    assert set(stats["by_genre"]) == {"house", "techno"}
    assert stats["phrase_histogram"][16] == 3
    db.close()
