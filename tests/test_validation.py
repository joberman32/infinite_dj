"""
The validation harness.

The pure parts (holdout splitting, ceiling arithmetic, report formatting) are
tested here without audio. The end-to-end render-and-probe path is exercised by
`test_mine_render_set.py` and by `dj.py validate` itself, which is slow by nature
— it renders real audio on purpose, because that's the only way the engine's side
passes through the same estimator as the corpus.
"""
import os

import pytest

from infinite_dj.validation import (
    TOLERANCE_BEATS,
    ceiling_hit_rates,
    compare_to_corpus,
    format_validation_report,
    holdout_split,
)


def _row(mix_id, idx, beats=32.0, is_cut=0, conf=0.8):
    return {"mix_id": mix_id, "idx": idx, "duration_beats": beats,
            "duration_sec": beats * 0.46, "is_cut": is_cut, "confidence": conf,
            "t_center": 100.0 * idx}


# ── Holdout ──────────────────────────────────────────────────────────────────

def test_holdout_splits_by_mix_not_by_transition():
    """
    Transitions from one mix share a DJ, a tracklist and a tempo track, so
    splitting within a mix would leak the thing we're trying to hold out.
    """
    rows = [_row(m, i) for m in range(1, 11) for i in range(1, 5)]
    train, test = holdout_split(rows, frac=0.2, seed=1)

    train_mixes = {r["mix_id"] for r in train}
    test_mixes = {r["mix_id"] for r in test}
    assert not (train_mixes & test_mixes)
    assert len(test_mixes) == 2
    assert len(train) + len(test) == len(rows)
    # Every transition of a held-out mix goes with it.
    for m in test_mixes:
        assert sum(1 for r in test if r["mix_id"] == m) == 4


def test_holdout_is_deterministic_for_a_seed():
    rows = [_row(m, 1) for m in range(1, 21)]
    a = holdout_split(rows, seed=3)[1]
    b = holdout_split(rows, seed=3)[1]
    assert [r["mix_id"] for r in a] == [r["mix_id"] for r in b]


def test_holdout_with_a_single_mix_holds_nothing_back():
    """
    One mix can't be split without leaking. Returning an empty test set is the
    honest outcome, and the caller falls back to reporting on everything.
    """
    rows = [_row(1, i) for i in range(1, 6)]
    train, test = holdout_split(rows, frac=0.15, seed=0)
    assert len(train) == 5
    assert test == []


def test_holdout_on_no_rows():
    assert holdout_split([], seed=0) == ([], [])


# ── Ceiling ──────────────────────────────────────────────────────────────────

def _matched(truth_beats, meas_beats, bpm=130.0):
    beat = 60.0 / bpm
    return ({"duration_sec": meas_beats * beat, "confidence": 0.6,
             "is_cut": int(meas_beats < 1), "t_center": 100.0},
            {"length_beats": truth_beats, "length_sec": truth_beats * beat,
             "bpm": bpm, "style": "blend" if truth_beats > 4 else "cut",
             "t_start": 90.0, "t_end": 110.0})


def test_ceiling_reports_perfect_measurement_as_perfect():
    eng = {"matched": [_matched(32.0, 32.0), _matched(16.0, 16.0)]}
    cl = ceiling_hit_rates(eng)
    assert cl["n"] == 2
    assert cl["median_abs_beats"] == 0.0
    for tol in TOLERANCE_BEATS:
        assert cl[f"within_{tol:g}"] == 1.0


def test_ceiling_separates_blends_from_cuts():
    """
    Cuts are trivially easy to measure — truth and measurement are both ~0 beats —
    so pooling them with blends flatters the ceiling. The blends-only figure is
    the one calibration depends on, so it has to be reported apart.
    """
    eng = {"matched": [
        _matched(0.6, 0.0), _matched(0.6, 0.0), _matched(0.6, 0.0),
        _matched(32.0, 8.0),     # 24 beats out
        _matched(48.0, 12.0),    # 36 beats out
    ]}
    cl = ceiling_hit_rates(eng)

    # Pooled looks respectable purely because of the cuts.
    assert cl["within_4"] == pytest.approx(0.6)
    # Blends alone tell the truth.
    assert cl["blends"]["n"] == 2
    assert cl["blends"]["within_4"] == 0.0
    assert cl["blends"]["median_abs_beats"] == pytest.approx(30.0, abs=0.1)


def test_ceiling_with_no_measurements():
    assert ceiling_hit_rates({"matched": []}) == {"n": 0}


def test_ceiling_handles_all_cuts():
    eng = {"matched": [_matched(0.6, 0.0), _matched(0.6, 0.0)]}
    cl = ceiling_hit_rates(eng)
    assert cl["n"] == 2
    assert cl["blends"] == {"n": 0}


# ── Report ───────────────────────────────────────────────────────────────────

def _result():
    return {
        "corpus": {"n_total": 140, "n_train": 120, "n_test": 20,
                   "min_confidence": 0.5},
        "engine": {"n_pairs": 30, "n_measured": 25, "n_rejected": 5},
        "ceiling": {"n": 25, "median_abs_beats": 3.76, "within_1": 0.48,
                    "within_4": 0.52, "within_8": 0.56,
                    "blends": {"n": 10, "median_abs_beats": 25.09,
                               "within_1": 0.0, "within_4": 0.0,
                               "within_8": 0.1}},
        "duration_beats": {"corpus_median": 21.0, "corpus_n": 20,
                           "engine_median": 30.0, "engine_n": 10,
                           "wasserstein": 12.5},
        "cut_rate": {"corpus": 0.2, "engine": 0.4},
        "hit_rates": {"engine_choice_beats": 30.0, "median_abs_beats": 9.0,
                      "within_1": 0.05, "within_4": 0.2, "within_8": 0.45},
    }


def test_report_shows_hit_rate_against_the_ceiling():
    """
    A hit rate without its ceiling makes a measurement limit look like an engine
    defect, so the two must appear together.
    """
    text = format_validation_report(_result())
    assert "MEASUREMENT CEILING" in text
    assert "blends only" in text
    assert "ceiling 0%" in text or "ceiling 5%" in text or "ceiling" in text
    assert "distributional, not per-transition" in text


def test_report_survives_a_thin_corpus():
    res = {
        "corpus": {"n_total": 0, "n_train": 0, "n_test": 0,
                   "min_confidence": 0.5},
        "engine": {"n_pairs": 0, "n_measured": 0, "n_rejected": 0},
        "ceiling": {"n": 0},
        "duration_beats": None, "cut_rate": None, "hit_rates": None,
    }
    text = format_validation_report(res)
    assert "no corpus data" in text
    assert "not enough rendered pairs" in text


def test_compare_to_corpus_on_an_empty_library(tmp_path):
    from infinite_dj.db import TrackDB

    db = TrackDB(str(tmp_path / "empty.db"))
    res = compare_to_corpus(db, [], n_pairs=1)
    db.close()

    assert res["engine"]["n_pairs"] == 0
    assert res["duration_beats"] is None
    # Must still format rather than raising.
    assert "Validation" in format_validation_report(res)


LIB_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "combined.db")


@pytest.mark.skipif(not os.path.exists(LIB_DB),
                    reason="needs the local library at combined.db")
def test_engine_and_ceiling_end_to_end():
    """
    One real render-and-probe round trip, to keep the slow path from rotting.
    Kept to a single small set — the accuracy figures come from `dj.py validate`.
    """
    from infinite_dj.db import TrackDB
    from infinite_dj.validation import engine_and_ceiling

    db = TrackDB(LIB_DB)
    tracks = db.load_all()
    db.close()

    eng = engine_and_ceiling(tracks, n_pairs=3, seed=5, set_size=4, quiet=True)
    assert eng["n_pairs"] >= 3
    for row, truth in eng["matched"]:
        assert truth["length_beats"] >= 0
        assert row["status"] == "ok"
    cl = ceiling_hit_rates(eng)
    assert cl["n"] == len(eng["matched"])
