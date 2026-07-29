"""
The calibration layer.

The load-bearing property is the first test: with no corpus, the engine must
behave exactly as it did before any of this existed. Calibration is an optional
refinement, and a refinement that changes behaviour when it's absent is a bug.
"""
import json

import pytest

from infinite_dj import calibration as cal
from infinite_dj.calibration import (
    MIN_N,
    PROBE_DEPENDENT,
    Calibration,
    Value,
    build_from_stats,
    load,
    save,
    to_dict,
)


@pytest.fixture(autouse=True)
def _clean_active():
    """Never let one test's calibration leak into another, or into the suite."""
    cal.reset()
    yield
    cal.reset()


# ── Defaults are today's constants ───────────────────────────────────────────

def test_defaults_match_the_hardcoded_constants():
    from infinite_dj.engine import MAX_DWELL_BARS, MIN_DWELL_BARS

    c = Calibration()
    assert c.get("blend_bars") == 16
    assert c.get("fade_bars") == 12
    assert c.get("build_bars") == 8
    assert c.get("swap_bars") == 8
    assert c.get("min_solo_bars") == 32
    assert c.get("min_dwell_bars") == MIN_DWELL_BARS
    assert c.get("max_dwell_bars") == MAX_DWELL_BARS
    assert c.get("harmonic_min_score") == 0.3


def test_missing_file_yields_defaults(tmp_path):
    c = load(str(tmp_path / "nope.json"))
    assert c.corpus == {}
    assert c.get("blend_bars") == 16


def test_malformed_file_yields_defaults(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    assert load(str(p)).get("blend_bars") == 16


def test_empty_values_yield_defaults(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"version": 1, "values": {}}))
    assert load(str(p)).get("min_solo_bars") == 32


def test_render_set_is_byte_identical_without_a_corpus(tmp_path):
    """
    The non-negotiable property: with no calibration, rendering must produce
    exactly the audio it produced before this layer existed.

    Rendered twice — once with the default (uncalibrated) path, once with the
    pre-calibration constants passed explicitly — and the samples compared.
    """
    import hashlib

    import numpy as np

    from test_plan_transition import _write_track

    from infinite_dj.mixer import render_set

    tracks = [
        _write_track(tmp_path / "a.wav", bpm=124.0, duration=150.0, seed=21),
        _write_track(tmp_path / "b.wav", bpm=125.0, duration=150.0, seed=22),
        _write_track(tmp_path / "c.wav", bpm=170.0, duration=150.0, seed=23),
    ]

    cal.set_active(Calibration())
    auto, sr1, m1, c1 = render_set(tracks)
    # 32 was the literal default before calibration was wired in.
    explicit, sr2, m2, c2 = render_set(tracks, min_solo_bars=32)

    assert sr1 == sr2
    assert hashlib.sha256(np.ascontiguousarray(auto).tobytes()).hexdigest() == \
        hashlib.sha256(np.ascontiguousarray(explicit).tobytes()).hexdigest()
    assert [(round(m.time, 4), m.style) for m in m1] == \
        [(round(m.time, 4), m.style) for m in m2]
    assert c1 == c2


def test_choose_transition_style_is_unchanged_without_a_corpus():
    """
    The style table must be exactly 16/12/8/8 bars when nothing is calibrated —
    the same numbers that were literals in the function before this layer existed.
    """
    from infinite_dj.mixer import choose_transition_style
    from infinite_dj.models import CuePoint

    cal.set_active(Calibration())

    def style(eo, ei):
        return choose_transition_style(CuePoint(0.0, "out", True, eo, 1.0),
                                       CuePoint(0.0, "in", True, ei, 1.0),
                                       beatmatched=True)

    assert style(0.3, 0.3).name == "blend" and style(0.3, 0.3).n_bars == 16
    assert style(0.9, 0.9).name == "swap" and style(0.9, 0.9).n_bars == 8
    assert style(0.8, 0.5).name == "fade" and style(0.8, 0.5).n_bars == 12
    assert style(0.5, 0.6).name == "build" and style(0.5, 0.6).n_bars == 8


# ── The min-n refusal ────────────────────────────────────────────────────────

def test_thin_evidence_is_refused():
    """
    A value backed by too few observations must fall back, not be applied. This is
    the whole degradation story: a corpus can grow field by field.
    """
    c = Calibration(blend_bars=Value(value=4.0, n=MIN_N - 1, source="p75"))
    assert c.get("blend_bars") == 16
    assert "default" in c.provenance("blend_bars")
    assert f"n={MIN_N - 1}" in c.provenance("blend_bars")


def test_well_supported_evidence_is_applied():
    c = Calibration(blend_bars=Value(value=22.0, n=MIN_N, source="p75"))
    assert c.get("blend_bars") == 22.0
    assert "mined" in c.provenance("blend_bars")


def test_probe_dependent_values_are_flagged_in_provenance():
    """
    Probe-derived numbers carry much weaker evidence than tracklist-derived ones,
    so the distinction has to be visible wherever a value is reported.
    """
    c = Calibration(blend_bars=Value(value=22.0, n=100, source="p75"),
                    max_dwell_bars=Value(value=120.0, n=100, source="p90"))
    assert "probe-dependent" in c.provenance("blend_bars")
    assert "probe-dependent" not in c.provenance("max_dwell_bars")
    assert "blend_bars" in PROBE_DEPENDENT
    assert "max_dwell_bars" not in PROBE_DEPENDENT


# ── Wiring ───────────────────────────────────────────────────────────────────

def test_calibrated_bars_reach_choose_transition_style():
    from infinite_dj.mixer import choose_transition_style
    from infinite_dj.models import CuePoint

    cal.set_active(Calibration(
        blend_bars=Value(value=24.0, n=50, source="p75"),
        swap_bars=Value(value=6.0, n=50, source="p25"),
    ))
    blend = choose_transition_style(CuePoint(0.0, "out", True, 0.3, 1.0),
                                    CuePoint(0.0, "in", True, 0.3, 1.0), True)
    swap = choose_transition_style(CuePoint(0.0, "out", True, 0.9, 1.0),
                                   CuePoint(0.0, "in", True, 0.9, 1.0), True)
    assert blend.n_bars == 24
    assert swap.n_bars == 6
    # Uncalibrated styles keep their defaults.
    fade = choose_transition_style(CuePoint(0.0, "out", True, 0.8, 1.0),
                                   CuePoint(0.0, "in", True, 0.5, 1.0), True)
    assert fade.n_bars == 12


def test_calibrated_bars_never_go_below_one():
    from infinite_dj.mixer import choose_transition_style
    from infinite_dj.models import CuePoint

    cal.set_active(Calibration(
        blend_bars=Value(value=0.0, n=50, source="p75")))
    s = choose_transition_style(CuePoint(0.0, "out", True, 0.3, 1.0),
                                CuePoint(0.0, "in", True, 0.3, 1.0), True)
    assert s.n_bars >= 1


def test_dwell_bounds_are_read_from_calibration():
    from infinite_dj.engine import MAX_DWELL_BARS, MIN_DWELL_BARS, _dwell_bounds

    cal.set_active(Calibration())
    assert _dwell_bounds() == (MIN_DWELL_BARS, MAX_DWELL_BARS)

    cal.set_active(Calibration(
        min_dwell_bars=Value(value=20.0, n=50, source="p10"),
        max_dwell_bars=Value(value=140.0, n=50, source="p90")))
    assert _dwell_bounds() == (20, 140)


def test_inverted_dwell_bounds_fall_back():
    """A thin or odd corpus could produce min > max; the scheduler must not break."""
    from infinite_dj.engine import MAX_DWELL_BARS, MIN_DWELL_BARS, _dwell_bounds

    cal.set_active(Calibration(
        min_dwell_bars=Value(value=200.0, n=50, source="p10"),
        max_dwell_bars=Value(value=50.0, n=50, source="p90")))
    assert _dwell_bounds() == (MIN_DWELL_BARS, MAX_DWELL_BARS)


def test_plan_transition_uses_calibrated_min_solo_bars(tmp_path):
    """
    The calibrated dwell must actually gate the exit.

    Asserted as the floor rather than "a bigger value exits later": the exit cue
    is chosen by *confidence* among those past the dwell, so raising the floor
    restricts the candidate set without necessarily moving the winner.
    """
    from test_plan_transition import _write_track  # reuse the synthetic fixture

    a = _write_track(tmp_path / "a.wav", bpm=124.0, duration=200.0, seed=11)
    b = _write_track(tmp_path / "b.wav", bpm=125.0, duration=200.0, seed=12)

    from infinite_dj.mixer import plan_transition

    bar_sec = (60.0 / a.bpm) * 4

    cal.set_active(Calibration(
        min_solo_bars=Value(value=30.0, n=50, source="p25")))
    gated = plan_transition(a, b, 0.0, a.duration)
    assert gated.cue_out_t >= 30 * bar_sec - 1e-6

    # A floor past every scored cue must still yield a usable exit (the downbeat
    # fallback) rather than raising or cueing before the dwell.
    cal.set_active(Calibration(
        min_solo_bars=Value(value=100.0, n=50, source="p25")))
    extreme = plan_transition(a, b, 0.0, a.duration)
    assert extreme.cue_out_t != gated.cue_out_t
    assert extreme.cue_out_t <= a.duration


# ── Round-trip and building from stats ───────────────────────────────────────

def test_round_trip_preserves_mined_values(tmp_path):
    c = Calibration(
        blend_bars=Value(value=21.5, n=88, source="p75", spread=(8.0, 32.0)),
        harmonic_min_score=Value(value=0.0, n=120, source="p10"),
        corpus={"n_mixes": 7, "n_transitions": 140, "n_accepted": 120},
    )
    p = str(tmp_path / "cal.json")
    save(c, p)
    back = load(p)

    assert back.get("blend_bars") == 21.5
    assert back.get("harmonic_min_score") == 0.0
    assert back.corpus["n_mixes"] == 7
    # Defaults aren't serialised — the file records only what was measured.
    assert "fade_bars" not in to_dict(c)["values"]
    assert back.get("fade_bars") == 12


def test_build_from_stats_maps_quantiles():
    stats = {
        "counts": {"n_mixes": 3, "n_transitions": 200, "n_accepted": 180},
        "n_at_confidence": 180,
        "cut_rate": 0.12,
        "duration_bars": {"n": 150, "mean": 14.0, "p10": 6.0, "p25": 8.0,
                          "p50": 14.0, "p75": 22.0, "p90": 30.0},
        "solo_bars": {"n": 140, "mean": 60.0, "p10": 20.0, "p25": 30.0,
                      "p50": 60.0, "p75": 80.0, "p90": 100.0},
        "dwell_bars": {"n": 180, "mean": 70.0, "p10": 34.0, "p25": 50.0,
                       "p50": 70.0, "p75": 90.0, "p90": 130.0},
        "camelot_score": {"n": 175, "mean": 0.4, "p10": 0.0, "p25": 0.0,
                          "p50": 0.3, "p75": 0.8, "p90": 0.9},
        "tempo_step_pct": {"n": 180, "mean": 0.5, "p10": -2.0, "p25": -1.0,
                           "p50": 0.2, "p75": 1.0, "p90": 2.0},
    }
    c = build_from_stats(stats)

    # blend takes the long tail, build/swap the short — the *ordering* of styles
    # stays a design choice, only the magnitudes are mined.
    assert c.get("blend_bars") == 22.0
    assert c.get("fade_bars") == 14.0
    assert c.get("build_bars") == 8.0
    assert c.get("swap_bars") == 8.0
    assert c.get("min_solo_bars") == 30.0
    assert c.get("min_dwell_bars") == 34.0
    assert c.get("max_dwell_bars") == 130.0
    assert c.get("harmonic_min_score") == 0.0
    assert c.corpus["n_mixes"] == 3


def test_build_from_stats_on_an_empty_corpus_yields_defaults():
    c = build_from_stats({"counts": {"n_mixes": 0, "n_transitions": 0,
                                     "n_accepted": 0},
                          "duration_bars": None, "solo_bars": None,
                          "dwell_bars": None, "camelot_score": None,
                          "tempo_step_pct": None, "cut_rate": None,
                          "n_at_confidence": 0})
    assert c.get("blend_bars") == 16
    assert c.get("min_solo_bars") == 32
    assert to_dict(c)["values"] == {}


def test_describe_reports_provenance():
    c = Calibration(blend_bars=Value(value=22.0, n=88, source="p75"))
    text = c.describe()
    assert "blend_bars" in text
    assert "mined" in text
    assert "fade_bars" in text and "default" in text
