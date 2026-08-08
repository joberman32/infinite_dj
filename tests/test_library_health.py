"""
Tests for the library health reports.

The triage grading is pure metadata logic, so it's tested against synthetic
`TrackMeta`. The gap report's thresholds are pinned against the mixer's real
gates — if `choose_transition_style` moves, these fail rather than silently
reporting the wrong thing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infinite_dj.library_health import (  # noqa: E402
    BLEND_MAX_ENERGY, MIN_BPM_CONFIDENCE, MIN_CUES_PER_TYPE, MIN_DOWNBEATS,
    MIN_DURATION_SEC, SWAP_MIN_ENERGY, _beatmatchable, _cluster_bpms,
    grid_quality, library_gaps, triage,
)
from infinite_dj.models import CuePoint, Section, TrackMeta  # noqa: E402


def _track(*, bpm=128.0, bpm_conf=1.0, key="8A", key_conf=0.9, dur=300.0,
           n_down=64, n_cues=4, cue_energy=0.5, phrase_aligned=True,
           sections=True, title="T", beats_span=1.0):
    """A synthetic analyzed track, healthy by default."""
    beat_dt = 60.0 / bpm
    n_beats = int(dur * beats_span / beat_dt)
    beats = [i * beat_dt for i in range(n_beats)]
    downbeats = [i * beat_dt * 4 for i in range(n_down)]
    cues = []
    for t in ("in", "out"):
        for i in range(n_cues):
            cues.append(CuePoint(timestamp=10.0 + i * 20, type=t,
                                 phrase_aligned=phrase_aligned,
                                 energy=cue_energy, confidence=0.9 - i * 0.1))
    return TrackMeta(
        file_path=f"/tmp/{title}.wav", title=title, duration=dur,
        bpm=bpm, bpm_confidence=bpm_conf, beats=beats, downbeats=downbeats,
        phrases=downbeats[::4], key=key, key_name="A minor",
        key_confidence=key_conf, energy_curve=[0.5] * int(dur),
        sections=[Section(0.0, dur, "steady", 0.5)] if sections else [],
        cue_points=cues, loudness=-12.0, analyzed_at="2026-08-08T00:00:00Z",
    )


# ── Triage grading ───────────────────────────────────────────────────────────

def test_healthy_track_is_good():
    r = grid_quality(_track())
    assert r.grade == "good"
    assert r.reasons == []
    assert r.keep


@pytest.mark.parametrize("kwargs,fragment", [
    ({"dur": MIN_DURATION_SEC - 1, "n_down": 8},  "too short"),
    ({"n_down": MIN_DOWNBEATS - 1},               "downbeats"),
    ({"bpm_conf": MIN_BPM_CONFIDENCE - 0.01},     "tempo confidence"),
    ({"bpm": 200.0},                              "outside"),
    ({"n_cues": MIN_CUES_PER_TYPE - 1},           "too few cues"),
    ({"beats_span": 0.4},                         "beat grid covers"),
])
def test_hard_failures_are_rejected(kwargs, fragment):
    r = grid_quality(_track(**kwargs))
    assert r.grade == "reject", r.reasons
    assert not r.keep
    assert any(fragment in reason for reason in r.reasons), r.reasons


@pytest.mark.parametrize("kwargs,fragment", [
    ({"key_conf": 0.05},          "key confidence"),
    ({"sections": False},         "structural sections"),
    ({"phrase_aligned": False},   "phrase-aligned"),
    ({"bpm": 91.0},               "octave-fold edge"),
])
def test_soft_failures_are_usable_not_rejected(kwargs, fragment):
    r = grid_quality(_track(**kwargs))
    assert r.grade == "usable", r.reasons
    assert r.keep, "a soft problem must not lose the track"
    assert any(fragment in reason for reason in r.reasons), r.reasons


def test_triage_sorts_worst_first():
    tracks = [_track(title="ok"),
              _track(title="bad", dur=10.0, n_down=2),
              _track(title="meh", sections=False)]
    grades = [r.grade for r in triage(tracks)]
    assert grades == ["reject", "usable", "good"]


def test_a_confidently_wrong_grid_is_invisible():
    """
    The documented blind spot: the stored grid is equidistant by construction,
    so a wrong-but-confident tempo grades `good`. If this ever starts failing,
    triage got smarter and the docstring needs updating.
    """
    assert grid_quality(_track(bpm=128.0, bpm_conf=1.0)).grade == "good"


# ── Beatmatch + clustering ───────────────────────────────────────────────────

def test_beatmatchable_matches_plan_transition_logic():
    from infinite_dj.mixer import MAX_STRETCH
    assert _beatmatchable(128.0, 128.0, MAX_STRETCH)
    assert _beatmatchable(128.0, 130.0, MAX_STRETCH)      # ~1.6%
    assert _beatmatchable(128.0, 64.0, MAX_STRETCH)       # double
    assert _beatmatchable(128.0, 256.0, MAX_STRETCH)      # half
    assert not _beatmatchable(128.0, 150.0, MAX_STRETCH)  # ~17%
    assert not _beatmatchable(0.0, 128.0, MAX_STRETCH)


def test_cluster_bpms_groups_neighbours():
    clusters = _cluster_bpms([126.0, 127.0, 128.0, 150.0], width=4.0)
    assert clusters[0][1] == 3
    assert 126.0 <= clusters[0][0] <= 128.0
    assert clusters[-1][1] == 1


# ── Gap report ───────────────────────────────────────────────────────────────

def test_gap_thresholds_mirror_the_mixer_gates():
    """
    `library_health` duplicates the two energy gates rather than importing
    them. This pins the duplication: if `choose_transition_style` changes its
    0.45 / 0.70 thresholds, this fails instead of the report going quietly wrong.
    """
    import inspect

    from infinite_dj import mixer
    src = inspect.getsource(mixer.choose_transition_style)
    assert f"< {BLEND_MAX_ENERGY}" in src, "blend gate moved in mixer.py"
    assert f"> {SWAP_MIN_ENERGY}" in src, "swap gate moved in mixer.py"


def test_the_gap_report_plans_rather_than_re_deriving():
    """
    The report must go through `plan_transition`. An earlier version called
    `choose_transition_style` on `best_cue_out` / `best_cue_in`, which the set
    renderer never uses: measured on the 25-track library, that pairing put
    median exit energy at 0.15 against the planner's 0.51, and reported `swap`
    unreachable when it really fires on 6.3% of pairs.

    If this ever needs to change, re-measure both paths first — the two answers
    differ enough to invert a conclusion.
    """
    import inspect

    src = inspect.getsource(library_gaps)
    assert "plan_transition" in src
    assert "choose_transition_style" not in src, \
        "the gap report must not re-derive style selection"
    assert "_strongest" not in src, \
        "the globally strongest cue is not the one the renderer exits on"


def test_unreachable_style_is_reported():
    """A library of only mid-energy cues can never trigger blend or swap."""
    tracks = [_track(title=f"t{i}", bpm=128.0, cue_energy=0.55) for i in range(4)]
    g = library_gaps(tracks, target_hours=1.0)
    assert g.style_counts.get("blend", 0) == 0
    assert g.style_counts.get("swap", 0) == 0
    assert any("blend" in f and "unreachable" in f for f in g.findings)
    assert any("swap" in f and "unreachable" in f for f in g.findings)


def test_tempo_scattered_library_reports_low_beatmatch():
    tracks = [_track(title=f"t{i}", bpm=bpm)
              for i, bpm in enumerate([95.0, 118.0, 137.0, 162.0])]
    g = library_gaps(tracks, target_hours=1.0)
    assert g.beatmatchable_frac == 0.0
    assert g.style_counts.get("cut", 0) > 0
    cut = [f for f in g.findings if "hard cuts" in f]
    assert cut, g.findings
    assert "100%" in cut[0], "the cut share is the number worth leading with"


def test_a_mostly_cutting_library_is_flagged_even_when_some_pairs_match():
    """
    Regression: the finding used to key off `beatmatchable_frac < 0.30`, which
    stayed silent on the reference library at 36% beatmatchable — while 64.5%
    of its transitions were hard cuts. The share that matters is the one you
    hear, and it is the complement.
    """
    tracks = [_track(title=f"t{i}", bpm=bpm)
              for i, bpm in enumerate([126.0, 127.0, 128.0, 129.0, 150.0, 95.0])]
    g = library_gaps(tracks, target_hours=1.0)
    assert 0.30 < g.beatmatchable_frac < 0.50, g.beatmatchable_frac
    assert any("hard cuts" in f for f in g.findings), g.findings


def test_dense_cluster_reports_high_beatmatch():
    tracks = [_track(title=f"t{i}", bpm=bpm)
              for i, bpm in enumerate([126.0, 127.0, 128.0, 129.0])]
    g = library_gaps(tracks, target_hours=1.0)
    assert g.beatmatchable_frac == 1.0
    assert g.style_counts.get("cut", 0) == 0


def test_hours_shortfall_is_reported_with_a_track_estimate():
    tracks = [_track(title=f"t{i}", dur=300.0) for i in range(4)]
    g = library_gaps(tracks, target_hours=24.0)
    assert g.total_hours == pytest.approx(4 * 300 / 3600, abs=0.01)
    short = [f for f in g.findings if "short of" in f]
    assert short and "more tracks" in short[0]


def test_energy_extremes_are_counted():
    lo = _track(title="lo", cue_energy=BLEND_MAX_ENERGY - 0.1)
    hi = _track(title="hi", cue_energy=SWAP_MIN_ENERGY + 0.1)
    g = library_gaps([lo, hi], target_hours=1.0)
    assert g.low_energy_tracks == 1
    assert g.high_energy_tracks == 1
