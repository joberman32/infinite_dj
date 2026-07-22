import numpy as np

from infinite_dj.models import Section, TrackMeta
from infinite_dj.mixer import _clap_cos, _track_sections, _track_splice_points


def _section(start, end, label, energy, emb=None):
    return Section(start=start, end=end, label=label, energy=energy, embedding=emb)


def _track(sections):
    return TrackMeta(
        file_path="t.wav", title="t", duration=400.0, bpm=128.0, bpm_confidence=1.0,
        beats=[], downbeats=[], phrases=[], key="8A", key_name="A minor",
        key_confidence=1.0, energy_curve=[0.5] * 400, sections=sections,
        cue_points=[], analyzed_at=0.0,
    )


def test_clap_cos_bounds():
    a = [1.0, 0.0, 0.0]
    assert abs(_clap_cos(a, a) - 1.0) < 1e-6
    assert abs(_clap_cos([1, 0, 0], [0, 1, 0])) < 1e-6


def test_track_sections_filters_short_segments():
    secs = [_section(0, 4, "intro", 0.3), _section(4, 200, "peak", 0.8)]
    kept = _track_sections(_track(secs), min_len_sec=8.0)
    assert kept == [secs[1]]  # the 4s intro is dropped


def test_track_sections_falls_back_when_all_short():
    secs = [_section(0, 3, "a", 0.3), _section(3, 6, "b", 0.8)]
    assert _track_sections(_track(secs), min_len_sec=8.0) == secs


def test_splice_points_are_farthest_first_by_clap():
    # Strongest energy is the [1,0,0] section; the [0,1,0] section is orthogonal
    # to it (most contrasting) and should come SECOND, before the near-duplicate.
    strong = _section(0, 100, "peak", 0.9, emb=[1.0, 0.0, 0.0])
    dup    = _section(100, 200, "peak", 0.8, emb=[0.98, 0.0, 0.05])   # ~ same timbre
    ortho  = _section(200, 300, "sparse", 0.4, emb=[0.0, 1.0, 0.0])    # contrasting
    order = _track_splice_points(_track([strong, dup, ortho]))
    assert order[0] == strong.start          # strongest first
    assert order[1] == ortho.start           # most contrasting next
    assert order[2] == dup.start             # near-duplicate last
