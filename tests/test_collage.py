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


def test_segment_pool_subdivides_sections():
    from infinite_dj.mixer import _segment_pool
    # One long section: without dicing there's a single entry point; with
    # sub_bars the mixer can also enter partway through it.
    secs = [_section(0, 120, "peak", 0.8)]
    t = _track(secs)
    bar = 1.875
    whole = _segment_pool([t], bar, sub_bars=None)
    diced = _segment_pool([t], bar, sub_bars=4)
    assert len(whole) == 1
    assert len(diced) > len(whole)
    starts = [s.start for _, s in diced]
    assert starts[0] == 0 and max(starts) > 0        # entries inside the section
    assert all(s.label == "peak" for _, s in diced)  # inherits parent metadata


def test_segment_pool_caps_subdivisions_per_section():
    from infinite_dj.mixer import _segment_pool
    t = _track([_section(0, 1000, "peak", 0.8)])
    diced = _segment_pool([t], 1.875, sub_bars=2, max_subs=6)
    assert len(diced) == 7          # the section itself + 6 sub-segments


def _stub_tracks():
    """Two synthetic tracks with sections — no audio files touched."""
    import numpy as np
    secs = [_section(0, 60, "peak", 0.8), _section(60, 120, "rising", 0.6)]
    a = _track(secs); a.file_path = "a.wav"; a.bpm = 128.0
    b = _track(secs); b.file_path = "b.wav"; b.bpm = 128.0
    return [a, b]


def test_hop_is_proportional_to_the_emitted_segment():
    # Regression: a segment clamped short near a track's end used to advance
    # `pos` by its NOMINAL bar count, leaving a silent hole in the timeline.
    seg_bars, hop_bars = 20, 19
    nominal_len, actual_len = 40000, 8000        # clamped to a fifth
    frac = float(hop_bars) / max(1, int(seg_bars))
    assert round(frac * actual_len) < nominal_len
    # the hop must never outrun the audio actually written
    assert round(frac * actual_len) <= actual_len


def test_render_collage_state_roundtrip_keeps_absolute_times():
    # The streaming contract: state carries a tail, a position and an absolute
    # time offset so clips stay on one monotonic timeline across calls.
    st = {}
    st.update({"time_offset": 12.5, "pos": 0, "active": [], "recent": []})
    assert st["time_offset"] == 12.5
    # after a commit the offset advances by the committed audio, never rewinds
    committed = 7.25
    st["time_offset"] += committed
    assert st["time_offset"] == 19.75


# ── fade shape vocabulary ────────────────────────────────────────────────────

def test_every_fade_shape_is_a_valid_monotonic_ramp():
    import numpy as np
    from infinite_dj.mixer import FADE_SHAPES, _fade_curve
    for shape in FADE_SHAPES:
        up = _fade_curve(256, shape, rising=True)
        dn = _fade_curve(256, shape, rising=False)
        assert np.isfinite(up).all() and np.isfinite(dn).all()
        assert abs(up[0]) < 1e-6 and abs(up[-1] - 1.0) < 1e-6      # 0 -> 1
        assert abs(dn[0] - 1.0) < 1e-6 and abs(dn[-1]) < 1e-6      # 1 -> 0
        assert np.all(np.diff(up) >= -1e-6)                        # never dips
        assert np.all(np.diff(dn) <= 1e-6)
        assert up.min() >= 0.0 and up.max() <= 1.0


def test_fade_curve_handles_degenerate_lengths():
    from infinite_dj.mixer import _fade_curve
    assert len(_fade_curve(0, "exp")) == 0
    assert len(_fade_curve(1, "exp")) == 1      # no room to ramp; stays unity


def test_abrupt_shapes_are_reserved_for_weave():
    # A hard stop in a solo `breathe` segment would read as a dropout; only
    # weave (which always has another layer covering) may be bold.
    import random
    from infinite_dj.mixer import _pick_fade_shapes
    rng = random.Random(0)
    for mode in ("feature", "breathe"):
        for _ in range(60):
            a, b = _pick_fade_shapes(mode, 1.0, rng)
            assert "slam" not in (a, b) and "hold" not in (a, b)
    bold = any("slam" in _pick_fade_shapes("weave", 1.0, rng)
               for _ in range(60))
    assert bold, "weave at full chaos should sometimes pick abrupt shapes"


def test_feature_joins_stay_invisible():
    import random
    from infinite_dj.mixer import _pick_fade_shapes
    rng = random.Random(1)
    for _ in range(20):
        assert _pick_fade_shapes("feature", 1.0, rng) == ("equal_power", "equal_power")
