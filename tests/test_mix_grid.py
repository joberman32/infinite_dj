"""
Piecewise tempo tracking over a whole mix. Everything here runs on synthesized
click tracks — no audio files, no library — so the segmentation can be checked
against tempos that are known exactly.
"""
import numpy as np

from infinite_dj.analyzer import BEAT_HOP, SR
from infinite_dj.mix_grid import (
    TempoSegment,
    bars_between,
    beats_between,
    bpm_at,
    mix_onset_envelope,
    track_tempo_segments,
)


def click_track(sections, sr=SR, seed=0):
    """
    Render `[(bpm, seconds), ...]` as clicks with a little noise.

    Phase is carried across section boundaries so the grid doesn't reset — a DJ
    nudging the tempo doesn't restart the bar.
    """
    rng = np.random.default_rng(seed)
    out = []
    carry = 0.0
    for bpm, dur in sections:
        n = int(dur * sr)
        buf = 0.002 * rng.standard_normal(n).astype(np.float32)
        beat = 60.0 / bpm
        t = carry
        while t < dur:
            s = int(t * sr)
            env = np.exp(-np.arange(min(1200, n - s)) / 200.0).astype(np.float32)
            if len(env):
                tt = np.arange(len(env)) / sr
                buf[s:s + len(env)] += env * (
                    np.sin(2 * np.pi * 60.0 * tt) + 0.4 * np.sin(2 * np.pi * 3000.0 * tt)
                ).astype(np.float32)
            t += beat
        carry = t - dur
        out.append(buf)
    return np.concatenate(out)


def test_constant_tempo_gives_one_segment():
    y = click_track([(124.0, 90.0)])
    segs = track_tempo_segments(mix_onset_envelope(y), duration=len(y) / SR)

    assert len(segs) == 1
    assert abs(segs[0].bpm - 124.0) < 0.5
    assert segs[0].start == 0.0
    assert abs(segs[0].end - 90.0) < 0.1


def test_tracks_a_tempo_change():
    """60s at 124 then 60s at 132 must split into two segments at the change."""
    y = click_track([(124.0, 60.0), (132.0, 60.0)])
    segs = track_tempo_segments(mix_onset_envelope(y), duration=len(y) / SR)

    assert len(segs) == 2, [(s.bpm, s.start, s.end) for s in segs]
    assert abs(segs[0].bpm - 124.0) < 0.5
    assert abs(segs[1].bpm - 132.0) < 0.5
    # Boundary resolution is HOP_SEC/2 — the tempo track is sampled every 10 s,
    # so the change can only be placed within +/-5 s of the truth. That's fine
    # for our purposes: tempo is always read from clean windows either side of a
    # transition, never from the boundary itself, and a 5 s misplacement moves
    # `beats_between` by well under 1%.
    assert abs(segs[0].end - 60.0) < 8.0


def test_segments_are_contiguous_and_cover_the_mix():
    y = click_track([(120.0, 60.0), (128.0, 60.0)])
    dur = len(y) / SR
    segs = track_tempo_segments(mix_onset_envelope(y), duration=dur)

    assert segs[0].start == 0.0
    assert abs(segs[-1].end - dur) < 0.01
    for a, b in zip(segs[:-1], segs[1:]):
        assert abs(a.end - b.start) < 1e-9


def test_ignores_jitter_below_the_split_threshold():
    """A tempo that wanders under SPLIT_PCT must not manufacture segments."""
    y = click_track([(124.0, 40.0), (124.3, 40.0), (124.0, 40.0)])
    segs = track_tempo_segments(mix_onset_envelope(y), duration=len(y) / SR)

    assert len(segs) == 1, [(s.bpm, s.start, s.end) for s in segs]


def test_silence_yields_no_segments():
    y = np.zeros(int(SR * 30), dtype=np.float32)
    assert track_tempo_segments(mix_onset_envelope(y), duration=30.0) == []


def test_short_input_returns_a_single_segment():
    """Shorter than one analysis window still has to produce a usable tempo."""
    y = click_track([(128.0, 20.0)])
    segs = track_tempo_segments(mix_onset_envelope(y), duration=len(y) / SR)

    assert len(segs) == 1
    assert abs(segs[0].bpm - 128.0) < 1.0


def test_beats_between_integrates_across_segments():
    """
    A span crossing a tempo change must count beats per segment, not apply one
    tempo to the whole span — that's what makes a transition length in beats
    comparable across a set.
    """
    segs = [TempoSegment(0.0, 60.0, 120.0, 0.0, 1.0),
            TempoSegment(60.0, 120.0, 240.0, 0.0, 1.0)]

    assert abs(beats_between(segs, 0.0, 60.0) - 120.0) < 1e-6     # 2 beats/sec
    assert abs(beats_between(segs, 60.0, 120.0) - 240.0) < 1e-6   # 4 beats/sec
    assert abs(beats_between(segs, 30.0, 90.0) - (60.0 + 120.0)) < 1e-6
    assert abs(bars_between(segs, 0.0, 60.0) - 30.0) < 1e-6
    assert beats_between(segs, 50.0, 50.0) == 0.0


def test_bpm_at_picks_the_containing_segment():
    segs = [TempoSegment(0.0, 60.0, 120.0, 0.0, 0.9),
            TempoSegment(60.0, 120.0, 128.0, 0.0, 0.8)]

    assert bpm_at(segs, 10.0)[0] == 120.0
    assert bpm_at(segs, 90.0)[0] == 128.0
    assert bpm_at(segs, 500.0)[0] == 128.0     # clamps to the nearest
    assert np.isnan(bpm_at([], 10.0)[0])
