"""
`render_collage` used to sum overlapping layers unweighted (`master[pos:end]
+= seg`), with the only safety net a single global peak scalar applied once
after an entire buffer/chunk was already summed. High Serendipity (layers=5,
chaos=1.0) pushed that scalar hard enough to read as clipping, and in the
streaming/radio path the scalar only ever dropped, permanently quietening a
long session after one dense passage. See CHANGELOG for the full writeup.

These tests pin the fix: per-layer equal-power gain applied at placement time
(`_layer_gain`), a local lookahead limiter (`_limiter`) instead of a single
whole-buffer scalar, and a residual streaming ceiling that can recover.
"""
import numpy as np
import soundfile as sf

from infinite_dj.mixer import _layer_gain, _limiter, render_collage
from infinite_dj.models import Section, TrackMeta

SR = 44100


def _write_track(path, bpm: float, duration: float, seed: int,
                 amplitude: float = 0.8, loudness=-12.0) -> TrackMeta:
    """A synthetic track with real audio on disk — decorrelated tone + noise
    per track (different pitch/seed) so overlapping layers approximate the
    independent-source assumption `_layer_gain`'s equal-power law relies on."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    beat = 60.0 / bpm

    audio = amplitude * 0.15 * rng.standard_normal(n).astype(np.float32)
    for k in range(int(duration / beat)):
        s = int(k * beat * SR)
        env = np.exp(-np.arange(min(4000, n - s)) / 900.0)
        audio[s:s + len(env)] += amplitude * 0.6 * env * np.sin(
            2 * np.pi * 55.0 * t[s:s + len(env)])
    audio += amplitude * 0.3 * np.sin(2 * np.pi * (220.0 + 37 * seed) * t)
    sf.write(path, np.stack([audio, audio], axis=1), SR, subtype="PCM_16")

    downbeats = [round(k * beat * 4, 3) for k in range(int(duration / (beat * 4)))]
    return TrackMeta(
        file_path=str(path), title=path.stem, duration=duration,
        bpm=bpm, bpm_confidence=0.9,
        beats=[round(k * beat, 3) for k in range(int(duration / beat))],
        downbeats=downbeats, phrases=downbeats[::8],
        key="8A", key_name="A minor", key_confidence=0.8,
        energy_curve=[0.6] * int(duration),
        sections=[Section(start=0.0, end=duration, label="peak", energy=0.7,
                          embedding=[float(seed % 7 == i) for i in range(7)])],
        cue_points=[], analyzed_at=0.0, loudness=loudness,
    )


def _tracks(tmp_path, n=6, amplitude=0.8, loudness=-12.0):
    return [_write_track(tmp_path / f"t{i}.wav", bpm=128.0, duration=60.0,
                         seed=i, amplitude=amplitude, loudness=loudness)
           for i in range(n)]


# ── _layer_gain ──────────────────────────────────────────────────────────────

def test_layer_gain_formula():
    assert _layer_gain(1) == 1.0
    assert abs(_layer_gain(4) - 0.5) < 1e-9
    vals = [_layer_gain(n) for n in range(1, 9)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))   # monotonically decreasing
    assert _layer_gain(0) == _layer_gain(1)               # floor at 1 voice


# ── _limiter ─────────────────────────────────────────────────────────────────

def test_limiter_respects_ceiling_and_recovers():
    n = SR * 2
    audio = np.zeros((n, 2), dtype=np.float32)
    audio[SR:SR + 4410] = 2.0   # a hot transient well above the ceiling
    limited, final_gain = _limiter(audio, SR, ceiling=0.95)
    assert np.abs(limited).max() <= 0.97          # small tolerance for control-rate interp
    assert np.abs(limited[:SR - 100]).max() < 1e-6  # silence before the transient stays silent
    assert final_gain > 0.99                       # gain has recovered by the end (signal is silent again)


def test_limiter_carries_gain_and_tail_across_streaming_calls():
    """A quiet call following a loud one should not reset to full gain instantly
    (the ceiling only recovers slowly) or lose the lookahead context."""
    n = SR
    loud = np.full((n, 2), 2.0, dtype=np.float32)
    limited1, gain1 = _limiter(loud, SR)
    assert gain1 < 1.0

    quiet = np.full((n, 2), 0.01, dtype=np.float32)
    limited2, gain2 = _limiter(quiet, SR, pre_tail=loud[-700:], start_gain=gain1)
    assert gain2 >= gain1   # recovering, not reset or stuck below where it started


# ── render_collage: peak ceiling + per-clip gain compensation ────────────────

def test_render_collage_insane_stays_under_ceiling_and_compensates_dense_layers(tmp_path):
    tracks = _tracks(tmp_path, n=6)
    audio, sr, markers, clips = render_collage(
        tracks, target_length_sec=40.0, layers=5, chaos=1.0, seed=7)

    assert np.abs(audio).max() <= 0.97

    # Recompute, from the clip list alone, how many voices were sounding when
    # each clip was placed (mirrors `place()`'s `len(active) + 1`), and check
    # every clip's stored `layer_gain` matches the equal-power formula for
    # that concurrency — i.e. the compensation actually engaged, not just the
    # final peak clamp.
    assert any(c["layer_gain"] < 1.0 for c in clips), \
        "layers=5/chaos=1.0 should produce at least one overlapping placement"
    for i, c in enumerate(clips):
        n_voices = 1 + sum(
            1 for j in range(i)
            if clips[j]["out_end"] > c["out_start"] and clips[j]["out_start"] <= c["out_start"]
        )
        expected = round(1.0 / np.sqrt(max(1, n_voices)), 3)
        assert c["layer_gain"] == expected, (i, c["layer_gain"], expected, n_voices)


def test_render_collage_streaming_ceiling_recovers_after_a_dense_block(tmp_path):
    """
    Regression for the 'gain only ever drops' bug: a dense block should duck
    the residual streaming ceiling, but a calmer block afterward must recover
    it rather than staying pinned for the rest of the session.
    """
    dense_tracks = _tracks(tmp_path, n=6, amplitude=1.0)
    state: dict = {}
    render_collage(dense_tracks, target_length_sec=20.0, layers=5, chaos=1.0,
                   seed=3, state=state)
    ducked_gain = state["gain"]
    assert ducked_gain < 1.0, "a dense enough block should have ducked the ceiling"

    # A calmer follow-up block (loudness=None so _loudness_match doesn't renormalize
    # it back up — this is genuinely quiet audio, not just fewer layers).
    calm_tracks = _tracks(tmp_path, n=2, amplitude=0.05, loudness=None)
    for _ in range(6):   # several quiet blocks: recovery is slow (~0.02/block) by design
        render_collage(calm_tracks, target_length_sec=20.0, layers=2, chaos=0.0,
                       seed=11, state=state)
    assert state["gain"] > ducked_gain
