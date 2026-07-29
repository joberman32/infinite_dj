"""
`plan_transition` is the cue+style decision lifted out of `render_set`'s loop so
the corpus validator can replay it without rendering audio. These tests pin the
extraction: a `plan_transition` loop must reproduce exactly what `render_set`
emits, so the two can never drift apart silently.
"""
import numpy as np
import soundfile as sf

from infinite_dj.mixer import MAX_STRETCH, plan_transition, render_set
from infinite_dj.models import CuePoint, Section, TrackMeta


SR = 44100


def _write_track(path, bpm: float, duration: float, seed: int) -> TrackMeta:
    """A synthetic track with a real beat grid, cue points and an energy curve."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    beat = 60.0 / bpm

    # Kick on every beat plus a little tonal content, so the mixer's 3-band
    # split has something to work with in every band.
    audio = 0.05 * rng.standard_normal(n).astype(np.float32)
    for k in range(int(duration / beat)):
        s = int(k * beat * SR)
        env = np.exp(-np.arange(min(4000, n - s)) / 900.0)
        audio[s:s + len(env)] += 0.6 * env * np.sin(
            2 * np.pi * 55.0 * t[s:s + len(env)])
    audio += 0.1 * np.sin(2 * np.pi * (220.0 + 30 * seed) * t)
    sf.write(path, np.stack([audio, audio], axis=1), SR, subtype="PCM_16")

    downbeats = [round(k * beat * 4, 3) for k in range(int(duration / (beat * 4)))]
    phrases = downbeats[::8]
    # Cue points spread across the track so the dwell filter has real choices.
    cues = []
    for frac, kind in ((0.05, "in"), (0.25, "in"), (0.45, "out"),
                       (0.65, "out"), (0.85, "out")):
        ts = min(downbeats, key=lambda d: abs(d - duration * frac))
        cues.append(CuePoint(timestamp=ts, type=kind, phrase_aligned=True,
                             energy=0.3 + 0.5 * frac, confidence=0.4 + frac / 2))
    return TrackMeta(
        file_path=str(path), title=path.stem, duration=duration,
        bpm=bpm, bpm_confidence=0.9,
        beats=[round(k * beat, 3) for k in range(int(duration / beat))],
        downbeats=downbeats, phrases=phrases,
        key="8A", key_name="A minor", key_confidence=0.8,
        energy_curve=[round(0.3 + 0.4 * (i / max(1, int(duration))), 4)
                      for i in range(int(duration))],
        sections=[Section(start=0.0, end=duration, label="steady", energy=0.5)],
        cue_points=cues, analyzed_at=0.0,
    )


def _tracks(tmp_path):
    # Two beatmatchable tempos and one far enough away to force a cut.
    return [
        _write_track(tmp_path / "a.wav", bpm=124.0, duration=150.0, seed=1),
        _write_track(tmp_path / "b.wav", bpm=126.0, duration=150.0, seed=2),
        _write_track(tmp_path / "c.wav", bpm=170.0, duration=150.0, seed=3),
    ]


def test_plan_transition_reproduces_render_set(tmp_path):
    """A plan_transition loop must predict render_set's markers exactly."""
    tracks = _tracks(tmp_path)
    audio, sr, markers, clips = render_set(tracks, min_solo_bars=4)

    assert len(markers) == len(tracks) - 1

    # Replay the same decisions without rendering. render_set starts the first
    # track at its entry cue and thereafter reads from the previous entry point.
    from infinite_dj.mixer import _set_entry_cue

    ci = _set_entry_cue(tracks[0])
    read_t = ci.timestamp if ci else 0.0
    for i, marker in enumerate(markers):
        planned = plan_transition(
            tracks[i], tracks[i + 1], read_t, tracks[i].duration,
            min_solo_bars=4, sim_threshold=0.82,
        )
        assert planned.style.name == marker.style, f"style drift at {i}"
        assert planned.beatmatched == (marker.method == "beatmatch")
        # render_set reports stretch as a percentage of the planned ratio.
        expected_pct = (planned.ratio - 1.0) * 100 if planned.beatmatched else 0.0
        assert abs(expected_pct - marker.stretch_pct) < 1e-6, f"ratio drift at {i}"
        read_t = planned.cue_in_t


def test_plan_transition_crossfade_length_matches_clips(tmp_path):
    """The style's bar count must explain the fade_out length render_set wrote."""
    tracks = _tracks(tmp_path)
    audio, sr, markers, clips = render_set(tracks, min_solo_bars=4)

    from infinite_dj.mixer import _set_entry_cue

    ci = _set_entry_cue(tracks[0])
    read_t = ci.timestamp if ci else 0.0
    for i in range(len(tracks) - 1):
        planned = plan_transition(
            tracks[i], tracks[i + 1], read_t, tracks[i].duration, min_solo_bars=4,
        )
        style = planned.style
        if style.is_cut:
            expected = style.cut_seconds
        else:
            expected = style.n_bars * (60.0 / tracks[i].bpm) * 4
        # clips[i]["fade_out"] is the realised crossfade; it can only be shorter
        # than planned if the outgoing track ran out of audio.
        assert clips[i]["fade_out"] <= expected + 0.01
        read_t = planned.cue_in_t


def test_beatmatch_decision_is_half_double_aware(tmp_path):
    """170 vs 86 BPM is a double-time match, not a cut."""
    a = _write_track(tmp_path / "x.wav", bpm=170.0, duration=60.0, seed=4)
    b = _write_track(tmp_path / "y.wav", bpm=86.0, duration=60.0, seed=5)
    planned = plan_transition(a, b, 0.0, a.duration, min_solo_bars=1)
    assert planned.beatmatched
    assert abs(planned.ratio - 1.0) <= MAX_STRETCH


def test_splice_mode_exits_inside_the_segment_window(tmp_path):
    """With a splice window the exit must land inside it, not after the dwell."""
    tracks = _tracks(tmp_path)
    a, b = tracks[0], tracks[1]
    planned = plan_transition(a, b, 10.0, a.duration, splice=(5.0, 25.0))
    assert 15.0 <= planned.cue_out_t <= 35.0


def test_dur_out_shortens_the_available_exit_window(tmp_path):
    """
    dur_out is the *loaded* audio length, not TrackMeta.duration — passing a
    short one must confine the exit, which is what protects render_set from
    cueing past the end of its buffer.
    """
    tracks = _tracks(tmp_path)
    a, b = tracks[0], tracks[1]
    planned = plan_transition(a, b, 0.0, 40.0, min_solo_bars=1)
    assert planned.cue_out_t <= 40.0
