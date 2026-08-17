"""
Camelot-wheel steps are 7 semitones apart (the circle of fifths) and 7 is
coprime to 12, so there's no "small shift, small wheel move" relationship: a
+/-1 semitone shift, measured against a track's OWN unshifted key, lands 5-7
Camelot hours away -- worse than doing nothing. But measured against an
ARBITRARY partner key (the actual question `pitch_shift_for_compatibility`
answers), a small shift routinely closes a large gap, because the partner may
already sit exactly where a small shift lands.

These tests pin the exhaustive result that justified building this feature at
all (78% of all ordered Camelot pairs improve within +/-3 semitones, median
needed shift +/-1) so a change to CAMELOT_MAP or camelot_compatibility can't
silently invalidate the premise, plus the function's specific contract:
smallest-|n_steps| tie-break, structurally-unreachable targets (parallel
major/minor and same-key both require a mode change, which pitch-shifting
can't do), and invalid-key handling.
"""
from collections import Counter

import pytest

from infinite_dj.harmony import (
    CAMELOT_MAP,
    CAMELOT_REVERSE,
    camelot_compatibility,
    pitch_shift_for_compatibility,
)

ALL_KEYS = list(CAMELOT_MAP.values())


def _shifted_key(key: str, n_steps: int) -> str:
    root, is_major = CAMELOT_REVERSE[key]
    return CAMELOT_MAP[((root + n_steps) % 12, is_major)]


# ── The premise, pinned exhaustively ────────────────────────────────────────

def test_the_premise_78_percent_of_pairs_improve_within_budget():
    """The measured result that justified building this feature at all."""
    n_pairs = n_improved = 0
    for a in ALL_KEYS:
        for b in ALL_KEYS:
            if a == b:
                continue
            n_pairs += 1
            if pitch_shift_for_compatibility(a, b) is not None:
                n_improved += 1
    assert n_pairs == 552
    assert n_improved == 432
    assert n_improved / n_pairs == pytest.approx(0.7826, abs=0.001)


def test_the_premise_median_needed_shift_is_one_semitone():
    shifts_needed = Counter()
    for a in ALL_KEYS:
        for b in ALL_KEYS:
            if a == b:
                continue
            result = pitch_shift_for_compatibility(a, b)
            if result is not None:
                shifts_needed[abs(result[0])] += 1
    # 240 of 432 improved pairs need only +/-1 semitone (measured in-session).
    assert shifts_needed[1] == 240
    assert shifts_needed[1] > shifts_needed[2] + shifts_needed[3]


def test_no_pair_is_ever_made_worse():
    """The function only ever returns a strict improvement, or None."""
    for a in ALL_KEYS:
        for b in ALL_KEYS:
            if a == b:
                continue
            baseline = camelot_compatibility(a, b)
            result = pitch_shift_for_compatibility(a, b)
            if result is not None:
                assert result[2] > baseline


# ── Specific worked examples ────────────────────────────────────────────────

def test_worked_example_two_steps_away_reaches_a_perfect_match():
    # 8B (C major) and 10B (D major) are 2 Camelot steps apart (baseline 0.6).
    # -2 semitones on 10B lands exactly on 8B.
    n_steps, new_key, score = pitch_shift_for_compatibility("8B", "10B")
    assert (n_steps, new_key, score) == (-2, "8B", 1.0)


def test_worked_example_four_steps_away_reaches_an_adjacent_key():
    # 8B and 12B are 4 Camelot steps apart -- outside every explicit tier,
    # baseline 0.0. +1 semitone on 12B lands on 7B, adjacent to 8B (0.8).
    n_steps, new_key, score = pitch_shift_for_compatibility("8B", "12B")
    assert (n_steps, new_key, score) == (1, "7B", 0.8)


def test_ties_break_toward_the_smallest_shift():
    """When two shifts reach the same best score, the smaller one wins."""
    for a in ALL_KEYS:
        for b in ALL_KEYS:
            if a == b:
                continue
            result = pitch_shift_for_compatibility(a, b)
            if result is None:
                continue
            n_steps, _, best_score = result
            for n in range(-3, 4):
                if n == 0 or n == n_steps:
                    continue
                alt_score = camelot_compatibility(a, _shifted_key(b, n))
                if alt_score == best_score:
                    assert abs(n_steps) <= abs(n)


# ── Structurally unreachable targets ────────────────────────────────────────

def test_parallel_major_minor_is_unreachable_by_pitch_shift():
    """Pitch-shifting transposes a key; it never changes major<->minor."""
    # 8B/8A (C major / A minor) are parallel: baseline 0.9, the best score
    # pitch-shifting alone could ever produce short of an exact key match.
    assert pitch_shift_for_compatibility("8B", "8A") is None


def test_same_key_has_nothing_left_to_improve():
    assert pitch_shift_for_compatibility("8B", "8B") is None


# ── Input handling ───────────────────────────────────────────────────────────

def test_missing_or_invalid_keys_return_none():
    assert pitch_shift_for_compatibility(None, "8B") is None
    assert pitch_shift_for_compatibility("8B", None) is None
    assert pitch_shift_for_compatibility("", "8B") is None
    assert pitch_shift_for_compatibility("8B", "not-a-key") is None


# ── mixer._time_stretch: the combined tempo+pitch render ───────────────────
#
# A shifted vocal has to keep its original timbre (formant preservation), and
# tempo+pitch have to land in ONE Rubber Band pass — two separate passes would
# double the processing and compound each stage's own quality loss. These
# tests render real tones through the actual function (not a mock of the
# math) and measure the result by FFT peak, the same way test_shelf_eq.py
# verifies its filters.

import subprocess

import numpy as np

from infinite_dj.mixer import _time_stretch

MSR = 44100


def _tone(freq: float, seconds: float = 3.0, sr: int = MSR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    mono = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _peak_freq(audio: np.ndarray, sr: int) -> float:
    mono = audio[:, 0]
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    return float(freqs[np.argmax(spectrum)])


def test_default_pitch_shift_is_a_true_no_op():
    """Every pre-existing call site omits the new parameter — it must
    reproduce the exact old behaviour, not just something close to it."""
    stereo = _tone(440)
    out = _time_stretch(stereo, MSR, 1.0)
    assert out is stereo   # identity, not just equality: zero overhead


def test_pure_tempo_stretch_leaves_pitch_unchanged():
    stereo = _tone(440)
    out = _time_stretch(stereo, MSR, 1.2, 0.0)
    assert abs(len(out) / len(stereo) - 1 / 1.2) < 0.01
    assert abs(_peak_freq(out, MSR) - 440.0) < 2.0


def test_pure_pitch_shift_leaves_duration_unchanged():
    stereo = _tone(440)
    out = _time_stretch(stereo, MSR, 1.0, 2.0)
    assert abs(len(out) / len(stereo) - 1.0) < 0.01
    assert abs(_peak_freq(out, MSR) - 440.0 * 2 ** (2 / 12)) < 2.0


def test_combined_tempo_and_pitch_shift_both_apply():
    """The actual crossfade case: stretched to match a tempo AND key-synced,
    together — not one silently dropped because the other short-circuited."""
    stereo = _tone(440)
    out = _time_stretch(stereo, MSR, 1.2, -2.0)
    assert abs(len(out) / len(stereo) - 1 / 1.2) < 0.01
    assert abs(_peak_freq(out, MSR) - 440.0 * 2 ** (-2 / 12)) < 2.0


def test_combined_case_is_one_rubberband_pass_per_channel_not_two():
    """Two passes would double the processing and compound quality loss —
    confirm --tempo and --pitch ride the same subprocess invocation."""
    calls = []
    orig = subprocess.check_call
    def spy(args, **kwargs):
        calls.append(args)
        return orig(args, **kwargs)
    subprocess.check_call = spy
    try:
        stereo = _tone(440, seconds=1.0)
        _time_stretch(stereo, MSR, 1.15, -1.5)
    finally:
        subprocess.check_call = orig

    assert len(calls) == 2   # one per stereo channel, same as tempo-only
    for args in calls:
        assert "--tempo" in args and "--pitch" in args and "--formant" in args


# ── sequencer.build_compatibility_graph: shift-aware scoring ───────────────

from infinite_dj.models import TrackMeta
from infinite_dj.sequencer import KEY_SHIFT_PENALTY_PER_SEMITONE, build_compatibility_graph


def _seq_track(key: str, path: str, bpm: float = 124.0) -> TrackMeta:
    return TrackMeta(
        file_path=f"/tmp/{path}.wav", title=path, duration=200.0,
        bpm=bpm, bpm_confidence=0.9, beats=[0.0], downbeats=[0.0],
        phrases=[0.0], key=key, key_name=key, key_confidence=0.9,
        energy_curve=[0.5] * 200, sections=[], cue_points=[], analyzed_at=0.0,
    )


def test_a_previously_excluded_pair_becomes_reachable():
    """8B and 12B score 0.0 unshifted — MIN_SCORE=0.3 would exclude the edge
    entirely. A key-sync shift is exactly what's supposed to rescue this."""
    tracks = [_seq_track("8B", "a"), _seq_track("12B", "b")]
    graph = build_compatibility_graph(tracks)
    edge = graph["/tmp/a.wav"][0]
    assert edge.key_shift_semitones == 1.0
    assert edge.harmonic > 0.3


def test_the_penalty_keeps_a_natural_match_ranked_above_a_shifted_one():
    """A pair that's already 0.8-compatible must still outrank one that only
    reaches 0.8 by spending a semitone of shift to get there."""
    tracks = [_seq_track("8B", "out"), _seq_track("9B", "natural"),
             _seq_track("12B", "shifted")]
    graph = build_compatibility_graph(tracks)
    edges = {e.track_b: e for e in graph["/tmp/out.wav"]}
    assert edges["/tmp/natural.wav"].key_shift_semitones is None
    assert edges["/tmp/shifted.wav"].key_shift_semitones == 1.0
    # Both reach the 0.8 tier, but the shifted one is docked.
    assert edges["/tmp/natural.wav"].harmonic == 0.8
    assert edges["/tmp/shifted.wav"].harmonic == pytest.approx(
        0.8 - KEY_SHIFT_PENALTY_PER_SEMITONE, abs=1e-6)
    assert edges["/tmp/natural.wav"].score > edges["/tmp/shifted.wav"].score


def test_an_already_good_pair_is_left_alone():
    """No shift recorded when nothing needed rescuing."""
    tracks = [_seq_track("8B", "a"), _seq_track("9B", "b")]
    graph = build_compatibility_graph(tracks)
    edge = graph["/tmp/a.wav"][0]
    assert edge.key_shift_semitones is None
    assert edge.harmonic == 0.8


# ── transpose_key + the snap-back regression ────────────────────────────────

from infinite_dj.harmony import transpose_key


def test_transpose_key_moves_the_root_and_keeps_the_mode():
    assert transpose_key("12B", 1) == "7B"     # matches pitch_shift_for_compatibility
    assert transpose_key("8B", 0) == "8B"
    assert transpose_key("8B", -12) == "8B"    # a full octave is the identity
    assert transpose_key("8A", 1)[-1] == "A"   # pitch shift never changes mode
    assert transpose_key("bad", 1) is None
    assert transpose_key(None, 1) is None


def test_the_promised_shift_actually_lands_where_it_says():
    """pitch_shift_for_compatibility's reported score must equal what you get
    by actually transposing — the two functions must not drift apart."""
    for a in ALL_KEYS:
        for b in ALL_KEYS:
            result = pitch_shift_for_compatibility(a, b)
            if result is None:
                continue
            n_steps, shifted_key, promised = result
            assert transpose_key(b, n_steps) == shifted_key
            assert camelot_compatibility(a, shifted_key) == promised


def test_a_key_synced_track_does_not_snap_back_after_the_crossfade(tmp_path):
    """The regression this fix exists for.

    Applying the shift only to the crossfade region left the incoming track
    snapping back to its native key the instant the blend ended — a fully
    audible key jump mid-track (measured at -0.98 semitones). Only TEMPO is
    per-transition by design; a semitone lurch is not an acceptable tradeoff
    the way a ~5% tempo nudge at a phrase boundary is.
    """
    import soundfile as sf
    from infinite_dj.mixer import render_set
    from infinite_dj.models import CuePoint, Section

    SR = 44100

    def make(path, bpm, key, tone_hz, dur=90.0):
        n = int(dur * SR)
        t = np.arange(n) / SR
        beat = 60.0 / bpm
        audio = 0.02 * np.random.RandomState(1).standard_normal(n).astype(np.float32)
        for k in range(int(dur / beat)):
            s = int(k * beat * SR)
            env = np.exp(-np.arange(min(4000, n - s)) / 900.0)
            audio[s:s + len(env)] += 0.5 * env * np.sin(2 * np.pi * 55.0 * t[s:s + len(env)])
        audio += 0.35 * np.sin(2 * np.pi * tone_hz * t)   # unambiguous fundamental
        sf.write(path, np.stack([audio, audio], axis=1), SR, subtype="PCM_16")
        db = [round(k * beat * 4, 3) for k in range(int(dur / (beat * 4)))]
        cues = [CuePoint(min(db, key=lambda d: abs(d - dur * f)), kind, True,
                        0.3 + 0.5 * f, 0.4 + f / 2)
                for f, kind in ((0.05, "in"), (0.45, "out"), (0.85, "out"))]
        return TrackMeta(
            file_path=str(path), title=str(path), duration=dur, bpm=bpm,
            bpm_confidence=0.9, beats=[round(k * beat, 3) for k in range(int(dur / beat))],
            downbeats=db, phrases=db[::8], key=key, key_name=key, key_confidence=0.9,
            energy_curve=[0.5] * int(dur),
            sections=[Section(0.0, dur, "steady", 0.5)], cue_points=cues, analyzed_at=0.0)

    # 8B -> 12B fires a +1 semitone shift; incoming's tone is 400 Hz.
    a = make(tmp_path / "a.wav", 124.0, "8B", 300.0)
    b = make(tmp_path / "b.wav", 124.0, "12B", 400.0)
    audio, sr, markers, clips = render_set([a, b], min_solo_bars=4)

    assert markers[0].pitch_shift_semitones == 1.0
    xf_end = clips[0]["out_end"]

    def fundamental(seg):
        spec = np.abs(np.fft.rfft(seg[:, 0] * np.hanning(len(seg))))
        fr = np.fft.rfftfreq(len(seg), 1 / sr)
        band = (fr > 250) & (fr < 600)
        return float(fr[band][np.argmax(spec[band])])

    during = fundamental(audio[int((xf_end - 1.6) * sr):int((xf_end - 0.1) * sr)])
    after = fundamental(audio[int((xf_end + 0.1) * sr):int((xf_end + 1.6) * sr)])

    # No audible jump at the boundary (a whole semitone is 100 cents; this
    # tolerance is just FFT bin resolution).
    assert abs(12 * np.log2(after / during)) < 0.2
    # And it really is still shifted, not silently reverted to 400 Hz native.
    assert abs(after - 400.0 * 2 ** (1 / 12)) < 8.0


def test_the_next_transition_plans_against_the_key_actually_sounding(tmp_path):
    """A track that was key-synced on the way in keeps its shifted key, so the
    NEXT transition must compare against that — not the analyzed key."""
    from infinite_dj.mixer import plan_transition
    from infinite_dj.models import CuePoint

    def meta(key, bpm=124.0, name="t"):
        return TrackMeta(
            file_path=f"/tmp/{name}.wav", title=name, duration=200.0, bpm=bpm,
            bpm_confidence=0.9, beats=[0.0], downbeats=[0.0], phrases=[0.0],
            key=key, key_name=key, key_confidence=0.9, energy_curve=[0.5] * 200,
            sections=[], cue_points=[CuePoint(5.0, "in", True, 0.5, 0.9),
                                     CuePoint(50.0, "out", True, 0.5, 0.9)],
            analyzed_at=0.0)

    out_track, in_track = meta("12B", name="out"), meta("8B", name="in")

    # Planned against the analyzed key (12B) vs. against the key it's actually
    # sounding in after a +1 shift (7B) must give different answers — that
    # difference is the whole point of threading out_key through.
    as_analyzed = plan_transition(out_track, in_track, 0.0, 200.0, min_solo_bars=1)
    as_sounding = plan_transition(out_track, in_track, 0.0, 200.0, min_solo_bars=1,
                                  out_key=transpose_key("12B", 1))
    assert as_analyzed.pitch_shift_semitones != as_sounding.pitch_shift_semitones
