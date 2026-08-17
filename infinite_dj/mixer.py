"""
Mixing engine — Phase 2.

Takes two analyzed tracks + their chosen cue points and renders
a beat-aligned, EQ-shaped crossfade transition.

Pipeline:
  1. Load both audio files at full SR
  2. Time-stretch the incoming track to match outgoing BPM
  3. Phase-align: find the nearest downbeat in incoming to target cue
  4. Build the crossfade region:
       bars 1-2:  full outgoing, fade incoming highs in
       bars 3-4:  cut outgoing bass, bring incoming bass in (bass swap)
       bars 5-6:  full incoming, fade outgoing highs out
  5. Render to file or return as numpy array
"""

import os
import hashlib
import numpy as np
import soundfile as sf
import librosa
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")

try:
    import pyrubberband as rb
    HAS_RUBBERBAND = True
except ImportError:
    HAS_RUBBERBAND = False

from .models import TrackMeta, CuePoint, Section
from .harmony import camelot_compatibility, pitch_shift_for_compatibility, transpose_key

# Full quality SR for output
MIX_SR = 44100


# ── Data types ────────────────────────────────────────────────────────────────

# Time-stretch beyond this fraction artifacts audibly (Rubber Band stays clean
# to ~6–8%). Past the budget we do a short "cut" instead of forcing a beatmatch.
MAX_STRETCH = 0.08


@dataclass
class TransitionPlan:
    """Everything needed to execute a single transition between two tracks."""
    track_out: TrackMeta
    track_in: TrackMeta
    cue_out: CuePoint          # exit point in track_out
    cue_in: CuePoint           # entry point in track_in
    n_mix_bars: int = 16       # how many bars the crossfade spans
    max_stretch: float = MAX_STRETCH
    bpm_target: float = 0.0    # resolved BPM to mix at (set on init)
    stretch_ratio: float = 1.0 # how much track_in is time-stretched
    beatmatched: bool = True   # False => tempos too far apart, do a cut
    method: str = "beatmatch"  # "beatmatch" | "cut"
    pitch_shift_semitones: float = 0.0  # key-sync shift applied to track_in

    def __post_init__(self):
        if self.bpm_target == 0.0:
            self.bpm_target = self.track_out.bpm

        out_bpm = self.track_out.bpm
        in_bpm  = self.track_in.bpm

        # Consider matching the incoming track directly, or at half/double time,
        # and take whichever needs the least stretch.
        candidates = [
            out_bpm / in_bpm,
            out_bpm / (in_bpm * 2.0),
            out_bpm / (in_bpm / 2.0),
        ]
        ratio = min(candidates, key=lambda r: abs(r - 1.0))

        if abs(ratio - 1.0) <= self.max_stretch:
            self.stretch_ratio = ratio
            self.beatmatched   = True
            self.method        = "beatmatch"
        else:
            # Too far to beatmatch cleanly — cut instead of mangling the tempo.
            self.stretch_ratio = 1.0
            self.beatmatched   = False
            self.method        = "cut"

        # Key-sync: only worth it when the two tracks actually overlap in a
        # beat-locked crossfade (a cut has no simultaneous harmonic content
        # to clash, so there's nothing to fix).
        self.pitch_shift_semitones = 0.0
        if self.beatmatched:
            shift = pitch_shift_for_compatibility(self.track_out.key, self.track_in.key)
            if shift is not None:
                self.pitch_shift_semitones = float(shift[0])


@dataclass
class MixResult:
    """Output of a rendered transition."""
    audio: np.ndarray      # stereo float32 array
    sr: int
    duration: float
    plan: TransitionPlan


# ── Audio I/O helpers ─────────────────────────────────────────────────────────

def _load_audio(file_path: str, sr: int = MIX_SR) -> tuple[np.ndarray, int]:
    """Load audio as stereo float32."""
    y, file_sr = sf.read(file_path, always_2d=True)
    if y.shape[1] == 1:
        y = np.hstack([y, y])  # mono -> stereo

    if file_sr != sr:
        # Resample each channel
        left  = librosa.resample(y[:, 0], orig_sr=file_sr, target_sr=sr)
        right = librosa.resample(y[:, 1], orig_sr=file_sr, target_sr=sr)
        y = np.stack([left, right], axis=1)

    return y.astype(np.float32), sr


def _time_to_samples(t: float, sr: int) -> int:
    return int(round(t * sr))


# ── Time-stretching ───────────────────────────────────────────────────────────

def _time_stretch(audio: np.ndarray, sr: int, ratio: float,
                  pitch_shift_semitones: float = 0.0) -> np.ndarray:
    """
    Time-stretch audio by `ratio` and, optionally, pitch-shift it by
    `pitch_shift_semitones` (key-sync, see harmony.pitch_shift_for_compatibility).
    ratio > 1.0 = slower (incoming track has lower BPM, stretch to match)
    ratio < 1.0 = faster (incoming track has higher BPM, compress)

    Both happen in a SINGLE Rubber Band pass when both are requested — two
    separate passes would double the processing and compound each stage's own
    quality loss, on top of the phase-mismatch a two-stage stretch/shift risks.
    Pitch-shifted with `--formant` (formant preservation), so a shifted vocal
    keeps its original timbre instead of "chipmunking".

    Uses pyrubberband (Rubber Band Library) for best quality. Falls back to
    librosa (mono, no formant preservation) when Rubber Band isn't installed.
    """
    no_stretch = abs(ratio - 1.0) < 0.001
    no_shift = abs(pitch_shift_semitones) < 0.001
    if no_stretch and no_shift:
        return audio  # Nothing to do

    if HAS_RUBBERBAND:
        if no_shift:
            left  = rb.time_stretch(audio[:, 0], sr, ratio)
            right = rb.time_stretch(audio[:, 1], sr, ratio)
        else:
            # rb.time_stretch() short-circuits on rate==1.0 before ever
            # touching rbargs, so a pure pitch-shift (no tempo change) has to
            # go through rb.pitch_shift() instead — its only guard is
            # n_steps==0. Passing `--tempo` through rbargs there still gets
            # both changes applied in the one rubberband invocation.
            rbargs = {"--formant": ""}
            if not no_stretch:
                rbargs["--tempo"] = ratio
            left  = rb.pitch_shift(audio[:, 0], sr, pitch_shift_semitones,
                                   rbargs=dict(rbargs))
            right = rb.pitch_shift(audio[:, 1], sr, pitch_shift_semitones,
                                   rbargs=dict(rbargs))
        return np.stack([left, right], axis=1).astype(np.float32)
    else:
        # Fallback: librosa phase vocoder (mono only, lower quality, no
        # formant preservation on the pitch-shifted path).
        mono = audio.mean(axis=1)
        if not no_stretch:
            mono = librosa.effects.time_stretch(mono, rate=1.0 / ratio)
        if not no_shift:
            mono = librosa.effects.pitch_shift(mono, sr=sr,
                                               n_steps=pitch_shift_semitones)
        stereo = np.stack([mono, mono], axis=1)
        return stereo.astype(np.float32)


# ── Phase alignment ───────────────────────────────────────────────────────────

def _find_nearest_downbeat(
    target_time: float,
    downbeats: list[float],
    search_window: float = 4.0,
    max_offset: Optional[float] = None,
) -> float:
    """
    After time-stretching, the downbeats shift. Find the downbeat
    in the stretched track that's closest to our target cue-in time,
    within a search window of ±search_window seconds.

    If the nearest downbeat is more than `max_offset` seconds away (e.g. half a
    bar), snapping there would break phrase alignment — keep the target instead.
    """
    candidates = [d for d in downbeats
                  if abs(d - target_time) <= search_window]
    if not candidates:
        return target_time  # fallback: use as-is
    best = min(candidates, key=lambda d: abs(d - target_time))
    if max_offset is not None and abs(best - target_time) > max_offset:
        return target_time
    return best


# ── Multiband EQ helpers ──────────────────────────────────────────────────────

# Three-band split points, modelling a DJ mixer's low/mid/high EQ. The bands are
# built by difference-of-lowpass so low+mid+high == the original signal exactly
# (no gain hole when all three are at unity).
LOW_CUT = 200.0     # bass / low-mid boundary
MID_CUT = 2600.0    # mid / treble boundary


def _apply_lowpass(audio: np.ndarray, sr: int, cutoff: float = 200.0) -> np.ndarray:
    """Extract bass content (below cutoff Hz)."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff / (sr / 2), btype='low', output='sos')
    left  = sosfilt(sos, audio[:, 0])
    right = sosfilt(sos, audio[:, 1])
    return np.stack([left, right], axis=1).astype(np.float32)


def _apply_highpass(audio: np.ndarray, sr: int, cutoff: float = 200.0) -> np.ndarray:
    """Extract mid+high content (above cutoff Hz)."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff / (sr / 2), btype='high', output='sos')
    left  = sosfilt(sos, audio[:, 0])
    right = sosfilt(sos, audio[:, 1])
    return np.stack([left, right], axis=1).astype(np.float32)


def _apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    return (audio * gain).astype(np.float32)


def _layer_gain(n_voices: int) -> float:
    """
    Equal-power headroom for a segment joining a stack of `n_voices` total
    concurrently sounding layers (including itself): 1/sqrt(N). Keeps summed
    power growing like ln(N) instead of N as `render_collage` stacks layers,
    without needing to know the future (which/how many layers join later).

    Known limitation, accepted: this is evaluated once, when a segment enters
    (see `place()`), so it doesn't adapt if layers around it end or new ones
    join afterward. `_limiter` is the backstop for whatever this
    approximation misses.
    """
    return 1.0 / np.sqrt(max(1, n_voices))


def _limiter(audio: np.ndarray, sr: int, ceiling: float = 0.95,
            lookahead_ms: float = 15.0, release_ms: float = 150.0,
            pre_tail: Optional[np.ndarray] = None,
            start_gain: float = 1.0, hop: int = 64) -> tuple:
    """
    Short-lookahead peak limiter: instant attack, smoothed release. Unlike a
    single whole-buffer peak scalar, this responds to the actual hot passage
    rather than dragging everything else in the buffer/chunk down with it.

    The envelope (forward-looking max of |audio| across channels) is computed
    at a downsampled control rate (every `hop` samples, ~1.5ms at 44.1kHz) via
    `scipy.ndimage.maximum_filter1d`, then linearly upsampled back to full
    rate — cheap even on a full-length render, since the inherently sequential
    release-smoothing loop only has to run at the control rate.

    `pre_tail` (the last `lookahead_ms` of raw audio from the previous call)
    and `start_gain` (the previous call's final gain) carry continuity across
    streaming calls, the same way `render_collage`'s `st["carry"]` does for
    the audio itself. Returns `(limited_audio, final_gain)`.
    """
    from scipy.ndimage import maximum_filter1d

    n = len(audio)
    if n == 0:
        return audio, start_gain

    look_n = max(1, int(sr * lookahead_ms / 1000.0))
    if pre_tail is not None and len(pre_tail):
        padded = np.concatenate([pre_tail[-look_n:], audio], axis=0)
        pad_n = min(look_n, len(pre_tail))
    else:
        padded = np.concatenate(
            [np.zeros((look_n, audio.shape[1]), dtype=audio.dtype), audio], axis=0)
        pad_n = look_n

    # Forward-looking envelope: at sample i we need the max over
    # [i, i+look_n), i.e. a max filter centred `look_n/2` samples ahead.
    mag = np.max(np.abs(padded), axis=1)
    env = maximum_filter1d(mag, size=look_n + 1, origin=-(look_n // 2))
    env = env[pad_n:pad_n + n]

    # Control-rate gain curve: instant attack (drop immediately to whatever
    # the lookahead peak demands), one-pole release back toward 1.0.
    ctrl_idx = np.arange(0, n, hop)
    ctrl_env = env[ctrl_idx]
    target = np.minimum(1.0, ceiling / np.maximum(ctrl_env, 1e-9))

    release_per_hop = np.exp(-hop / (sr * (release_ms / 1000.0)))
    ctrl_gain = np.empty(len(ctrl_idx), dtype=np.float64)
    g = float(start_gain)
    for i, tgt in enumerate(target):
        if tgt < g:
            g = tgt                                   # instant attack
        else:
            g = tgt + (g - tgt) * release_per_hop      # smoothed release
        ctrl_gain[i] = g
    final_gain = float(g)

    gain_curve = np.interp(np.arange(n), ctrl_idx, ctrl_gain)
    limited = (audio * gain_curve[:, None]).astype(np.float32)
    return limited, final_gain


def _split3(audio: np.ndarray, sr: int) -> tuple:
    """
    Split into (low, mid, high) bands by difference-of-lowpass so the three sum
    back to the input exactly. Used for offline (whole-region) rendering.
    """
    lp_low = _apply_lowpass(audio, sr, LOW_CUT)
    lp_mid = _apply_lowpass(audio, sr, MID_CUT)
    low  = lp_low
    mid  = lp_mid - lp_low
    high = audio[:len(lp_mid)] - lp_mid
    return low, mid, high


@dataclass
class CrossfadeFilterState:
    """Stateful 3-band split for a chunked crossfade.

    ``sosfilt`` resets to silence when called without ``zi``.  That is fine for
    an offline region rendered in one call, but produces a filter transient at
    every producer chunk in the real-time engine.  This object keeps the two
    lowpass states (per source/channel) for the lifetime of a live transition,
    so chunked rendering matches a single continuous render sample-for-sample.
    Bands are reconstructed as low = lp(LOW), mid = lp(MID) - lp(LOW),
    high = x - lp(MID).
    """
    low_sos: np.ndarray
    mid_sos: np.ndarray
    out_low_zi: np.ndarray
    out_mid_zi: np.ndarray
    in_low_zi: np.ndarray
    in_mid_zi: np.ndarray

    @classmethod
    def create(cls, sr: int = MIX_SR):
        from scipy.signal import butter
        low_sos = butter(4, LOW_CUT / (sr / 2), btype="low", output="sos")
        mid_sos = butter(4, MID_CUT / (sr / 2), btype="low", output="sos")
        low_shape = (low_sos.shape[0], 2, 2)   # (sections, delay-elems, channels)
        mid_shape = (mid_sos.shape[0], 2, 2)
        return cls(
            low_sos=low_sos, mid_sos=mid_sos,
            out_low_zi=np.zeros(low_shape, dtype=np.float64),
            out_mid_zi=np.zeros(mid_shape, dtype=np.float64),
            in_low_zi=np.zeros(low_shape, dtype=np.float64),
            in_mid_zi=np.zeros(mid_shape, dtype=np.float64),
        )

    @staticmethod
    def _filter(audio: np.ndarray, sos: np.ndarray, zi: np.ndarray) -> tuple:
        from scipy.signal import sosfilt
        filtered = np.empty_like(audio, dtype=np.float32)
        next_zi = np.empty_like(zi)
        for channel in range(2):
            filtered[:, channel], next_zi[:, :, channel] = sosfilt(
                sos, audio[:, channel], zi=zi[:, :, channel]
            )
        return filtered, next_zi

    def split(self, out_audio: np.ndarray, in_audio: np.ndarray) -> tuple:
        o_lplow, self.out_low_zi = self._filter(out_audio, self.low_sos, self.out_low_zi)
        o_lpmid, self.out_mid_zi = self._filter(out_audio, self.mid_sos, self.out_mid_zi)
        i_lplow, self.in_low_zi  = self._filter(in_audio,  self.low_sos, self.in_low_zi)
        i_lpmid, self.in_mid_zi  = self._filter(in_audio,  self.mid_sos, self.in_mid_zi)
        o_low, o_mid, o_high = o_lplow, o_lpmid - o_lplow, out_audio - o_lpmid
        i_low, i_mid, i_high = i_lplow, i_lpmid - i_lplow, in_audio - i_lpmid
        return o_low, o_mid, o_high, i_low, i_mid, i_high


# ── Shelving EQ (real DJ-mixer topology) ──────────────────────────────────────
#
# WHY THIS EXISTS. The `_split3` path above splits into three bands and re-sums
# them with independent gains. That reconstructs perfectly at unity, but the
# bands are causal IIR outputs with *different phase*, so they only cancel
# correctly when their gains are equal. Scale them differently — which is the
# entire point of a DJ EQ — and the crossover region misbehaves.
#
# Measured on `_blend` with a 60 Hz tone (see CHANGELOG 2026-08-15): at
# crossfade phase 0.7 the bass-swap lane reads 0.00, meaning "outgoing bass
# fully cut", yet 0.33 amplitude (-9.6 dB) of the outgoing kick survives —
# it rides out on the *mid* band's envelope instead. Sweeping a tone through
# the "low band killed" state boosts 80-200 Hz by up to +5 dB rather than
# removing it. The single-source bass swap that the whole design rests on
# ("only one kick ever plays") therefore does not happen.
#
# A real DJ mixer's EQ is not a band-split. It is a cascade of minimum-phase
# shelving/peaking filters applied to the signal itself, where the magnitude
# response *is* the intended curve — no re-summing, so nothing to mis-cancel.
# That is what this section implements. (Topology idea from Vande Veire &
# De Bie's auto-DJ; their code is AGPL, so this is a fresh implementation from
# the published RBJ Audio EQ Cookbook formulas.)

# Which EQ a crossfade uses when the caller doesn't supply a filter state:
#   "shelf" — minimum-phase shelving cascade (this section); the bass swap works
#   "split" — the legacy `_split3` band-split, kept reachable for A/B and
#             because `CrossfadeFilterState` callers still select it explicitly
EQ_TOPOLOGY = "shelf"

SHELF_LOW_HZ  = LOW_CUT     # low-shelf corner (bass knob)
SHELF_HIGH_HZ = MID_CUT     # high-shelf corner (treble knob)
SHELF_MID_HZ  = float(np.sqrt(LOW_CUT * MID_CUT))  # ~721 Hz, geometric centre
SHELF_MID_Q   = 0.50        # mid bell width (see the note below)
SHELF_SLOPE   = 1.00        # RBJ shelf `S`: steepest that stays monotonic
SHELF_STAGES  = 2           # identical shelves cascaded, each at 1/N of the dB
EQ_FLOOR_DB   = -30.0       # a killed band is attenuated to this, not to -inf
EQ_CTRL_HOP   = 512         # samples between coefficient updates (~12ms @44.1k)

# SHELF_STAGES / EQ_FLOOR_DB were picked by measuring the transition width, not
# by ear. Cascading N shelves at dB/N gives a steeper skirt than one shelf at
# the full dB, so the kill reaches the kick without dragging the low-mids with
# it. At matched bass attenuation (60 Hz -> 0.065):
#
#   1 x -24 dB   60Hz 0.067   400Hz 0.710   900Hz 0.981
#   2 x -12 dB   60Hz 0.065   400Hz 0.813   900Hz 0.991   <- same kill, less collateral
#
# The floor is the depth at full kill. -30 dB puts the silenced kick ~30 dB
# under the incoming one, comfortably masked, while costing only ~2.5 dB at
# 400 Hz. Deeper floors buy inaudible extra kill for real low-mid loss
# (2 x -20 dB: 60Hz 0.011 but 400Hz 0.619).
#
# SHELF_MID_Q trades how evenly the mid lane covers its 3.7-octave band
# against how far the bell's skirts reach into its neighbours. Measured with
# the mid lane at 0.5 (spread across 300-2400 Hz) and at full kill (60 Hz
# survival when the mid alone is killed):
#
#   Q      mid spread     60Hz when mid killed
#   0.35        0.22                      0.60
#   0.50        0.31                      0.73   <- chosen
#   0.70        0.38                      0.83
#   1.00        0.43                      0.90
#
# Lower Q is flatter, and flatness is what runs constantly: the mid lane is
# the *primary midrange crossfade*, so an uneven bell turns a level change
# into a notch at 720 Hz. The collateral column only bites when the mid is
# killed with both neighbours wide open, which the current profiles never
# reach (their mid lead is <=0.50 while the bass swap completes at >=0.62).
# 0.50 favours the case that always happens without going all the way to the
# 0.30 a 3.7-octave bell would theoretically use, leaving headroom if someone
# later adds a style with a much longer mid lead.


def _biquad_low_shelf(sr: int, f0: float, gain_db: float,
                      slope: Optional[float] = None) -> np.ndarray:
    """RBJ low-shelf biquad as a single normalized SOS row."""
    slope = SHELF_SLOPE if slope is None else slope
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    tsa = 2.0 * np.sqrt(A) * alpha
    b0 =        A * ((A + 1.0) - (A - 1.0) * cw + tsa)
    b1 =  2.0 * A * ((A - 1.0) - (A + 1.0) * cw)
    b2 =        A * ((A + 1.0) - (A - 1.0) * cw - tsa)
    a0 =            ((A + 1.0) + (A - 1.0) * cw + tsa)
    a1 =     -2.0 * ((A - 1.0) + (A + 1.0) * cw)
    a2 =            ((A + 1.0) + (A - 1.0) * cw - tsa)
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def _biquad_high_shelf(sr: int, f0: float, gain_db: float,
                       slope: Optional[float] = None) -> np.ndarray:
    """RBJ high-shelf biquad as a single normalized SOS row."""
    slope = SHELF_SLOPE if slope is None else slope
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    tsa = 2.0 * np.sqrt(A) * alpha
    b0 =        A * ((A + 1.0) + (A - 1.0) * cw + tsa)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cw)
    b2 =        A * ((A + 1.0) + (A - 1.0) * cw - tsa)
    a0 =            ((A + 1.0) - (A - 1.0) * cw + tsa)
    a1 =      2.0 * ((A - 1.0) - (A + 1.0) * cw)
    a2 =            ((A + 1.0) - (A - 1.0) * cw - tsa)
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def _biquad_peaking(sr: int, f0: float, gain_db: float,
                    q: Optional[float] = None) -> np.ndarray:
    """RBJ peaking-EQ biquad as a single normalized SOS row."""
    q = SHELF_MID_Q if q is None else q
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def _n_sections() -> int:
    """Biquads in one cascade: N low shelves + 1 mid bell + N high shelves."""
    return 2 * SHELF_STAGES + 1


def _shelf_sos(sr: int, g_low: float, g_mid: float, g_high: float) -> np.ndarray:
    """
    Build the EQ cascade for one set of *relative* band gains.

    Gains are linear in [0, 1] (a DJ EQ cuts; it never boosts here). They are
    relative — see `_split_overall`, which factors the common level out first —
    so all-equal gains give an exactly flat cascade rather than three filters
    that happen to cancel.

    Each shelf is realised as `SHELF_STAGES` identical biquads at 1/N of the
    requested dB, which is steeper than one biquad at the full dB (see the
    measurements beside `SHELF_STAGES`). The mid stays a single bell: halving
    a peaking filter's gain also changes its effective width, so the same
    trick doesn't transfer cleanly.
    """
    def db(g):
        return max(EQ_FLOOR_DB, 20.0 * np.log10(max(float(g), 1e-9)))
    n = max(1, SHELF_STAGES)
    low  = _biquad_low_shelf(sr, SHELF_LOW_HZ, db(g_low) / n)
    high = _biquad_high_shelf(sr, SHELF_HIGH_HZ, db(g_high) / n)
    return np.vstack([low] * n
                     + [_biquad_peaking(sr, SHELF_MID_HZ, db(g_mid))]
                     + [high] * n)


def _split_overall(g_low: np.ndarray, g_mid: np.ndarray, g_high: np.ndarray) -> tuple:
    """
    Factor per-sample band lanes into (overall_level, relative band gains).

    The lanes carry two things at once: how loud this track is overall, and how
    it is shaped. A shelf cascade can only express the shaping — it bottoms out
    at `EQ_FLOOR_DB` and can never reach true silence, yet the incoming track's
    lanes are all 0.0 at phase 0 and it must contribute *nothing* rather than a
    floor's worth of bleed. Pulling `overall = max(lanes)` out as a plain scalar
    gain restores both exactness properties the band-split had for free:

      - all lanes equal  -> flat cascade x overall   (no spectral colouring)
      - all lanes zero   -> overall 0                (true digital silence)
    """
    overall = np.maximum.reduce([g_low, g_mid, g_high])
    safe = np.maximum(overall, 1e-9)
    return overall, g_low / safe, g_mid / safe, g_high / safe


def _hop_segments(pos: int, n: int, hop: int):
    """
    Split a chunk into segments on the *global* control grid.

    Segment boundaries are multiples of `hop` in absolute crossfade position,
    never relative to this chunk. That is what keeps a real-time chunked render
    sample-identical to one continuous offline render: where the producer
    happens to cut its chunks cannot shift the coefficient-update grid.
    """
    i = 0
    while i < n:
        j = min(i + hop - ((pos + i) % hop), n)
        yield i, j
        i = j


@dataclass
class ShelfEQState:
    """
    Time-varying shelving EQ for one source, safe to drive in chunks.

    Biquad coefficients are recomputed once per `EQ_CTRL_HOP` samples from the
    automation lanes and held across the segment, with filter state (`zi`)
    carried through — the same streaming discipline `CrossfadeFilterState`
    uses. Coefficient updates every ~12 ms are far faster than the lanes move
    (they traverse a whole 8-16 bar crossfade), so the automation stays smooth
    without paying for per-sample coefficient computation in a recursive filter.
    """
    sr: int
    hop: int
    pos: int
    zi: np.ndarray                      # (sections, 2 delay elems, 2 channels)
    sos: Optional[np.ndarray] = None    # coefficients in force, carried across chunks

    @classmethod
    def create(cls, sr: int = MIX_SR, hop: int = EQ_CTRL_HOP):
        return cls(sr=sr, hop=hop, pos=0,
                   zi=np.zeros((_n_sections(), 2, 2), dtype=np.float64))

    def process(self, audio: np.ndarray, g_low: np.ndarray,
                g_mid: np.ndarray, g_high: np.ndarray) -> np.ndarray:
        """Apply the automated EQ to `audio` given per-sample band lanes."""
        from scipy.signal import sosfilt
        n = len(audio)
        if n == 0:
            return audio
        overall, r_low, r_mid, r_high = _split_overall(g_low, g_mid, g_high)
        out = np.empty_like(audio, dtype=np.float32)
        for i, j in _hop_segments(self.pos, n, self.hop):
            # Recompute only when this segment begins on a true grid boundary.
            # A chunk that starts mid-segment must keep using the coefficients
            # derived at that segment's global start — which lies in an earlier
            # chunk, so they have to be carried in state, exactly like `zi`.
            # Sampling the lane at the chunk's own start instead would make the
            # render depend on where the producer happened to cut its chunks.
            if self.sos is None or (self.pos + i) % self.hop == 0:
                self.sos = _shelf_sos(self.sr, r_low[i], r_mid[i], r_high[i])
            filtered, self.zi = sosfilt(self.sos, audio[i:j], axis=0, zi=self.zi)
            out[i:j] = filtered
        self.pos += n
        return (out * overall.reshape(-1, 1)).astype(np.float32)


@dataclass
class ShelfCrossfadeState:
    """
    The two `ShelfEQState`s of one crossfade, so the shelf topology plugs into
    the same "create once per transition, pass to every chunk" contract the
    engine already uses for `CrossfadeFilterState`.
    """
    out_eq: ShelfEQState
    in_eq: ShelfEQState

    @classmethod
    def create(cls, sr: int = MIX_SR, hop: int = EQ_CTRL_HOP):
        return cls(out_eq=ShelfEQState.create(sr, hop),
                   in_eq=ShelfEQState.create(sr, hop))


def make_crossfade_state(sr: int = MIX_SR):
    """
    The per-transition filter state matching the configured EQ topology.

    Callers that stream a crossfade in chunks (the real-time engine) go through
    this rather than naming a state class, so flipping `EQ_TOPOLOGY` switches
    live playback and offline rendering together instead of leaving the engine
    pinned to whichever class it happened to import.
    """
    if EQ_TOPOLOGY == "shelf":
        return ShelfCrossfadeState.create(sr)
    return CrossfadeFilterState.create(sr)


# ── Crossfade shapes ──────────────────────────────────────────────────────────

def _equal_power_fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Equal-power crossfade curves of length n.
    Returns (fade_out, fade_in) — both maintain constant perceived loudness.
    """
    t = np.linspace(0, np.pi / 2, n)
    fade_out = np.cos(t).astype(np.float32)
    fade_in  = np.sin(t).astype(np.float32)
    return fade_out, fade_in


def _linear_fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    fade_out = np.linspace(1, 0, n, dtype=np.float32)
    fade_in  = np.linspace(0, 1, n, dtype=np.float32)
    return fade_out, fade_in


# ── Fade shapes ───────────────────────────────────────────────────────────────
#
# The collage overlap-adds independent segments rather than crossfading a pair,
# so each segment carries its own in/out envelope. Giving those envelopes a
# vocabulary — and letting the two ends differ — is what turns a uniform
# "everything fades the same way" collage into one with gesture.

FADE_SHAPES = ("equal_power", "linear", "exp", "log", "scurve", "hold", "slam")


def _fade_curve(n: int, shape: str = "equal_power", rising: bool = True) -> np.ndarray:
    """An n-sample gain ramp: 0→1 when `rising`, else 1→0."""
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float32)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if not rising:
        x = x[::-1]                      # mirror: same curve, run backwards
    if shape == "linear":
        g = x
    elif shape == "exp":                 # sneaks in late, then arrives
        g = x ** 2.5
    elif shape == "log":                 # arrives at once, then settles
        g = 1.0 - (1.0 - x) ** 2.5
    elif shape == "scurve":              # gentle at both ends
        g = x * x * (3.0 - 2.0 * x)
    elif shape == "hold":                # mostly full, quick knee at the edge
        g = np.clip(x * 4.0, 0.0, 1.0)
    elif shape == "slam":                # a few ms of ramp: a cut without a click
        g = np.clip(x * 12.0, 0.0, 1.0)
    else:                                # equal_power — constant perceived level
        g = np.sin(x * (np.pi / 2.0))
    return g.astype(np.float32)


# ── Per-layer EQ moves ────────────────────────────────────────────────────────
#
# `_blend` rides a 3-band EQ across a *pair* of tracks, which doesn't fit the
# collage's N-way overlap-add. These are the per-layer equivalent: each segment
# carries its own low/mid/high envelope over its lifetime, so a layer can open
# up like a DJ lifting a high-pass, or stack over an existing groove without
# fighting it for the low end.
#
# "flat" returns the segment untouched, so the common case costs nothing — the
# band split is only paid for when a gesture actually uses it.

EQ_MOVES = ("flat", "sweep_in", "sweep_out", "bass_first", "lowcut")


def _split3_zerophase(audio: np.ndarray, sr: int) -> tuple:
    """
    Zero-phase (low, mid, high) split, for offline whole-segment processing.

    `_split3` filters causally, which is required for the real-time engine's
    incremental `CrossfadeFilterState` — but a causal IIR phase-shifts each
    band, so turning one *down* does not cancel it. Measured: attenuating the
    low band 85% causally *raised* low-band energy to 1.13x; zero-phase gives
    the intended 0.24x. The collage has whole segments in hand, so it can and
    should filter zero-phase.
    """
    from scipy.signal import butter, sosfiltfilt
    lo_sos = butter(4, LOW_CUT / (sr / 2), btype="low", output="sos")
    mid_sos = butter(4, MID_CUT / (sr / 2), btype="low", output="sos")
    lp_low = sosfiltfilt(lo_sos, audio, axis=0)
    lp_mid = sosfiltfilt(mid_sos, audio, axis=0)
    return (lp_low.astype(np.float32),
            (lp_mid - lp_low).astype(np.float32),
            (audio - lp_mid).astype(np.float32))


def _apply_eq_move(seg: np.ndarray, sr: int, move: str, fade_len: int) -> np.ndarray:
    """Apply a per-band gain envelope to one segment."""
    # sosfiltfilt pads internally, so very short slices can't be filtered.
    if move == "flat" or len(seg) < 64:
        return seg
    low, mid, high = _split3_zerophase(seg, sr)
    n = len(seg)
    lo_g = np.ones(n, dtype=np.float32)
    hi_g = np.ones(n, dtype=np.float32)
    span = int(np.clip(fade_len * 3, 1, n))
    if move == "sweep_in":               # highs first, low end swings in after
        lo_g[:span] = _fade_curve(span, "scurve", rising=True)
    elif move == "sweep_out":            # low end leaves early, highs linger
        lo_g[-span:] = _fade_curve(span, "scurve", rising=False)
    elif move == "bass_first":           # low end lands, the top arrives after
        hi_g[:span] = _fade_curve(span, "scurve", rising=True)
    elif move == "lowcut":               # never claims the low end at all
        lo_g[:] = 0.15                   # a trace, so it doesn't sound hollow
    return (low * lo_g.reshape(-1, 1) + mid
            + high * hi_g.reshape(-1, 1)).astype(np.float32)


def _pick_eq_move(mode: str, chaos: float, n_active: int, rng) -> str:
    """
    Choose a layer's EQ gesture. Feature joins stay flat so contiguous sections
    of one track don't audibly change tone mid-track. When several layers are
    already sounding, a new one is more likely to enter low-cut — the expressive
    version of "one source holds the bottom", chosen per layer rather than
    imposed as a global rule.
    """
    if mode == "feature":
        return "flat"
    if mode == "breathe":
        return "sweep_in" if rng.random() < 0.3 else "flat"
    if n_active >= 2 and rng.random() < 0.5 + 0.3 * chaos:
        return "lowcut"
    r = rng.random()
    if r < 0.35 * chaos:
        return "sweep_in"
    if r < 0.50 * chaos:
        return rng.choice(("sweep_out", "bass_first"))
    return "flat"


def _pick_fade_shapes(mode: str, chaos: float, rng) -> tuple:
    """
    (fade_in, fade_out) shapes for one placement.

    Bold, abrupt shapes are reserved for `weave`, where heavy overlap means
    another layer is always covering — a hard stop in a solo `breathe` segment
    would just read as a dropout.
    """
    if mode == "feature":
        # Contiguous sections of one track: the join should be invisible.
        return "equal_power", "equal_power"
    if mode == "breathe":
        # One thing at a time — let it swell in and settle out.
        return rng.choice(("scurve", "exp")), rng.choice(("scurve", "equal_power"))
    if rng.random() < chaos * 0.6:       # weave, feeling bold
        return (rng.choice(("slam", "hold", "log")),
                rng.choice(("hold", "slam", "equal_power")))
    return rng.choice(("equal_power", "scurve", "exp")), "equal_power"


def _loudness_match(
    audio: np.ndarray,
    src_loudness: Optional[float],
    dst_loudness: Optional[float],
    max_gain_db: float = 9.0,
) -> np.ndarray:
    """
    Scale `audio` (whose integrated loudness is `src_loudness` dBFS) toward
    `dst_loudness` so the two tracks sit at the same level through a transition.
    Gain is clamped so a very quiet track isn't blown up into clipping/noise.
    """
    if src_loudness is None or dst_loudness is None:
        return audio
    gain_db = float(np.clip(dst_loudness - src_loudness, -max_gain_db, max_gain_db))
    if abs(gain_db) < 0.1:
        return audio
    return (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)


# ── Breakpoint EQ automation ───────────────────────────────────────────────────
#
# A crossfade is described by automation "lanes": piecewise-linear breakpoints
# over the crossfade phase (0 → 1). Each track has a volume lane plus one gain
# lane per EQ band (low/mid/high), so a style is just a table of curves — much
# more expressive than a few scalar knobs, and the natural way to model a DJ
# riding three EQ faders. (Idea from Vande Veire & De Bie's auto-DJ; their code
# is AGPL, so this is a fresh implementation.)

Lane = tuple  # ((phase, value), ...) with phase ascending, spanning 0..1


def _sample_lane(points: Lane, phase: np.ndarray) -> np.ndarray:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return np.interp(phase, xs, ys).astype(np.float32)


@dataclass
class TransitionProfile:
    """Per-track volume + 3-band gain automation for a crossfade."""
    out_vol: Lane; out_low: Lane; out_mid: Lane; out_high: Lane
    in_vol: Lane;  in_low: Lane;  in_mid: Lane;  in_high: Lane


def _fade_out(hold: float = 0.0) -> Lane:
    """1.0 held until `hold`, then an equal-power fall to 0 by phase 1."""
    hold = float(np.clip(hold, 0.0, 0.8))
    if hold <= 0.0:
        return ((0.0, 1.0), (0.5, 0.707), (1.0, 0.0))
    return ((0.0, 1.0), (hold, 1.0), ((hold + 1.0) / 2.0, 0.707), (1.0, 0.0))


def _fade_in(lead: float = 0.0) -> Lane:
    """0.0 held until `lead`, then an equal-power rise to 1 by phase 1."""
    lead = float(np.clip(lead, 0.0, 0.8))
    if lead <= 0.0:
        return ((0.0, 0.0), (0.5, 0.707), (1.0, 1.0))
    return ((0.0, 0.0), (lead, 0.0), ((lead + 1.0) / 2.0, 0.707), (1.0, 1.0))


def _make_profile(cp: float, in_mid_lead: float, in_high_lead: float,
                  out_mid_hold: float = 0.0, out_high_hold: float = 0.0,
                  bass_w: float = 0.12) -> TransitionProfile:
    """
    Build a standard crossfade profile: the bass swaps from the outgoing to the
    incoming track over a short window centred on `cp` (only one kick ever
    plays), while the mid and high bands crossfade with independent timing — so
    e.g. a swap can bring the incoming hats in early while holding its mids back.
    """
    cp = float(np.clip(cp, bass_w, 1.0 - bass_w))
    out_low = ((0.0, 1.0), (cp - bass_w, 1.0), (cp + bass_w, 0.0), (1.0, 0.0))
    in_low  = ((0.0, 0.0), (cp - bass_w, 0.0), (cp + bass_w, 1.0), (1.0, 1.0))
    flat = ((0.0, 1.0), (1.0, 1.0))
    return TransitionProfile(
        out_vol=flat, out_low=out_low,
        out_mid=_fade_out(out_mid_hold), out_high=_fade_out(out_high_hold),
        in_vol=flat, in_low=in_low,
        in_mid=_fade_in(in_mid_lead), in_high=_fade_in(in_high_lead),
    )


@dataclass
class TransitionStyle:
    """
    How a crossfade behaves, chosen from the two tracks' dynamics at the mix
    point. Different exits/entries want different lengths and EQ automation.
    """
    name: str
    n_bars: int                # crossfade length (cumulative time)
    high_slope: float = 1.0    # legacy scalar knobs (used to build a default
    in_high_delay: float = 0.0 # profile when `profile` is None)
    bass_swap_center: float = 0.5
    bass_swap_width: float = 0.10
    is_cut: bool = False       # short time-based fade instead of a bar-based blend
    cut_seconds: float = 0.30  # length of that fade when is_cut
    profile: Optional[TransitionProfile] = None  # breakpoint EQ automation


def _default_profile(style: TransitionStyle) -> TransitionProfile:
    """Build a profile from a style's legacy scalar knobs (back-compat)."""
    return _make_profile(style.bass_swap_center,
                         in_mid_lead=style.in_high_delay,
                         in_high_lead=style.in_high_delay)



# `fade` vs `build` is decided below by the sign of `eo - ei`. Mining real DJ
# mixes (CHANGELOG, 2026-08-08) found that comparison carries almost no signal:
# median margin 0.055 across the 141/213 beatmatched pairs that reach it, 46%
# inside 0.05, 12% inside 0.01 — `_match_entry` picks the incoming cue by
# *minimizing* |eo - ei|, so it equalizes the two energies and then the style
# branches on the sign of the difference it just drove toward zero. Below this
# margin the sign is measurement noise, not a musical decision, so it is not
# trustworthy as one; TIE_MARGIN gates a deliberate, reproducible draw instead
# (see `_seeded_unit`) rather than letting floating-point noise decide.
TIE_MARGIN = 0.08


def _seeded_unit(*parts) -> float:
    """
    Deterministic pseudo-random float in [0, 1) from stable parts. Same inputs
    always give the same output (needed so `plan_transition` stays a pure,
    replayable function for the corpus validator and for `gaps`/`triage`), but
    the output is decorrelated from tiny numeric noise in the inputs the way a
    direct sign comparison isn't. Not `hash()` — that's salted per-process.
    """
    key = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_transition_style(out_cue, in_cue, beatmatched: bool,
                            high_sim_threshold: float = 0.82,
                            seed_extra: tuple = ()) -> TransitionStyle:
    """
    Pick a crossfade style from the energy and CLAP vector similarity of the
    outgoing exit and incoming entry. Beat-locked pairs get real blends shaped
    to their dynamics; tempo-incompatible pairs get a short cut.

    `high_sim_threshold` is the CLAP-similarity cutoff above which a smooth long
    blend is forced; callers pass a per-library calibrated value (the fixed
    0.82 default is only a fallback — on a real library it fires on ~30% of
    pairs and doesn't discriminate).

    `seed_extra` lets a caller (e.g. `plan_transition`, which knows the two
    tracks and how many times this pair has already transitioned in the current
    set) salt the `fade`/`build` tie-break so a repeated pair doesn't land on
    the same coin every time. Omit it and the draw is keyed on the cues alone.
    """
    if not beatmatched:
        # Tempos clash — a long overlap would phase two grooves through each
        # other. Keep it near-instant (just enough to avoid a click).
        return TransitionStyle("cut", n_bars=0, is_cut=True, cut_seconds=0.30)

    try:
        from .sequencer import cue_cosine_similarity
        sim = cue_cosine_similarity(out_cue, in_cue)
    except Exception:
        sim = None

    eo = out_cue.energy if out_cue else 0.5
    ei = in_cue.energy if in_cue else 0.5

    def styled(name, n_bars, cp, mid_lead, high_lead,
               out_mid_hold=0.0, out_high_hold=0.0):
        return TransitionStyle(
            name, n_bars,
            profile=_make_profile(cp, mid_lead, high_lead,
                                  out_mid_hold, out_high_hold),
        )

    # Crossfade lengths come from the calibration layer, which falls back to
    # these same 16/12/8/8 bars when no mined corpus is present. The per-band
    # automation (cp / leads) stays hand-set: corpus measurement can't resolve
    # per-band phase — see infinite_dj/calibration.py.
    from .calibration import value as _cal

    def bars(name: str) -> int:
        return max(1, int(round(_cal(name))))

    if (sim is not None and sim >= high_sim_threshold) or (eo < 0.45 and ei < 0.45):
        # High textural similarity or both sparse: long, symmetric smooth blend.
        return styled("blend", bars("blend_bars"),
                      cp=0.55, mid_lead=0.12, high_lead=0.12)
    if eo > 0.70 and ei > 0.70:
        # Drop → drop: bring the incoming HATS in early over the outgoing groove,
        # hold its MIDS back to avoid clash, then swap the bass mid-way.
        return styled("swap", bars("swap_bars"),
                      cp=0.50, mid_lead=0.50, high_lead=0.10,
                      out_mid_hold=0.50, out_high_hold=0.30)
    margin = eo - ei
    if abs(margin) < TIE_MARGIN:
        # Too close to call from energy alone. Weight a deterministic draw by
        # the (weak) signal we do have — an exact tie is a 50/50 coin, and the
        # draw tilts toward whichever side the margin already leans as it
        # approaches TIE_MARGIN, where the sign comparison below takes over.
        p_fade = 0.5 + (margin / TIE_MARGIN) * 0.5
        draw = _seeded_unit(
            getattr(out_cue, "timestamp", -1.0), getattr(in_cue, "timestamp", -1.0),
            round(eo, 4), round(ei, 4), *seed_extra,
        )
        is_fade = draw < p_fade
    else:
        is_fade = margin >= 0
    if is_fade:
        # Busier → calmer: gentle medium fade, incoming eased in.
        return styled("fade", bars("fade_bars"),
                      cp=0.55, mid_lead=0.30, high_lead=0.20)
    # Calmer → rising: bring the incoming up sooner across all bands.
    return styled("build", bars("build_bars"),
                  cp=0.40, mid_lead=0.10, high_lead=0.05)



def _blend(
    out_chunk: np.ndarray,
    in_chunk: np.ndarray,
    phase,                       # scalar or per-sample array in [0, 1]
    sr: int = MIX_SR,
    style: Optional[TransitionStyle] = None,
    bass_cutoff: float = 200.0,
    filter_state: Optional[CrossfadeFilterState] = None,
) -> np.ndarray:
    """
    Shared DJ-style EQ blend used by both the offline renderer and the real-time
    engine. `phase` is 0 at the start of the crossfade and 1 at the end.

      - Highs/mids: crossfade shaped by the style — the incoming can be delayed
        and eased in (`in_high_delay`, `high_slope`) so its percussion doesn't
        clash with the outgoing groove.
      - Bass: single source at a time — swapped over a short window
        (`bass_swap_center`/`width`), so only one kick drum ever plays.
    """
    n = min(len(out_chunk), len(in_chunk))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out_c = out_chunk[:n]
    in_c  = in_chunk[:n]

    if style is None:
        style = TransitionStyle("default", 0)
    prof = style.profile or _default_profile(style)

    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim == 0:
        phase = np.full(n, float(phase), dtype=np.float32)
    phase = phase[:n]

    def lane(l):
        return _sample_lane(l, phase)

    def g(l):
        return lane(l).reshape(-1, 1)

    # Which EQ topology: an explicitly supplied state always wins (the caller
    # has already committed to one for this transition), otherwise EQ_TOPOLOGY.
    if isinstance(filter_state, ShelfCrossfadeState):
        shelf_state = filter_state
    elif filter_state is None and EQ_TOPOLOGY == "shelf":
        shelf_state = ShelfCrossfadeState.create(sr)
    else:
        shelf_state = None

    if shelf_state is not None:
        # Minimum-phase shelving EQ: the magnitude response *is* the automation
        # curve, so a killed band is genuinely gone rather than phase-shifted
        # into its neighbour.
        out_mix = g(prof.out_vol) * shelf_state.out_eq.process(
            out_c, lane(prof.out_low), lane(prof.out_mid), lane(prof.out_high))
        in_mix = g(prof.in_vol) * shelf_state.in_eq.process(
            in_c, lane(prof.in_low), lane(prof.in_mid), lane(prof.in_high))
        return (out_mix + in_mix).astype(np.float32)

    if filter_state is None:
        o_low, o_mid, o_high = _split3(out_c, sr)
        i_low, i_mid, i_high = _split3(in_c, sr)
    else:
        o_low, o_mid, o_high, i_low, i_mid, i_high = filter_state.split(out_c, in_c)

    # Each track = volume × (per-band gain × band), a 3-fader DJ EQ ridden by
    # the style's breakpoint automation.
    out_mix = g(prof.out_vol) * (
        g(prof.out_low) * o_low + g(prof.out_mid) * o_mid + g(prof.out_high) * o_high)
    in_mix = g(prof.in_vol) * (
        g(prof.in_low) * i_low + g(prof.in_mid) * i_mid + g(prof.in_high) * i_high)

    return (out_mix + in_mix).astype(np.float32)


# ── Main transition renderer ──────────────────────────────────────────────────

def render_transition(plan: TransitionPlan, sr: int = MIX_SR) -> MixResult:
    """
    Render the crossfade transition described by a TransitionPlan.

    Output structure:
      [tail of track_out] + [mix region] + [head of track_in]

    The mix region uses a DJ-style EQ transition:
      Phase 1 (bars 1-2): Outgoing full, incoming highs only fading in
      Phase 2 (bars 3-4): Bass swap — cut outgoing bass, bring incoming bass
      Phase 3 (bars 5-6): Incoming full, outgoing highs fading out
    """
    print(f"  Loading audio files...")
    audio_out, _ = _load_audio(plan.track_out.file_path, sr)
    audio_in,  _ = _load_audio(plan.track_in.file_path, sr)

    # Loudness-match the incoming track to the outgoing one so levels don't jump
    audio_in = _loudness_match(
        audio_in, plan.track_in.loudness, plan.track_out.loudness
    )

    # Time-stretch incoming track to match outgoing BPM, and key-sync it if a
    # shift improves harmonic compatibility (only when beatmatching — a cut
    # has no simultaneous harmonic content to fix). Combined in one pass.
    needs_stretch = plan.beatmatched and abs(plan.stretch_ratio - 1.0) > 0.001
    needs_shift = plan.beatmatched and abs(plan.pitch_shift_semitones) > 0.001
    if needs_stretch or needs_shift:
        if needs_stretch:
            pct = (plan.stretch_ratio - 1.0) * 100
            print(f"  Time-stretching incoming track by {pct:+.1f}% to match {plan.bpm_target:.1f} BPM...")
        if needs_shift:
            print(f"  Key-syncing incoming track by {plan.pitch_shift_semitones:+.0f} semitones...")
        audio_in = _time_stretch(audio_in, sr, plan.stretch_ratio,
                                 plan.pitch_shift_semitones)
        # A downbeat at native time d lands at d / ratio after stretching
        # (ratio > 1 speeds the track up, so beats move earlier). Pitch-shift
        # doesn't move anything in time, so this stays valid alongside it.
        stretched_downbeats = [d / plan.stretch_ratio
                               for d in plan.track_in.downbeats]
    else:
        stretched_downbeats = list(plan.track_in.downbeats)

    # Calculate bar duration at target BPM
    bar_duration = (60.0 / plan.bpm_target) * 4   # 4 beats per bar
    # A cut is short (tempos don't match, so there's nothing to ride); a
    # beatmatch gets the full configured length.
    n_bars = plan.n_mix_bars if plan.beatmatched else min(4, plan.n_mix_bars)
    mix_duration = bar_duration * n_bars

    # Find exact sample positions
    out_cue_sample = _time_to_samples(plan.cue_out.timestamp, sr)
    mix_region_samples = _time_to_samples(mix_duration, sr)

    # Find best aligned entry in incoming track (reject snaps > half a bar away)
    aligned_in_time = _find_nearest_downbeat(
        plan.cue_in.timestamp, stretched_downbeats, max_offset=bar_duration / 2.0
    )
    in_cue_sample = _time_to_samples(aligned_in_time, sr)

    print(f"  {plan.method.upper()}: OUT cue {plan.cue_out.timestamp:.1f}s → "
          f"IN cue {aligned_in_time:.1f}s")
    print(f"  Mix region: {mix_duration:.1f}s ({n_bars} bars at {plan.bpm_target:.1f} BPM)")

    # ── Extract audio regions ─────────────────────────────────────────────────

    # Pre-mix tail: 8 bars of outgoing track leading into the cue
    pre_bars = 8
    pre_duration_samples = _time_to_samples(bar_duration * pre_bars, sr)
    pre_start = max(0, out_cue_sample - pre_duration_samples)
    pre_region = audio_out[pre_start:out_cue_sample]

    # Mix region from outgoing track
    out_mix_end = min(len(audio_out), out_cue_sample + mix_region_samples)
    out_mix = audio_out[out_cue_sample:out_mix_end]

    # Mix region from incoming track
    in_mix_end = min(len(audio_in), in_cue_sample + mix_region_samples)
    in_mix = audio_in[in_cue_sample:in_mix_end]

    # Match lengths
    mix_len = min(len(out_mix), len(in_mix), mix_region_samples)
    out_mix = out_mix[:mix_len]
    in_mix  = in_mix[:mix_len]

    # Post-mix head: remaining incoming track after the mix region
    post_region = audio_in[in_cue_sample + mix_len:]

    # ── Build the crossfade ────────────────────────────────────────────────────

    print(f"  Rendering {plan.method} crossfade...")

    if mix_len <= 0:
        mix = np.zeros((0, 2), dtype=np.float32)
    elif plan.beatmatched:
        # Beat-locked EQ blend: equal-power highs + single-source bass swap,
        # driven by a smooth phase ramp across the whole region.
        phase = np.linspace(0.0, 1.0, mix_len, dtype=np.float32)
        mix = _blend(out_mix, in_mix, phase, sr)
    else:
        # Cut: tempos don't lock, so just an equal-power full-spectrum fade.
        fo, fi = _equal_power_fade(mix_len)
        mix = out_mix * fo.reshape(-1, 1) + in_mix * fi.reshape(-1, 1)

    # Small headroom reduction before the final normalize
    mix = _apply_gain(mix, 0.9)

    # ── Assemble final output ─────────────────────────────────────────────────

    # Fade the pre-region in smoothly over 0.5s
    fade_in_samples = min(int(0.5 * sr), len(pre_region))
    if len(pre_region) > 0 and fade_in_samples > 0:
        fade = np.linspace(0, 1, fade_in_samples, dtype=np.float32).reshape(-1, 1)
        pre_region = pre_region.copy()
        pre_region[:fade_in_samples] *= fade

    # Fade the post-region out over 1s at the end
    fade_out_samples = min(int(1.0 * sr), len(post_region))
    if len(post_region) > 0 and fade_out_samples > 0:
        post_region = post_region.copy()
        fade = np.linspace(1, 0, fade_out_samples, dtype=np.float32).reshape(-1, 1)
        post_region[-fade_out_samples:] *= fade

    # Concatenate everything
    parts = []
    if len(pre_region) > 0:
        parts.append(pre_region)
    parts.append(mix)
    if len(post_region) > 0:
        parts.append(post_region[:_time_to_samples(bar_duration * 16, sr)])  # 16 bars of post

    output = np.concatenate(parts, axis=0)

    # Final peak normalize
    peak = np.abs(output).max()
    if peak > 0.95:
        output = output * (0.93 / peak)

    duration = len(output) / sr
    return MixResult(audio=output, sr=sr, duration=duration, plan=plan)


def write_mix(result: MixResult, out_path: str):
    """Write a MixResult to a WAV file (16-bit, matching the source library)."""
    sf.write(out_path, result.audio, result.sr, subtype='PCM_16')
    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Written: {out_path} ({result.duration:.1f}s, {mb:.1f} MB)")


# ── Full-set renderer (continuous timeline) ────────────────────────────────────

# Every track is normalized toward this level so the whole set sits consistently.
MASTER_LOUDNESS = -12.0   # dBFS


@dataclass
class SetMarker:
    """Where a transition lands in the rendered set, for reporting."""
    time: float            # seconds into the output
    label: str             # "Track A → Track B"
    method: str            # "beatmatch" | "cut"
    stretch_pct: float
    style: str = ""        # transition style name
    pitch_shift_semitones: float = 0.0  # key-sync shift applied to the incoming track


def _best_out_cue_after(track: TrackMeta, min_t: float, max_t: float):
    """Strongest OUT cue in (min_t, max_t); None if the track has none there."""
    outs = [c for c in track.cue_points
            if c.type == "out" and min_t <= c.timestamp < max_t]
    return max(outs, key=lambda c: c.confidence) if outs else None


def _set_entry_cue(track: TrackMeta, max_frac: float = 0.4):
    """
    Where a track should ENTER in a linear set: the strongest IN cue in its
    early portion, so it plays a full solo before it has to mix out again. (The
    cue detector scores IN points across the whole track for arbitrary re-entry;
    for a set we want an intro/early phrase, not a great entry near the end.)
    """
    limit = max(1.0, (track.duration or 0.0) * max_frac)
    early = [c for c in track.cue_points if c.type == "in" and c.timestamp <= limit]
    if early:
        return max(early, key=lambda c: c.confidence)
    ins = [c for c in track.cue_points if c.type == "in"]
    return min(ins, key=lambda c: c.timestamp) if ins else None


def _energy_at(track: TrackMeta, t: float) -> float:
    ec = track.energy_curve
    return float(ec[min(int(t), len(ec) - 1)]) if ec else 0.5


def _pick_exit_cue(track: TrackMeta, min_t: float, max_t: float,
                   groove_floor: float = 0.4):
    """
    Best OUT cue past the dwell that still has a groove going (energy >= floor),
    so the outgoing carries a beat into the crossfade instead of exiting from a
    dead valley. Falls back to the strongest available cue.
    """
    outs = [c for c in track.cue_points
            if c.type == "out" and min_t <= c.timestamp < max_t]
    if not outs:
        return None
    groovy = [c for c in outs if c.energy >= groove_floor]
    return max(groovy or outs, key=lambda c: c.confidence)


def _clap_cos(a, b) -> float:
    """Cosine similarity of two CLAP embedding vectors."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def _track_sections(track: TrackMeta, min_len_sec: float = 8.0) -> list:
    """The track's structural sections long enough to splice (all, if none pass)."""
    if not track.sections:
        return []
    return [s for s in track.sections if (s.end - s.start) >= min_len_sec] \
        or list(track.sections)


def _segment_pool(tracks: list, bar: float, sub_bars: Optional[int] = None,
                  max_subs: int = 6, min_len_sec: float = 8.0) -> list:
    """
    Candidate ``(track, section)`` entry points for the collage.

    Normally one per structural section. With `sub_bars`, each section is also
    diced into sub-segments every `sub_bars` bars (capped at `max_subs` per
    section so the pool stays tractable), so the mixer can enter *partway
    through* a section — many more, shorter splices. Sub-segments inherit their
    parent's CLAP embedding and label: timbre is near-constant inside a section,
    which is what makes it a section.
    """
    pool = []
    for t in tracks:
        for s in _track_sections(t, min_len_sec):
            pool.append((t, s))
            if not sub_bars:
                continue
            step = sub_bars * bar
            k, made = s.start + step, 0
            while k + step <= s.end and made < max_subs:
                pool.append((t, Section(start=round(k, 3), end=s.end,
                                        label=s.label, energy=s.energy,
                                        embedding=s.embedding)))
                k += step
                made += 1
    return pool


def _track_splice_points(track: TrackMeta, min_len_sec: float = 8.0) -> list:
    """
    Ordered entry points for splicing a track, one per *distinct structural
    segment* from the analyzer's novelty segmentation — so recurring splices are
    musically different parts, chosen with sensitivity to dynamics.

    When sections carry CLAP embeddings they are ordered for maximal timbral
    distinctness (farthest-first from the strongest); otherwise by energy. Falls
    back to scored cue points, then downbeats.
    """
    segs = _track_sections(track, min_len_sec)
    if segs:
        if len(segs) > 1 and all(s.embedding for s in segs):
            remaining = list(segs)
            ordered = [max(remaining, key=lambda s: s.energy)]
            remaining.remove(ordered[0])
            while remaining:
                nxt = min(remaining,
                          key=lambda s: max(_clap_cos(s.embedding, o.embedding) for o in ordered))
                ordered.append(nxt)
                remaining.remove(nxt)
            return [s.start for s in ordered]
        return [s.start for s in sorted(segs, key=lambda s: -s.energy)]
    cues = sorted({c.timestamp for c in track.cue_points})
    if cues:
        return cues
    return [track.downbeats[0]] if track.downbeats else [0.0]


def _pick_splice_exit(cur_t: TrackMeta, nxt_t: TrackMeta, min_t: float, max_t: float):
    """
    For splice mode: choose the OUT cue in the [min_t, max_t] window whose CLAP
    embedding best matches the next track's entry cue — a serendipitous splice
    point where the two textures connect. Falls back to the strongest cue (or
    the latest one) when embeddings are absent.
    """
    outs = [c for c in cur_t.cue_points if c.type == "out" and min_t <= c.timestamp <= max_t]
    if not outs:
        return None
    nxt_in = best_cue_in(nxt_t)
    try:
        from .sequencer import cue_cosine_similarity
        scored = [(c, cue_cosine_similarity(c, nxt_in)) for c in outs]
        if any(s is not None for _, s in scored):
            return max(scored, key=lambda cs: cs[1] if cs[1] is not None else -1.0)[0]
    except Exception:
        pass
    return max(outs, key=lambda c: c.confidence)


def _match_entry(track: TrackMeta, target_energy: float,
                 min_frac: float = 0.02, max_frac: float = 0.55,
                 groove_floor: float = 0.35) -> CuePoint:
    """
    For a beat-locked blend, enter the incoming where its groove is *already
    going* at an energy close to the outgoing exit — so two rhythms overlap and
    lock, rather than overlapping a groove with a silent intro. Searches the
    track's own downbeats (not just the low-energy scored IN cues), preferring
    phrase-aligned ones whose energy matches the target.
    """
    dur = track.duration or 0.0
    phrases = track.phrases or []
    lo, hi = dur * min_frac, max(dur * max_frac, 1.0)
    cands = [d for d in track.downbeats if lo <= d <= hi]
    if not cands:
        return _set_entry_cue(track) or CuePoint(0.0, "in", True, 0.5, 0.1)

    def near_phrase(d):
        return min((abs(d - p) for p in phrases), default=99.0) < 1.0

    def score(d):
        e = _energy_at(track, d)
        s = -abs(e - target_energy)          # match the outgoing's energy
        if e >= groove_floor:
            s += 0.3                          # prefer an established groove
        if near_phrase(d):
            s += 0.2                          # prefer phrase boundaries
        return s

    best = max(cands, key=score)
    return CuePoint(timestamp=round(best, 3), type="in",
                    phrase_aligned=near_phrase(best),
                    energy=round(_energy_at(track, best), 3), confidence=0.5)


@dataclass(frozen=True)
class TransitionPlanned:
    """
    The decisions a transition needs, before any of them touch samples: where the
    outgoing leaves, where the incoming enters, and how the crossfade is shaped.
    """
    cue_out: Optional[CuePoint]   # None when no scored OUT cue was available
    cue_out_t: float              # seconds into the OUTGOING track
    cue_in: CuePoint
    cue_in_t: float               # seconds into the INCOMING track, downbeat-snapped
    style: TransitionStyle
    ratio: float                  # out_bpm / in_bpm, half/double folded
    beatmatched: bool
    pitch_shift_semitones: float = 0.0  # key-sync shift for track_in, 0 if none/not beatmatched


def plan_transition(
    track_out: TrackMeta,
    track_in: TrackMeta,
    read_t: float,
    dur_out: float,
    *,
    min_solo_bars: Optional[int] = None,
    max_stretch: float = MAX_STRETCH,
    sim_threshold: float = 0.82,
    splice: Optional[tuple] = None,
    occurrence: int = 0,
    out_key: Optional[str] = None,
) -> TransitionPlanned:
    """
    Choose the exit cue, entry cue and crossfade style for one A→B transition.

    Extracted from `render_set`'s loop so the same decisions can be replayed
    outside a render — the corpus validator compares these against what real DJs
    did, and the calibration report needs them without producing audio.

    `out_key` overrides the outgoing track's analyzed key. A track that was
    itself key-synced on the way in keeps playing in its shifted key, so the
    set loop passes the key actually sounding; callers planning a one-off
    transition can leave it None and get `track_out.key`.

    `read_t` is where the outgoing track is currently playing from and `dur_out`
    the length of its *loaded* audio (which differs slightly from the analyzed
    `duration`, so it has to be passed in rather than read off the TrackMeta).
    `splice` is `(min_seg_sec, max_seg_sec)` for collage mode, else None.
    `min_solo_bars` defaults to the calibrated value (32 without a corpus).
    `occurrence` is how many times this ordered (track_out, track_in) pair has
    already transitioned earlier in the same set/session (radio mode can repeat
    pairs over a long run) — it salts the `fade`/`build` tie-break in
    `choose_transition_style` so a repeated pair doesn't draw the same coin
    every time. Callers that don't track repeats can leave it at 0.

    Sample-domain clamping stays in the caller: it depends on the rendered buffer
    length and, for the tail clamp, on the style this function returns.
    """
    if min_solo_bars is None:
        from .calibration import value as _cal
        min_solo_bars = int(round(_cal("min_solo_bars")))
    out_bpm = track_out.bpm
    out_bar_sec = (60.0 / out_bpm) * 4

    # ── Tempo match: bring incoming to the OUTGOING track's native tempo ───────
    ratios = [out_bpm / track_in.bpm,
              out_bpm / (track_in.bpm * 2.0),
              out_bpm / (track_in.bpm / 2.0)]
    ratio = min(ratios, key=lambda r: abs(r - 1.0))
    beatmatched = abs(ratio - 1.0) <= max_stretch

    # ── Outgoing exit ─────────────────────────────────────────────────────────
    if splice is not None:
        min_seg_sec, max_seg_sec = splice
        # Short segment: exit within [min_seg, max_seg] at a CLAP-serendipitous
        # splice point that connects to the next track's texture.
        min_exit_t = read_t + (min_seg_sec or 0.0)
        max_exit_t = min(read_t + max_seg_sec, dur_out)
        out_cue = _pick_splice_exit(track_out, track_in, min_exit_t, max_exit_t)
        if out_cue is not None:
            cue_out_t = out_cue.timestamp
        else:
            inwin = [d for d in track_out.downbeats if min_exit_t <= d <= max_exit_t]
            cue_out_t = inwin[-1] if inwin else max(read_t, max_exit_t - out_bar_sec)
    else:
        # Breathe, then leave at a strong cue that still has a groove (so the
        # outgoing carries a beat into the crossfade).
        min_exit_t = read_t + min_solo_bars * out_bar_sec
        out_cue = _pick_exit_cue(track_out, min_exit_t, dur_out)
        if out_cue is not None:
            cue_out_t = out_cue.timestamp
        else:
            later = [d for d in track_out.downbeats if min_exit_t <= d < dur_out]
            cue_out_t = later[0] if later else max(read_t, dur_out - out_bar_sec)

    # ── Incoming entry ────────────────────────────────────────────────────────
    # Beat-locked transitions enter where the incoming groove matches the exit's
    # energy (two rhythms lock); cuts just enter early.
    target_e = out_cue.energy if out_cue else 0.5
    if beatmatched:
        in_cue = _match_entry(track_in, target_e)
    else:
        in_cue = _set_entry_cue(track_in) or CuePoint(0.0, "in", True, 0.5, 0.1)
    cue_in_t = _find_nearest_downbeat(
        in_cue.timestamp, track_in.downbeats,
        max_offset=(60.0 / track_in.bpm) * 4 / 2.0,
    )

    # ── Pick a crossfade style from the two tracks' dynamics ──────────────────
    style = choose_transition_style(
        out_cue, in_cue, beatmatched, high_sim_threshold=sim_threshold,
        seed_extra=(track_out.file_path, track_in.file_path, occurrence),
    )

    # ── Key-sync: only when there's an actual beat-locked overlap to fix ──────
    pitch_shift_semitones = 0.0
    if beatmatched:
        shift = pitch_shift_for_compatibility(out_key or track_out.key, track_in.key)
        if shift is not None:
            pitch_shift_semitones = float(shift[0])

    return TransitionPlanned(cue_out=out_cue, cue_out_t=cue_out_t,
                             cue_in=in_cue, cue_in_t=cue_in_t,
                             style=style, ratio=ratio, beatmatched=beatmatched,
                             pitch_shift_semitones=pitch_shift_semitones)


def render_set(
    tracks: list,
    n_mix_bars: int = 16,
    sr: int = MIX_SR,
    max_stretch: float = MAX_STRETCH,
    min_solo_bars: Optional[int] = None,
    min_seg_sec: Optional[float] = None,
    max_seg_sec: Optional[float] = None,
    target_length_sec: Optional[float] = None,
) -> tuple:
    """
    Render an ordered list of tracks into ONE continuous set.

    Two modes:
      - Full-set (default): each track breathes (>= min_solo_bars) and exits at
        a genuinely strong, phrase-aligned OUT cue.
      - Splice (when `max_seg_sec` is given): each track plays only a short
        segment bounded by [min_seg_sec, max_seg_sec], exiting at a CLAP-chosen
        cue that connects to the next track's texture. Rendering stops once
        `target_length_sec` is reached — a collage of many short segments.

    Transitions adapt via `choose_transition_style`; each track plays at its own
    native tempo, only the incoming crossfade region is stretched to lock beats.
    `min_solo_bars` defaults to the calibrated value (32 without a corpus).
    Returns (audio, sr, [SetMarker, ...], [clip, ...]).
    """
    if len(tracks) < 2:
        raise ValueError("Need at least 2 tracks to render a set")

    if min_solo_bars is None:
        from .calibration import value as _cal
        min_solo_bars = int(round(_cal("min_solo_bars")))

    splice = max_seg_sec is not None

    def load_matched(t):
        a, _ = _load_audio(t.file_path, sr)
        return _loudness_match(a, t.loudness, MASTER_LOUDNESS)

    # Calibrate the "high textural similarity → smooth blend" cutoff to this
    # library's own CLAP distribution (None/fallback if no embeddings).
    try:
        from .sequencer import library_sim_threshold
        sim_threshold = library_sim_threshold(tracks) or 0.82
    except Exception:
        sim_threshold = 0.82

    # Current track state, all in that track's own native timeline.
    cur_t     = tracks[0]
    cur_audio = load_matched(cur_t)
    ci   = _set_entry_cue(cur_t)   # start near the top, not at a late re-entry cue
    read = _time_to_samples(
        (ci.timestamp if ci else (cur_t.downbeats[0] if cur_t.downbeats else 0.0)), sr
    )

    master: list = []
    markers: list = []
    clips: list = []          # per-track segments on the output timeline (for the UI)
    written = 0
    cur_start = 0             # output sample where the current track became audible
    prev_fade_in = 0.0        # fade-in of the current track (from the previous xfade)
    pair_occurrence: dict = {}  # (out path, in path) -> times already transitioned
    # Semitones the CURRENT track is being pitched by. A key-synced track keeps
    # playing in its shifted key for its whole time on air, so the next
    # transition has to plan against the key actually sounding — not the
    # analyzed one. Measured from each track's own native key, so offsets
    # never accumulate down the set.
    cur_key_offset = 0.0

    for i in range(len(tracks) - 1):
        solo_start = written  # output sample where this track's solo begins
        nxt_t     = tracks[i + 1]
        out_bpm   = cur_t.bpm
        out_bar_sec = (60.0 / out_bpm) * 4
        out_bar_s = _time_to_samples(out_bar_sec, sr)
        cur_dur   = len(cur_audio) / sr
        read_t    = read / sr

        # ── Cue + style selection (shared with the corpus validator) ───────────
        pair_key = (cur_t.file_path, nxt_t.file_path)
        occurrence = pair_occurrence.get(pair_key, 0)
        pair_occurrence[pair_key] = occurrence + 1
        planned = plan_transition(
            cur_t, nxt_t, read_t, cur_dur,
            min_solo_bars=min_solo_bars, max_stretch=max_stretch,
            sim_threshold=sim_threshold,
            splice=((min_seg_sec, max_seg_sec) if splice else None),
            occurrence=occurrence,
            out_key=transpose_key(cur_t.key, cur_key_offset) if cur_key_offset else None,
        )
        ratio, beatmatched = planned.ratio, planned.beatmatched
        pitch_shift = planned.pitch_shift_semitones
        out_cue, cue_out_t = planned.cue_out, planned.cue_out_t
        in_cue, style = planned.cue_in, planned.style

        cue_out_sample = max(read, _time_to_samples(cue_out_t, sr))
        in_sample = _time_to_samples(planned.cue_in_t, sr)

        nxt_audio = load_matched(nxt_t)
        in_bar_s  = _time_to_samples((60.0 / nxt_t.bpm) * 4, sr)

        # ── Build the crossfade regions ────────────────────────────────────────
        # Crossfades run their full style length (up to n_mix_bars) even in
        # splice mode — a short segment that's mostly crossfade is the point.
        n_bars = style.n_bars

        # Guarantee the outgoing track has a full overlap's worth of audio after
        # the exit cue — otherwise a cue near the track end collapses the blend
        # into a 1-2s stub (a "quick fade" instead of a real crossfade).
        mix_out_s = (_time_to_samples(style.cut_seconds, sr) if style.is_cut
                     else n_bars * out_bar_s)
        cue_out_sample = min(cue_out_sample, max(read, len(cur_audio) - mix_out_s))

        if style.is_cut:
            # Short time-based fade — no long overlap of unsynced tempos, so
            # nothing simultaneous to key-sync either.
            out_mix = cur_audio[cue_out_sample:cue_out_sample + mix_out_s]
            in_mix  = nxt_audio[in_sample:in_sample + mix_out_s]
            ratio   = 1.0
            pitch_shift = 0.0
        else:
            out_mix = cur_audio[cue_out_sample:cue_out_sample + mix_out_s]
            in_region = nxt_audio[in_sample:in_sample + n_bars * in_bar_s]
            if beatmatched and (abs(ratio - 1.0) > 0.001 or abs(pitch_shift) > 0.001):
                # ONE pass from the native audio: tempo and key together, so
                # the crossfade region is never processed twice.
                in_mix = _time_stretch(in_region, sr, ratio, pitch_shift)   # → outgoing tempo + key
            else:
                ratio = 1.0
                in_mix = in_region

        # A key-synced track has to KEEP its new key once the crossfade ends —
        # otherwise it snaps back to native the instant the blend finishes,
        # which is a fully audible key jump mid-track (measured at -0.98
        # semitones before this was fixed; see CHANGELOG 2026-08-17). Only the
        # tempo stretch is per-transition, by design: a ~5% tempo nudge at a
        # phrase boundary is the accepted tradeoff, a semitone is not.
        # Pitch-shifting preserves duration exactly, so `read` keeps indexing
        # the same sample positions as the native audio.
        if abs(pitch_shift) > 0.001:
            nxt_audio_after = _time_stretch(nxt_audio, sr, 1.0, pitch_shift)
        else:
            nxt_audio_after = nxt_audio

        m = min(len(out_mix), len(in_mix))
        out_mix, in_mix = out_mix[:m], in_mix[:m]

        # ── 1. Solo section of the current track up to the crossfade ───────────
        solo = cur_audio[read:cue_out_sample]
        if len(solo) > 0:
            master.append(solo)

        # ── 2. Crossfade ───────────────────────────────────────────────────────
        if m > 0:
            if style.is_cut:
                fo, fi = _equal_power_fade(m)
                xf = out_mix * fo.reshape(-1, 1) + in_mix * fi.reshape(-1, 1)
            else:
                phase = np.linspace(0.0, 1.0, m, dtype=np.float32)
                xf = _blend(out_mix, in_mix, phase, sr, style)
            xf = _apply_gain(xf, 0.9)
            markers.append(SetMarker(
                time=(written + len(solo)) / sr,
                label=f"{cur_t.title} → {nxt_t.title}",
                method="beatmatch" if beatmatched else "cut",
                stretch_pct=(ratio - 1.0) * 100.0,
                style=style.name,
                pitch_shift_semitones=pitch_shift,
            ))
            master.append(xf)
            written += len(solo) + len(xf)
        else:
            written += len(solo)

        # Record the outgoing track's clip: audible from cur_start through the end
        # of this crossfade, fading out over the crossfade region.
        xf_len = len(xf) if m > 0 else 0
        clips.append({
            "track": cur_t.file_path, "title": cur_t.title,
            "out_start": round(cur_start / sr, 3),
            "out_end": round((solo_start + len(solo) + xf_len) / sr, 3),
            "fade_in": round(prev_fade_in, 3), "fade_out": round(xf_len / sr, 3),
            "mode": style.name, "section": "",
            "bpm": round(cur_t.bpm, 1), "key": cur_t.key,
        })
        cur_start = solo_start + len(solo)   # incoming enters at the crossfade start
        prev_fade_in = xf_len / sr

        # ── Advance: incoming becomes current, continuing at its native tempo
        # but KEEPING any key-sync shift it entered with.
        consumed_in = int(round(m * ratio))
        cur_t     = nxt_t
        cur_audio = nxt_audio_after
        cur_key_offset = pitch_shift
        read      = in_sample + consumed_in

        # Splice mode: stop once we've filled the target length.
        if target_length_sec is not None and written / sr >= target_length_sec:
            break

    # ── Final track tail (capped) with a gentle fade-out ───────────────────────
    if splice:
        # One last short segment so we don't end mid-crossfade.
        max_tail = _time_to_samples(min(max_seg_sec, (min_seg_sec or 20.0)), sr)
    else:
        max_tail = _time_to_samples((60.0 / cur_t.bpm) * 4 * 32, sr)   # up to 32 bars
    tail = cur_audio[read:read + max_tail].copy()
    fade_s = min(int(2.0 * sr), len(tail))
    if fade_s > 0:
        tail[-fade_s:] *= np.linspace(1.0, 0.0, fade_s, dtype=np.float32).reshape(-1, 1)
    if len(tail) > 0:
        master.append(tail)

    # Final (current) track's clip runs to the end of the output.
    clips.append({
        "track": cur_t.file_path, "title": cur_t.title,
        "out_start": round(cur_start / sr, 3),
        "out_end": round((written + len(tail)) / sr, 3),
        "fade_in": round(prev_fade_in, 3),
        "fade_out": round(min(2.0, len(tail) / sr), 3),
        "mode": "end", "section": "",
        "bpm": round(cur_t.bpm, 1), "key": cur_t.key,
    })

    output = np.concatenate(master, axis=0)
    peak = np.abs(output).max()
    if peak > 0.95:
        output = output * (0.95 / peak)

    return output, sr, markers, clips


def render_collage(
    tracks: list,
    sr: int = MIX_SR,
    target_length_sec: float = 300.0,
    layers: int = 3,
    min_seg_bars: int = 8,
    max_seg_bars: int = 24,
    seed: Optional[int] = None,
    chaos: float = 0.0,
    state: Optional[dict] = None,
) -> tuple:
    """
    Structured, variable-pace overlap-add collage.

    Everything is locked to one tempo (pool median, octave-folded) and entered on
    a shared downbeat grid, so overlapping segments stay beat-aligned. Rather than
    a fixed cadence, the collage is composed into *movements* whose editing pace
    ebbs and flows:

      - feature: one track holds, playing several of its own sections contiguously
        in natural time order (reads as the track condensed).
      - weave:   rapid, heavily-overlapping segments chosen by TIMBRE — either
        CONTRASTING (CLAP-farthest) or COMPLEMENTARY (CLAP-nearest) to what's
        already sounding, within or across tracks.
      - breathe: a long segment plays mostly alone before the next.

    `chaos` (0..1) is the wildness master. As it rises: weave crowds out the
    calmer movements, segments get shorter, hops shrink (so more layers sound at
    once and switching is rapid), sections get diced into sub-segments so the
    mixer can enter partway through one, and picks alternate between
    contrasting and complementary timbre instead of always contrasting.

    Returns (audio, sr, [SetMarker, ...], clips).
    """
    import random
    if len(tracks) < 2:
        raise ValueError("Need at least 2 tracks for a collage")

    # `state` makes this renderer resumable: pass the same dict back in and the
    # collage continues where it left off, carrying the uncommitted tail and the
    # sounding layers across calls. Radio mode uses it; offline calls omit it
    # and get exactly the old behaviour.
    st = state if state is not None else {}
    resuming = bool(st)
    rng = st.get("rng") or random.Random(seed)

    bpms    = sorted(t.bpm for t in tracks)
    set_bpm = bpms[len(bpms) // 2]
    bar     = (60.0 / set_bpm) * 4
    bar_s   = _time_to_samples(bar, sr)
    total_s = _time_to_samples(target_length_sec, sr)
    # Generous tail: a segment played at a slow native tempo runs longer than
    # the same bar count on the median-tempo grid.
    master  = np.zeros((total_s + max_seg_bars * bar_s * 3 + sr
                        + int(st.get("pos", 0)), 2), dtype=np.float32)

    # Decode + loudness-match once per track, at its NATIVE tempo. Stretching is
    # per-segment and only happens when a segment has to beat-lock to audio
    # that's already sounding (see `emit`) — stretching whole tracks up front
    # cost ~9.6s each and dragged every track to one median tempo.
    cache = st.setdefault("cache", {})
    def get_track(t):
        if t.file_path not in cache:
            audio, _ = _load_audio(t.file_path, sr)
            audio = _loudness_match(audio, t.loudness, MASTER_LOUDNESS)
            cache[t.file_path] = (audio, list(t.downbeats))
        return cache[t.file_path]

    stretch_cache: dict = st.setdefault("stretch_cache", {})
    def _stretched(path, audio, start, n, ratio):
        """Stretch one slice, memoised — repeats are common in a long session."""
        key = (path, start, n, round(ratio, 4))
        got = stretch_cache.get(key)
        if got is None:
            got = _time_stretch(audio[start:start + n], sr, ratio)
            if len(stretch_cache) < 256:
                stretch_cache[key] = got
        return got

    # Tempo the next overlapping segment must lock to. A segment entering alone
    # plays native and becomes the new reference — what a DJ actually does.
    ref = st.setdefault("ref", {"bpm": set_bpm, "bar_s": bar_s})
    stats = st.setdefault("stats", {"stretched": 0, "native": 0})

    def _eff_bpm(t, target):
        """The track's tempo folded by half/double toward `target`."""
        return min((t.bpm, t.bpm * 2.0, t.bpm / 2.0),
                   key=lambda b: abs(target / b - 1.0))

    def emit(t, section, seg_bars, fade_bars, lock,
             shape_in="equal_power", shape_out="equal_power", eq_move="flat"):
        """
        Slice `seg_bars` bars at the track's native tempo. When `lock`, stretch
        that slice to the reference tempo so it beat-locks with what's already
        sounding; otherwise let it play native (far cheaper, and unmangled).
        Returns (audio, eff_bpm, stretched, fade_sec) or None.
        """
        audio, downs = get_track(t)
        # Fold toward the pool median, NOT the live reference: folding toward a
        # reference that each native segment then re-sets lets the tempo halve
        # repeatedly (128 -> 64 -> 32...), which across a resumed session makes
        # segments too long to fit and silently drops them.
        eff = _eff_bpm(t, set_bpm)
        nat_bar = (60.0 / eff) * 4
        nat_bar_s = _time_to_samples(nat_bar, sr)
        entry = _find_nearest_downbeat(section.start, downs, max_offset=nat_bar / 2.0)
        es = _time_to_samples(entry, sr)
        # Clamp to what's left of the track rather than rejecting: a rejected
        # placement would leave a hole, since `pos` advances regardless.
        avail = len(audio) - es
        if avail < nat_bar_s:
            return None
        n_native = min(int(seg_bars) * nat_bar_s, avail)

        ratio = ref["bpm"] / eff
        # Skip the stretch when the drift across this segment is imperceptible
        # (well under a 16th note) — Rubber Band is a subprocess, so not calling
        # it is the cheapest optimisation available.
        drift = (n_native / sr) * abs(ratio - 1.0)
        if lock and drift > 0.025:
            seg = _stretched(t.file_path, audio, es, n_native, ratio)
            stats["stretched"] += 1
            stretched = True
        else:
            seg = audio[es:es + n_native]
            stats["native"] += 1
            stretched = False

        n = len(seg)
        if n < 2:
            return None
        fi = int(np.clip(int(fade_bars) * (n / max(1, int(seg_bars))), 1, n // 2))
        # Band gains first (on the raw slice), then the overall amplitude
        # envelope — filtering a signal that's already fading would smear the
        # gesture across the fade.
        seg = _apply_eq_move(seg, sr, eq_move, fi)
        env = np.ones(n, dtype=np.float32)
        env[:fi]  = _fade_curve(fi, shape_in, rising=True)
        env[-fi:] = _fade_curve(fi, shape_out, rising=False)
        # `seg` may be a view into the cached track (or a cached stretch), so
        # multiply into a new array rather than in place.
        return ((seg * env.reshape(-1, 1)).astype(np.float32), eff, stretched, fi / sr)

    # Carry the uncommitted tail from a previous call in front of this buffer,
    # so segments that straddled the commit point continue seamlessly.
    carry = st.get("carry")
    if carry is not None and len(carry):
        n = min(len(carry), len(master))
        master[:n] += carry[:n]

    chaos = float(np.clip(chaos, 0.0, 1.0))
    # High chaos dices sections so we can enter partway through one, and lowers
    # the length floor so a splice can be a couple of bars.
    sub_bars = max(2, int(round(8 - 6 * chaos))) if chaos >= 0.3 else None
    min_len = 8.0 - 5.0 * chaos
    # Entries normally advance a whole bar, which keeps every layer's downbeats
    # on one grid. Very short splices can't stack that way (a 2-bar segment over
    # a 1-bar hop is only 2 deep), so high chaos allows beat-granular hops:
    # beats still line up, only the downbeat phase shifts between layers.
    # (the concrete bar/beat sizes are taken per-placement from `ref`, since the
    # reference tempo now follows whatever is actually sounding)
    sub_beat_hops = chaos >= 0.5
    seg_pool = _segment_pool(tracks, bar, sub_bars=sub_bars, min_len_sec=min_len)
    have_emb = any(s.embedding for _, s in seg_pool)

    # Placement state. On resume, `pos`/`active` are rebased so they line up with
    # the carried tail now sitting at the front of this buffer.
    markers, clips = [], []
    pos    = int(st.get("pos", 0))
    active = list(st.get("active", []))
    recent = list(st.get("recent", []))
    time_offset = float(st.get("time_offset", 0.0))   # seconds already committed

    def prune():
        active[:] = [a for a in active if a[1] > pos]

    def place(t, section, seg_bars, fade_bars, mode, hop_bars):
        """Overlap-add one segment at `pos`, log it, advance `pos` by hop_bars."""
        nonlocal pos
        prune()
        # Lock only when the overlap is *material*. A feature/breathe hop leaves
        # a 1-2 bar join at the end of a 70-bar segment; stretching the whole
        # segment to beat-lock 3% of it is exactly the waste that "only stretch
        # when transitioning" is meant to avoid — a brief join crossfades fine
        # at native tempo. A weave hop overlaps ~80% and does need locking.
        overlap = max([e for (_, e) in active], default=pos) - pos
        est_len = max(1, int(seg_bars) * ref["bar_s"])
        lock = overlap > 0.25 * est_len
        shape_in, shape_out = _pick_fade_shapes(mode, chaos, rng)
        eq_move = _pick_eq_move(mode, chaos, len(active), rng)
        got = emit(t, section, seg_bars, fade_bars, lock,
                   shape_in, shape_out, eq_move)
        seg, eff, was_stretched, fade_sec = got if got else (None, 0.0, False, 0.0)
        end = min(pos + len(seg), len(master)) if seg is not None else pos
        if seg is not None and end > pos:
            # Equal-power headroom for however many layers this segment joins
            # (itself + whatever `active` hasn't ended yet) — see _layer_gain.
            n_voices = len(active) + 1
            g = _layer_gain(n_voices)
            master[pos:end] += seg[:end - pos] * g
            markers.append(SetMarker(
                time=pos / sr,
                label=f"{t.title.split(' - ')[-1][:30]}  [{section.label} @{section.start:.0f}s]",
                method=mode, stretch_pct=0.0, style=mode))
            clips.append({
                "track": t.file_path, "title": t.title,
                "out_start": round(time_offset + pos / sr, 3),
                "out_end": round(time_offset + (pos + len(seg)) / sr, 3),
                "fade_in": round(fade_sec, 3),
                "fade_out": round(fade_sec, 3),
                "mode": mode, "section": section.label,
                "fade_shape": f"{shape_in}/{shape_out}", "eq": eq_move,
                "bpm": round(eff, 1), "key": t.key,
                "layer_gain": round(g, 3),
            })
            active.append((section.embedding, pos + len(seg)))
            recent.append(t.file_path)
            if not was_stretched:
                # It played native, so it now sets the tempo others lock to.
                ref["bpm"] = eff
                ref["bar_s"] = _time_to_samples((60.0 / eff) * 4, sr)

        # Hops follow the *current* reference tempo, so the grid tracks whatever
        # is actually sounding rather than a fixed global median.
        bar_s_ref = ref["bar_s"]
        beat_ref = max(1, int(round(bar_s_ref / 4)))
        hop_floor_s = beat_ref if sub_beat_hops else bar_s_ref
        if seg is None:
            # Nothing was placed (e.g. the entry ran off the end of the track).
            # Nudge minimally so the next pick fills this space instead of
            # leaving a hole in the timeline.
            pos += beat_ref
            return
        # Hop as a *proportion of the segment actually emitted*. A segment
        # clamped short (near the end of a track) would otherwise advance `pos`
        # by its nominal bar count and leave a hole where no audio was written.
        frac = float(hop_bars) / max(1, int(seg_bars))
        step = max(hop_floor_s, int(round(frac * len(seg))))
        if sub_beat_hops:
            # Fractional-bar hops would land layers off-beat against each other,
            # so snap to the shared beat grid. (Only here: a bar isn't always an
            # exact 4 sample-beats, so quantising whole-bar hops would drift.)
            step = max(hop_floor_s, int(round(step / beat_ref)) * beat_ref)
        pos += step

    def pick_next(last_key, complement=False):
        """
        A (track, section) to layer next, chosen by timbre against what's
        already sounding: `complement=True` picks the CLAP-nearest (textures
        that sit together), otherwise the CLAP-farthest (deliberate contrast).
        """
        prune()
        act = [e for (e, _) in active if e is not None]
        cands = [ts for ts in seg_pool if ts[0].file_path not in recent[-2:]] or seg_pool
        # Diced pools get large; scoring a random slice keeps picks cheap and
        # adds variety without changing the character of the choice.
        if len(cands) > 300:
            cands = rng.sample(cands, 300)
        if have_emb and act:
            def sim(ts):
                if not ts[1].embedding:
                    return 0.0 if complement else 1.0      # unknown ranks last
                return max(_clap_cos(ts[1].embedding, e) for e in act)
            ranked = sorted(cands, key=sim, reverse=complement)
            top = ranked[:6]
            if last_key:  # among the best timbre matches, favour harmonic fit
                top.sort(key=lambda ts: -camelot_compatibility(last_key, ts[0].key))
            return top[0]
        return rng.choice(cands)

    # Extend by `target_length_sec` *from wherever we are* when resuming.
    loop_target = pos + total_s if resuming else total_s
    guard = 0
    while pos < loop_target and guard < 100000:
        guard += 1
        prune()
        # Radio is endless, so hold the mid-set weighting rather than replaying
        # an intro/outro arc on every extend.
        progress = 0.5 if resuming else pos / total_s
        calm = 1.0 - 0.85 * chaos          # chaos crowds out the calm movements
        w_weave   = 1.5 + (1.5 if 0.2 <= progress <= 0.85 else 0.0) + 5.0 * chaos
        w_feature = 1.2 * calm
        w_breathe = (1.0 + (1.5 if (progress < 0.2 or progress > 0.85) else 0.0)) * calm
        mode = rng.choices(["weave", "feature", "breathe"],
                           weights=[w_weave, w_feature, w_breathe])[0]

        if mode == "feature":
            t = rng.choice([x for x in tracks if x.file_path not in recent[-1:]] or tracks)
            secs = sorted(_track_sections(t), key=lambda s: s.start)
            if not secs:
                pos += bar_s
                continue
            K = min(len(secs), rng.randint(2, 4))
            start_i = rng.randint(0, max(0, len(secs) - K))
            for s in secs[start_i:start_i + K]:
                seg_bars = int(np.clip((s.end - s.start) / bar, min_seg_bars, max_seg_bars))
                # Contiguous: hop ≈ segment length minus a short 2-bar join.
                place(t, s, seg_bars, fade_bars=2, mode="feature",
                      hop_bars=max(1, seg_bars - 2))
                if pos >= loop_target:
                    break

        elif mode == "weave":
            last_key = None
            # Longer weave runs and shorter segments as chaos rises.
            runs = rng.randint(3, 6 + int(8 * chaos))
            hi_bars = max(min_seg_bars,
                          int(((min_seg_bars + max_seg_bars) // 2) * (1.0 - 0.6 * chaos)))
            # More simultaneous layers => hop is a smaller slice of the segment.
            divisor = max(1, layers) + int(round(2 * chaos))
            for _ in range(runs):
                # Alternate contrast with complementary blends as chaos rises.
                t, s = pick_next(last_key, complement=rng.random() < 0.5 * chaos)
                last_key = t.key
                seg_bars = rng.randint(min_seg_bars, hi_bars)
                # Fractional hop: on short splices an integer bar hop would
                # floor to 1 and cap the stack depth regardless of `divisor`.
                place(t, s, seg_bars, fade_bars=max(2, seg_bars // 2), mode="weave",
                      hop_bars=max(0.25, seg_bars / divisor))
                if pos >= loop_target:
                    break

        else:  # breathe
            for _ in range(rng.randint(1, 2)):
                t, s = pick_next(None)
                seg_bars = rng.randint((min_seg_bars + max_seg_bars) // 2, max_seg_bars)
                place(t, s, seg_bars, fade_bars=max(2, seg_bars // 4), mode="breathe",
                      hop_bars=max(1, seg_bars - 1))  # minimal overlap
                if pos >= loop_target:
                    break

    if state is not None:
        # ── Streaming (radio) ────────────────────────────────────────────────
        # Placements only ever write at or after `pos`, and `pos` only moves
        # forward, so everything before it is already final. The tails of
        # segments that extend past it are carried into the next call.
        safe = pos
        out = master[:safe].copy()

        tail_end = min(len(master), max([e for (_, e) in active] + [pos]))
        st["carry"]  = master[safe:tail_end].copy()
        st["pos"]    = 0
        st["active"] = [(e, end - safe) for (e, end) in active if end > safe]
        st["recent"] = recent[-8:]
        st["rng"]    = rng
        st["time_offset"] = time_offset + safe / sr

        # A local lookahead limiter does the real peak control now (it reacts
        # to the actual hot passage, not the whole committed block), so this
        # scalar only has to trim whatever it didn't already catch. Because
        # that residual should now be small, the ceiling can recover after a
        # dense passage instead of staying pinned for the rest of the radio
        # session — the instant-duck/slow-release shape a limiter itself
        # uses, one level up. (Previously this only ever dropped, to avoid
        # the pumping a per-block *normalize* would cause; that risk doesn't
        # apply to a slow exponential recovery riding on top of the limiter.)
        if len(out):
            raw_tail = out[-int(sr * 0.015):].copy()   # pre-limiter, for next call's lookahead
            out, limiter_gain = _limiter(
                out, sr, pre_tail=st.get("limiter_tail"),
                start_gain=st.get("limiter_gain", 1.0))
            st["limiter_tail"] = raw_tail
            st["limiter_gain"] = limiter_gain

            gain = st.get("gain", 1.0)
            peak = float(np.abs(out).max()) * gain
            if peak > 0.95:
                gain = 0.95 / max(1e-9, float(np.abs(out).max()))
            else:
                gain = min(1.0, gain + 0.02)   # slow recovery, ~minutes to unwind a duck
            st["gain"] = gain
            out = np.clip(out * gain, -0.99, 0.99).astype(np.float32)
        return out, sr, markers, clips

    # Trim trailing silence and fade the very end.
    nz = np.nonzero(np.abs(master).sum(axis=1))[0]
    output = master[:nz[-1] + 1] if len(nz) else master[:total_s]
    fade_s = min(int(2.0 * sr), len(output))
    if fade_s > 0:
        output[-fade_s:] *= np.linspace(1.0, 0.0, fade_s, dtype=np.float32).reshape(-1, 1)
    if len(output):
        output, _ = _limiter(output, sr)
    # Cheap final defensive clamp — should rarely fire once the limiter above
    # is doing the real work; kept as insurance against float rounding and
    # lookahead edge effects at the very start/end of the render.
    peak = np.abs(output).max()
    if peak > 0.95:
        output = output * (0.95 / peak)
    return output, sr, markers, clips


# ── Auto-select best cue points ───────────────────────────────────────────────

def best_cue_out(track: TrackMeta) -> Optional[CuePoint]:
    outs = [c for c in track.cue_points if c.type == "out"]
    return max(outs, key=lambda c: c.confidence) if outs else None


def best_cue_in(track: TrackMeta) -> Optional[CuePoint]:
    ins = [c for c in track.cue_points if c.type == "in"]
    return max(ins, key=lambda c: c.confidence) if ins else None
