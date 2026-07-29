"""
Piecewise tempo tracking for whole DJ mixes.

`analyzer._compute_beats` assumes one constant tempo for a file, which is right
for a single track and wrong for a 60-minute mix: the DJ nudges the master tempo
across the set, and a single estimate averages a 122 BPM opener with a 128 BPM
peak into a number that describes neither.

So we slide a window over the mix's onset envelope, refine a tempo per window
with the *same* estimator the analyzer uses, and cut the result into constant-
tempo segments. `analyzer.py` needs no changes — `_refine_tempo_phase` is reused
verbatim, once per window for the tempo track and once per segment for a
segment-global phase.

Two things worth knowing before using this:

  - Duration in beats needs tempo but *not* phase, so the hard part of grid
    tracking (phase) isn't on the critical path for the corpus's primary
    statistic. Don't over-trust `TempoSegment.phase`.
  - Never estimate tempo inside a transition. Where two tracks overlap the
    onset envelope is a superposition of two grids; if they're beatmatched it
    reinforces harmlessly, and if they aren't the estimate is garbage. Callers
    measure tempo from the clean regions either side.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import librosa
import numpy as np

from .analyzer import BEAT_HOP, SR, _fold_octave, _refine_tempo_phase

# A window has to hold enough bars for autocorrelation to lock on; 30 s is ~15
# bars at 124 BPM. The hop sets how precisely a tempo change can be located.
WIN_SEC   = 30.0
HOP_SEC   = 10.0

# Relative tempo step that counts as a real change rather than estimator jitter.
# `_refine_tempo_phase` searches a 0.02 BPM grid, so ~0.6% at 124 BPM is a
# comfortable few grid steps above the noise floor.
SPLIT_PCT = 0.006

# A window whose refined tempo is this far from the running median is an octave
# flip or an overlap artifact, not a tempo change. `_fold_octave` puts everything
# in [90, 180), but a window near a boundary can still land on the wrong side.
OUTLIER_PCT = 0.15


@dataclass(frozen=True)
class TempoSegment:
    """A stretch of the mix held at one tempo."""
    start: float
    end: float
    bpm: float
    phase: float          # seconds; first beat of the segment's rigid grid
    confidence: float

    @property
    def beat_sec(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_sec(self) -> float:
        return (60.0 / self.bpm) * 4.0

    def to_dict(self) -> dict:
        return asdict(self)


def mix_onset_envelope(y: np.ndarray, sr: int = SR, hop: int = BEAT_HOP) -> np.ndarray:
    """
    Onset strength for a whole mix, computed once.

    This is the only whole-file spectral pass we can afford: at BEAT_HOP it is
    one float per 11.6 ms (~310k floats for an hour), where a whole-file STFT at
    hop 512 would be hundreds of megabytes.
    """
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


def _window_tempo(oenv_slice: np.ndarray, sr: int, hop: int) -> tuple:
    """(bpm, phase, confidence) for one window, or (nan, 0, 0) if unusable."""
    if len(oenv_slice) < 8 or float(oenv_slice.sum()) <= 0.0:
        return float("nan"), 0.0, 0.0
    try:
        hint = librosa.beat.beat_track(onset_envelope=oenv_slice, sr=sr,
                                       hop_length=hop)[0]
        hint = float(np.atleast_1d(hint)[0])
    except Exception:
        hint = 120.0
    if not np.isfinite(hint) or hint <= 0:
        hint = 120.0
    bpm, phase, conf = _refine_tempo_phase(oenv_slice, sr, hop, hint)
    return float(bpm), float(phase), float(conf)


def _reject_octave_outliers(bpms: np.ndarray) -> np.ndarray:
    """
    NaN out windows that sit far from the run's median tempo.

    A DJ set moves a few percent; a window that reads 15% off has locked onto a
    half/double or straddles a tempo-clashing overlap. Folding it back in would
    corrupt both the segmentation and the median.
    """
    out = bpms.copy()
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return out
    med = float(np.median(finite))
    for _ in range(2):   # one pass can be skewed by a cluster of bad windows
        bad = np.isfinite(out) & (np.abs(out - med) / med > OUTLIER_PCT)
        out[bad] = np.nan
        finite = out[np.isfinite(out)]
        if finite.size == 0:
            return out
        med = float(np.median(finite))
    return out


def _fill_nans(bpms: np.ndarray) -> np.ndarray:
    """Linearly interpolate over rejected windows so the track stays contiguous."""
    x = np.arange(len(bpms), dtype=float)
    ok = np.isfinite(bpms)
    if not ok.any():
        return np.full_like(bpms, np.nan)
    return np.interp(x, x[ok], bpms[ok])


def _median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Odd-kernel median filter with edge replication (no scipy.signal needed)."""
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=-1)


def _split_indices(bpms: np.ndarray, split_pct: float, k: int = 3) -> list:
    """
    Window indices where the tempo level shifts.

    Compares the median of the k windows before a candidate against the k after,
    which is far steadier than a first difference: a single noisy window can't
    manufacture a split, and a genuine step shows up in every k-vs-k comparison
    across it. Only local maxima of the shift are kept, so one tempo change
    yields one boundary rather than a run of adjacent ones.
    """
    n = len(bpms)
    if n < 2 * k + 2:
        return []
    shift = np.zeros(n)
    for i in range(k, n - k + 1):
        before = float(np.median(bpms[i - k:i]))
        after = float(np.median(bpms[i:i + k]))
        if before > 0:
            shift[i] = abs(after - before) / before

    cands = [i for i in range(k, n - k + 1) if shift[i] > split_pct]
    splits = []
    for i in cands:
        lo, hi = max(0, i - k), min(n, i + k + 1)
        if shift[i] >= shift[lo:hi].max() - 1e-12:
            if not splits or (i - splits[-1]) >= k:
                splits.append(i)
    return splits


def track_tempo_segments(
    onset_env: np.ndarray,
    duration: Optional[float] = None,
    sr: int = SR,
    hop: int = BEAT_HOP,
    win_sec: float = WIN_SEC,
    hop_sec: float = HOP_SEC,
    split_pct: float = SPLIT_PCT,
) -> list:
    """
    Cut a mix into constant-tempo segments.

    `onset_env` is the whole-mix envelope from `mix_onset_envelope`. Returns
    segments covering [0, duration] with no gaps; a mix at one tempo throughout
    returns a single segment.
    """
    if duration is None:
        duration = len(onset_env) * hop / sr
    if len(onset_env) < 8 or duration <= 0:
        return []

    fps = sr / hop
    win_f, hop_f = int(win_sec * fps), max(1, int(hop_sec * fps))

    if len(onset_env) <= win_f:
        bpm, phase, conf = _window_tempo(onset_env, sr, hop)
        if not np.isfinite(bpm):
            return []
        return [TempoSegment(0.0, float(duration), bpm, phase, conf)]

    starts = list(range(0, len(onset_env) - win_f + 1, hop_f))
    bpms = np.array([_window_tempo(onset_env[s:s + win_f], sr, hop)[0]
                     for s in starts])

    bpms = _reject_octave_outliers(bpms)
    if not np.isfinite(bpms).any():
        return []
    bpms = _median_filter(_fill_nans(bpms), k=5)

    # Window index -> the time at that window's centre.
    centres = np.array([(s + win_f / 2.0) / fps for s in starts])
    bounds = [0.0] + [float(centres[i]) for i in _split_indices(bpms, split_pct)] \
             + [float(duration)]
    bounds = sorted(set(round(b, 3) for b in bounds))

    segments = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 1.0:
            continue
        i0 = int(np.clip(a * fps, 0, len(onset_env) - 1))
        i1 = int(np.clip(b * fps, i0 + 1, len(onset_env)))
        # A segment-global refinement: the ±6% search lands back on the segment's
        # own tempo, and gives the phase a per-window estimate can't.
        bpm, phase, conf = _window_tempo(onset_env[i0:i1], sr, hop)
        if not np.isfinite(bpm):
            in_seg = [j for j, c in enumerate(centres) if a <= c < b]
            if not in_seg:
                continue
            bpm, phase, conf = float(np.median(bpms[in_seg])), 0.0, 0.0
        segments.append(TempoSegment(start=a, end=b, bpm=bpm,
                                     phase=a + phase, confidence=conf))
    return segments


def bpm_at(segments: list, t: float) -> tuple:
    """(bpm, confidence) at time `t`; falls back to the nearest segment."""
    if not segments:
        return float("nan"), 0.0
    for s in segments:
        if s.start <= t < s.end:
            return s.bpm, s.confidence
    nearest = min(segments, key=lambda s: min(abs(t - s.start), abs(t - s.end)))
    return nearest.bpm, nearest.confidence


def beats_between(segments: list, t0: float, t1: float) -> float:
    """
    Beats elapsed between two times, integrated across tempo segments.

    This — not `(t1 - t0) / beat_at_t0` — is what makes a transition length in
    beats meaningful when it straddles a tempo change.
    """
    if t1 <= t0:
        return 0.0
    if not segments:
        return float("nan")
    total = 0.0
    for s in segments:
        lo, hi = max(t0, s.start), min(t1, s.end)
        if hi > lo:
            total += (hi - lo) / s.beat_sec
    # Times outside the segment span (shouldn't happen, but don't return 0).
    covered = sum(max(0.0, min(t1, s.end) - max(t0, s.start)) for s in segments)
    if covered < (t1 - t0) - 1e-6:
        bpm, _ = bpm_at(segments, t0)
        total += ((t1 - t0) - covered) / (60.0 / bpm)
    return total


def bars_between(segments: list, t0: float, t1: float) -> float:
    """Bars (4 beats) elapsed between two times."""
    return beats_between(segments, t0, t1) / 4.0


def segments_from_dicts(rows: list) -> list:
    """Rehydrate segments persisted as JSON."""
    return [TempoSegment(**r) for r in rows]
