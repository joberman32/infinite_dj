"""
Measuring one transition in a DJ mix, from the mix audio alone.

We have mix audio and a tracklist, but not the source tracks — so mix-to-track
subsequence DTW (the usual approach) is unavailable, and the transition has to
be recovered from the mixture itself.

## Why this isn't a similarity ramp

The obvious method — compare each frame to a "clean A" reference and a "clean B"
reference and read the transition width off the ratio — fails, and fails
quietly. Against this repo's own `_blend` with an exactly-known 16-bar
crossfade it recovers **3.7 bars**.

The cause is structural rather than a tuning problem: in dance music the low
band dominates spectral energy, and the low band is a *switch, not a ramp*.
`_make_profile` swaps the bass single-source over `+/-bass_w = 0.12` of phase,
and real DJs bass-swap the same way, for the same reason (only one kick at a
time). So a broadband ratio measures the *bass-swap width* and reports it as the
crossfade width, on every four-to-the-floor mix in the corpus.

Hence two departures:

  - **Measure per band.** Mid and high carry the crossfade duration; low
    carries only the swap *centre* (its width is meaningless — a logistic can't
    represent a switch, and fitting one overestimates the span ~2.3x). Bands
    reuse mixer.LOW_CUT/MID_CUT so mined values are directly comparable to
    `_make_profile` arguments.
  - **Model the mixture, don't compare to it.** During an overlap the mix is
    literally `x = g_A*A + g_B*B`, so solve a 2-source non-negative
    decomposition per frame against the clean reference spectra. That yields the
    actual fader position instead of a monotone-ish proxy, and its residual is a
    principled confidence signal a similarity ratio doesn't have.

Note `alpha` is formed in the **amplitude** domain (`sqrt`): the decomposition's
coefficients are powers, while the profile lanes we compare against are
amplitude gains, and mixing the two introduces a systematic shape error.

## What not to reach for

MFCC is log-compressed, so a mixture's MFCC is not a convex combination of its
sources' — the alpha it produces is squashed and non-monotone by construction.
CLAP embeddings are 8-second windows, ~4 bars at 128 BPM, coarser than the whole
quantity being measured. Both are wrong tools here; use linear power spectra.

## Guarding it

Separability alone is not enough. In testing a band with rho=0.897 returned a
centre 784 phase-units outside the crossfade — a plausible-looking condition-
number threshold would have accepted it. The strongest guard is cross-band
agreement, because the outlier is identifiable by *disagreement* with no
absolute threshold needed. See `_REJECT` ordering in `probe_transition`.

## What the measured span is, and isn't

The recovered span is **not** a style's `n_bars`, and no fixed factor converts
between them. A crossfade's per-band alpha travels its middle 80% in ~0.74 of
the crossfade for a symmetric blend but ~0.42 for the `swap` profile's mid lane,
and the mapping depends on automation shape we don't get to observe. So don't
try to recover `n_bars`; compare like with like, and let the estimator's bias
cancel on both sides.

Measured accuracy against known crossfades, per band:

  - **Symmetric profiles**: mid and high recover the span to +6% for a fully
    symmetric `blend`, degrading with asymmetry — +12% for `build`
    (`mid_lead=0.10`) and +20% for `fade` (`mid_lead=0.30`).
  - **Asymmetric profiles** (`swap`, which holds the incoming mids back while
    bringing its highs in early): the mid span is over-reported ~2x. Cause,
    confirmed by rebuilding the same mix with `in_high_lead=0.50`: the incoming
    track's early *highs* leak into the mid analysis slice through STFT spread
    of impulsive hats — not through the band filters, which are ~70 dB down
    there. That is irreducible for percussive content, and real hats are
    impulsive too.
  - **Low band: centre only.** Its span reads ~2.6x high, as a logistic can't
    represent a switch.
  - **Per-band phase is NOT recoverable.** This is a negative result worth
    stating plainly, because it's tempting to assume otherwise. Absolute band
    centres come back accurate to ~±2 s on a 15-30 s crossfade, which sounds
    fine — but the inter-band separations that would calibrate `cp`,
    `in_mid_lead` and `in_high_lead` are *themselves* of that order, so the
    measurement cannot resolve them. Sweeping `cp` from 0.35 to 0.72 (an 11.5 s
    move of the true bass-swap centre) moved the measured centre 1.6 s:
    monotone, but ~7x compressed, with no usable dynamic range against real-
    world noise. Raising the low band's frequency resolution to 8192 did not
    fix it, and flipped the apparent band ordering — which is how we know an
    earlier "ordering recovered correctly" result was luck, not signal.
    Treat `band_phase` as diagnostic output, not a calibration target.

By contrast **duration is very well behaved**: measured span over nominal
`n_bars` held at 0.787 +/- 0.008 across 4, 8, 12, 16, 24 and 32 bars for a
symmetric profile. Near-perfectly linear, so it both calibrates and inverts.

Consequence for validation: the engine's side of any comparison must be
**rendered and probed through this same pipeline**, not computed analytically
from its automation lanes. The analytic lanes contain no spectral leakage, so an
analytic comparison would attribute the leakage bias to the DJ. `profile_*`
below are for interpretation and reporting, not for scoring.

This module is deliberately I/O-free: pure functions over arrays, no file
reading and no database. That is what lets the whole test suite run on
synthesized audio with exactly-known ground truth.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import librosa
import numpy as np

from .mixer import LOW_CUT, MID_CUT, _sample_lane

# Analysis resolution for the probe. 2048 at 22050 Hz is ~93 ms windows hopped
# every 23 ms — fine enough to place a bass swap, coarse enough that a 30-second
# crossfade is ~1300 frames rather than a memory problem.
N_FFT = 2048
HOP = 512

# The low band gets its own, much longer transform. At 2048 the 20-160 Hz slice
# is only ~13 bins of 10.8 Hz, so two basslines a few semitones apart smear into
# each other and the decomposition is ill-posed in practice even where the
# reference correlation looks acceptable — measured reference rho 0.229, and a
# bass swap whose true centre moved 11.5 s was measured as moving 1.2 s. At 8192
# the same slice is 52 bins of 2.7 Hz and rho falls to 0.053.
#
# Bass wants frequency resolution; hats want time resolution. There's no single
# window that serves both, so don't try to pick one.
N_FFT_LOW = 8192

BANDS = ("low", "mid", "high")

# The probe's bands sample the *interior* of each of the mixer's three bands,
# inset away from the crossovers rather than butting up against them, on the
# principle that a DJ mixer's EQ split is a filter and not a brick wall.
#
# Honest caveat: this is a precaution, not a demonstrated fix. On band-limited
# synthetic material it changes the recovered numbers by nothing measurable,
# because that material has guard bands already. It should help on real tracks,
# which don't — but that remains unverified until there's a real corpus.
#
# Mined values stay directly comparable to `_make_profile` arguments: the same
# three lanes, sampled where each one is least ambiguous.
BAND_INSET_LO = 1.5    # a band starts this far above the crossover below it
BAND_INSET_HI = 0.8    # ...and ends this fraction of the crossover above it

# ── Reject thresholds ────────────────────────────────────────────────────────
# A band whose two references are this correlated can't be decomposed: the Gram
# matrix is near-singular and the split between sources is arbitrary. This is a
# *pre-filter*, not a sufficient guard.
MAX_RHO = 0.5
# Least-squares fit quality for the logistic ramp.
MIN_R2 = 0.5
# Mean model residual over the transition; above this the two-source model isn't
# explaining the audio (effects, a third deck, an echo-out).
MAX_RESIDUAL = 0.6
# The fitted ramp must actually travel. Below this the bands are too alike to
# tell apart — which is exactly the long, smooth, timbrally-matched blend we
# care most about, so this rejection is a known source of short-bias.
MIN_TRAVEL = 0.30
# Cross-band centre agreement, in multiples of the median absolute deviation.
AGREE_MAD = 3.0
# A ramp narrower than this many beats, *with a good fit*, is a cut rather than
# a blend. Narrow with a bad fit is a rejection.
CUT_MAX_BEATS = 1.0
# Hard sanity bound on a plausible transition.
MAX_DURATION_BEATS = 256.0


@dataclass(frozen=True)
class BandFit:
    """The logistic ramp recovered from one frequency band."""
    band: str
    rho: float              # reference correlation; high == inseparable
    center_t: float         # seconds, absolute in the mix
    width_sec: float        # logistic scale
    span_1090: float        # seconds from 10% to 90% of the fitted travel
    travel: float           # fitted upper asymptote minus lower
    r2: float
    residual: float         # mean two-source model residual over the ramp
    ok: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    """Everything measured at one announced boundary."""
    status: str                       # "ok" | "rejected"
    reject_reason: Optional[str] = None
    announced_t: float = 0.0
    t_start: float = float("nan")
    t_end: float = float("nan")
    t_center: float = float("nan")
    t_bass: Optional[float] = None
    duration_sec: float = float("nan")
    duration_beats: float = float("nan")
    duration_bars: float = float("nan")
    is_cut: bool = False
    bands: dict = field(default_factory=dict)        # band -> BandFit
    band_phase: dict = field(default_factory=dict)   # band -> 50% phase in [t_start,t_end]
    confidence: float = 0.0
    sub_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bands"] = {k: (v.to_dict() if isinstance(v, BandFit) else v)
                      for k, v in self.bands.items()}
        return d


def _reject(announced_t: float, reason: str, **extra) -> ProbeResult:
    return ProbeResult(status="rejected", reject_reason=reason,
                       announced_t=announced_t, **extra)


# ── Spectral plumbing ────────────────────────────────────────────────────────

def power_spectrogram(y: np.ndarray, sr: int, n_fft: int = N_FFT,
                      hop: int = HOP) -> tuple:
    """(power spectrogram (F,T), frame times, frequencies)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.abs(librosa.stft(np.asarray(y, dtype=np.float32),
                                n_fft=n_fft, hop_length=hop)) ** 2
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S.astype(np.float64), times, freqs


def band_slice(freqs: np.ndarray, band: str) -> slice:
    """
    Frequency-bin slice for a band's interior.

    Anchored on the mixer's own crossover points (`LOW_CUT`/`MID_CUT`) but inset
    away from them, so each band is sampled where one automation lane dominates
    rather than in the filter transition region where two do. See the note on
    `BAND_INSET_LO`/`BAND_INSET_HI`.
    """
    if band == "low":
        lo, hi = 20.0, LOW_CUT * BAND_INSET_HI
    elif band == "mid":
        lo, hi = LOW_CUT * BAND_INSET_LO, MID_CUT * BAND_INSET_HI
    elif band == "high":
        lo, hi = MID_CUT * BAND_INSET_LO, float(freqs[-1]) + 1.0
    else:
        raise ValueError(f"unknown band {band!r}")
    idx = np.where((freqs >= lo) & (freqs < hi))[0]
    if idx.size == 0:
        return slice(0, 0)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def reference_correlation(ref_a: np.ndarray, ref_b: np.ndarray) -> float:
    """
    Cosine similarity of two reference spectra.

    1.0 means the two tracks are spectrally identical in this band and the
    decomposition is degenerate — there is no fact of the matter about which
    source contributed what.
    """
    na, nb = np.linalg.norm(ref_a), np.linalg.norm(ref_b)
    if na <= 0 or nb <= 0:
        return 1.0
    return float(np.clip(ref_a @ ref_b / (na * nb), 0.0, 1.0))


def unmix_two_source(S: np.ndarray, ref_a: np.ndarray,
                     ref_b: np.ndarray) -> tuple:
    """
    Per-frame non-negative decomposition of `S` onto two reference spectra.

    Returns `(a, b, residual)`, each length T. `a`/`b` are the power
    contributions of source A and B; `residual` is the fraction of each frame's
    energy the two-source model fails to explain.

    With only two variables the Gram matrix is 2x2 and constant across frames,
    so the whole thing is one solve plus one matmul. Non-negativity is handled
    exactly rather than by clipping: for two variables, if the unconstrained
    solution has a negative component, the constrained optimum lies on the
    corresponding axis, so we fall back to the closed-form 1-variable solution.
    """
    ref_a = np.asarray(ref_a, dtype=np.float64)
    ref_b = np.asarray(ref_b, dtype=np.float64)
    aa, bb, ab = ref_a @ ref_a, ref_b @ ref_b, ref_a @ ref_b
    T = S.shape[1]
    if aa <= 0 or bb <= 0:
        return np.zeros(T), np.zeros(T), np.ones(T)

    ra_s, rb_s = ref_a @ S, ref_b @ S            # (T,), (T,)
    det = aa * bb - ab * ab
    if abs(det) < 1e-12 * aa * bb:
        # Degenerate: references are collinear. Project onto A alone and let the
        # separability guard reject this band.
        a = np.clip(ra_s / aa, 0.0, None)
        b = np.zeros(T)
    else:
        a = (bb * ra_s - ab * rb_s) / det
        b = (aa * rb_s - ab * ra_s) / det
        # Exact 2-variable NNLS: a negative component means the optimum sits on
        # the other axis.
        neg_a, neg_b = a < 0, b < 0
        if neg_a.any():
            a[neg_a] = 0.0
            b[neg_a] = np.clip(rb_s[neg_a] / bb, 0.0, None)
        if neg_b.any():
            b[neg_b] = 0.0
            a[neg_b] = np.clip(ra_s[neg_b] / aa, 0.0, None)

    recon_sq = (a * a * aa) + (b * b * bb) + (2 * a * b * ab)
    frame_sq = np.einsum("ft,ft->t", S, S)
    cross = 2.0 * (a * ra_s + b * rb_s)
    resid_sq = np.clip(frame_sq - cross + recon_sq, 0.0, None)
    residual = np.sqrt(resid_sq) / (np.sqrt(frame_sq) + 1e-12)
    return a, b, np.clip(residual, 0.0, 1.0)


def mixing_alpha(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fader position from the decomposition's power coefficients.

    The sqrt matters: `a`/`b` are powers while the automation lanes we compare
    against are amplitude gains, so comparing power-domain alpha to a lane
    introduces a systematic shape error.
    """
    sa, sb = np.sqrt(np.clip(a, 0.0, None)), np.sqrt(np.clip(b, 0.0, None))
    return sb / (sa + sb + 1e-12)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Short median filter to kill frame-level noise without smearing the ramp."""
    n = int(n)
    if n <= 1 or len(x) < n:
        return x
    if n % 2 == 0:
        n += 1
    pad = n // 2
    padded = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, n), axis=-1)


def _pool(x: np.ndarray, n: int) -> np.ndarray:
    """Moving sum over `n` frames, length-preserving with edge replication."""
    n = int(n)
    if n <= 1 or len(x) < n:
        return x
    if n % 2 == 0:
        n += 1
    pad = n // 2
    padded = np.pad(x, pad, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, n).sum(axis=-1)


def pooled_alpha(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Fader position from power coefficients pooled over `n` frames.

    Pooling `a` and `b` *before* forming the ratio is the principled aggregation
    for a band whose content is sparse in time — hats, percussion, stabs. `a`
    and `b` are energies, so a moving sum answers "how much did each source
    contribute over this window", which automatically ignores frames where the
    band is silent. Smoothing the alpha *ratio* instead gives every frame an
    equal vote, including the ones between hats where alpha is the quotient of
    two noise floors, and that is what makes a percussive band unusable.
    """
    return mixing_alpha(_pool(a, n), _pool(b, n))


# ── Ramp fitting ─────────────────────────────────────────────────────────────

_LN9 = float(np.log(9.0))


def _logistic(t, lo, hi, tc, w):
    # Clip the exponent rather than letting it overflow: the fitter explores
    # extreme widths early on, and a warning per evaluation is just noise.
    z = np.clip((t - tc) / w, -60.0, 60.0)
    return lo + (hi - lo) / (1.0 + np.exp(-z))


def fit_ramp(alpha: np.ndarray, times: np.ndarray,
             weights: Optional[np.ndarray] = None) -> tuple:
    """
    Fit `alpha(t) ~ lo + (hi-lo)*sigmoid((t-tc)/w)`.

    Returns `(center_t, width_sec, span_1090, travel, r2)`.

    Parametric rather than threshold-crossing because a real alpha curve is
    noisy and can dip back below its 10% crossing partway through; a fit
    degrades gracefully where a crossing search reports nonsense. The asymptotes
    are free (bounded to [0,1]) so contamination that stops alpha reaching 0 or
    1 shows up as reduced `travel` instead of a distorted width.

    `weights` should be per-frame band energy. A frame with no energy in this
    band carries no information about the fader position — its alpha is the
    ratio of two noise floors — so letting it vote equally is what wrecks the
    fit on percussive bands, where content is sparse in time. Weighting is not
    cosmetic here: it's the difference between a usable and an unusable high
    band on real music.

    10%-to-90% of the fitted travel spans `2*ln(9)*w`.
    """
    from scipy.optimize import curve_fit

    n = len(alpha)
    if n < 8:
        return float("nan"), float("nan"), float("nan"), 0.0, 0.0

    if weights is None:
        w_arr = np.ones(n)
    else:
        w_arr = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
        if w_arr.max() <= 0:
            w_arr = np.ones(n)
        w_arr = w_arr / w_arr.max()
    # curve_fit minimises sum(((y - f)/sigma)^2), so sigma ~ 1/sqrt(weight).
    sigma = 1.0 / np.sqrt(w_arr + 1e-3)

    span = float(times[-1] - times[0]) or 1.0
    # Seed the centre at the steepest rise, which beats a midpoint crossing when
    # the curve is noisy at its ends.
    k = max(1, n // 50)
    grad = np.gradient(_smooth(alpha, k))
    tc0 = float(times[int(np.argmax(grad))])
    p0 = [float(np.percentile(alpha, 5)), float(np.percentile(alpha, 95)),
          tc0, max(span / 20.0, 1e-3)]
    bounds = ([0.0, 0.0, float(times[0]) - span, 1e-3],
              [1.0, 1.0, float(times[-1]) + span, span])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_logistic, times, alpha, p0=p0, sigma=sigma,
                                bounds=bounds, maxfev=6000)
    except Exception:
        return float("nan"), float("nan"), float("nan"), 0.0, 0.0

    lo, hi, tc, w = (float(v) for v in popt)
    pred = _logistic(times, lo, hi, tc, w)
    # Weighted R^2, so fit quality is judged on the frames the fit was asked to
    # explain rather than on silent ones.
    mean_a = float(np.average(alpha, weights=w_arr))
    ss_res = float(np.sum(w_arr * (alpha - pred) ** 2))
    ss_tot = float(np.sum(w_arr * (alpha - mean_a) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return tc, w, 2.0 * _LN9 * w, hi - lo, float(np.clip(r2, 0.0, 1.0))


# ── The engine's own crossfade, measured the same way ────────────────────────
#
# The corpus statistic is "10-90% span of band alpha", which is NOT the same
# number as a style's nominal `n_bars`: for `_make_profile(0.55, 0.12, 0.12)`
# the alpha derived from the lanes travels its middle 80% in ~0.74 of the
# crossfade. Comparing a mined span against a nominal bar count would be a ~20%
# apples-to-oranges error that would silently corrupt every calibrated value.
#
# So both sides of any comparison — and the ground truth in the tests — run
# through these, which read the engine's automation lanes analytically.

def profile_band_alpha(profile, band: str, n: int = 4096) -> np.ndarray:
    """`alpha(phase)` for one band, computed from a TransitionProfile's lanes."""
    phase = np.linspace(0.0, 1.0, n)
    g_out = (_sample_lane(getattr(profile, f"out_{band}"), phase)
             * _sample_lane(profile.out_vol, phase))
    g_in = (_sample_lane(getattr(profile, f"in_{band}"), phase)
            * _sample_lane(profile.in_vol, phase))
    return np.asarray(g_in / (g_out + g_in + 1e-12), dtype=np.float64)


def _crossing(alpha: np.ndarray, level: float) -> float:
    """Phase at which `alpha` first reaches `level`, linearly interpolated."""
    n = len(alpha)
    idx = np.where(alpha >= level)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return 0.0
    a0, a1 = alpha[i - 1], alpha[i]
    frac = 0.0 if a1 == a0 else (level - a0) / (a1 - a0)
    return (i - 1 + frac) / (n - 1)


def profile_span_1090(profile, band: str) -> float:
    """
    Width of a band's alpha travel, in phase units, 10% to 90%.

    Multiply by the crossfade's length in seconds/bars for a number directly
    comparable to `ProbeResult.duration_sec` / `duration_bars`.
    """
    alpha = profile_band_alpha(profile, band)
    lo, hi = float(alpha[0]), float(alpha[-1])
    if hi - lo < 1e-6:
        return float("nan")
    return _crossing(alpha, lo + 0.9 * (hi - lo)) - _crossing(alpha, lo + 0.1 * (hi - lo))


def profile_band_phase(profile, band: str) -> float:
    """Phase at which a band's alpha reaches its halfway point (the band's `cp`)."""
    alpha = profile_band_alpha(profile, band)
    lo, hi = float(alpha[0]), float(alpha[-1])
    if hi - lo < 1e-6:
        return float("nan")
    return _crossing(alpha, lo + 0.5 * (hi - lo))


def reference_purity(S: np.ndarray, sl: slice) -> float:
    """
    How spectrally self-consistent a reference window is, in [0, 1].

    Split the window in half and compare the halves: a window that straddles a
    transition or a structural break has two different spectra in it and makes a
    contaminated reference. Cheap, and it catches the failure that otherwise
    silently biases every measurement at this boundary.
    """
    seg = S[:, sl]
    if seg.shape[1] < 4:
        return 0.0
    half = seg.shape[1] // 2
    u = seg[:, :half].mean(axis=1)
    v = seg[:, half:].mean(axis=1)
    return reference_correlation(u, v)


# ── The probe ────────────────────────────────────────────────────────────────

def probe_transition(
    y: np.ndarray,
    sr: int,
    announced_t: float,
    ref_a: tuple,
    ref_b: tuple,
    bpm: float = 128.0,
    search_back: float = 90.0,
    search_fwd: float = 45.0,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> ProbeResult:
    """
    Measure the transition near `announced_t`.

    `y` is mono mix audio, `ref_a`/`ref_b` are `(start, end)` second windows of
    clean outgoing and clean incoming audio, and `bpm` converts the result to
    beats. `announced_t` is treated as a *noisy anchor*, never as the transition
    start: tracklists usually mark "where you first hear B", which is already
    inside the overlap, so the search window is deliberately wider backwards.
    """
    dur = len(y) / sr
    lo_t = max(0.0, announced_t - search_back)
    hi_t = min(dur, announced_t + search_fwd)
    if hi_t - lo_t < 5.0:
        return _reject(announced_t, "search_window_too_short")

    # Cover the references and the search window in one STFT — we never take a
    # whole-file transform, which at hop 512 would be hundreds of MB per mix.
    a0, a1 = ref_a
    b0, b1 = ref_b
    if a1 - a0 < 5.0 or b1 - b0 < 5.0:
        return _reject(announced_t, "reference_window_too_short")

    seg_lo = max(0.0, min(a0, lo_t))
    seg_hi = min(dur, max(b1, hi_t))
    chunk = y[int(seg_lo * sr):int(seg_hi * sr)]

    # Two transforms, because bass and hats want opposite trade-offs. Only the
    # boundary's neighbourhood is transformed — a whole-file STFT at hop 512
    # would run to hundreds of MB per mix.
    views = {}
    for key, nf in (("wide", n_fft), ("narrow", N_FFT_LOW)):
        if nf == n_fft and key == "narrow":
            views[key] = views["wide"]
            continue
        S_k, t_k, f_k = power_spectrogram(chunk, sr, nf, hop)
        views[key] = (S_k, t_k + seg_lo, f_k)
    band_view = {"low": "narrow", "mid": "wide", "high": "wide"}

    if views["wide"][0].shape[1] < 16:
        return _reject(announced_t, "too_few_frames")

    def windows(times):
        def win(t0, t1):
            i0 = int(np.searchsorted(times, t0))
            i1 = int(np.searchsorted(times, t1))
            return slice(i0, max(i1, i0 + 1))
        return win(a0, a1), win(b0, b1), win(lo_t, hi_t)

    _, wide_times, _ = views["wide"]
    sl_a, sl_b, sl_search = windows(wide_times)
    if sl_search.stop - sl_search.start < 16:
        return _reject(announced_t, "too_few_frames")

    beat_sec = 60.0 / bpm if bpm and bpm > 0 else 60.0 / 128.0
    purity = min(reference_purity(views["wide"][0], sl_a),
                 reference_purity(views["wide"][0], sl_b))

    def fit_bands(pool_frames: int) -> tuple:
        """Per-band ramp fits at a given coefficient-pooling resolution."""
        fits: dict = {}
        for band in BANDS:
            S, times, freqs = views[band_view[band]]
            b_sl_a, b_sl_b, b_sl_search = windows(times)
            bs = band_slice(freqs, band)
            if bs.stop - bs.start < 2:
                fits[band] = BandFit(band, 1.0, float("nan"), float("nan"),
                                     float("nan"), 0.0, 0.0, 1.0, False,
                                     "empty_band")
                continue
            Sb = S[bs]
            ra = Sb[:, b_sl_a].mean(axis=1)
            rb = Sb[:, b_sl_b].mean(axis=1)
            rho = reference_correlation(ra, rb)

            Sx = Sb[:, b_sl_search]
            a, b, resid = unmix_two_source(Sx, ra, rb)
            alpha = pooled_alpha(a, b, pool_frames)
            # Frames with little energy in this band say nothing about the fader
            # position; weight the fit (and the residual) by band energy so
            # sparse percussive bands stay usable.
            energy = Sx.sum(axis=0)
            tc, w, span, travel, r2 = fit_ramp(alpha, times[b_sl_search],
                                               weights=energy)
            wsum = float(energy.sum())
            residual = float(np.dot(resid, energy) / wsum) if wsum > 0 \
                else float(np.mean(resid))

            reason = ""
            if rho > MAX_RHO:
                reason = "low_separability"
            elif not np.isfinite(tc) or not np.isfinite(span):
                reason = "fit_failed"
            elif not (lo_t <= tc <= hi_t):
                # The workhorse guard: a blown-up fit lands far outside the window.
                reason = "center_out_of_window"
            elif travel < MIN_TRAVEL:
                reason = "low_travel"
            elif r2 < MIN_R2:
                reason = "poor_fit"
            elif residual > MAX_RESIDUAL:
                reason = "high_residual"
            fits[band] = BandFit(band, rho, tc, w, span, travel, r2, residual,
                                 reason == "", reason)
        return fits, [f for f in fits.values() if f.ok]

    # ── Per-band fits, at a resolution matched to what's being measured ──────
    # Pooling the power coefficients over a beat is what keeps a sparse
    # percussive band measurable, and costs ~3% of a typical crossfade's length.
    # But it also smears a sub-beat event: a 0.30 s cut measures 1.23 beats
    # instead of 0.62. So if the first pass comes back short, re-fit finely — a
    # cut is a broadband, energy-dense change that needs no pooling to see.
    coarse_pool = max(1, int(beat_sec * sr / hop))
    fits, surviving = fit_bands(coarse_pool)
    if surviving:
        coarse = np.median([f.span_1090 for f in surviving
                            if f.band in ("mid", "high")] or
                           [f.span_1090 for f in surviving])
        if np.isfinite(coarse) and coarse < 4.0 * beat_sec:
            fine_pool = max(1, int(0.125 * beat_sec * sr / hop))
            if fine_pool < coarse_pool:
                fine_fits, fine_surviving = fit_bands(fine_pool)
                if fine_surviving:
                    fits, surviving = fine_fits, fine_surviving
    if not surviving:
        # Report the most informative reason rather than a generic failure.
        reasons = [f.reason for f in fits.values() if f.reason]
        order = ["low_separability", "low_travel", "center_out_of_window",
                 "poor_fit", "high_residual", "fit_failed", "empty_band"]
        pick = next((r for r in order if r in reasons), "no_band_fit")
        return _reject(announced_t, pick,
                       bands=fits, sub_scores={"refs": purity})

    # ── Cross-band agreement ─────────────────────────────────────────────────
    # The strongest guard, and the only threshold-free one: an outlier band is
    # identifiable purely by disagreeing with the others.
    centres = np.array([f.center_t for f in surviving])
    med = float(np.median(centres))
    mad = float(np.median(np.abs(centres - med)))

    # The tolerance has to scale with the transition, not with the beat. Two
    # reasons, both learned the hard way:
    #
    #  - For a symmetric style the mid and high lanes are *identical*, so the MAD
    #    of three centres collapses to exactly 0 and any floor near a beat
    #    rejects the low band every time — losing the bass-swap centre, which is
    #    the single most valuable thing here.
    #  - Bands are *supposed* to disagree. `cp` versus the mid lead is a real
    #    offset of a good fraction of the crossfade; that's the quantity being
    #    measured, not an error.
    #
    # A generous tolerance still catches what this guard exists for: the observed
    # failure was a band whose centre landed ~700 s outside a ~30 s window.
    provisional = [f for f in surviving if f.band in ("mid", "high")] or surviving
    prov_span = float(np.median([f.span_1090 for f in provisional]))
    if not np.isfinite(prov_span):
        prov_span = 0.0
    tol = max(AGREE_MAD * mad, 0.4 * prov_span, 2.0 * beat_sec)
    agreed = [f for f in surviving if abs(f.center_t - med) <= tol]
    for f in surviving:
        if f not in agreed:
            fits[f.band] = BandFit(f.band, f.rho, f.center_t, f.width_sec,
                                   f.span_1090, f.travel, f.r2, f.residual,
                                   False, "band_outlier")

    duration_bands = [f for f in agreed if f.band in ("mid", "high")]
    if not duration_bands:
        # Low alone gives a centre but no duration — the bass swap is a switch.
        return _reject(announced_t, "no_duration_band", bands=fits,
                       sub_scores={"refs": purity})
    if len(agreed) < 2:
        return _reject(announced_t, "insufficient_band_agreement", bands=fits,
                       sub_scores={"refs": purity})

    # ── Combine ──────────────────────────────────────────────────────────────
    # Duration comes from mid/high; the low band contributes only its centre.
    span = float(np.median([f.span_1090 for f in duration_bands]))
    t_center = float(np.median([f.center_t for f in duration_bands]))
    duration_beats = span / beat_sec

    is_cut = duration_beats < CUT_MAX_BEATS
    if is_cut:
        span, duration_beats = 0.0, 0.0
    elif duration_beats > MAX_DURATION_BEATS:
        return _reject(announced_t, "implausible_duration", bands=fits,
                       sub_scores={"refs": purity})

    t_start, t_end = t_center - span / 2.0, t_center + span / 2.0
    band_phase = {}
    if span > 0:
        for f in agreed:
            band_phase[f.band] = round((f.center_t - t_start) / span, 4)
    low = fits.get("low")
    t_bass = low.center_t if (low is not None and low.ok) else None

    agree_score = float(np.clip(1.0 - (mad / (2.0 * beat_sec)), 0.0, 1.0)) \
        if len(surviving) > 1 else 0.5
    sub = {
        "sep": round(float(np.clip(1.0 - max(f.rho for f in agreed), 0.0, 1.0)), 4),
        "fit": round(float(np.mean([f.r2 for f in agreed])), 4),
        "travel": round(float(np.clip(np.mean([f.travel for f in agreed]), 0.0, 1.0)), 4),
        "resid": round(float(np.clip(1.0 - np.mean([f.residual for f in agreed]), 0.0, 1.0)), 4),
        "agree": round(agree_score, 4),
        "refs": round(float(purity), 4),
    }
    vals = np.array([max(v, 1e-6) for v in sub.values()])
    confidence = float(np.exp(np.mean(np.log(vals))))

    return ProbeResult(
        status="ok", announced_t=announced_t,
        t_start=round(t_start, 3), t_end=round(t_end, 3),
        t_center=round(t_center, 3),
        t_bass=(round(t_bass, 3) if t_bass is not None else None),
        duration_sec=round(span, 3),
        duration_beats=round(duration_beats, 3),
        duration_bars=round(duration_beats / 4.0, 3),
        is_cut=is_cut, bands=fits, band_phase=band_phase,
        confidence=round(confidence, 4), sub_scores=sub,
    )
