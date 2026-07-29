"""
The measurement DSP, checked against transitions whose parameters are known
exactly.

Everything here is synthesized in numpy and blended with the mixer's *own*
`_blend` + `_make_profile`, so the crossfade under test is the real code path
with ground truth we chose. No audio files, no library, runs in seconds.

Two things this harness is deliberately careful about:

  - **Ground truth is the 10-90% span of the profile's alpha, not `n_bars`.**
    Those differ by ~26%, and testing against the wrong one would bake a
    systematic error into every calibrated value.
  - **Both synthetic tracks must differ in all three bands.** A generator that
    leaves a band identical makes that band's reference correlation 1.0, the
    decomposition degenerate, and the duration assertion vacuous — so
    `test_synthetic_tracks_are_separable` guards the fixture itself.
"""
import numpy as np
import pytest

from infinite_dj.mixer import (
    TransitionStyle,
    _blend,
    _make_profile,
    choose_transition_style,
)
from infinite_dj.transition_probe import (
    BANDS,
    band_slice,
    fit_ramp,
    mixing_alpha,
    power_spectrogram,
    probe_transition,
    profile_band_phase,
    profile_span_1090,
    reference_correlation,
    unmix_two_source,
)

SR = 22050
BPM = 124.0
BAR = (60.0 / BPM) * 4


# ── Synthetic material ───────────────────────────────────────────────────────

# Guard bands. The probe slices at LOW_CUT=200 / MID_CUT=2600, and a Butterworth
# is not a brick wall — a component placed right at a boundary leaks into the
# neighbouring slice, where it is driven by a *different* automation lane. That
# leak is what defeats per-band timing recovery, so the fixture keeps each
# component well inside its band.
LOW_TOP = 130.0
MID_LO, MID_HI = 400.0, 2000.0
HIGH_LO = 3500.0


def _band_limit(y, sr, lo=None, hi=None, order=8):
    """Confine a component to its intended band with a steep Butterworth."""
    from scipy.signal import butter, sosfilt

    nyq = sr / 2
    if lo and hi:
        sos = butter(order, [lo / nyq, min(hi, nyq * 0.99) / nyq],
                     btype="bandpass", output="sos")
    elif lo:
        sos = butter(order, lo / nyq, btype="highpass", output="sos")
    else:
        sos = butter(order, hi / nyq, btype="lowpass", output="sos")
    return sosfilt(sos, y)


def synth_track(duration, kick_f, bass_f, chord_f, hat_f, seed, sr=SR, bpm=BPM):
    """
    A "track" with distinct content in the low, mid and high bands.

    Each component is band-limited to the band it belongs in. That matters: a
    naive synthetic kick is a broadband click whose harmonics dominate the mid
    band, so the "mid" spectral slice ends up driven by the *low* automation
    lane and the per-band timing test measures nothing. (Real tracks do leak
    across bands, which is a genuine limit on per-band phase recovery — but
    that's a property to characterise separately, not to bake into the fixture
    that's meant to isolate lane timing.)

    Different frequencies per track keep the two references separable in every
    band; `test_synthetic_tracks_are_separable` enforces it.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    beat = 60.0 / bpm

    # Low: kick transients plus a steady sub.
    low = np.zeros(n, dtype=np.float64)
    for k in range(int(duration / beat)):
        s = int(k * beat * sr)
        m = min(int(0.25 * sr), n - s)
        if m <= 0:
            break
        env = np.exp(-np.arange(m) / (0.05 * sr))
        low[s:s + m] += 0.9 * env * np.sin(2 * np.pi * kick_f * t[:m])
    low += 0.35 * np.sin(2 * np.pi * bass_f * t)

    # Mid: a three-note chord stack — the band that carries crossfade duration.
    mid = np.zeros(n, dtype=np.float64)
    for mult, amp in ((1.0, 0.30), (1.26, 0.22), (1.5, 0.18)):
        mid += amp * np.sin(2 * np.pi * chord_f * mult * t)

    # High: hats on the offbeat. Sparse in time, which is exactly why the ramp
    # fit has to be energy-weighted.
    high = np.zeros(n, dtype=np.float64)
    for k in range(int(duration / (beat / 2))):
        s = int((k * beat / 2 + beat / 2) * sr)
        m = min(int(0.06 * sr), n - s)
        if m <= 0:
            break
        env = np.exp(-np.arange(m) / (0.012 * sr))
        carrier = np.sin(2 * np.pi * hat_f * t[:m])
        high[s:s + m] += 1.4 * env * carrier * (1 + 0.3 * rng.standard_normal(m))

    y = (_band_limit(low, sr, hi=LOW_TOP)
         + _band_limit(mid, sr, lo=MID_LO, hi=MID_HI)
         + _band_limit(high, sr, lo=HIGH_LO))
    y += 0.001 * rng.standard_normal(n)
    y /= np.abs(y).max() + 1e-9
    return np.stack([y, y], axis=1).astype(np.float32)


TRACK_A = dict(kick_f=48.0, bass_f=72.0, chord_f=500.0, hat_f=4800.0, seed=1)
TRACK_B = dict(kick_f=66.0, bass_f=104.0, chord_f=1050.0, hat_f=8800.0, seed=2)


def build_mix(style, pre=50.0, post=50.0, sr=SR, bpm=BPM,
              a_spec=None, b_spec=None, xfade_sec=None):
    """
    A -> crossfade -> B, using the mixer's real `_blend`.

    Returns `(mono_mix, truth)` where truth carries the exact crossfade bounds
    and the profile-derived span for each band.
    """
    a_spec = dict(a_spec or TRACK_A)
    b_spec = dict(b_spec or TRACK_B)
    if xfade_sec is None:
        xfade_sec = (style.cut_seconds if style.is_cut else style.n_bars * BAR)

    total = pre + xfade_sec + post
    a = synth_track(duration=total, sr=sr, bpm=bpm, **a_spec)
    b = synth_track(duration=total, sr=sr, bpm=bpm, **b_spec)

    n_pre, n_x = int(pre * sr), int(xfade_sec * sr)
    phase = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
    blended = _blend(a[n_pre:n_pre + n_x], b[:n_x], phase, sr=sr, style=style)

    mix = np.concatenate([a[:n_pre], blended, b[n_x:n_x + int(post * sr)]])
    mono = mix.mean(axis=1).astype(np.float32)

    prof = style.profile
    span = {bd: (profile_span_1090(prof, bd) * xfade_sec if prof else np.nan)
            for bd in BANDS}
    phase = {bd: (profile_band_phase(prof, bd) if prof else np.nan)
             for bd in BANDS}
    truth = {
        # Crossfade bounds as the renderer laid them down.
        "t_start": pre,
        "t_end": pre + xfade_sec,
        "xfade_sec": xfade_sec,
        "ref_a": (max(0.0, pre - 40.0), pre - 5.0),
        "ref_b": (pre + xfade_sec + 5.0, pre + xfade_sec + 40.0),
        "span": span,
        "phase": phase,
        # What the probe actually reports: the mid band's 10%..90% travel, which
        # is narrower than the crossfade and offset by the mid lane's own
        # timing. Comparing r.t_start against `t_start` above would be comparing
        # two different quantities.
        "probe_t_start": pre + phase["mid"] * xfade_sec - span["mid"] / 2,
        "probe_t_end": pre + phase["mid"] * xfade_sec + span["mid"] / 2,
    }
    return mono, truth


def blend_style(name, n_bars, cp, mid_lead, high_lead,
                out_mid_hold=0.0, out_high_hold=0.0):
    return TransitionStyle(name, n_bars,
                           profile=_make_profile(cp, mid_lead, high_lead,
                                                 out_mid_hold, out_high_hold))


BLEND_16 = blend_style("blend", 16, 0.55, 0.12, 0.12)
SWAP_8 = blend_style("swap", 8, 0.50, 0.50, 0.10, 0.50, 0.30)


def run_probe(mono, truth, announced=None, **kw):
    announced = truth["t_end"] if announced is None else announced
    return probe_transition(mono, SR, announced,
                            ref_a=truth["ref_a"], ref_b=truth["ref_b"],
                            bpm=BPM, **kw)


# ── Fixture self-checks ──────────────────────────────────────────────────────

def test_synthetic_tracks_are_separable():
    """
    Guard the fixture, not the code: if the two tracks share a band, that band's
    reference correlation is ~1, its decomposition is degenerate, and every
    duration assertion below becomes vacuous.
    """
    a = synth_track(20.0, **TRACK_A)
    b = synth_track(20.0, **TRACK_B)
    Sa, _, freqs = power_spectrogram(a.mean(axis=1), SR)
    Sb, _, _ = power_spectrogram(b.mean(axis=1), SR)

    for band in BANDS:
        sl = band_slice(freqs, band)
        rho = reference_correlation(Sa[sl].mean(axis=1), Sb[sl].mean(axis=1))
        assert rho < 0.5, f"{band} band not separable (rho={rho:.4f})"


def test_profile_span_is_not_the_bar_count():
    """
    The statistic we mine differs from a style's nominal length. Pinning the
    ratio here stops anyone 'simplifying' the validator to compare against
    n_bars.
    """
    span = profile_span_1090(BLEND_16.profile, "mid")
    assert 0.6 < span < 0.9, span


# ── Core recovery ────────────────────────────────────────────────────────────

def test_recovers_a_16_bar_blend():
    mono, truth = build_mix(BLEND_16)
    r = run_probe(mono, truth)

    assert r.status == "ok", r.reject_reason
    assert not r.is_cut
    expected_bars = truth["span"]["mid"] / BAR
    assert abs(r.duration_bars - expected_bars) < 2.0, (r.duration_bars, expected_bars)
    assert abs(r.t_center - (truth["t_start"] + truth["t_end"]) / 2) < 2.0 * BAR


def test_recovers_a_short_swap_relative_to_a_blend():
    """
    A swap is materially shorter than a blend, and that ordering — which is what
    calibration keys off — must survive measurement.

    Deliberately *not* asserted against the swap's absolute mid span: see
    `test_asymmetric_lanes_overestimate_the_mid_span` for why.
    """
    mono, truth = build_mix(SWAP_8)
    r = run_probe(mono, truth)
    assert r.status == "ok", r.reject_reason

    blend_mono, blend_truth = build_mix(BLEND_16)
    rb = run_probe(blend_mono, blend_truth)
    assert rb.status == "ok", rb.reject_reason

    assert r.duration_bars < rb.duration_bars
    # The 8-bar style must not be measured as long as the 16-bar one.
    assert r.duration_bars < 0.75 * rb.duration_bars


def test_asymmetric_lanes_overestimate_the_mid_span():
    """
    A known, characterised bias — pinned so it can't drift unnoticed.

    `swap` holds the incoming mids back (`in_mid_lead=0.50`) while bringing its
    highs in early (`in_high_lead=0.10`). Those early highs leak into the mid
    analysis slice through STFT spread of impulsive hats, so the mid band reads
    ~2x its true span. Rebuilding the identical mix with the highs held back
    too removes the effect entirely, which is what identifies the cause.

    This is why the validator must render-and-probe the engine's side rather
    than reading its automation lanes analytically: the bias only cancels if
    both sides pass through the same spectral analysis.
    """
    mono, truth = build_mix(SWAP_8)
    r = run_probe(mono, truth)
    assert r.status == "ok", r.reject_reason
    over = r.bands["mid"].span_1090 / truth["span"]["mid"]
    assert 1.6 < over < 2.8, over

    # Same profile, but with the incoming highs held back to match the mids.
    held = blend_style("swap_held", 8, 0.50, 0.50, 0.50, 0.50, 0.30)
    mono2, truth2 = build_mix(held)
    r2 = run_probe(mono2, truth2)
    assert r2.status == "ok", r2.reject_reason
    over2 = r2.bands["mid"].span_1090 / truth2["span"]["mid"]
    assert over2 < 1.3, (over, over2)


def test_duration_is_linear_in_crossfade_length():
    """
    The property calibration actually rests on: measured span is a near-perfect
    linear function of the crossfade's true length, so the estimator's bias is a
    single stable factor rather than something shape-dependent.

    Measured 0.787 +/- 0.008 of nominal across a 8x range of lengths.
    """
    ratios = []
    for n_bars in (4, 8, 12, 16, 24, 32):
        style = blend_style("blend", n_bars, 0.55, 0.12, 0.12)
        mono, truth = build_mix(style)
        r = run_probe(mono, truth)
        assert r.status == "ok", (n_bars, r.reject_reason)
        ratios.append(r.duration_bars / n_bars)

    assert min(ratios) > 0.70 and max(ratios) < 0.85, ratios
    # Stability of the factor is the point — that's what makes it invertible.
    assert (max(ratios) - min(ratios)) < 0.04, ratios


def test_recovers_high_band_span_for_a_swap():
    """
    The band whose lane actually spans the crossfade is measured accurately even
    for an asymmetric profile — it's the *narrow* lanes that suffer.
    """
    mono, truth = build_mix(SWAP_8)
    r = run_probe(mono, truth)
    assert r.status == "ok", r.reject_reason
    assert r.bands["high"].ok
    ratio = r.bands["high"].span_1090 / truth["span"]["high"]
    assert 0.85 < ratio < 1.25, ratio


def test_detects_a_cut():
    """A near-instant fade is a valid measurement, not a failure."""
    style = TransitionStyle("cut", n_bars=0, is_cut=True, cut_seconds=0.30,
                            profile=_make_profile(0.5, 0.0, 0.0, bass_w=0.02))
    mono, truth = build_mix(style, xfade_sec=0.30)
    r = run_probe(mono, truth)

    assert r.status == "ok", r.reject_reason
    assert r.is_cut
    assert r.duration_beats == 0.0


def test_band_centres_land_within_a_bar():
    """
    Absolute band centres are accurate — it's their *differences* that aren't
    (see the next test). All three should land within a bar of truth.
    """
    mono, truth = build_mix(SWAP_8)
    r = run_probe(mono, truth)
    assert r.status == "ok", r.reject_reason

    for band, fit in r.bands.items():
        if not fit.ok:
            continue
        expected = truth["t_start"] + truth["phase"][band] * truth["xfade_sec"]
        assert abs(fit.center_t - expected) < 1.5 * BAR, (band, fit.center_t,
                                                          expected)


def test_band_phase_does_not_track_cp():
    """
    A pinned negative result: `band_phase` is diagnostic output, not a
    calibration target.

    Sweeping the bass-swap centre across most of its usable range moves the true
    low-band centre by ~11 s but the measured one by under 3 s. The relationship
    is monotone, so it looks encouraging in isolation — but the compression
    leaves no dynamic range to survive real-world noise, and the inter-band
    separations that would calibrate `_make_profile` are the same size as the
    measurement error.

    If someone later improves the estimator enough to break this test, that is
    good news and the calibration scope should widen accordingly.
    """
    measured, true_centres = [], []
    for cp in (0.35, 0.50, 0.65, 0.72):
        style = blend_style("blend", 16, cp, 0.12, 0.12)
        mono, truth = build_mix(style)
        r = run_probe(mono, truth)
        assert r.status == "ok", (cp, r.reject_reason)
        assert r.bands["low"].ok, cp
        measured.append(r.bands["low"].center_t)
        true_centres.append(truth["t_start"] + cp * truth["xfade_sec"])

    true_range = max(true_centres) - min(true_centres)
    meas_range = max(measured) - min(measured)
    assert true_range > 10.0, true_range
    # Compressed by more than 3x — the finding this test exists to pin.
    assert meas_range < true_range / 3.0, (meas_range, true_range)


@pytest.mark.parametrize("offset", [-30.0, -10.0, 0.0, 10.0, 25.0])
def test_robust_to_announced_timestamp_jitter(offset):
    """
    Tracklist timestamps are noisy anchors — usually marking "where you first
    hear B", which is already inside the overlap. The measurement must not move
    with them.
    """
    mono, truth = build_mix(BLEND_16)
    announced = round(truth["t_end"] + offset)     # also rounds to the second
    r = run_probe(mono, truth, announced=announced)

    assert r.status == "ok", (offset, r.reject_reason)
    assert abs(r.t_start - truth["probe_t_start"]) < 2.0 * BAR, (offset, r.t_start)
    assert abs(r.t_end - truth["probe_t_end"]) < 2.0 * BAR, (offset, r.t_end)


# ── Rejections ───────────────────────────────────────────────────────────────

def test_rejects_a_track_blended_with_itself():
    """
    The failure that motivated the whole guard stack: with no spectral contrast
    the fit is unconstrained, and the probe must say so rather than emit a
    confident number.
    """
    mono, truth = build_mix(BLEND_16, a_spec=TRACK_A, b_spec=TRACK_A)
    r = run_probe(mono, truth)

    assert r.status == "rejected"
    assert r.reject_reason in ("low_separability", "low_travel"), r.reject_reason


def test_rejects_an_incoherent_band():
    """
    Reproduces the case a naive condition-number threshold waves through: one
    band whose fit lands far outside the search window. Cross-band agreement
    must drop it while the measurement survives on the others.
    """
    mono, truth = build_mix(BLEND_16)
    # Overwrite the high band with wideband noise everywhere, so it carries no
    # usable A-vs-B ramp while low and mid are untouched.
    from scipy.signal import butter, sosfilt

    from infinite_dj.mixer import MID_CUT

    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(mono))
    sos = butter(4, MID_CUT / (SR / 2), btype="highpass", output="sos")
    mono = (mono + 0.35 * sosfilt(sos, noise)).astype(np.float32)

    r = run_probe(mono, truth)
    assert r.status == "ok", r.reject_reason
    assert not r.bands["high"].ok
    assert r.bands["mid"].ok


def test_rejects_when_references_are_too_short():
    mono, truth = build_mix(BLEND_16)
    r = probe_transition(mono, SR, truth["t_end"],
                         ref_a=(10.0, 12.0), ref_b=(120.0, 121.0), bpm=BPM)
    assert r.status == "rejected"
    assert r.reject_reason == "reference_window_too_short"


def test_adversarial_effects_tail_rejects_or_stays_bounded():
    """
    Real transitions include echo-outs and filter sweeps the two-source model
    can't represent. The probe may reject those, but it must not report a
    confident wrong duration.
    """
    mono, truth = build_mix(BLEND_16)
    sr = SR
    # A 4-bar decaying echo of the outgoing, pasted over the incoming.
    tail_start = int(truth["t_end"] * sr)
    tail_n = int(4 * BAR * sr)
    src = mono[tail_start - tail_n:tail_start]
    if len(src) == tail_n and tail_start + tail_n <= len(mono):
        decay = np.exp(-np.linspace(0, 3, tail_n)).astype(np.float32)
        mono = mono.copy()
        mono[tail_start:tail_start + tail_n] += 0.5 * src * decay

    r = run_probe(mono, truth)
    if r.status == "ok":
        expected = truth["span"]["mid"] / BAR
        assert abs(r.duration_bars - expected) < 6.0, (r.duration_bars, expected)


# ── Unit-level ───────────────────────────────────────────────────────────────

def test_unmix_recovers_known_coefficients():
    """A synthetic mixture of two non-negative spectra must decompose exactly."""
    rng = np.random.default_rng(3)
    ra = np.abs(rng.standard_normal(64)) + 0.1
    rb = np.abs(rng.standard_normal(64)) + 0.1
    coef_a = np.array([1.0, 0.7, 0.3, 0.0])
    coef_b = np.array([0.0, 0.3, 0.7, 1.0])
    S = np.outer(ra, coef_a) + np.outer(rb, coef_b)

    a, b, resid = unmix_two_source(S, ra, rb)
    assert np.allclose(a, coef_a, atol=1e-6)
    assert np.allclose(b, coef_b, atol=1e-6)
    assert np.all(resid < 1e-6)


def test_unmix_is_non_negative():
    """Frames containing only A must not produce a negative B coefficient."""
    rng = np.random.default_rng(4)
    ra = np.abs(rng.standard_normal(32)) + 0.1
    rb = np.abs(rng.standard_normal(32)) + 0.1
    S = np.outer(ra, np.array([1.0, 2.0, 3.0]))

    a, b, _ = unmix_two_source(S, ra, rb)
    assert np.all(b >= 0.0)
    assert np.all(a >= 0.0)


def test_mixing_alpha_is_amplitude_domain():
    """Equal powers give alpha 0.5; a 4:1 power ratio gives 2:1 in amplitude."""
    a = np.array([1.0, 4.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    alpha = mixing_alpha(a, b)
    assert abs(alpha[0] - 0.5) < 1e-6
    assert abs(alpha[1] - (1.0 / 3.0)) < 1e-6
    assert abs(alpha[2] - 1.0) < 1e-6


def test_fit_ramp_recovers_a_known_logistic():
    times = np.linspace(0.0, 60.0, 1200)
    truth_tc, truth_w = 30.0, 2.5
    alpha = 1.0 / (1.0 + np.exp(-(times - truth_tc) / truth_w))

    tc, w, span, travel, r2 = fit_ramp(alpha, times)
    assert abs(tc - truth_tc) < 0.2
    assert abs(w - truth_w) < 0.2
    assert abs(span - 2 * np.log(9) * truth_w) < 0.5
    assert travel > 0.9 and r2 > 0.99


def test_fit_ramp_reports_low_travel_for_a_flat_curve():
    times = np.linspace(0.0, 60.0, 600)
    alpha = np.full_like(times, 0.5)
    _, _, _, travel, _ = fit_ramp(alpha, times)
    assert travel < 0.3
