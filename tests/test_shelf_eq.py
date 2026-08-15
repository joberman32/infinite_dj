"""
The crossfade EQ used to be a band-split (`_split3`) whose three bands were
re-summed with independent gains. That reconstructs perfectly at unity, but the
bands are causal IIR outputs with different phase, so they only cancel when
their gains are equal — and a DJ EQ exists precisely to make them unequal.

Measured consequence (see CHANGELOG 2026-08-15): at crossfade phase 0.7 the
bass-swap lane reads 0.00 — "outgoing bass fully cut" — yet 0.33 amplitude of
the outgoing kick survived, riding out on the mid band's envelope. Killing the
low band boosted 80-200 Hz by up to +5 dB instead of removing it, so the
single-source bass swap the design rests on ("only one kick ever plays") did
not happen.

These tests pin the shelving replacement: a minimum-phase cascade whose
magnitude response *is* the automation curve, the exactness properties that
justified the band-split in the first place (identity, silence, no colouring),
and the streaming discipline the real-time engine needs.
"""
import numpy as np
import pytest

import infinite_dj.mixer as M
from infinite_dj.mixer import (
    CrossfadeFilterState,
    ShelfCrossfadeState,
    ShelfEQState,
    TransitionStyle,
    _blend,
    _make_profile,
    _sample_lane,
    _shelf_sos,
    make_crossfade_state,
)

SR = M.MIX_SR
SECONDS = 2


def _tone(freq: float, seconds: int = SECONDS) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    s = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([s, s], axis=1)


def _lanes(value: float, n: int) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def _amplitude_at(audio: np.ndarray, freq: float) -> float:
    """Steady-state amplitude at `freq`, skipping the filter's startup second."""
    tail = audio[SR:, 0]
    spectrum = np.fft.rfft(tail)
    freqs = np.fft.rfftfreq(len(tail), 1 / SR)
    return float(np.abs(spectrum[np.argmin(np.abs(freqs - freq))]))


# ── Exactness properties the band-split had, which the shelf must not lose ────

def test_unity_lanes_pass_the_signal_through_untouched():
    """All three knobs open must be a true bypass, not near-unity filtering."""
    signal = (_tone(60) + _tone(900) + _tone(7000)) / 3.0
    state = ShelfEQState.create(SR)
    n = len(signal)

    out = state.process(signal, _lanes(1.0, n), _lanes(1.0, n), _lanes(1.0, n))

    assert np.max(np.abs(out - signal)) < 1e-6


def test_all_lanes_closed_gives_true_silence():
    """A shelf floors at EQ_FLOOR_DB, so silence has to come from elsewhere.

    `_split_overall` factors the common level out as a plain scalar, which is
    what makes this exact — the incoming track's lanes are all 0.0 at phase 0,
    and it must contribute nothing at all rather than -30 dB of bleed.
    """
    signal = (_tone(60) + _tone(900)) / 2.0
    state = ShelfEQState.create(SR)
    n = len(signal)

    out = state.process(signal, _lanes(0.0, n), _lanes(0.0, n), _lanes(0.0, n))

    assert np.max(np.abs(out)) == 0.0


def test_equal_lanes_scale_without_colouring_the_spectrum():
    """Three knobs pulled down together is a level change, not an EQ move."""
    signal = (_tone(60) + _tone(900) + _tone(7000)) / 3.0
    state = ShelfEQState.create(SR)
    n = len(signal)

    out = state.process(signal, _lanes(0.5, n), _lanes(0.5, n), _lanes(0.5, n))

    assert np.max(np.abs(out - 0.5 * signal)) < 1e-6


# ── The actual fix ───────────────────────────────────────────────────────────

def test_killing_the_low_lane_removes_the_bass():
    signal = (_tone(60) + _tone(900) + _tone(7000)) / 3.0
    reference = _amplitude_at(signal, 60)
    state = ShelfEQState.create(SR)
    n = len(signal)

    out = state.process(signal, _lanes(0.0, n), _lanes(1.0, n), _lanes(1.0, n))

    assert _amplitude_at(out, 60) / reference < 0.05


def test_killing_the_low_lane_leaves_the_upper_bands_alone():
    """The kill has to be surgical enough not to gut the mids with it."""
    signal = (_tone(60) + _tone(900) + _tone(7000)) / 3.0
    ref_mid, ref_high = _amplitude_at(signal, 900), _amplitude_at(signal, 7000)
    state = ShelfEQState.create(SR)
    n = len(signal)

    out = state.process(signal, _lanes(0.0, n), _lanes(1.0, n), _lanes(1.0, n))

    assert _amplitude_at(out, 900) / ref_mid > 0.90
    assert _amplitude_at(out, 7000) / ref_high > 0.95


def test_band_split_leaks_bass_where_the_shelf_does_not():
    """The regression this replaced — pinned so it can't quietly come back.

    Both topologies are asked for the same thing (low fully cut, mid and high
    open). The band-split leaves most of the kick behind; the shelf doesn't.
    """
    signal = _tone(60)
    reference = _amplitude_at(signal, 60)
    low, mid, high = M._split3(signal, SR)

    split_survivor = _amplitude_at(0.0 * low + mid + high, 60) / reference
    shelf_state = ShelfEQState.create(SR)
    n = len(signal)
    shelf_survivor = _amplitude_at(
        shelf_state.process(signal, _lanes(0.0, n), _lanes(1.0, n), _lanes(1.0, n)),
        60,
    ) / reference

    assert split_survivor > 0.5           # the bug: "cut" bass mostly survives
    assert shelf_survivor < 0.05          # the fix


def test_bass_swap_tracks_the_automation_curve():
    """The headline fix, stated as the property that actually matters.

    Whatever the low lane says at a given phase is what the 60 Hz content
    should measure. The band-split deviated by up to 0.33 of full amplitude;
    the shelf must follow the curve closely across the whole crossfade.
    """
    outgoing = _tone(60)
    silent = np.zeros_like(outgoing)          # isolate the outgoing path
    profile = _make_profile(0.5, 0.2, 0.1)
    style = TransitionStyle("test", 8, profile=profile)
    reference = _amplitude_at(outgoing, 60)

    worst = 0.0
    for phase in np.arange(0.0, 1.01, 0.1):
        intended = float(_sample_lane(
            profile.out_low, np.array([phase], dtype=np.float32))[0])
        mixed = _blend(outgoing, silent, float(phase), SR, style,
                       filter_state=ShelfCrossfadeState.create(SR))
        worst = max(worst, abs(_amplitude_at(mixed, 60) / reference - intended))

    assert worst < 0.05


def test_cascaded_shelves_are_steeper_than_one_shelf_of_the_same_depth():
    """Why SHELF_STAGES exists: same kill, less collateral on the low-mids."""
    from scipy.signal import sosfreqz

    def response(sos, freq):
        _, h = sosfreqz(sos, worN=[2 * np.pi * freq / SR])
        return float(np.abs(h[0]))

    single = M._biquad_low_shelf(SR, M.SHELF_LOW_HZ, -24.0).reshape(1, 6)
    halved = M._biquad_low_shelf(SR, M.SHELF_LOW_HZ, -12.0)
    doubled = np.vstack([halved, halved])

    # Matched where it counts (the kick), better where it shouldn't reach.
    assert response(doubled, 60) == pytest.approx(response(single, 60), abs=0.01)
    assert response(doubled, 400) > response(single, 400) + 0.05


def test_a_killed_band_stops_at_the_configured_floor():
    """Attenuation is bounded, so a 'kill' can't become a numerical cliff."""
    sos = _shelf_sos(SR, 0.0, 1.0, 1.0)
    from scipy.signal import sosfreqz

    _, h = sosfreqz(sos, worN=[2 * np.pi * 20 / SR])
    floor_linear = 10.0 ** (M.EQ_FLOOR_DB / 20.0)

    assert float(np.abs(h[0])) == pytest.approx(floor_linear, rel=0.25)


# ── Streaming discipline ─────────────────────────────────────────────────────

def test_make_crossfade_state_follows_the_configured_topology(monkeypatch):
    monkeypatch.setattr(M, "EQ_TOPOLOGY", "shelf")
    assert isinstance(make_crossfade_state(SR), ShelfCrossfadeState)

    monkeypatch.setattr(M, "EQ_TOPOLOGY", "split")
    assert isinstance(make_crossfade_state(SR), CrossfadeFilterState)


def test_an_explicit_state_overrides_the_configured_topology(monkeypatch):
    """A transition already under way keeps its topology mid-flight.

    The engine builds one state per transition and passes it to every chunk;
    if flipping EQ_TOPOLOGY could change topology partway, a live crossfade
    would switch EQ under itself.
    """
    monkeypatch.setattr(M, "EQ_TOPOLOGY", "shelf")
    signal, silent = _tone(60), np.zeros_like(_tone(60))
    style = TransitionStyle("test", 8, profile=_make_profile(0.5, 0.2, 0.1))
    reference = _amplitude_at(signal, 60)

    # Hand it a split state at a phase where the split path is known to leak.
    mixed = _blend(signal, silent, 0.75, SR, style,
                   filter_state=CrossfadeFilterState.create(SR))

    assert _amplitude_at(mixed, 60) / reference > 0.1


def test_the_mid_lane_fades_the_whole_band_not_just_its_centre():
    """The mid lane is the primary midrange crossfade, so it must read as a
    level change across the band rather than a notch at the bell's centre.

    `SHELF_MID_Q` trades this evenness against how far the bell reaches into
    its neighbours; the numbers behind the choice are recorded beside the
    constant. This pins the property that motivated it.
    """
    from scipy.signal import sosfreqz

    sos = _shelf_sos(SR, 1.0, 0.5, 1.0)
    freqs = np.array([300, 500, 720, 1200, 1800, 2400], dtype=float)
    _, h = sosfreqz(sos, worN=2 * np.pi * freqs / SR)
    response = np.abs(h)

    assert response.max() - response.min() < 0.35     # reasonably even
    assert response.max() < 0.85                      # every probe actually moved
