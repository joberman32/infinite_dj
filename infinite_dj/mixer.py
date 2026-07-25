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
from .harmony import camelot_compatibility

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

def _time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """
    Time-stretch audio by `ratio` without changing pitch.
    ratio > 1.0 = slower (incoming track has lower BPM, stretch to match)
    ratio < 1.0 = faster (incoming track has higher BPM, compress)

    Uses pyrubberband (Rubber Band Library) for best quality.
    Falls back to librosa phase vocoder for ratios within 6%.
    """
    if abs(ratio - 1.0) < 0.001:
        return audio  # No stretch needed

    if HAS_RUBBERBAND:
        left  = rb.time_stretch(audio[:, 0], sr, ratio)
        right = rb.time_stretch(audio[:, 1], sr, ratio)
        return np.stack([left, right], axis=1).astype(np.float32)
    else:
        # Fallback: librosa phase vocoder (mono only, lower quality)
        mono = audio.mean(axis=1)
        stretched = librosa.effects.time_stretch(mono, rate=1.0/ratio)
        stereo = np.stack([stretched, stretched], axis=1)
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


def choose_transition_style(out_cue, in_cue, beatmatched: bool,
                            high_sim_threshold: float = 0.82) -> TransitionStyle:
    """
    Pick a crossfade style from the energy and CLAP vector similarity of the
    outgoing exit and incoming entry. Beat-locked pairs get real blends shaped
    to their dynamics; tempo-incompatible pairs get a short cut.

    `high_sim_threshold` is the CLAP-similarity cutoff above which a smooth long
    blend is forced; callers pass a per-library calibrated value (the fixed
    0.82 default is only a fallback — on a real library it fires on ~30% of
    pairs and doesn't discriminate).
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

    if (sim is not None and sim >= high_sim_threshold) or (eo < 0.45 and ei < 0.45):
        # High textural similarity or both sparse: long, symmetric smooth blend.
        return styled("blend", 16, cp=0.55, mid_lead=0.12, high_lead=0.12)
    if eo > 0.70 and ei > 0.70:
        # Drop → drop: bring the incoming HATS in early over the outgoing groove,
        # hold its MIDS back to avoid clash, then swap the bass mid-way.
        return styled("swap", 8, cp=0.50, mid_lead=0.50, high_lead=0.10,
                      out_mid_hold=0.50, out_high_hold=0.30)
    if eo >= ei:
        # Busier → calmer: gentle medium fade, incoming eased in.
        return styled("fade", 12, cp=0.55, mid_lead=0.30, high_lead=0.20)
    # Calmer → rising: bring the incoming up sooner across all bands.
    return styled("build", 8, cp=0.40, mid_lead=0.10, high_lead=0.05)



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

    if filter_state is None:
        o_low, o_mid, o_high = _split3(out_c, sr)
        i_low, i_mid, i_high = _split3(in_c, sr)
    else:
        o_low, o_mid, o_high, i_low, i_mid, i_high = filter_state.split(out_c, in_c)

    def g(lane):
        return _sample_lane(lane, phase).reshape(-1, 1)

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

    # Time-stretch incoming track to match outgoing BPM (only when beatmatching)
    if plan.beatmatched and abs(plan.stretch_ratio - 1.0) > 0.001:
        pct = (plan.stretch_ratio - 1.0) * 100
        print(f"  Time-stretching incoming track by {pct:+.1f}% to match {plan.bpm_target:.1f} BPM...")
        audio_in = _time_stretch(audio_in, sr, plan.stretch_ratio)
        # A downbeat at native time d lands at d / ratio after stretching
        # (ratio > 1 speeds the track up, so beats move earlier).
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


def render_set(
    tracks: list,
    n_mix_bars: int = 16,
    sr: int = MIX_SR,
    max_stretch: float = MAX_STRETCH,
    min_solo_bars: int = 32,
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
    Returns (audio, sr, [SetMarker, ...]).
    """
    if len(tracks) < 2:
        raise ValueError("Need at least 2 tracks to render a set")

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

    for i in range(len(tracks) - 1):
        solo_start = written  # output sample where this track's solo begins
        nxt_t     = tracks[i + 1]
        out_bpm   = cur_t.bpm
        out_bar_sec = (60.0 / out_bpm) * 4
        out_bar_s = _time_to_samples(out_bar_sec, sr)
        cur_dur   = len(cur_audio) / sr
        read_t    = read / sr

        # ── Tempo match: bring incoming to the OUTGOING track's native tempo ───
        ratios = [out_bpm / nxt_t.bpm,
                  out_bpm / (nxt_t.bpm * 2.0),
                  out_bpm / (nxt_t.bpm / 2.0)]
        ratio = min(ratios, key=lambda r: abs(r - 1.0))
        beatmatched = abs(ratio - 1.0) <= max_stretch

        # ── Outgoing exit ──────────────────────────────────────────────────────
        if splice:
            # Short segment: exit within [min_seg, max_seg] at a CLAP-serendipitous
            # splice point that connects to the next track's texture.
            min_exit_t = read_t + (min_seg_sec or 0.0)
            max_exit_t = min(read_t + max_seg_sec, cur_dur)
            out_cue = _pick_splice_exit(cur_t, nxt_t, min_exit_t, max_exit_t)
            if out_cue is not None:
                cue_out_t = out_cue.timestamp
            else:
                inwin = [d for d in cur_t.downbeats if min_exit_t <= d <= max_exit_t]
                cue_out_t = inwin[-1] if inwin else max(read_t, max_exit_t - out_bar_sec)
        else:
            # Breathe, then leave at a strong cue that still has a groove (so the
            # outgoing carries a beat into the crossfade).
            min_exit_t = read_t + min_solo_bars * out_bar_sec
            out_cue = _pick_exit_cue(cur_t, min_exit_t, cur_dur)
            if out_cue is not None:
                cue_out_t = out_cue.timestamp
            else:
                later = [d for d in cur_t.downbeats if min_exit_t <= d < cur_dur]
                cue_out_t = later[0] if later else max(read_t, cur_dur - out_bar_sec)
        cue_out_sample = max(read, _time_to_samples(cue_out_t, sr))
        target_e = out_cue.energy if out_cue else 0.5

        # ── Incoming entry ─────────────────────────────────────────────────────
        # Beat-locked transitions enter where the incoming groove matches the
        # exit's energy (two rhythms lock); cuts just enter early.
        if beatmatched:
            in_cue = _match_entry(nxt_t, target_e)
        else:
            in_cue = _set_entry_cue(nxt_t) or CuePoint(0.0, "in", True, 0.5, 0.1)
        in_time = _find_nearest_downbeat(
            in_cue.timestamp, nxt_t.downbeats, max_offset=(60.0 / nxt_t.bpm) * 4 / 2.0
        )
        in_sample = _time_to_samples(in_time, sr)

        # ── Pick a crossfade style from the two tracks' dynamics ───────────────
        style = choose_transition_style(out_cue, in_cue, beatmatched,
                                        high_sim_threshold=sim_threshold)

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
            # Short time-based fade — no long overlap of unsynced tempos.
            out_mix = cur_audio[cue_out_sample:cue_out_sample + mix_out_s]
            in_mix  = nxt_audio[in_sample:in_sample + mix_out_s]
            ratio   = 1.0
        else:
            out_mix = cur_audio[cue_out_sample:cue_out_sample + mix_out_s]
            in_region = nxt_audio[in_sample:in_sample + n_bars * in_bar_s]
            if beatmatched and abs(ratio - 1.0) > 0.001:
                in_mix = _time_stretch(in_region, sr, ratio)   # → outgoing tempo
            else:
                ratio = 1.0
                in_mix = in_region

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

        # ── Advance: incoming becomes current, continuing at its native tempo ──
        consumed_in = int(round(m * ratio))
        cur_t     = nxt_t
        cur_audio = nxt_audio
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

    def emit(t, section, seg_bars, fade_bars, lock):
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
        env = np.ones(n, dtype=np.float32)
        env[:fi]  = np.sin(np.linspace(0.0, np.pi / 2.0, fi))
        env[-fi:] = np.cos(np.linspace(0.0, np.pi / 2.0, fi))
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
        lock = len(active) > 0        # something is sounding -> must beat-lock
        got = emit(t, section, seg_bars, fade_bars, lock)
        seg, eff, was_stretched, fade_sec = got if got else (None, 0.0, False, 0.0)
        end = min(pos + len(seg), len(master)) if seg is not None else pos
        if seg is not None and end > pos:
            master[pos:end] += seg[:end - pos]
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
                "bpm": round(eff, 1), "key": t.key,
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

        # Ceiling that only ever drops, so committed blocks join without the
        # level pumping a per-block normalise would cause.
        gain = st.get("gain", 1.0)
        if len(out):
            peak = float(np.abs(out).max()) * gain
            if peak > 0.95:
                gain = 0.95 / max(1e-9, float(np.abs(out).max()))
            st["gain"] = gain
            out = np.clip(out * gain, -0.99, 0.99).astype(np.float32)
        return out, sr, markers, clips

    # Trim trailing silence and fade the very end.
    nz = np.nonzero(np.abs(master).sum(axis=1))[0]
    output = master[:nz[-1] + 1] if len(nz) else master[:total_s]
    fade_s = min(int(2.0 * sr), len(output))
    if fade_s > 0:
        output[-fade_s:] *= np.linspace(1.0, 0.0, fade_s, dtype=np.float32).reshape(-1, 1)
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
