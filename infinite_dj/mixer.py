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

from .models import TrackMeta, CuePoint

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


def _blend(
    out_chunk: np.ndarray,
    in_chunk: np.ndarray,
    phase,                       # scalar or per-sample array in [0, 1]
    sr: int = MIX_SR,
    bass_cutoff: float = 200.0,
) -> np.ndarray:
    """
    Shared DJ-style EQ blend used by both the offline renderer and the real-time
    engine. `phase` is 0 at the start of the crossfade and 1 at the end.

      - Highs/mids: equal-power crossfade across the whole region (constant
        perceived loudness, no doubling of both tracks at full).
      - Bass: single source at a time — the outgoing low end is held until the
        midpoint, then swapped to the incoming over a short window, so only one
        kick drum is ever playing (avoids the muddy two-track pile-up).
    """
    n = min(len(out_chunk), len(in_chunk))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out_c = out_chunk[:n]
    in_c  = in_chunk[:n]

    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim == 0:
        phase = np.full(n, float(phase), dtype=np.float32)
    phase = phase[:n].reshape(-1, 1)

    out_bass = _apply_lowpass(out_c, sr, bass_cutoff)
    out_high = _apply_highpass(out_c, sr, bass_cutoff)
    in_bass  = _apply_lowpass(in_c, sr, bass_cutoff)
    in_high  = _apply_highpass(in_c, sr, bass_cutoff)

    # Equal-power high/mid crossfade
    a = phase * (np.pi / 2.0)
    highs = out_high * np.cos(a) + in_high * np.sin(a)

    # Single-source bass swap over a narrow window centered on the midpoint
    swap = np.clip((phase - 0.45) / 0.10, 0.0, 1.0) * (np.pi / 2.0)
    bass = out_bass * np.cos(swap) + in_bass * np.sin(swap)

    return (highs + bass).astype(np.float32)


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
    """Write a MixResult to a WAV file."""
    sf.write(out_path, result.audio, result.sr, subtype='PCM_24')
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


def render_set(
    tracks: list,
    n_mix_bars: int = 16,
    sr: int = MIX_SR,
    max_stretch: float = MAX_STRETCH,
) -> tuple:
    """
    Render an ordered list of tracks into ONE continuous set.

    Unlike stitching independent transition clips, this lays every track on a
    single master timeline: each track plays solo (at its own native tempo) from
    its entry to its exit cue, and consecutive tracks overlap only during a
    crossfade. Only the final track fades out — nothing cuts to silence mid-set.

    Beatmatching is per-transition: the incoming track's crossfade region is
    time-stretched to the *outgoing* track's native tempo so the overlap is
    beat-locked, then the incoming continues its solo at its own tempo. This
    avoids the tempo drift of a global lock and keeps cuts to only those pairs
    genuinely too far apart to match. Returns (audio, sr, [SetMarker, ...]).
    """
    if len(tracks) < 2:
        raise ValueError("Need at least 2 tracks to render a set")

    def load_matched(t):
        a, _ = _load_audio(t.file_path, sr)
        return _loudness_match(a, t.loudness, MASTER_LOUDNESS)

    # Current track state, all in that track's own native timeline.
    cur_t     = tracks[0]
    cur_audio = load_matched(cur_t)
    ci   = best_cue_in(cur_t)
    read = _time_to_samples(
        (ci.timestamp if ci else (cur_t.downbeats[0] if cur_t.downbeats else 0.0)), sr
    )

    master: list = []
    markers: list = []
    written = 0

    for i in range(len(tracks) - 1):
        nxt_t = tracks[i + 1]
        out_bpm  = cur_t.bpm
        out_bar_s = _time_to_samples((60.0 / out_bpm) * 4, sr)
        min_solo  = out_bar_s * 8    # at least 8 bars solo before mixing out

        # ── Outgoing cue: a downbeat with room for a solo section ──────────────
        db_samples = [_time_to_samples(d, sr) for d in cur_t.downbeats]
        best_out = best_cue_out(cur_t)
        best_out_sample = _time_to_samples(best_out.timestamp, sr) if best_out else -1
        lower = read + min_solo
        cands = [d for d in db_samples if lower <= d < len(cur_audio)]
        if cands:
            cue_out_sample = (
                min(cands, key=lambda d: abs(d - best_out_sample))
                if best_out_sample >= lower else cands[0]
            )
        else:
            cue_out_sample = max(read, len(cur_audio) - 1)

        # ── Tempo match: bring incoming to the OUTGOING track's native tempo ───
        ratios = [out_bpm / nxt_t.bpm,
                  out_bpm / (nxt_t.bpm * 2.0),
                  out_bpm / (nxt_t.bpm / 2.0)]
        ratio = min(ratios, key=lambda r: abs(r - 1.0))
        beatmatched = abs(ratio - 1.0) <= max_stretch
        n_bars = n_mix_bars if beatmatched else min(4, n_mix_bars)

        nxt_audio = load_matched(nxt_t)
        in_bar_s  = _time_to_samples((60.0 / nxt_t.bpm) * 4, sr)

        # ── Incoming entry point (native downbeat) ─────────────────────────────
        cin = best_cue_in(nxt_t)
        in_time = cin.timestamp if cin else (nxt_t.downbeats[0] if nxt_t.downbeats else 0.0)
        in_time = _find_nearest_downbeat(
            in_time, nxt_t.downbeats, max_offset=(60.0 / nxt_t.bpm) * 4 / 2.0
        )
        in_sample = _time_to_samples(in_time, sr)

        # ── Crossfade regions ──────────────────────────────────────────────────
        out_mix = cur_audio[cue_out_sample:cue_out_sample + n_bars * out_bar_s]
        in_region = nxt_audio[in_sample:in_sample + n_bars * in_bar_s]
        if beatmatched and abs(ratio - 1.0) > 0.001:
            # Stretch the incoming region to the outgoing tempo (bars now align)
            in_mix = _time_stretch(in_region, sr, ratio)
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
            if beatmatched:
                phase = np.linspace(0.0, 1.0, m, dtype=np.float32)
                xf = _blend(out_mix, in_mix, phase, sr)
            else:
                fo, fi = _equal_power_fade(m)
                xf = out_mix * fo.reshape(-1, 1) + in_mix * fi.reshape(-1, 1)
            xf = _apply_gain(xf, 0.9)
            markers.append(SetMarker(
                time=(written + len(solo)) / sr,
                label=f"{cur_t.title} → {nxt_t.title}",
                method="beatmatch" if beatmatched else "cut",
                stretch_pct=(ratio - 1.0) * 100.0,
            ))
            master.append(xf)
            written += len(solo) + len(xf)
        else:
            written += len(solo)

        # ── Advance: incoming becomes current, continuing at its native tempo ──
        # Native samples of the incoming consumed by the (stretched) crossfade.
        consumed_in = int(round(m * ratio))
        cur_t     = nxt_t
        cur_audio = nxt_audio
        read      = in_sample + consumed_in

    # ── Final track tail (capped) with a gentle fade-out ───────────────────────
    max_tail = _time_to_samples((60.0 / cur_t.bpm) * 4 * 32, sr)   # up to 32 bars
    tail = cur_audio[read:read + max_tail].copy()
    fade_s = min(int(2.0 * sr), len(tail))
    if fade_s > 0:
        tail[-fade_s:] *= np.linspace(1.0, 0.0, fade_s, dtype=np.float32).reshape(-1, 1)
    if len(tail) > 0:
        master.append(tail)

    output = np.concatenate(master, axis=0)
    peak = np.abs(output).max()
    if peak > 0.95:
        output = output * (0.95 / peak)

    return output, sr, markers


# ── Auto-select best cue points ───────────────────────────────────────────────

def best_cue_out(track: TrackMeta) -> Optional[CuePoint]:
    outs = [c for c in track.cue_points if c.type == "out"]
    return max(outs, key=lambda c: c.confidence) if outs else None


def best_cue_in(track: TrackMeta) -> Optional[CuePoint]:
    ins = [c for c in track.cue_points if c.type == "in"]
    return max(ins, key=lambda c: c.confidence) if ins else None
