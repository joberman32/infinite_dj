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

@dataclass
class TransitionPlan:
    """Everything needed to execute a single transition between two tracks."""
    track_out: TrackMeta
    track_in: TrackMeta
    cue_out: CuePoint          # exit point in track_out
    cue_in: CuePoint           # entry point in track_in
    n_mix_bars: int = 8        # how many bars the crossfade spans
    bpm_target: float = 0.0   # resolved BPM to mix at (set on init)
    stretch_ratio: float = 1.0 # how much track_in is time-stretched

    def __post_init__(self):
        if self.bpm_target == 0.0:
            self.bpm_target = self.track_out.bpm
        self.stretch_ratio = self.track_out.bpm / self.track_in.bpm


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
    search_window: float = 4.0
) -> float:
    """
    After time-stretching, the downbeats shift. Find the downbeat
    in the stretched track that's closest to our target cue-in time,
    within a search window of ±search_window seconds.
    """
    candidates = [d for d in downbeats
                  if abs(d - target_time) <= search_window]
    if not candidates:
        return target_time  # fallback: use as-is
    return min(candidates, key=lambda d: abs(d - target_time))


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

    # Time-stretch incoming track to match outgoing BPM
    if abs(plan.stretch_ratio - 1.0) > 0.001:
        pct = (plan.stretch_ratio - 1.0) * 100
        print(f"  Time-stretching incoming track by {pct:+.1f}% to match {plan.bpm_target:.1f} BPM...")
        audio_in = _time_stretch(audio_in, sr, plan.stretch_ratio)

        # Adjust downbeat timestamps after stretching
        stretched_downbeats = [d * plan.stretch_ratio
                               for d in plan.track_in.downbeats]
    else:
        stretched_downbeats = list(plan.track_in.downbeats)

    # Calculate bar duration at target BPM
    bar_duration = (60.0 / plan.bpm_target) * 4   # 4 beats per bar
    mix_duration = bar_duration * plan.n_mix_bars

    # Find exact sample positions
    out_cue_sample = _time_to_samples(plan.cue_out.timestamp, sr)
    mix_region_samples = _time_to_samples(mix_duration, sr)

    # Find best aligned entry in incoming track
    aligned_in_time = _find_nearest_downbeat(
        plan.cue_in.timestamp, stretched_downbeats
    )
    in_cue_sample = _time_to_samples(aligned_in_time, sr)

    print(f"  OUT cue: {plan.cue_out.timestamp:.1f}s → IN cue: {aligned_in_time:.1f}s (stretched)")
    print(f"  Mix region: {mix_duration:.1f}s ({plan.n_mix_bars} bars at {plan.bpm_target:.1f} BPM)")

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

    # ── Build EQ-shaped crossfade ─────────────────────────────────────────────
    #
    # We split the mix region into 3 phases:
    #   Phase 1: bars 1-N/3   — outgoing full + incoming highs fade in
    #   Phase 2: bars N/3-2N/3 — bass swap
    #   Phase 3: bars 2N/3-N  — incoming full + outgoing highs fade out
    #
    # This mimics how a DJ uses EQ knobs on a mixer.

    print(f"  Rendering EQ crossfade...")

    phase_len = mix_len // 3
    p1 = slice(0,           phase_len)
    p2 = slice(phase_len,   phase_len * 2)
    p3 = slice(phase_len*2, mix_len)

    # Separate bass and highs for both tracks
    out_bass = _apply_lowpass(out_mix, sr)
    out_high = _apply_highpass(out_mix, sr)
    in_bass  = _apply_lowpass(in_mix, sr)
    in_high  = _apply_highpass(in_mix, sr)

    mix = np.zeros((mix_len, 2), dtype=np.float32)

    # Phase 1: outgoing full, incoming highs fade in
    fo1, fi1 = _equal_power_fade(phase_len)
    fo1 = fo1.reshape(-1, 1)
    fi1 = fi1.reshape(-1, 1)
    mix[p1] = (
        out_mix[p1]                      +   # outgoing full
        in_high[p1] * fi1                    # incoming highs fade in
    )

    # Phase 2: bass swap
    # outgoing: highs stay, bass fades out
    # incoming: highs already in, bass fades in
    fo2, fi2 = _equal_power_fade(phase_len)
    fo2 = fo2.reshape(-1, 1)
    fi2 = fi2.reshape(-1, 1)
    mix[p2] = (
        out_high[p2]                     +   # outgoing highs (still full)
        out_bass[p2] * fo2               +   # outgoing bass fading out
        in_high[p2]                      +   # incoming highs (full)
        in_bass[p2] * fi2                    # incoming bass fading in
    )

    # Phase 3: incoming full, outgoing highs fade out
    p3_len = mix_len - phase_len * 2
    fo3, _ = _equal_power_fade(p3_len)
    fo3 = fo3.reshape(-1, 1)
    mix[p3] = (
        out_high[p3] * fo3               +   # outgoing highs fade out
        in_mix[p3]                           # incoming full
    )

    # Small gain reduction during mix peak to prevent clipping
    mix = _apply_gain(mix, 0.85)

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


# ── Auto-select best cue points ───────────────────────────────────────────────

def best_cue_out(track: TrackMeta) -> Optional[CuePoint]:
    outs = [c for c in track.cue_points if c.type == "out"]
    return max(outs, key=lambda c: c.confidence) if outs else None


def best_cue_in(track: TrackMeta) -> Optional[CuePoint]:
    ins = [c for c in track.cue_points if c.type == "in"]
    return max(ins, key=lambda c: c.confidence) if ins else None
