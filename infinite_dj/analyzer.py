"""
Core analysis engine.

Takes an audio file path, runs the full analysis pipeline,
and returns a TrackMeta object. This is the expensive step —
everything downstream uses cached results from the DB.
"""

import os
import time
import warnings
import numpy as np
import librosa
import librosa.segment

from .models import TrackMeta, Section
from .harmony import detect_key
from .cue_detector import detect_cue_points

warnings.filterwarnings("ignore")  # librosa is chatty

# ── Constants ────────────────────────────────────────────────────────────────
SR          = 22050    # sample rate for analysis (mono, downsampled)
HOP_LENGTH  = 512      # librosa default hop
N_MELS      = 128

# Tempo octave-normalization band. beat_track frequently locks to half- or
# double-time; we fold the detected tempo into a single octave [MIN, 2*MIN)
# so half/double errors resolve to a consistent representative tempo and the
# beat grid is re-gridded to match (not just the BPM number relabeled).
BPM_MIN     = 90.0
BPM_MAX     = 180.0    # == 2 * BPM_MIN (one octave)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _frames_to_times(frames, sr=SR, hop_length=HOP_LENGTH):
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length).tolist()


def _octave_normalize(bpm: float, beat_times: list) -> tuple[float, list]:
    """
    Fold a detected tempo into the [BPM_MIN, BPM_MAX) octave and re-grid the
    beats to match. beat_track often locks to half- or double-time; relabeling
    the BPM alone would leave the beat spacing wrong, so we actually insert
    midpoint beats (when doubling) or drop every other beat (when halving).
    """
    beats = list(beat_times)

    guard = 0
    while bpm < BPM_MIN and guard < 4:
        # Double: insert a beat halfway between each adjacent pair
        doubled = []
        for i in range(len(beats) - 1):
            doubled.append(beats[i])
            doubled.append((beats[i] + beats[i + 1]) / 2.0)
        if beats:
            doubled.append(beats[-1])
        beats = doubled
        bpm *= 2.0
        guard += 1

    while bpm >= BPM_MAX and guard < 8:
        # Halve: keep every other beat
        beats = beats[::2]
        bpm /= 2.0
        guard += 1

    return bpm, beats


def _anchor_downbeats(beat_times: list, onset_env: np.ndarray, sr: int, hop: int) -> list:
    """
    Pick the bar-phase (which of every 4 beats is beat 1) whose beats carry the
    most onset energy, so downbeats land on the actual musical bar starts rather
    than on an arbitrary offset. Returns every 4th beat from that phase.
    """
    if len(beat_times) < 4:
        return list(beat_times)

    frames = librosa.time_to_frames(np.asarray(beat_times), sr=sr, hop_length=hop)
    frames = np.clip(frames, 0, len(onset_env) - 1)
    strengths = onset_env[frames]

    best_off, best_sum = 0, -np.inf
    for off in range(4):
        s = float(strengths[off::4].sum())
        if s > best_sum:
            best_sum, best_off = s, off

    return list(np.asarray(beat_times)[best_off::4])


def _compute_beats(y, sr):
    """
    Returns (bpm, bpm_confidence, beat_times, downbeat_times).

    Tempo is octave-normalized into the house/techno band with the beat grid
    re-gridded to match, then downbeats are anchored to the onset-strongest
    bar phase.
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    beat_times = _frames_to_times(beat_frames)

    # Fold half/double-time detections into one octave, re-gridding beats
    bpm, beat_times = _octave_normalize(bpm, beat_times)

    # Estimate confidence from beat consistency
    if len(beat_times) > 4:
        intervals = np.diff(beat_times)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
        bpm_confidence = float(np.clip(1.0 - cv * 2, 0, 1))
    else:
        bpm_confidence = 0.0

    # Anchor downbeats to the strongest bar phase
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    downbeats = _anchor_downbeats(beat_times, onset_env, sr, HOP_LENGTH)

    return bpm, bpm_confidence, beat_times, list(downbeats)


def _compute_phrases(downbeats: list, phrase_bars: int = 8) -> list:
    """Every `phrase_bars` downbeats is a phrase boundary."""
    return downbeats[::phrase_bars]


def _compute_energy_curve(y, sr) -> list:
    """
    Normalized RMS energy, one value per second.
    Returns list of floats in [0, 1].
    """
    frame_length = sr           # 1 second frames
    hop = sr // 2               # 0.5s hop, then resample to 1s

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]

    # Resample to exactly 1 value per second
    duration_sec = int(len(y) / sr)
    if len(rms) > 0:
        indices = np.linspace(0, len(rms) - 1, duration_sec).astype(int)
        rms_1s = rms[indices]
    else:
        rms_1s = rms

    # Normalize
    max_rms = rms_1s.max() + 1e-8
    normalized = (rms_1s / max_rms).tolist()
    return [round(float(v), 4) for v in normalized]


def _compute_sections(y, sr, energy_curve: list, duration: float) -> list:
    """
    Detect structural sections using spectral novelty on MFCCs.
    Fast approach: compute a novelty curve from MFCC differences,
    find peaks as boundaries, label by energy.
    """
    # MFCCs at reduced resolution for speed
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH * 4)

    # Novelty function: L2 norm of frame-to-frame MFCC difference
    diff = np.diff(mfcc, axis=1)
    novelty = np.linalg.norm(diff, axis=0)

    # Smooth novelty curve
    kernel = np.hanning(max(3, int(len(novelty) * 0.05)))
    kernel /= kernel.sum()
    novelty_smooth = np.convolve(novelty, kernel, mode='same')

    # Find peaks (candidate boundaries)
    from scipy.signal import find_peaks
    min_dist = max(1, int(len(novelty_smooth) * 0.1))  # at least 10% apart
    peaks, _ = find_peaks(novelty_smooth, distance=min_dist,
                          height=np.percentile(novelty_smooth, 60))

    # Convert peak frames to times
    hop_ratio = HOP_LENGTH * 4
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_ratio).tolist()

    # Limit to reasonable number of sections
    n_target = max(3, min(8, int(duration / 30)))
    if len(peak_times) > n_target - 1:
        # Keep the highest-novelty peaks
        peak_scores = novelty_smooth[peaks]
        top_idx = np.argsort(peak_scores)[::-1][:n_target - 1]
        peak_times = sorted([peak_times[i] for i in top_idx])

    boundaries = [0.0] + peak_times + [duration]

    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]

        t_start = max(0, int(start))
        t_end   = min(len(energy_curve), int(end))
        seg_energy = float(np.mean(energy_curve[t_start:t_end])) if t_end > t_start else 0.0

        pos = start / duration
        if pos < 0.12 and seg_energy < 0.45:
            label = "intro"
        elif pos > 0.85 and seg_energy < 0.5:
            label = "outro"
        elif seg_energy > 0.75:
            label = "drop"
        elif seg_energy > 0.5:
            label = "build"
        elif seg_energy < 0.3:
            label = "breakdown"
        else:
            label = "body"

        sections.append(Section(
            start=round(start, 2),
            end=round(end, 2),
            label=label,
            energy=round(seg_energy, 3),
        ))

    return sections


def _compute_spectral_flatness(y, sr):
    """Spectral flatness per frame — used in cue point scoring."""
    return librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH)[0]


def _compute_loudness(y) -> float:
    """
    Integrated loudness as full-signal RMS in dBFS (negative). Used to gain-match
    tracks at transitions so levels don't jump. Relative, so the analysis-rate
    mono signal is fine.
    """
    rms = float(np.sqrt(np.mean(np.square(y))))
    return round(20.0 * np.log10(rms + 1e-9), 2)


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_track(file_path: str) -> TrackMeta:
    """
    Full analysis pipeline for a single audio file.

    Steps:
      1. Load audio (mono, 22050 Hz)
      2. Compute BPM, beats, downbeats
      3. Compute phrase boundaries
      4. Detect key via Krumhansl-Schmuckler
      5. Compute energy curve (1s resolution)
      6. Detect structural sections
      7. Score cue points at all downbeats

    Returns a TrackMeta object ready to be persisted to the DB.
    """
    print(f"  Loading:    {os.path.basename(file_path)}")
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    duration = float(len(y) / sr)

    print(f"  BPM/beats...", end=" ", flush=True)
    bpm, bpm_conf, beats, downbeats = _compute_beats(y, sr)
    print(f"{bpm:.1f} BPM ({len(beats)} beats)")

    print(f"  Phrases...", end=" ", flush=True)
    phrases = _compute_phrases(downbeats)
    print(f"{len(phrases)} phrase boundaries")

    print(f"  Key...", end=" ", flush=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    camelot, key_name, key_conf = detect_key(chroma)
    print(f"{key_name} ({camelot}), confidence {key_conf:.2f}")

    print(f"  Energy...", end=" ", flush=True)
    energy_curve = _compute_energy_curve(y, sr)
    print(f"{len(energy_curve)}s curve")

    print(f"  Sections...", end=" ", flush=True)
    sections = _compute_sections(y, sr, energy_curve, duration)
    print(f"{len(sections)} sections: {[s.label for s in sections]}")

    loudness = _compute_loudness(y)

    print(f"  Cue points...", end=" ", flush=True)
    spectral_flatness = _compute_spectral_flatness(y, sr)
    cue_points = detect_cue_points(
        downbeats=downbeats,
        phrases=phrases,
        energy_curve=energy_curve,
        spectral_flatness=spectral_flatness,
        sr=SR,
        hop_length=HOP_LENGTH,
        duration=duration,
        top_k=5,
    )
    n_in  = sum(1 for c in cue_points if c.type == "in")
    n_out = sum(1 for c in cue_points if c.type == "out")
    print(f"{n_in} IN, {n_out} OUT")

    title = os.path.splitext(os.path.basename(file_path))[0]

    return TrackMeta(
        file_path=os.path.abspath(file_path),
        title=title,
        duration=round(duration, 2),
        bpm=round(bpm, 2),
        bpm_confidence=round(bpm_conf, 3),
        beats=[round(b, 3) for b in beats],
        downbeats=[round(d, 3) for d in downbeats],
        phrases=[round(p, 3) for p in phrases],
        key=camelot,
        key_name=key_name,
        key_confidence=round(key_conf, 3),
        energy_curve=energy_curve,
        sections=sections,
        cue_points=cue_points,
        analyzed_at=time.time(),
        loudness=loudness,
    )
