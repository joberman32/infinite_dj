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

# Beat tracking uses a finer hop for tempo/phase precision (~11.6 ms/frame).
BEAT_HOP    = 256

# Tempo octave band. We estimate the dominant tempo over a wide range, then fold
# it into a single octave [MIN, 2*MIN) so half/double detections resolve to a
# consistent representative tempo.
BPM_MIN     = 90.0
BPM_MAX     = 180.0    # == 2 * BPM_MIN (one octave)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _frames_to_times(frames, sr=SR, hop_length=HOP_LENGTH):
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length).tolist()


def _adaptive_mean(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(int(n)) / float(n), mode="same")


def _fold_octave(bpm: float) -> float:
    """Fold a tempo into [BPM_MIN, BPM_MAX) by doubling/halving."""
    guard = 0
    while bpm < BPM_MIN and guard < 8:
        bpm *= 2.0; guard += 1
    while bpm >= BPM_MAX and guard < 8:
        bpm /= 2.0; guard += 1
    return bpm


def _refine_tempo_phase(onset_env: np.ndarray, sr: int, hop: int,
                        tempo_hint: float) -> tuple:
    """
    DJ-grade tempo + phase: build a rigid, perfectly equidistant beat grid
    (constant tempo assumed, as real electronic tracks are) instead of the
    beat-to-beat wobble a dynamic beat tracker produces — that wobble is what
    makes two tracks drift in and out of phase over a long crossfade.

    We trust librosa's perceptually-anchored tempo for the *metrical level*
    (which octave/subdivision), then refine it to a precise constant value and
    find the global beat phase by autocorrelation — the precision that makes a
    rigid grid actually line up. (Method reimplemented from Vande Veire & De
    Bie's auto-DJ, not copied.)

    Returns (bpm, phase_seconds, confidence).
    """
    oenv = np.clip(onset_env - _adaptive_mean(onset_env, 16), 0.0, None)
    if oenv.sum() <= 0 or len(oenv) < 8:
        return _fold_octave(tempo_hint or 120.0), 0.0, 0.0

    ac = np.correlate(oenv, oenv, mode="full")[len(oenv) - 1:]
    n = len(ac)

    # Refine the tempo on a fine grid within a narrow window of the hint (so the
    # octave stays librosa's choice, only the precise value is sharpened).
    center = _fold_octave(tempo_hint or 120.0)
    bpms = np.arange(center * 0.94, center * 1.06, 0.02)
    scores = np.zeros(len(bpms))
    for i, bpm in enumerate(bpms):
        lag = 60.0 * sr / (bpm * hop)
        if lag < 1:
            continue
        idx = np.round(np.arange(lag, n, lag)).astype(int)
        idx = idx[idx < n]
        if len(idx):
            scores[i] = ac[idx].mean()
    best = int(np.argmax(scores))
    tempo = _fold_octave(float(bpms[best]))
    conf = float(np.clip(scores[best] / (scores.mean() + 1e-9) - 1.0, 0.0, 1.0))

    # Global beat phase: the offset whose beat positions carry the most onset.
    period = 60.0 * sr / (tempo * hop)       # frames per beat
    best_phase, best_pscore = 0.0, -1.0
    for ph in np.arange(0.0, period, 0.5):
        idx = np.round(np.arange(ph, len(oenv), period)).astype(int)
        idx = idx[idx < len(oenv)]
        s = float(oenv[idx].sum()) if len(idx) else 0.0
        if s > best_pscore:
            best_pscore, best_phase = s, ph
    phase_sec = best_phase * hop / sr

    return tempo, phase_sec, conf


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

    Beats are a rigid equidistant grid from a fine tempo + explicit phase
    estimate (drift-free for beatmatching), with downbeats anchored to the
    onset-strongest bar phase.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=BEAT_HOP)
    # librosa's perceptual tempo prior picks a sensible metrical level; we then
    # refine it to a precise constant tempo + explicit phase for a rigid grid.
    tempo_hint = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    tempo_hint = float(np.atleast_1d(tempo_hint)[0])
    tempo, phase_sec, conf = _refine_tempo_phase(onset_env, sr, BEAT_HOP, tempo_hint)

    duration = len(y) / sr
    beat_sec = 60.0 / tempo
    beat_times = list(np.arange(phase_sec, max(phase_sec, duration - beat_sec), beat_sec))

    downbeats = _anchor_downbeats(beat_times, onset_env, sr, BEAT_HOP)
    return tempo, conf, beat_times, list(downbeats)


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
        seg = energy_curve[t_start:t_end]
        seg_energy = float(np.mean(seg)) if len(seg) else 0.0

        # Energy trend within the section (rising / falling / flat), so labels
        # describe dynamics rather than genre-specific structure.
        if len(seg) >= 6:
            third = max(1, len(seg) // 3)
            trend = float(np.mean(seg[-third:]) - np.mean(seg[:third]))
        else:
            trend = 0.0

        # Genre-neutral vocabulary: works for ambient / IDM / rock, not just EDM.
        pos = start / duration
        if pos < 0.12 and seg_energy < 0.45:
            label = "intro"
        elif pos > 0.85 and seg_energy < 0.5:
            label = "outro"
        elif seg_energy < 0.30:
            label = "sparse"
        elif trend > 0.08:
            label = "rising"
        elif trend < -0.08:
            label = "falling"
        elif seg_energy > 0.70:
            label = "peak"
        else:
            label = "steady"

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

def analyze_track(file_path: str, verbose: bool = True) -> TrackMeta:
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
    def log(msg, end="\n", flush=False):
        if verbose:
            print(msg, end=end, flush=flush)

    log(f"  Loading:    {os.path.basename(file_path)}")
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    duration = float(len(y) / sr)

    log(f"  BPM/beats...", end=" ", flush=True)
    bpm, bpm_conf, beats, downbeats = _compute_beats(y, sr)
    log(f"{bpm:.1f} BPM ({len(beats)} beats)")

    log(f"  Phrases...", end=" ", flush=True)
    phrases = _compute_phrases(downbeats)
    log(f"{len(phrases)} phrase boundaries")

    log(f"  Key...", end=" ", flush=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    camelot, key_name, key_conf = detect_key(chroma)
    log(f"{key_name} ({camelot}), confidence {key_conf:.2f}")

    log(f"  Energy...", end=" ", flush=True)
    energy_curve = _compute_energy_curve(y, sr)
    log(f"{len(energy_curve)}s curve")

    log(f"  Sections...", end=" ", flush=True)
    sections = _compute_sections(y, sr, energy_curve, duration)
    log(f"{len(sections)} sections: {[s.label for s in sections]}")

    loudness = _compute_loudness(y)

    log(f"  Cue points...", end=" ", flush=True)
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
        y=y,
    )
    n_in  = sum(1 for c in cue_points if c.type == "in")
    n_out = sum(1 for c in cue_points if c.type == "out")
    n_emb = sum(1 for c in cue_points if c.embedding is not None)
    log(f"{n_in} IN, {n_out} OUT ({n_emb} CLAP embedded)")

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

