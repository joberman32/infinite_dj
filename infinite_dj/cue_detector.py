"""
Cue point detection.

Scores every downbeat as a potential IN or OUT transition point using
a multi-factor heuristic. Good cue points share properties with how
professional DJs select transition moments.

Phase 3: supports FULL-TRACK scanning — any downbeat, anywhere in the
track, can be a valid OUT point. The scoring function heavily weights
phrase boundaries and energy valleys regardless of track position.
"""

import numpy as np
from typing import List
from .models import CuePoint


def _energy_at(timestamp: float, energy_curve: List[float]) -> float:
    idx = min(int(timestamp), len(energy_curve) - 1)
    return energy_curve[idx]


def _is_energy_valley(
    timestamp: float,
    energy_curve: List[float],
    window: float = 8.0
) -> float:
    t = int(timestamp)
    n = len(energy_curve)
    w = int(window)
    local  = energy_curve[max(0, t - 2): min(n, t + 2)]
    before = energy_curve[max(0, t - w): max(0, t - 2)]
    after  = energy_curve[min(n, t + 2): min(n, t + w)]
    if not local:
        return 0.0
    local_mean = np.mean(local)
    context = []
    if before: context.append(np.mean(before))
    if after:  context.append(np.mean(after))
    if not context:
        return 0.0
    drop = np.mean(context) - local_mean
    return float(np.clip(drop * 4, 0, 1))


def _is_phrase_boundary(timestamp: float, phrases: List[float], tol: float = 0.5) -> float:
    if not phrases:
        return 0.0
    dists = [abs(timestamp - p) for p in phrases]
    return float(np.clip(1.0 - min(dists) / tol, 0, 1))


def _energy_trend(timestamp: float, energy_curve: List[float], window: float = 4.0) -> float:
    t = int(timestamp)
    n = len(energy_curve)
    w = int(window)
    before = energy_curve[max(0, t - w): t]
    after  = energy_curve[t: min(n, t + w)]
    if not before or not after:
        return 0.0
    return float(np.mean(after) - np.mean(before))


def _spectral_flatness_score(
    timestamp: float,
    spectral_flatness: np.ndarray,
    sr: int,
    hop_length: int
) -> float:
    frame = int(timestamp * sr / hop_length)
    frame = min(frame, len(spectral_flatness) - 1)
    val = spectral_flatness[frame]
    return float(np.clip(1.0 - abs(val - 0.2) * 5, 0, 1))


def score_out_point(
    timestamp: float,
    energy_curve: List[float],
    phrases: List[float],
    spectral_flatness: np.ndarray,
    sr: int,
    hop_length: int,
    duration: float,
    mid_track: bool = False,   # Phase 3: allow mid-track OUT points
) -> float:
    """
    Score a timestamp as an OUT cue point.

    Phase 3 change: mid_track=True relaxes the position constraints
    so any phrase boundary/energy valley in the track is eligible,
    not just the final 15-85% window. This enables arbitrary mid-track
    transitions — the engine can jump out of a track at any good moment.
    """
    if not mid_track:
        if timestamp < duration * 0.2 or timestamp > duration * 0.85:
            return 0.0
    else:
        # Only reject the very beginning (no musical content yet)
        if timestamp < 4.0:
            return 0.0

    phrase_score = _is_phrase_boundary(timestamp, phrases) * 3.5
    valley_score = _is_energy_valley(timestamp, energy_curve) * 2.5
    trend_score  = -_energy_trend(timestamp, energy_curve) * 1.5
    flat_score   = _spectral_flatness_score(timestamp, spectral_flatness, sr, hop_length) * 1.0
    energy_abs   = (1.0 - _energy_at(timestamp, energy_curve)) * 0.5

    raw = phrase_score + valley_score + trend_score + flat_score + energy_abs
    max_possible = 3.5 + 2.5 + 1.5 + 1.0 + 0.5
    return float(np.clip(raw / max_possible, 0, 1))


def score_in_point(
    timestamp: float,
    energy_curve: List[float],
    phrases: List[float],
    spectral_flatness: np.ndarray,
    sr: int,
    hop_length: int,
    duration: float,
    mid_track: bool = False,   # Phase 3: allow mid-track IN points
) -> float:
    """
    Score a timestamp as an IN cue point.

    Phase 3: mid_track=True allows entering a track at any phrase
    boundary, not just the first half. This makes tracks reusable
    as sources at multiple entry points.
    """
    if not mid_track:
        if timestamp > duration * 0.5:
            return 0.0
        if timestamp < 2.0:
            timestamp = max(timestamp, 0.0)
    else:
        if timestamp < 2.0:
            return 0.0
        if timestamp > duration * 0.8:
            return 0.0

    phrase_score = _is_phrase_boundary(timestamp, phrases) * 3.5
    trend_score  = _energy_trend(timestamp, energy_curve) * 1.5
    valley_score = _is_energy_valley(timestamp, energy_curve) * 1.5
    flat_score   = _spectral_flatness_score(timestamp, spectral_flatness, sr, hop_length) * 1.0
    energy_abs   = (1.0 - _energy_at(timestamp, energy_curve)) * 0.5

    raw = phrase_score + trend_score + valley_score + flat_score + energy_abs
    max_possible = 3.5 + 1.5 + 1.5 + 1.0 + 0.5
    return float(np.clip(raw / max_possible, 0, 1))


def detect_cue_points(
    downbeats: List[float],
    phrases: List[float],
    energy_curve: List[float],
    spectral_flatness: np.ndarray,
    sr: int,
    hop_length: int,
    duration: float,
    top_k: int = 5,
    mid_track: bool = True,    # Phase 3 default: scan full track
) -> List[CuePoint]:
    """
    Score all downbeats as potential IN/OUT cue points.
    Returns top_k of each type, sorted by timestamp.
    """
    cue_points = []

    for ts in downbeats:
        energy   = _energy_at(ts, energy_curve)
        on_phrase = _is_phrase_boundary(ts, phrases) > 0.8

        out_score = score_out_point(
            ts, energy_curve, phrases, spectral_flatness,
            sr, hop_length, duration, mid_track=mid_track
        )
        in_score = score_in_point(
            ts, energy_curve, phrases, spectral_flatness,
            sr, hop_length, duration, mid_track=mid_track
        )

        if out_score > 0.1:
            cue_points.append(CuePoint(
                timestamp=round(ts, 3),
                type="out",
                phrase_aligned=on_phrase,
                energy=round(energy, 3),
                confidence=round(out_score, 3),
            ))
        if in_score > 0.1:
            cue_points.append(CuePoint(
                timestamp=round(ts, 3),
                type="in",
                phrase_aligned=on_phrase,
                energy=round(energy, 3),
                confidence=round(in_score, 3),
            ))

    outs = sorted([c for c in cue_points if c.type == "out"],
                  key=lambda c: c.confidence, reverse=True)[:top_k]
    ins  = sorted([c for c in cue_points if c.type == "in"],
                  key=lambda c: c.confidence, reverse=True)[:top_k]

    return sorted(outs + ins, key=lambda c: c.timestamp)

