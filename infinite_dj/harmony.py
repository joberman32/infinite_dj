"""
Harmonic analysis utilities.

Key detection via Krumhansl-Schmuckler algorithm on chroma features.
Camelot wheel for DJ-standard harmonic compatibility scoring.
"""

import numpy as np

# Krumhansl-Schmuckler key profiles
# Major and minor tonal hierarchies (12 pitch classes starting from C)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot wheel mapping: (root_semitone, is_major) -> camelot_key
# Built from the standard Camelot/Open Key system
CAMELOT_MAP = {
    # Major keys (B suffix)
    (0,  True):  '8B',   # C major
    (7,  True):  '9B',   # G major
    (2,  True):  '10B',  # D major
    (9,  True):  '11B',  # A major
    (4,  True):  '12B',  # E major
    (11, True):  '1B',   # B major
    (6,  True):  '2B',   # F#/Gb major
    (1,  True):  '3B',   # Db major
    (8,  True):  '4B',   # Ab major
    (3,  True):  '5B',   # Eb major
    (10, True):  '6B',   # Bb major
    (5,  True):  '7B',   # F major
    # Minor keys (A suffix)
    (9,  False): '8A',   # A minor
    (4,  False): '9A',   # E minor
    (11, False): '10A',  # B minor
    (6,  False): '11A',  # F# minor
    (1,  False): '12A',  # C# minor
    (8,  False): '1A',   # G# minor
    (3,  False): '2A',   # D# minor
    (10, False): '3A',   # Bb minor
    (5,  False): '4A',   # F minor
    (0,  False): '5A',   # C minor
    (7,  False): '6A',   # G minor
    (2,  False): '7A',   # D minor
}

# Reverse map: camelot_key -> (root_semitone, is_major)
CAMELOT_REVERSE = {v: k for k, v in CAMELOT_MAP.items()}


def detect_key(chroma: np.ndarray) -> tuple[str, str, float]:
    """
    Detect key from a chroma feature matrix using Krumhansl-Schmuckler.

    Args:
        chroma: (12, T) chroma matrix from librosa

    Returns:
        (camelot_key, key_name, confidence)
        e.g. ("8B", "C major", 0.87)
    """
    # Mean chroma vector across time
    mean_chroma = chroma.mean(axis=1)
    mean_chroma = mean_chroma / (mean_chroma.sum() + 1e-8)

    best_score = -np.inf
    best_root = 0
    best_is_major = True

    for root in range(12):
        # Rotate profiles to this root
        major_profile = np.roll(KS_MAJOR, root)
        minor_profile = np.roll(KS_MINOR, root)

        # Pearson correlation
        major_corr = np.corrcoef(mean_chroma, major_profile)[0, 1]
        minor_corr = np.corrcoef(mean_chroma, minor_profile)[0, 1]

        if major_corr > best_score:
            best_score = major_corr
            best_root = root
            best_is_major = True

        if minor_corr > best_score:
            best_score = minor_corr
            best_root = root
            best_is_major = False

    # Confidence: normalize correlation to 0-1
    confidence = float(np.clip((best_score + 1) / 2, 0, 1))

    camelot = CAMELOT_MAP[(best_root, best_is_major)]
    mode = "major" if best_is_major else "minor"
    key_name = f"{NOTE_NAMES[best_root]} {mode}"

    return camelot, key_name, confidence


def camelot_compatibility(key_a: str, key_b: str) -> float:
    """
    Score harmonic compatibility between two Camelot keys.
    Returns a value in [0, 1] where 1 = perfect match.

    DJ harmonic mixing rules:
      1.0 — same key
      0.9 — same number, different letter (parallel major/minor)
      0.8 — adjacent number on wheel (±1, perfect 5th relationship)
      0.6 — two steps (±2)
      0.3 — three steps (energy shift, usable but dramatic)
      0.0 — everything else
    """
    if key_a == key_b:
        return 1.0

    def parse(k):
        letter = k[-1]           # 'A' or 'B'
        number = int(k[:-1])     # 1-12
        return number, letter

    num_a, let_a = parse(key_a)
    num_b, let_b = parse(key_b)

    # Parallel major/minor (same number, different letter)
    if num_a == num_b and let_a != let_b:
        return 0.9

    # Wheel distance (circular, 1-12)
    diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))

    if let_a == let_b:
        # Same mode (A-A or B-B)
        if diff == 1:
            return 0.8
        if diff == 2:
            return 0.6
        if diff == 3:
            return 0.3
    else:
        # Cross-mode adjacency
        if diff <= 1:
            return 0.5

    return 0.0


def bpm_compatibility(bpm_a: float, bpm_b: float) -> float:
    """
    Score BPM compatibility for beatmatching.
    Time-stretching is clean up to ~6% without audible artifacts.
    Doubling/halving handles genre crossovers (e.g. 90 BPM hip-hop -> 180 BPM drum & bass).
    """
    def ratio_score(ratio):
        pct = abs(ratio - 1.0)
        if pct <= 0.06:
            return 1.0 - (pct / 0.06) * 0.2   # 1.0 -> 0.8 within clean range
        if pct <= 0.12:
            return 0.8 - ((pct - 0.06) / 0.06) * 0.5  # degrades to 0.3
        return 0.0

    direct = ratio_score(bpm_b / bpm_a)
    double = ratio_score((bpm_b * 2) / bpm_a)
    half   = ratio_score((bpm_b / 2) / bpm_a)

    return max(direct, double, half)
