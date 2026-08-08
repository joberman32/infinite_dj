"""
Library health: is this library one the engine can actually mix?

Two reports, both derived entirely from cached `TrackMeta` — no audio is
touched, so both are fast enough to run over a few hundred tracks.

  * `triage`  — per-track: can the engine mix THIS track well?
  * `gaps`    — per-library: what is missing that the engine needs?

**What triage can and cannot see.** `analyzer._refine_tempo_phase` lays beats
down as a perfectly equidistant grid (`arange(phase, dur, 60/bpm)`), so the
stored grid is regular *by construction* — you cannot detect a wobbly source
by measuring the grid. What is detectable from metadata: the analyzer's own
`bpm_confidence`, implausible tempo placement, structural deficiency (too few
downbeats/phrases/cues to mix with), and durations too short to hold a solo
plus a blend. A confidently-wrong grid is invisible here and needs ears.

The gap report exists because the engine's behaviour is gated on library
*composition*, not size. `choose_transition_style` only returns `blend` when
both cue energies are below `BLEND_MAX_ENERGY`, and only returns `swap` when
both are above `SWAP_MIN_ENERGY` — so a library with no energy extremes can
never produce two of the four styles, no matter what the mixer does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ── Triage thresholds ────────────────────────────────────────────────────────
# Calibrated against the existing hand-picked library, which is the only
# known-good reference available. Deliberately permissive: the cost of a false
# reject (losing a fine track) is higher than a false accept (one odd mix).
MIN_BPM_CONFIDENCE   = 0.35   # below this the tempo is a guess
MIN_KEY_CONFIDENCE   = 0.20   # below this, harmonic mixing is noise
MIN_DURATION_SEC     = 90.0   # shorter can't hold a solo + a blend
MIN_DOWNBEATS        = 16     # 16 bars: the shortest usable dwell
MIN_CUES_PER_TYPE    = 2      # need somewhere to enter and somewhere to leave
BPM_EDGE_MARGIN      = 3.0    # within this of the fold boundary => suspect

# Mirrors `analyzer.BPM_MIN` / `BPM_MAX`; duplicated as floats to keep this
# module importable without pulling in librosa.
BPM_MIN, BPM_MAX = 90.0, 180.0


@dataclass
class GridReport:
    """One track's fitness for mixing."""
    title: str
    file_path: str
    grade: str                       # "good" | "usable" | "reject"
    reasons: List[str] = field(default_factory=list)
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    key: str = ""
    duration: float = 0.0

    @property
    def keep(self) -> bool:
        return self.grade != "reject"


def grid_quality(track) -> GridReport:
    """
    Grade one analyzed track. `reject` means the engine cannot mix it sensibly;
    `usable` means it will work but something is weak enough to hear.
    """
    hard: List[str] = []      # → reject
    soft: List[str] = []      # → usable

    dur = float(getattr(track, "duration", 0.0) or 0.0)
    if dur < MIN_DURATION_SEC:
        hard.append(f"too short ({dur:.0f}s < {MIN_DURATION_SEC:.0f}s)")

    downbeats = list(getattr(track, "downbeats", None) or [])
    if len(downbeats) < MIN_DOWNBEATS:
        hard.append(f"only {len(downbeats)} downbeats (< {MIN_DOWNBEATS})")

    conf = float(getattr(track, "bpm_confidence", 0.0) or 0.0)
    if conf < MIN_BPM_CONFIDENCE:
        hard.append(f"tempo confidence {conf:.2f} (< {MIN_BPM_CONFIDENCE})")

    bpm = float(getattr(track, "bpm", 0.0) or 0.0)
    if not (BPM_MIN <= bpm < BPM_MAX):
        hard.append(f"bpm {bpm:.1f} outside [{BPM_MIN:.0f}, {BPM_MAX:.0f})")
    elif bpm < BPM_MIN + BPM_EDGE_MARGIN or bpm > BPM_MAX - BPM_EDGE_MARGIN:
        # Sitting on the octave-fold boundary usually means the metrical level
        # was ambiguous and the fold could have gone either way.
        soft.append(f"bpm {bpm:.1f} on the octave-fold edge")

    # A grid that doesn't span the audio means beat tracking gave up partway.
    beats = list(getattr(track, "beats", None) or [])
    if beats and dur > 0:
        covered = (beats[-1] - beats[0]) / dur
        if covered < 0.60:
            hard.append(f"beat grid covers only {covered:.0%} of the track")

    cues = list(getattr(track, "cue_points", None) or [])
    n_in  = sum(1 for c in cues if c.type == "in")
    n_out = sum(1 for c in cues if c.type == "out")
    if n_in < MIN_CUES_PER_TYPE or n_out < MIN_CUES_PER_TYPE:
        hard.append(f"too few cues ({n_in} in / {n_out} out)")
    elif not any(c.phrase_aligned for c in cues):
        soft.append("no phrase-aligned cue points")

    kconf = float(getattr(track, "key_confidence", 0.0) or 0.0)
    if kconf < MIN_KEY_CONFIDENCE:
        soft.append(f"key confidence {kconf:.2f} — harmonic matching unreliable")

    if not list(getattr(track, "sections", None) or []):
        soft.append("no structural sections — splices fall back to cue points")

    grade = "reject" if hard else ("usable" if soft else "good")
    return GridReport(
        title=getattr(track, "title", "?"),
        file_path=getattr(track, "file_path", ""),
        grade=grade, reasons=hard + soft,
        bpm=bpm, bpm_confidence=conf,
        key=getattr(track, "key", "") or "", duration=dur,
    )


def triage(tracks: List) -> List[GridReport]:
    """Grade every track, worst first so problems are visible at the top."""
    order = {"reject": 0, "usable": 1, "good": 2}
    return sorted((grid_quality(t) for t in tracks),
                  key=lambda r: (order[r.grade], -len(r.reasons)))


# ── Gap report ───────────────────────────────────────────────────────────────

# These mirror `mixer.choose_transition_style`'s gates. They are duplicated
# rather than imported so this module stays light, and pinned by a test that
# fails if the mixer's thresholds move.
BLEND_MAX_ENERGY = 0.45
SWAP_MIN_ENERGY  = 0.70


@dataclass
class GapReport:
    n_tracks: int
    total_hours: float
    target_hours: float
    beatmatchable_frac: float
    bpm_clusters: List[tuple]          # (centre_bpm, n_tracks)
    camelot_coverage: int              # distinct keys present (of 24)
    low_energy_tracks: int             # have a cue under BLEND_MAX_ENERGY
    high_energy_tracks: int            # have a cue over SWAP_MIN_ENERGY
    style_counts: dict                 # style name -> reachable pair count
    findings: List[str] = field(default_factory=list)


def _beatmatchable(bpm_a: float, bpm_b: float, max_stretch: float) -> bool:
    """Exactly `plan_transition`'s test: nearest of direct/half/double."""
    if bpm_a <= 0 or bpm_b <= 0:
        return False
    ratios = [bpm_a / bpm_b, bpm_a / (bpm_b * 2.0), bpm_a / (bpm_b / 2.0)]
    return abs(min(ratios, key=lambda r: abs(r - 1.0)) - 1.0) <= max_stretch


def _cluster_bpms(bpms: List[float], width: float = 4.0) -> List[tuple]:
    """Greedy 1-D clustering: tracks within `width` BPM belong together."""
    clusters: List[tuple] = []
    for b in sorted(bpms):
        if clusters and b - clusters[-1][0] <= width:
            centre, n = clusters[-1]
            clusters[-1] = ((centre * n + b) / (n + 1), n + 1)
        else:
            clusters.append((b, 1))
    return sorted(((round(c, 1), n) for c, n in clusters),
                  key=lambda x: -x[1])


def library_gaps(tracks: List, target_hours: float = 24.0,
                 max_pairs: int = 4000) -> GapReport:
    """
    What is this library missing? Reports against the engine's real gates, so
    every finding maps to a behaviour you would or wouldn't hear.

    `max_pairs` bounds the pairwise style simulation: it samples evenly rather
    than running every ordered pair on a large library.
    """
    from .mixer import MAX_STRETCH, choose_transition_style

    n = len(tracks)
    total_h = sum(float(t.duration or 0.0) for t in tracks) / 3600.0
    bpms = [float(t.bpm) for t in tracks if getattr(t, "bpm", 0)]

    # Beatmatchable fraction over ordered pairs (direction matters: the
    # outgoing track sets the reference tempo).
    pairs = [(a, b) for i, a in enumerate(tracks) for j, b in enumerate(tracks)
             if i != j]
    step = max(1, len(pairs) // max_pairs)
    sampled = pairs[::step]
    n_bm = sum(1 for a, b in sampled if _beatmatchable(a.bpm, b.bpm, MAX_STRETCH))
    bm_frac = n_bm / len(sampled) if sampled else 0.0

    # Style reachability: call the real chooser so this can't drift from it.
    style_counts: dict = {}
    for a, b in sampled:
        out_c = _strongest(a, "out")
        in_c  = _strongest(b, "in")
        if out_c is None or in_c is None:
            continue
        bm = _beatmatchable(a.bpm, b.bpm, MAX_STRETCH)
        name = choose_transition_style(out_c, in_c, bm).name
        style_counts[name] = style_counts.get(name, 0) + 1

    lo = sum(1 for t in tracks
             if any(c.energy < BLEND_MAX_ENERGY for c in (t.cue_points or [])))
    hi = sum(1 for t in tracks
             if any(c.energy > SWAP_MIN_ENERGY for c in (t.cue_points or [])))
    keys = {t.key for t in tracks if getattr(t, "key", None)}
    clusters = _cluster_bpms(bpms)

    findings: List[str] = []
    if total_h < target_hours:
        need = target_hours - total_h
        avg_min = (total_h * 60 / n) if n else 5.5
        findings.append(
            f"{need:.1f}h short of {target_hours:.0f}h "
            f"(~{int(need * 60 / max(avg_min, 0.1))} more tracks at {avg_min:.1f} min avg)")
    if bm_frac < 0.30:
        findings.append(
            f"only {bm_frac:.0%} of pairs beatmatch — most transitions will be "
            f"hard cuts; add tracks near {clusters[0][0]:.0f} BPM" if clusters
            else f"only {bm_frac:.0%} of pairs beatmatch")
    for style in ("blend", "swap", "fade", "build"):
        if style_counts.get(style, 0) == 0:
            findings.append(f"style '{style}' is unreachable — no pair triggers it")
    if lo < max(3, n // 10):
        findings.append(
            f"only {lo} tracks have a cue below {BLEND_MAX_ENERGY} energy — "
            "long smooth blends will be rare")
    if hi < max(3, n // 10):
        findings.append(
            f"only {hi} tracks have a cue above {SWAP_MIN_ENERGY} energy — "
            "drop-to-drop swaps will be rare")
    if len(keys) < 12:
        findings.append(
            f"{len(keys)}/24 Camelot keys present — the harmonic gate has "
            "little to choose from")

    return GapReport(
        n_tracks=n, total_hours=round(total_h, 2), target_hours=target_hours,
        beatmatchable_frac=round(bm_frac, 3), bpm_clusters=clusters[:6],
        camelot_coverage=len(keys), low_energy_tracks=lo, high_energy_tracks=hi,
        style_counts=style_counts, findings=findings,
    )


def _strongest(track, cue_type: str):
    """Highest-confidence cue of a type, which is what the mixer prefers."""
    cues = [c for c in (getattr(track, "cue_points", None) or [])
            if c.type == cue_type]
    return max(cues, key=lambda c: c.confidence) if cues else None
