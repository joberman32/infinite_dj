"""
Track sequencer — Phase 2.

Builds a weighted compatibility graph over all analyzed tracks and
finds an ordered sequence for mixing. Supports:
  - Greedy: always pick the best next track
  - Energy arc: shape the set toward a target energy curve
  - Avoid repeats: configurable cooldown window
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from .models import TrackMeta, CuePoint
from .harmony import camelot_compatibility, bpm_compatibility, pitch_shift_for_compatibility

# Docked from a key-sync shift's improved harmonic score so a naturally
# compatible pair still outranks a shifted one at the same nominal tier —
# small enough that it never erases the improvement itself: the smallest gap
# between any two camelot_compatibility tiers is 0.1 (0.5 -> 0.6), and the
# maximum shift budget is 3 semitones, so the worst case (0.09 docked) still
# nets a gain. See tests/test_key_shift.py.
KEY_SHIFT_PENALTY_PER_SEMITONE = 0.03


def cue_cosine_similarity(c1: CuePoint, c2: CuePoint) -> Optional[float]:
    """
    Cosine similarity between two cue points' timbre vectors.

    Prefers the 512D CLAP `.embedding` when both cues have one; falls back to
    the pooled-MFCC `.timbre` vector otherwise (populated on every analysis —
    no extra install required, see infinite_dj/embeddings.py). The two vector
    kinds are never mixed: if one cue only has CLAP and the other only has
    MFCC, that's treated the same as neither having a vector, because cosine
    similarity across two different embedding spaces is not a meaningful
    number even though `np.dot` would happily compute one.
    """
    if not c1 or not c2:
        return None
    if c1.embedding and c2.embedding:
        v1 = np.array(c1.embedding, dtype=np.float32)
        v2 = np.array(c2.embedding, dtype=np.float32)
    elif c1.timbre and c2.timbre:
        v1 = np.array(c1.timbre, dtype=np.float32)
        v2 = np.array(c2.timbre, dtype=np.float32)
    else:
        return None
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    sim = float(np.dot(v1, v2) / (norm1 * norm2))
    return round(float(np.clip(sim, -1.0, 1.0)), 4)


def find_best_cue_pair(
    track_out: TrackMeta,
    track_in: TrackMeta
) -> tuple[Optional[CuePoint], Optional[CuePoint], float]:
    """
    Find the optimal (cue_out, cue_in) pair between two tracks.
    Uses cue timbre similarity (CLAP, or pooled MFCC as fallback) if available,
    combined with phrase alignment & confidence.
    Returns (cue_out, cue_in, match_score).
    """
    outs = [c for c in track_out.cue_points if c.type == "out"]
    ins  = [c for c in track_in.cue_points if c.type == "in"]

    if not outs or not ins:
        c_out = outs[0] if outs else (track_out.cue_points[0] if track_out.cue_points else None)
        c_in  = ins[0]  if ins  else (track_in.cue_points[0]  if track_in.cue_points  else None)
        return c_out, c_in, 0.5

    best_pair = (outs[0], ins[0])
    best_score = -1.0

    for c_out in outs:
        for c_in in ins:
            sim = cue_cosine_similarity(c_out, c_in)
            if sim is not None:
                phrase_bonus = 0.2 if (c_out.phrase_aligned and c_in.phrase_aligned) else 0.0
                score = 0.6 * sim + 0.2 * (c_out.confidence + c_in.confidence) / 2.0 + phrase_bonus
            else:
                phrase_bonus = 0.3 if (c_out.phrase_aligned and c_in.phrase_aligned) else 0.0
                score = 0.7 * (c_out.confidence + c_in.confidence) / 2.0 + phrase_bonus

            if score > best_score:
                best_score = score
                best_pair = (c_out, c_in)

    return best_pair[0], best_pair[1], round(best_score, 3)


@dataclass
class CompatibilityEdge:
    track_a: str   # file_path
    track_b: str
    harmonic: float             # possibly key-sync-improved, see key_shift_semitones
    rhythmic: float
    score: float   # weighted composite
    cue_similarity: Optional[float] = None
    key_shift_semitones: Optional[float] = None  # set when `harmonic` assumes a pitch shift


@dataclass
class Sequence:
    tracks: List[TrackMeta]
    edges: List[CompatibilityEdge]   # edge[i] = transition from tracks[i] to tracks[i+1]
    total_duration: float

    def describe(self):
        print(f"\nSequence: {len(self.tracks)} tracks, {self.total_duration/60:.1f} min")
        print(f"{'─'*65}")
        for i, t in enumerate(self.tracks):
            dur = f"{int(t.duration//60)}:{int(t.duration%60):02d}"
            if i < len(self.edges):
                e = self.edges[i]
                sim_str = f" sim={e.cue_similarity:.2f}" if e.cue_similarity is not None else ""
                shift_str = (f" key{e.key_shift_semitones:+.0f}"
                            if e.key_shift_semitones else "")
                arrow = f"→ harm={e.harmonic:.2f}{shift_str} rhythm={e.rhythmic:.2f}{sim_str} score={e.score:.2f}"
            else:
                arrow = "(end)"
            print(f"  {i+1:>2}. [{t.key} {t.bpm:.0f}bpm {dur}] {t.title[:40]}")
            if arrow != "(end)":
                print(f"      {arrow}")


def build_compatibility_graph(
    tracks: List[TrackMeta],
    harm_weight: float = 0.6,
    rhythm_weight: float = 0.4,
) -> Dict[str, List[CompatibilityEdge]]:
    """
    Build a full directed compatibility graph.
    Each track -> list of edges to all other tracks, sorted by score descending.
    """
    graph: Dict[str, List[CompatibilityEdge]] = {t.file_path: [] for t in tracks}

    for i, a in enumerate(tracks):
        for j, b in enumerate(tracks):
            if i == j:
                continue
            harm   = camelot_compatibility(a.key, b.key)
            rhythm = bpm_compatibility(a.bpm, b.bpm)

            # Key-sync: b can be pitch-shifted onto a better Camelot match at
            # render time (mixer.plan_transition derives the exact same shift
            # from the same two keys, so this and the render never disagree).
            # Docked by KEY_SHIFT_PENALTY_PER_SEMITONE so an already-compatible
            # pair still outranks one that needs a shift to reach it.
            key_shift = None
            shift = pitch_shift_for_compatibility(a.key, b.key)
            if shift is not None:
                n_steps, _, shifted_score = shift
                penalized = shifted_score - KEY_SHIFT_PENALTY_PER_SEMITONE * abs(n_steps)
                if penalized > harm:
                    harm = penalized
                    key_shift = float(n_steps)

            c_out, c_in, _ = find_best_cue_pair(a, b)
            sim = cue_cosine_similarity(c_out, c_in) if (c_out and c_in) else None

            if sim is not None:
                score = 0.4 * harm + 0.3 * rhythm + 0.3 * max(0.0, sim)
            else:
                score = harm_weight * harm + rhythm_weight * rhythm

            graph[a.file_path].append(CompatibilityEdge(
                track_a=a.file_path,
                track_b=b.file_path,
                harmonic=round(harm, 3),
                rhythmic=round(rhythm, 3),
                score=round(score, 3),
                cue_similarity=sim,
                key_shift_semitones=key_shift,
            ))

        # Sort edges by score descending
        graph[a.file_path].sort(key=lambda e: -e.score)

    return graph



def sequence_greedy(
    tracks: List[TrackMeta],
    start: Optional[TrackMeta] = None,
    n_tracks: Optional[int] = None,
    min_score: Optional[float] = None,
    cooldown: int = 5,
    seed: Optional[int] = None,
) -> Sequence:
    """
    Greedy sequencer: always pick the highest-scoring unplayed next track.

    Args:
        tracks:     Full library of analyzed tracks.
        start:      Starting track (random if None).
        n_tracks:   How many tracks in the sequence (all if None).
        min_score:  Minimum compatibility score to accept a transition.
                    If no track meets this, picks the best available. Defaults to
                    the calibrated value (0.3 without a mined corpus) — real DJs
                    mix out of key considerably more than the Camelot ladder
                    implies, so a corpus is likely to loosen this.
        cooldown:   Don't revisit a track until this many tracks have played.
        seed:       Random seed for reproducibility.
    """
    if min_score is None:
        from .calibration import value as _cal
        min_score = float(_cal("harmonic_min_score"))
    if seed is not None:
        random.seed(seed)

    if not tracks:
        return Sequence(tracks=[], edges=[], total_duration=0)

    n = n_tracks or len(tracks)
    graph = build_compatibility_graph(tracks)
    track_map = {t.file_path: t for t in tracks}

    # Start track
    current = start or random.choice(tracks)
    sequence_tracks = [current]
    sequence_edges  = []
    recent = [current.file_path]   # cooldown window

    while len(sequence_tracks) < n:
        candidates = graph[current.file_path]

        # Filter out recently played tracks
        candidates = [e for e in candidates if e.track_b not in recent[-cooldown:]]

        if not candidates:
            break  # Exhausted options

        # Pick best candidate (or best above min_score)
        good = [e for e in candidates if e.score >= min_score]
        chosen_edge = good[0] if good else candidates[0]

        next_track = track_map[chosen_edge.track_b]
        sequence_tracks.append(next_track)
        sequence_edges.append(chosen_edge)
        recent.append(next_track.file_path)
        current = next_track

    total_duration = sum(t.duration for t in sequence_tracks)
    return Sequence(
        tracks=sequence_tracks,
        edges=sequence_edges,
        total_duration=total_duration,
    )


def _beatmatchable(a_bpm: float, b_bpm: float, max_stretch: float = 0.08) -> bool:
    """True if b can be beat-locked to a within the stretch budget (incl. half/double)."""
    ratios = [a_bpm / b_bpm, a_bpm / (b_bpm * 2.0), a_bpm / (b_bpm / 2.0)]
    return min(abs(r - 1.0) for r in ratios) <= max_stretch


def _percentile_ranker(values):
    """
    Return f(v) -> percentile of v within `values`, in [0, 1]. Cue-timbre
    similarities on a real library sit in a compressed band (e.g. 0.36–0.93, median ~0.77),
    so a raw threshold barely discriminates; ranking against the library's own
    distribution turns it into a usable per-library signal.
    """
    import bisect
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return lambda v: 0.5
    m = len(vals)
    return lambda v: 0.5 if v is None else bisect.bisect_right(vals, v) / m


def library_sim_threshold(tracks: List[TrackMeta], pct: float = 85.0) -> Optional[float]:
    """
    A per-library "high textural similarity" cutoff: the given percentile of all
    pairwise best-cue timbre similarities. None if the library has no embeddings
    (CLAP or MFCC) on its cue points.
    """
    sims = []
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            c_out, c_in, _ = find_best_cue_pair(a, b)
            s = cue_cosine_similarity(c_out, c_in)
            if s is not None:
                sims.append(s)
    if not sims:
        return None
    return float(np.percentile(sims, pct))


def sequence_for_mixing(
    tracks: List[TrackMeta],
    arc: str = "peak",
    n_tracks: Optional[int] = None,
    max_stretch: float = 0.08,
    seed: Optional[int] = None,
    allow_repeats: bool = False,
    cooldown: int = 4,
    stochastic: bool = False,
) -> Sequence:
    """
    Order tracks for a smooth mixed set: strongly prefer beat-matchable
    (tempo-compatible) neighbours so the render uses gentle blends rather than
    hard cuts, while still respecting harmony, energy arc and cue timbre
    (CLAP if installed, pooled MFCC otherwise).

    This is the sequencer `render-set` uses — it trades some energy-arc
    precision for far fewer jarring tempo cuts.

    With `allow_repeats=True` (splice mode) tracks may recur after a `cooldown`
    window and `n_tracks` may exceed the library size, so a long collage can be
    built from a small pool.
    """
    if seed is not None:
        random.seed(seed)
    if not tracks:
        return Sequence(tracks=[], edges=[], total_duration=0)

    n = n_tracks or len(tracks)
    graph = build_compatibility_graph(tracks)
    track_map = {t.file_path: t for t in tracks}

    # Per-library timbre ranker: turns the compressed similarity band into a
    # discriminating [0,1] signal for timbral track selection (None if no
    # embeddings at all, in which case timbre simply doesn't contribute).
    all_sims = [e.cue_similarity for edges in graph.values()
                for e in edges if e.cue_similarity is not None]
    sim_rank = _percentile_ranker(all_sims) if all_sims else None

    # Energy-arc target (same shapes as sequence_energy_arc)
    positions = np.linspace(0, 1, n)
    if arc == "build":
        target = 0.2 + positions * 0.8
    elif arc == "steady":
        target = np.full(n, 0.7)
    elif arc == "wave":
        target = np.clip(0.4 + 0.5 * np.sin(positions * 2 * np.pi), 0, 1)
    else:  # peak
        target = np.where(positions < 0.6,
                          0.3 + positions / 0.6 * 0.7,
                          1.0 - (positions - 0.6) / 0.4 * 0.6)

    def mean_energy(t):
        return float(np.mean(t.energy_curve)) if t.energy_curve else 0.5

    def energy_fit(t, pos):
        return max(0.0, 1.0 - abs(target[min(pos, n - 1)] - mean_energy(t)) * 2)

    # Start near the arc's opening energy
    current = max(tracks, key=lambda t: energy_fit(t, 0))
    seq_tracks, seq_edges = [current], []
    used = {current.file_path}

    while len(seq_tracks) < n:
        pos = len(seq_tracks)
        if allow_repeats:
            # Splice mode: exclude only the last `cooldown` tracks, so a long
            # collage can revisit the pool for variety.
            recent = {t.file_path for t in seq_tracks[-cooldown:]}
            candidates = [e for e in graph[current.file_path]
                          if e.track_b not in recent]
        else:
            # Full set: a permutation — prefer unused, repeat only if stuck.
            candidates = [e for e in graph[current.file_path]
                          if e.track_b not in used]
        if not candidates:
            candidates = graph[current.file_path]
        if not candidates:
            break

        def score(e):
            t = track_map[e.track_b]
            # Beatmatchability dominates so the render avoids cuts; harmony,
            # energy-arc fit, and cue-timbre similarity break ties.
            bm = 1.0 if _beatmatchable(current.bpm, t.bpm, max_stretch) else 0.0
            s = 2.0 * bm + e.harmonic + 0.5 * energy_fit(t, pos)
            if sim_rank is not None and e.cue_similarity is not None:
                s += 0.75 * sim_rank(e.cue_similarity)
            return s

        if stochastic and len(candidates) > 1:
            # Collage: wander the library — sample from the top candidates
            # weighted by score rather than always taking the argmax, so it
            # doesn't loop the same compatible cluster.
            ranked = sorted(candidates, key=score, reverse=True)[:5]
            weights = np.array([score(e) for e in ranked], dtype=float)
            weights = weights - weights.min() + 0.1
            chosen = ranked[int(np.random.choice(len(ranked), p=weights / weights.sum()))]
        else:
            chosen = max(candidates, key=score)
        nxt = track_map[chosen.track_b]
        seq_tracks.append(nxt)
        seq_edges.append(chosen)
        used.add(nxt.file_path)
        current = nxt

    return Sequence(tracks=seq_tracks, edges=seq_edges,
                    total_duration=sum(t.duration for t in seq_tracks))


def sequence_energy_arc(
    tracks: List[TrackMeta],
    arc: str = "peak",
    n_tracks: Optional[int] = None,
    seed: Optional[int] = None,
) -> Sequence:
    """
    Energy-arc sequencer: shapes the set toward a target energy narrative.

    arc options:
      "peak"      — build → peak → cool down (classic DJ set shape)
      "steady"    — consistent energy throughout
      "build"     — continuously building
      "wave"      — two peaks with a valley between

    Scores candidate tracks by both compatibility AND how well they
    fit the target energy at each position in the sequence.
    """
    if seed is not None:
        random.seed(seed)

    n = n_tracks or len(tracks)
    graph = build_compatibility_graph(tracks)
    track_map = {t.file_path: t for t in tracks}

    # Build target energy curve
    positions = np.linspace(0, 1, n)
    if arc == "peak":
        # Ramp up to 0.7, peak at 0.6 position, cool down
        target = np.where(
            positions < 0.6,
            0.3 + positions / 0.6 * 0.7,
            1.0 - (positions - 0.6) / 0.4 * 0.6
        )
    elif arc == "steady":
        target = np.full(n, 0.7)
    elif arc == "build":
        target = 0.2 + positions * 0.8
    elif arc == "wave":
        target = 0.4 + 0.5 * np.sin(positions * 2 * np.pi)
        target = np.clip(target, 0, 1)
    else:
        target = np.full(n, 0.7)

    def track_mean_energy(t: TrackMeta) -> float:
        if t.energy_curve:
            return float(np.mean(t.energy_curve))
        return 0.5

    def energy_score(t: TrackMeta, position: int) -> float:
        target_e = target[min(position, n - 1)]
        actual_e = track_mean_energy(t)
        return max(0, 1.0 - abs(target_e - actual_e) * 2)

    # Start with a track that fits position 0's energy target
    def start_score(t):
        return energy_score(t, 0)

    current = max(tracks, key=start_score)
    sequence_tracks = [current]
    sequence_edges  = []
    recent = [current.file_path]
    cooldown = 5

    while len(sequence_tracks) < n:
        pos = len(sequence_tracks)
        candidates = graph[current.file_path]
        candidates = [e for e in candidates if e.track_b not in recent[-cooldown:]]

        if not candidates:
            break

        # Combined score: compatibility + energy arc fit
        def combined_score(e):
            t = track_map[e.track_b]
            return 0.5 * e.score + 0.5 * energy_score(t, pos)

        chosen_edge = max(candidates, key=combined_score)
        next_track = track_map[chosen_edge.track_b]

        sequence_tracks.append(next_track)
        sequence_edges.append(chosen_edge)
        recent.append(next_track.file_path)
        current = next_track

    total_duration = sum(t.duration for t in sequence_tracks)
    return Sequence(
        tracks=sequence_tracks,
        edges=sequence_edges,
        total_duration=total_duration,
    )
