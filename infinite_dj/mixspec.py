"""
Mix specification: library browsing + spec → render mapping.

The web setup pane lets a user pick a track pool and dial in two axes:

  * **Pace** — how long things play before moving on (a preset that sets
    min/max segment lengths, with an optional raw override).
  * **Serendipity** — how wild/nonlinear the mixing is. This is the master:
    it selects the renderer and the parameter ranges, and at ``insane`` it
    deliberately supersedes Pace (segment bounds are ignored).

`render_from_spec` is the single place those choices become renderer
arguments, so the mapping stays testable without touching audio.
"""

import os
import random
import re
from typing import List, Optional

from .models import TrackMeta
from .timeline import _short_title, _track_id

# Serendipity levels, least → most adventurous.
SERENDIPITY = ("low", "medium", "high", "insane")

# Pace presets → (min_seconds, max_seconds) a segment may play.
PACE_PRESETS = {
    "long-form": (90.0, 240.0),
    "flowing":   (45.0, 120.0),
    "shifting":  (25.0, 75.0),
    "choppy":    (12.0, 40.0),
}
DEFAULT_PACE = "flowing"

# Seconds per bar is tempo-dependent; ~2s/bar at 128 BPM is a good conversion
# for turning the Pace presets into the collage's bar-based bounds.
_SEC_PER_BAR = 1.875


# ── Library browsing ─────────────────────────────────────────────────────────

def _artist_album(track: TrackMeta) -> tuple:
    """
    Best-effort (artist, album) for a track.

    Libraries here come in two shapes: a folder named ``"Artist - Album"``, or
    a slug folder whose files carry ``"Artist - Album - NN Title"``. Try the
    folder first, then the filename, then fall back to the folder name.
    """
    folder = os.path.basename(os.path.dirname(track.file_path))
    if " - " in folder:
        artist, album = folder.split(" - ", 1)
        return artist.strip(), album.strip()

    stem = os.path.splitext(os.path.basename(track.file_path))[0]
    parts = [p.strip() for p in stem.split(" - ")]
    if len(parts) >= 3:
        return parts[0], parts[1]

    pretty = re.sub(r"[-_]+", " ", folder).strip()
    return (pretty or "Unknown artist"), (pretty or "Unknown album")


def library_groups(tracks: List[TrackMeta]) -> list:
    """
    Group tracks as artist → album → tracks for the setup pane. Track ids match
    the timeline's ids so a selection maps straight onto rendered clips.
    """
    by_artist: dict = {}
    for t in tracks:
        artist, album = _artist_album(t)
        albums = by_artist.setdefault(artist, {})
        albums.setdefault(album, []).append({
            "id": _track_id(t.file_path),
            "title": _short_title(t.title),
            "bpm": round(t.bpm, 1),
            "key": t.key,
            "duration": round(t.duration, 1),
        })

    out = []
    for artist in sorted(by_artist):
        albums = []
        for album in sorted(by_artist[artist]):
            items = sorted(by_artist[artist][album], key=lambda x: x["title"])
            albums.append({"album": album, "tracks": items})
        out.append({
            "artist": artist,
            "albums": albums,
            "count": sum(len(a["tracks"]) for a in albums),
        })
    return out


# ── Spec → render parameters ─────────────────────────────────────────────────

def _seg_bounds(spec: dict) -> tuple:
    """Resolve (min_sec, max_sec) from the Pace preset or a raw override."""
    preset = spec.get("pace") or DEFAULT_PACE
    lo, hi = PACE_PRESETS.get(preset, PACE_PRESETS[DEFAULT_PACE])
    raw_min, raw_max = spec.get("min_sec"), spec.get("max_sec")
    if raw_min is not None and raw_max is not None:
        try:
            lo, hi = float(raw_min), float(raw_max)
        except (TypeError, ValueError):
            pass
    lo = max(4.0, lo)
    hi = max(lo + 4.0, hi)
    return lo, hi


def resolve_params(spec: dict) -> dict:
    """
    Turn a UI spec into concrete renderer arguments (no audio touched).

    Returns ``{"renderer": "set"|"collage", "kwargs": {...}, "seg": (lo, hi)}``.
    Kept separate from `render_from_spec` so the mapping is unit-testable.
    """
    level = (spec.get("serendipity") or "medium").lower()
    if level not in SERENDIPITY:
        level = "medium"
    length_sec = float(spec.get("length_min") or 10.0) * 60.0
    lo, hi = _seg_bounds(spec)

    if level == "low":
        # Full-set feel: whole tracks, long solos, standard crossfades. The
        # length cap applies (render_set stops once filled), and the minimum
        # solo scales down for short mixes so a 3-minute request doesn't
        # overshoot to 8 (each track otherwise holds the 32-bar default).
        solo_bars = int(min(32, max(8, length_sec / 4 / _SEC_PER_BAR)))
        return {"renderer": "set", "seg": (lo, hi),
                "kwargs": {"n_mix_bars": 16, "target_length_sec": length_sec,
                           "min_solo_bars": solo_bars}}

    if level == "medium":
        # Sequential splices bounded by Pace.
        return {"renderer": "set", "seg": (lo, hi),
                "kwargs": {"min_seg_sec": lo, "max_seg_sec": hi,
                           "target_length_sec": length_sec}}

    if level == "high":
        # Structured collage; segment bounds still follow Pace.
        return {"renderer": "collage", "seg": (lo, hi),
                "kwargs": {"target_length_sec": length_sec, "layers": 3,
                           "min_seg_bars": max(2, int(lo / _SEC_PER_BAR)),
                           "max_seg_bars": max(4, int(hi / _SEC_PER_BAR)),
                           "seed": spec.get("seed")}}

    # insane — Pace is deliberately ignored: sub-segment freely, layer heavily,
    # and pick a fresh seed every time so no two mixes are alike.
    return {"renderer": "collage", "seg": None,
            "kwargs": {"target_length_sec": length_sec, "layers": 4,
                       "min_seg_bars": 2, "max_seg_bars": 32,
                       "seed": spec.get("seed", random.randint(0, 10 ** 6))}}


def select_tracks(spec: dict, tracks: List[TrackMeta]) -> List[TrackMeta]:
    """Filter the library to the spec's selected track ids (all if unset)."""
    ids = spec.get("track_ids")
    if not ids:
        return list(tracks)
    wanted = set(ids)
    return [t for t in tracks if _track_id(t.file_path) in wanted]


def render_from_spec(spec: dict, tracks: List[TrackMeta]) -> tuple:
    """
    Render a mix from a UI spec. Returns ``(audio, sr, clips, pool)`` where
    `pool` is the tracks actually used (needed to build the timeline).
    """
    from .mixer import render_collage, render_set
    from .sequencer import sequence_for_mixing

    pool = select_tracks(spec, tracks)
    if len(pool) < 2:
        raise ValueError("Select at least 2 tracks to build a mix.")

    plan = resolve_params(spec)
    kwargs = plan["kwargs"]

    if plan["renderer"] == "collage":
        audio, sr, _markers, clips = render_collage(pool, **kwargs)
    else:
        # render_set needs an ordered sequence; a splice may revisit tracks.
        splicing = "max_seg_sec" in kwargs
        if splicing:
            lo, hi = plan["seg"]
            n = max(2, int(kwargs["target_length_sec"] / max(1.0, (lo + hi) / 2)) + 4)
            seq = sequence_for_mixing(pool, arc="steady", n_tracks=n,
                                      allow_repeats=True,
                                      cooldown=min(4, len(pool) - 1))
        else:
            seq = sequence_for_mixing(pool, arc="peak", n_tracks=len(pool))
        audio, sr, _markers, clips = render_set(seq.tracks, **kwargs)

    return audio, sr, clips, pool
