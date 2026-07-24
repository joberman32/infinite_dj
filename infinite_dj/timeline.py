"""
Timeline export for the web player.

Turns a renderer's `clips` (per-track segments on the output timeline, produced by
`render_set` / `render_collage` in mixer.py) plus the track metadata into a compact
JSON the browser dashboard consumes. The web player computes a `PlayerState(t)`
from this — which clips are sounding at time t, any active crossfade, and what's
next — synced to `audio.currentTime`. The same shape can later be emitted live by
the real-time engine.
"""

import hashlib
import json
import os
import re
from typing import List

import numpy as np

from .models import TrackMeta


def _track_id(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()[:8]


def _color_for(file_path: str) -> str:
    """Stable, pleasant HSL colour per track (hue from a hash of its path)."""
    h = int(hashlib.md5(file_path.encode()).hexdigest(), 16)
    hue = h % 360
    return f"hsl({hue}, 62%, 58%)"


def _short_title(title: str) -> str:
    """Turn archive/catalog filenames into a concise, human-facing track name."""
    title = os.path.splitext(os.path.basename(title))[0]
    title = title.replace("_", " ")
    title = re.sub(r"\s*-\s*", " - ", title)
    title = re.sub(r"^[a-z][a-z0-9-]*\d+\s+\d+\s+", "", title,
                   flags=re.IGNORECASE)
    title = re.sub(r"^\d{1,3}\s+", "", title)
    # Prefer the song portion of "Artist - Track" and longer catalog strings.
    title = title.split(" - ")[-1]
    return re.sub(r"\s+", " ", title).strip()


def build_timeline(clips: List[dict], tracks: List[TrackMeta],
                   duration: float, sr: int = 44100) -> dict:
    """
    Build the timeline dict. `clips` reference tracks by file_path; the output
    references them by a short id and carries per-track metadata (title, bpm,
    key, energy, colour) for the dashboard. Embeddings are never included.
    """
    by_path = {t.file_path: t for t in tracks}

    tracks_out: dict = {}
    for c in clips:
        p = c["track"]
        tid = _track_id(p)
        if tid not in tracks_out:
            t = by_path.get(p)
            tracks_out[tid] = {
                "id": tid,
                "title": _short_title(t.title) if t else _short_title(c.get("title", p)),
                "bpm": round(t.bpm, 1) if t else c.get("bpm", 0.0),
                "key": t.key if t else c.get("key", ""),
                "key_name": t.key_name if t else "",
                "energy": round(float(np.mean(t.energy_curve)), 3)
                if (t and t.energy_curve) else 0.5,
                "color": _color_for(p),
            }

    clips_out = []
    for c in clips:
        clips_out.append({
            "track": _track_id(c["track"]),
            "start": c["out_start"],
            "end": c["out_end"],
            "fade_in": c.get("fade_in", 0.0),
            "fade_out": c.get("fade_out", 0.0),
            "mode": c.get("mode", ""),
            "section": c.get("section", ""),
            "bpm": c.get("bpm", 0.0),
            "key": c.get("key", ""),
        })
    clips_out.sort(key=lambda c: c["start"])

    return {
        "version": 1,
        "duration": round(float(duration), 3),
        "sr": sr,
        "tracks": tracks_out,
        "clips": clips_out,
    }


def write_timeline(path: str, clips: List[dict], tracks: List[TrackMeta],
                   duration: float, sr: int = 44100) -> dict:
    data = build_timeline(clips, tracks, duration, sr)
    with open(path, "w") as f:
        json.dump(data, f)
    return data
