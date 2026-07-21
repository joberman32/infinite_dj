"""
Data models for the Infinite DJ analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class CuePoint:
    """A detected transition point within a track."""
    timestamp: float          # seconds from start
    type: str                 # "in", "out", "loop"
    phrase_aligned: bool      # whether this lands on a phrase boundary
    energy: float             # normalized RMS energy at this point (0-1)
    confidence: float         # composite score from cue detector (0-1)

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Section:
    """A detected structural segment of a track."""
    start: float
    end: float
    label: str        # "intro", "build", "drop", "breakdown", "outro", "body"
    energy: float     # mean normalized RMS energy in this section

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class TrackMeta:
    """Complete analysis metadata for a single audio file."""

    # Identity
    file_path: str
    title: str
    duration: float           # seconds

    # Rhythm
    bpm: float
    bpm_confidence: float     # 0-1
    beats: List[float]        # timestamps of every beat
    downbeats: List[float]    # timestamps of every bar (beat 1)
    phrases: List[float]      # timestamps of every 8-bar phrase boundary

    # Harmony
    key: str                  # Camelot notation: e.g. "8B", "5A"
    key_name: str             # Human readable: e.g. "C major", "A minor"
    key_confidence: float     # 0-1

    # Energy
    energy_curve: List[float] # normalized RMS per second, length == int(duration)

    # Structure
    sections: List[Section]
    cue_points: List[CuePoint]

    # Meta
    analyzed_at: float        # unix timestamp of analysis

    def to_dict(self):
        return {
            **{k: v for k, v in self.__dict__.items()
               if k not in ("sections", "cue_points")},
            "sections": [s.to_dict() for s in self.sections],
            "cue_points": [c.to_dict() for c in self.cue_points],
        }

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        # Strip DB-only columns not in the dataclass
        for extra in ("id", "file_hash"):
            d.pop(extra, None)
        d["sections"] = [Section.from_dict(s) for s in json.loads(d["sections"])]
        d["cue_points"] = [CuePoint.from_dict(c) for c in json.loads(d["cue_points"])]
        d["beats"] = json.loads(d["beats"])
        d["downbeats"] = json.loads(d["downbeats"])
        d["phrases"] = json.loads(d["phrases"])
        d["energy_curve"] = json.loads(d["energy_curve"])
        return cls(**d)
