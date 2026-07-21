"""Infinite DJ — auto-mixing engine for local music libraries."""

from .models import TrackMeta, CuePoint, Section
from .analyzer import analyze_track
from .db import TrackDB
from .harmony import camelot_compatibility, bpm_compatibility
from .mixer import (
    TransitionPlan, MixResult,
    render_transition, write_mix,
    best_cue_out, best_cue_in,
)
from .sequencer import (
    Sequence, CompatibilityEdge,
    build_compatibility_graph,
    sequence_greedy,
    sequence_energy_arc,
)
