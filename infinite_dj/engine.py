"""
Real-time streaming engine — Phase 3.

Architecture:
  StreamEngine
  ├── _producer_thread  — decodes audio, applies crossfades, fills ring buffer
  ├── _scheduler_thread — lookahead: monitors position, queues transitions
  ├── _audio_callback   — sounddevice pulls from ring buffer (runs on audio thread)
  └── Console UI        — live display of track/cue/upcoming info

The producer and scheduler communicate via a TransitionQueue. When the
scheduler decides it's time to mix, it posts a TransitionEvent. The producer
picks it up and executes the EQ crossfade inline in the audio stream.

Mid-track arbitrary mixing: the scheduler continuously scans upcoming OUT
cue points in the current track and decides whether to fire early (before
the track ends naturally) based on:
  - Cue point confidence score
  - Compatibility of the next track
  - How long we've been in the current track (min/max dwell time)
"""

import os
import sys
import time
import threading
import numpy as np
import soundfile as sf
import librosa
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import warnings

warnings.filterwarnings("ignore")

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

try:
    import pyrubberband as rb
    HAS_RUBBERBAND = True
except ImportError:
    HAS_RUBBERBAND = False

from .models import TrackMeta, CuePoint
from .mixer import (
    _load_audio, _time_stretch, _apply_lowpass, _apply_highpass,
    _equal_power_fade, _apply_gain, _find_nearest_downbeat,
    _blend, _loudness_match, choose_transition_style,
    TransitionPlan, CrossfadeFilterState, MAX_STRETCH, MASTER_LOUDNESS,
    MIX_SR as SR
)
from .sequencer import build_compatibility_graph, sequence_energy_arc
from .harmony import camelot_compatibility, bpm_compatibility


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PlaybackState:
    track: Optional[TrackMeta] = None
    position: float = 0.0          # producer/render seconds into current track
    playback_position: float = 0.0 # seconds emitted to the output device/session
    underruns: int = 0
    is_mixing: bool = False
    mix_progress: float = 0.0      # 0-1 during an active crossfade
    next_track: Optional[TrackMeta] = None
    next_cue_in: Optional[CuePoint] = None
    scheduled_out: Optional[CuePoint] = None
    total_played: float = 0.0      # total seconds played this session
    tracks_played: int = 0


@dataclass
class TransitionEvent:
    """Posted by scheduler, consumed by producer."""
    incoming_track: TrackMeta
    cue_in: CuePoint
    cue_out: Optional[CuePoint] = None   # exit cue (for choosing a style)
    n_bars: int = 8
    trigger_immediately: bool = False  # True = fire now regardless of position


@dataclass
class PreparedIncoming:
    """Incoming audio prepared off the real-time producer path."""
    current_path: str
    incoming_track: TrackMeta
    cue_in: CuePoint
    native_audio: np.ndarray
    stretched_audio: np.ndarray
    stretched_start_frame: int
    ratio: float
    beatmatched: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

CHUNK_FRAMES = 4096    # frames per producer iteration (~93ms at 44.1kHz)
BUFFER_SECONDS = 8.0   # ring buffer size
BUFFER_FRAMES  = int(BUFFER_SECONDS * SR)
LOOKAHEAD_BARS = 16    # scheduler looks this many bars ahead for OUT cues
MIN_DWELL_BARS = 32    # minimum bars to play before mixing out (let tracks breathe)
MAX_DWELL_BARS = 96    # force transition after this many bars if none found


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _audible_track_position(render_position: float, queued_frames: int) -> float:
    """Convert the producer cursor to the listener's latency-compensated time."""
    return max(0.0, render_position - queued_frames / SR)


class AudioRingBuffer:
    """Single-producer/single-consumer fixed-capacity stereo ring buffer.

    The audio callback never allocates or waits for a mutex. CPython's GIL
    makes the index updates atomic for this single producer/consumer use; the
    producer is the only side that waits when the buffer is full.
    """
    def __init__(self, capacity_frames: int):
        self.capacity = capacity_frames
        self.data = np.zeros((capacity_frames, 2), dtype=np.float32)
        self.read_frame = 0
        self.write_frame = 0

    @property
    def available_frames(self) -> int:
        return self.write_frame - self.read_frame

    @property
    def free_frames(self) -> int:
        return self.capacity - self.available_frames

    def write(self, chunk: np.ndarray) -> bool:
        n = len(chunk)
        if n > self.free_frames:
            return False
        start = self.write_frame % self.capacity
        first = min(n, self.capacity - start)
        self.data[start:start + first] = chunk[:first]
        if first < n:
            self.data[:n - first] = chunk[first:]
        self.write_frame += n
        return True

    def read_into(self, out: np.ndarray, frames: int) -> int:
        """Read into callback-owned output and return real (non-silent) frames."""
        n = min(frames, self.available_frames)
        if n:
            start = self.read_frame % self.capacity
            first = min(n, self.capacity - start)
            out[:first] = self.data[start:start + first]
            if first < n:
                out[first:n] = self.data[:n - first]
            self.read_frame += n
        if n < frames:
            out[n:frames].fill(0.0)
        return n

    def discard(self, frames: int) -> int:
        """Advance the consumer without allocating an output buffer (headless)."""
        n = min(frames, self.available_frames)
        self.read_frame += n
        return n


def _pick_next_track(
    current: TrackMeta,
    library: List[TrackMeta],
    graph: dict,
    recently_played: List[str],
    cooldown: int = 4,
) -> Optional[TrackMeta]:
    """Pick the best next track given current compatibility and recency."""
    track_map = {t.file_path: t for t in library}
    candidates = graph.get(current.file_path, [])
    candidates = [e for e in candidates
                  if e.track_b not in recently_played[-cooldown:]]
    if not candidates:
        candidates = graph.get(current.file_path, [])  # relax cooldown
    if not candidates:
        return None
    return track_map.get(candidates[0].track_b)


def _pick_best_in_cue(track: TrackMeta, after: float = 0.0) -> Optional[CuePoint]:
    """Pick the highest-confidence IN cue point after a given timestamp."""
    ins = [c for c in track.cue_points if c.type == "in" and c.timestamp >= after]
    if not ins:
        # Fallback: first downbeat
        if track.downbeats:
            from .models import CuePoint as CP
            return CP(timestamp=track.downbeats[0], type="in",
                      phrase_aligned=True, energy=0.5, confidence=0.1)
        return None
    return max(ins, key=lambda c: c.confidence)


def _resolve_stretch(out_bpm: float, in_bpm: float,
                     max_stretch: float = MAX_STRETCH) -> tuple:
    """
    Least-stretch ratio (considering half/double time) to bring the incoming
    tempo to the outgoing one. Returns (ratio, beatmatched). If the best match
    still exceeds the budget, beatmatched is False and the caller should cut.
    """
    ratios = [out_bpm / in_bpm, out_bpm / (in_bpm * 2.0), out_bpm / (in_bpm / 2.0)]
    ratio = min(ratios, key=lambda r: abs(r - 1.0))
    if abs(ratio - 1.0) <= max_stretch:
        return ratio, True
    return 1.0, False


def _transition_start_time(
    event: TransitionEvent,
    current_track: TrackMeta,
    producer_position: float,
) -> float:
    """Resolve a transition event to an outgoing-track timeline position.

    Cue points produced by the scheduler are downbeats, so preserving their
    timestamp keeps the audible handoff phrase-aligned.  The fallback paths
    cover events without a cue (max-dwell) and explicit user skips.
    """
    if event.trigger_immediately:
        return producer_position
    if event.cue_out is not None and event.cue_out.timestamp >= producer_position:
        return event.cue_out.timestamp
    future = [d for d in current_track.downbeats if d >= producer_position]
    return future[0] if future else producer_position


def _build_crossfade_chunk(
    out_audio: np.ndarray,
    in_audio:  np.ndarray,
    phase,                # scalar or per-sample values from 0.0 to 1.0
    sr: int = SR,
    style=None,
    filter_state: Optional[CrossfadeFilterState] = None,
) -> np.ndarray:
    """
    Render one chunk of the crossfade using the shared, style-aware EQ blend
    (or a plain equal-power fade for cuts), so the real-time engine and the
    offline renderer sound identical.
    """
    n = min(len(out_audio), len(in_audio))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    o, ii = out_audio[:n], in_audio[:n]
    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim == 0:
        phase = np.full(n, float(phase), dtype=np.float32)
    else:
        phase = phase[:n]

    if style is not None and style.is_cut:
        p = phase.reshape(-1, 1)
        fo, fi = np.cos(p * np.pi / 2), np.sin(p * np.pi / 2)
        return _apply_gain((o * fo + ii * fi).astype(np.float32), 0.9)
    return _apply_gain(_blend(o, ii, phase, sr, style, filter_state=filter_state), 0.9)


# ── Stream Engine ─────────────────────────────────────────────────────────────

class StreamEngine:
    """
    Real-time streaming DJ engine.

    Usage:
        engine = StreamEngine(library, arc="peak")
        engine.start()          # begins playback
        engine.skip()           # force transition now
        engine.stop()           # clean shutdown
    """

    def __init__(
        self,
        library: List[TrackMeta],
        arc: str = "peak",
        on_track_change: Optional[Callable] = None,
        output_file: Optional[str] = None,  # if set, write to file instead of speakers
        max_duration: Optional[float] = None,  # stop after N seconds (for testing)
    ):
        if not library:
            raise ValueError("Library is empty")

        self.library    = library
        self.arc        = arc
        self.on_track_change = on_track_change
        self.output_file = output_file
        self.max_duration = max_duration

        # Compatibility graph (built once)
        self.graph      = build_compatibility_graph(library)
        self.track_map  = {t.file_path: t for t in library}

        # State
        self.state      = PlaybackState()
        self.running    = False
        self._lock      = threading.Lock()

        # Fixed, preallocated SPSC ring.  The callback reads from it without a
        # mutex or temporary allocations; only the producer waits for capacity.
        self._ring = AudioRingBuffer(BUFFER_FRAMES + CHUNK_FRAMES)

        # Transition queue: producer reads this
        self._transition_queue: Optional[TransitionEvent] = None
        self._transition_lock  = threading.Lock()

        # Incoming tracks are decoded and stretched by a background worker,
        # never by the producer thread that has to keep the audio buffer full.
        self._prepare_lock = threading.Lock()
        self._prepared_incoming: Optional[PreparedIncoming] = None
        self._preparing_key: Optional[tuple] = None
        self._preparation_thread = None

        # Recently played track paths (for cooldown)
        self._recently_played: List[str] = []

        # Threads
        self._producer_thread  = None
        self._scheduler_thread = None
        self._audio_stream     = None

        # File output mode
        self._output_chunks: List[np.ndarray] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, first_track: Optional[TrackMeta] = None):
        """Start the engine. Blocks until stop() is called."""
        self.running = True

        # Pick starting track
        if first_track:
            self.state.track = first_track
        else:
            seq = sequence_energy_arc(self.library, arc=self.arc, n_tracks=1)
            self.state.track = seq.tracks[0] if seq.tracks else self.library[0]

        print(f"\n{'═'*60}")
        print(f"  INFINITE DJ  —  {len(self.library)} tracks  —  arc: {self.arc}")
        print(f"{'═'*60}\n")

        # Pre-load first track
        self._load_current_track()

        # Start threads
        self._producer_thread  = threading.Thread(target=self._producer, daemon=True)
        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._producer_thread.start()
        self._scheduler_thread.start()

        # Start audio output
        if HAS_AUDIO and self.output_file is None:
            self._start_audio_stream()
            try:
                while self.running:
                    time.sleep(0.1)
                    self._render_ui()
                    if self.max_duration and self.state.total_played >= self.max_duration:
                        self.stop()
            except KeyboardInterrupt:
                self.stop()
        else:
            # Headless / file-output mode: run until done
            self._run_headless()

    def skip(self):
        """Force an immediate transition to the next track."""
        with self._transition_lock:
            if self.state.next_track:
                evt = TransitionEvent(
                    incoming_track=self.state.next_track,
                    cue_in=self.state.next_cue_in or _pick_best_in_cue(self.state.next_track),
                    trigger_immediately=True,
                )
                self._transition_queue = evt
                print("\n  [SKIP] Forcing transition...")

    def stop(self):
        """Clean shutdown."""
        self.running = False
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
        if self.output_file and self._output_chunks:
            self._write_output_file()
        print("\n  [STOPPED]")

    # ── Internal state ────────────────────────────────────────────────────────

    def _load_matched(self, t: TrackMeta) -> np.ndarray:
        """Load a track's audio, normalized to the shared set loudness."""
        audio, _ = _load_audio(t.file_path, SR)
        return _loudness_match(audio, t.loudness, MASTER_LOUDNESS)

    def _load_current_track(self):
        """Load current track audio into memory."""
        t = self.state.track
        if not t:
            return
        print(f"  ▶ Loading: {t.title} [{t.key}, {t.bpm:.0f} BPM]")
        self._current_audio = self._load_matched(t)
        self._current_pos_frame = 0
        self._recently_played.append(t.file_path)
        self.state.position = 0.0
        self.state.tracks_played += 1
        if self.on_track_change:
            self.on_track_change(t)

    def _prepare_key(self, current: TrackMeta, incoming: TrackMeta,
                     cue_in: CuePoint) -> tuple:
        return (current.file_path, incoming.file_path, cue_in.timestamp)

    def _request_incoming_prepare(self, current: TrackMeta, incoming: TrackMeta,
                                  cue_in: Optional[CuePoint]):
        """Begin decoding/stretching the selected next track in the background."""
        if cue_in is None:
            return
        key = self._prepare_key(current, incoming, cue_in)
        with self._prepare_lock:
            if (self._prepared_incoming is not None
                    and self._prepare_key(current, self._prepared_incoming.incoming_track,
                                          self._prepared_incoming.cue_in) == key):
                return
            if self._preparing_key == key:
                return
            # Avoid competing full-track Rubber Band jobs. The scheduler normally
            # chooses one next track per current track, so this is only relevant
            # for an explicit skip while a stale preparation is still finishing.
            if self._preparation_thread is not None and self._preparation_thread.is_alive():
                return
            self._preparing_key = key

        def worker():
            try:
                native = self._load_matched(incoming)
                ratio, beatmatched = _resolve_stretch(current.bpm, incoming.bpm)
                if beatmatched and abs(ratio - 1.0) > 0.001:
                    stretched = _time_stretch(native, SR, ratio)
                    stretched_downbeats = [d / ratio for d in incoming.downbeats]
                else:
                    ratio = 1.0
                    stretched = native
                    stretched_downbeats = list(incoming.downbeats)

                in_bar = (60.0 / incoming.bpm) * 4
                aligned_t = _find_nearest_downbeat(
                    cue_in.timestamp / ratio, stretched_downbeats,
                    max_offset=in_bar / 2.0,
                )
                prepared = PreparedIncoming(
                    current_path=current.file_path,
                    incoming_track=incoming,
                    cue_in=cue_in,
                    native_audio=native,
                    stretched_audio=stretched,
                    stretched_start_frame=int(aligned_t * SR),
                    ratio=ratio,
                    beatmatched=beatmatched,
                )
                with self._prepare_lock:
                    if self._preparing_key == key:
                        self._prepared_incoming = prepared
            except Exception as exc:
                print(f"\n  [PREP ERROR] {incoming.title}: {exc}")
            finally:
                with self._prepare_lock:
                    if self._preparing_key == key:
                        self._preparing_key = None

        self._preparation_thread = threading.Thread(
            target=worker, name="infinite-dj-prep", daemon=True
        )
        self._preparation_thread.start()

    def _prepared_for(self, current: TrackMeta, event: TransitionEvent) -> Optional[PreparedIncoming]:
        """Return prepared audio only when it belongs to this exact handoff."""
        with self._prepare_lock:
            prepared = self._prepared_incoming
            if prepared is None or event.cue_in is None:
                return None
            if (prepared.current_path == current.file_path
                    and prepared.incoming_track.file_path == event.incoming_track.file_path
                    and prepared.cue_in.timestamp == event.cue_in.timestamp):
                return prepared
        return None

    # ── Producer thread ───────────────────────────────────────────────────────

    def _producer(self):
        """
        Continuously reads audio frames and pushes to the ring buffer.
        Executes crossfades when a TransitionEvent is pending.
        """
        current_audio = self._current_audio
        pos = 0  # frame position in current_audio

        in_audio        = None   # incoming (stretched-to-outgoing-tempo) audio
        in_native       = None   # incoming native audio, for post-mix handoff
        in_pos          = 0      # frame in in_audio
        xfade_total     = 0      # total crossfade frames
        xfade_done      = 0      # frames completed in crossfade
        active_ratio    = 1.0    # stretch applied to the incoming crossfade
        active_incoming = None
        active_style    = None   # TransitionStyle for the current crossfade
        active_filters  = None   # persistent EQ state during a crossfade
        pending         = None   # transition prepared but not yet started

        while self.running:
            # Check for a queued transition
            with self._transition_lock:
                evt = self._transition_queue
                if evt:
                    self._transition_queue = None

            # Turn a ready background preparation into a pending transition.  Do
            # not decode or stretch here: blocking this thread risks draining the
            # audio ring buffer and causing an underrun.
            if evt is not None and not self.state.is_mixing and pending is None:
                prepared = self._prepared_for(self.state.track, evt)
                if prepared is None:
                    # Leave the event pending while normal playback continues.
                    # The preloader was started when the scheduler selected this
                    # track, and a late preparation safely falls back to a future
                    # downbeat through _transition_start_time.
                    self._request_incoming_prepare(
                        self.state.track, evt.incoming_track, evt.cue_in
                    )
                    with self._transition_lock:
                        if self._transition_queue is None:
                            self._transition_queue = evt
                else:
                    inc = prepared.incoming_track
                    style = choose_transition_style(
                        evt.cue_out, evt.cue_in, prepared.beatmatched
                    )
                    if prepared.beatmatched and abs(prepared.ratio - 1.0) > 0.001:
                        print(f"\n  ⟶ Transition [{style.name}] to: {inc.title} "
                              f"({(prepared.ratio-1.0)*100:+.1f}% tempo)")
                    elif not prepared.beatmatched:
                        print(f"\n  ⟶ Transition [cut] to: {inc.title} (tempos too far apart)")

                    bar_frames = int((60.0 / self.state.track.bpm) * 4 * SR)
                    xfade_total_frames = (int(style.cut_seconds * SR) if style.is_cut
                                          else bar_frames * style.n_bars)
                    now_t = pos / SR
                    start_t = _transition_start_time(evt, self.state.track, now_t)
                    pending = {
                        "incoming":     inc,
                        "in_stretched": prepared.stretched_audio,
                        "in_native":    prepared.native_audio,
                        "in_pos":       prepared.stretched_start_frame,
                        "ratio":        prepared.ratio,
                        "style":        style,
                        "xfade_total":  max(1, xfade_total_frames),
                        "start_frame":  int(round(start_t * SR)),
                    }

            # Activate the prepared transition once we reach the downbeat
            if (pending is not None and not self.state.is_mixing
                    and pos >= pending["start_frame"]):
                in_audio        = pending["in_stretched"]
                in_native       = pending["in_native"]
                in_pos          = pending["in_pos"]
                xfade_total     = pending["xfade_total"]
                xfade_done      = 0
                active_ratio    = pending["ratio"]
                active_incoming = pending["incoming"]
                active_style    = pending["style"]
                active_filters  = (None if active_style.is_cut
                                   else CrossfadeFilterState.create(SR))
                self.state.is_mixing  = True
                self.state.next_track = active_incoming
                pending = None

            # Generate next chunk
            chunk_size = CHUNK_FRAMES

            if self.state.is_mixing and in_audio is not None:
                # In crossfade: blend out_chunk and in_chunk
                out_end = min(pos + chunk_size, len(current_audio))
                in_end  = min(in_pos + chunk_size, len(in_audio))
                actual  = min(out_end - pos, in_end - in_pos, chunk_size)

                if actual <= 0:
                    # Ran out — hand off to incoming at native tempo
                    current_audio, pos = self._handoff_native(
                        in_native, in_pos, active_ratio, active_incoming)
                    in_audio = in_native = None
                    self.state.is_mixing = False
                    continue

                out_chunk = current_audio[pos:pos + actual]
                in_chunk  = in_audio[in_pos:in_pos + actual]
                # A per-sample phase ramp avoids the audible 93 ms gain steps
                # caused by applying one scalar phase to the whole chunk.
                phase = np.minimum(
                    1.0,
                    (xfade_done + np.arange(actual, dtype=np.float32))
                    / max(1, xfade_total - 1),
                )

                chunk = _build_crossfade_chunk(
                    out_chunk, in_chunk, phase, SR, active_style, active_filters
                )
                xfade_done += actual
                in_pos     += actual
                pos        += actual

                self.state.mix_progress = phase

                if xfade_done >= xfade_total:
                    # Crossfade complete — continue incoming at its native tempo
                    current_audio, pos = self._handoff_native(
                        in_native, in_pos, active_ratio, active_incoming)
                    in_audio = in_native = None
                    active_filters = None
                    self.state.is_mixing = False
            else:
                # Normal playback
                end = min(pos + chunk_size, len(current_audio))
                if end <= pos:
                    # Track exhausted — wait for scheduler or pick next
                    current_audio, pos = self._handle_track_end()
                    continue
                chunk = current_audio[pos:end]
                pos = end

            # Update playback position
            self.state.position = pos / SR

            # Push to buffer
            self._push_buffer(chunk)

            # Throttle only on the producer side; the callback never blocks.
            while self._ring.free_frames < CHUNK_FRAMES and self.running:
                time.sleep(0.02)

    def _push_buffer(self, chunk: np.ndarray):
        while not self._ring.write(chunk) and self.running:
            time.sleep(0.002)
        if self.output_file is not None:
            self._output_chunks.append(chunk.copy())

    def _pop_buffer(self, n_frames: int) -> np.ndarray:
        """Pull frames for non-real-time consumers such as headless output."""
        out = np.zeros((n_frames, 2), dtype=np.float32)
        actual = self._ring.read_into(out, n_frames)
        self._record_playback(actual, actual)
        return out

    def _record_playback(self, requested_frames: int, actual_frames: int):
        """Advance the audible/session clock from the consumer side only."""
        self.state.playback_position += requested_frames / SR
        self.state.total_played = self.state.playback_position
        if actual_frames < requested_frames:
            self.state.underruns += 1

    def _handoff_native(self, in_native, in_pos, ratio, incoming):
        """
        After a crossfade completes, continue the incoming track at its own
        native tempo (the crossfade region was stretched to the outgoing tempo).
        Returns (current_audio, pos) for the producer to resume from.
        """
        self._finish_transition(incoming)
        native_pos = int(round(in_pos * ratio))   # stretched frame → native frame
        return in_native, native_pos

    def _finish_transition(self, incoming: TrackMeta):
        """Called when a crossfade completes."""
        self.state.track        = incoming
        self.state.next_track   = None
        self.state.next_cue_in  = None
        self.state.scheduled_out = None
        self._recently_played.append(incoming.file_path)
        self.state.tracks_played += 1
        if self.on_track_change:
            self.on_track_change(incoming)
        print(f"\n  ✓ Now playing: {incoming.title} [{incoming.key}, {incoming.bpm:.0f} BPM]")

    def _handle_track_end(self) -> tuple[np.ndarray, int]:
        """Track ran out with no scheduled transition. Pick next immediately."""
        current = self.state.track
        next_t = _pick_next_track(
            current, self.library, self.graph, self._recently_played
        )
        if not next_t:
            next_t = self.library[0]  # fallback: restart library

        cue_in = _pick_best_in_cue(next_t)
        print(f"\n  ↩ Track ended, jumping to: {next_t.title}")

        event = TransitionEvent(next_t, cue_in) if cue_in else None
        prepared = self._prepared_for(current, event) if event else None
        if prepared is not None:
            native_pos = int(round(prepared.stretched_start_frame * prepared.ratio))
            self._current_audio = prepared.native_audio
            self._finish_transition(next_t)
            self.state.position = native_pos / SR
            return self._current_audio, native_pos

        # This should only occur when no candidate could be preloaded (for
        # example, a preparation error). Keep the engine alive, but the normal
        # scheduler path above never performs this I/O on the producer thread.
        self._current_audio = self._load_matched(next_t)
        entry_pos = int(round((cue_in.timestamp if cue_in else 0.0) * SR))
        self._finish_transition(next_t)
        self.state.position = entry_pos / SR
        return self._current_audio, entry_pos

    # ── Scheduler thread ──────────────────────────────────────────────────────

    def _scheduler(self):
        """
        Lookahead scheduler. Runs every ~500ms.

        Each tick:
          1. Pre-select the next track (always have one ready)
          2. Look at upcoming OUT cue points in the current track
          3. If a high-confidence OUT point is coming within LOOKAHEAD_BARS,
             and we've been playing long enough (MIN_DWELL_BARS), schedule
             a transition
          4. Also enforce MAX_DWELL_BARS as a hard cap
        """
        while self.running:
            time.sleep(0.5)

            if not self.state.track or self.state.is_mixing:
                continue

            current   = self.state.track
            # The producer intentionally runs ahead of the listener by the
            # ring-buffer latency. Use audible time for dwell and cue policy;
            # the producer still maps the chosen cue back to its render cursor
            # when it writes the actual transition frames.
            pos = _audible_track_position(
                self.state.position, self._ring.available_frames
            )
            bar_dur   = (60.0 / current.bpm) * 4
            bars_played = pos / bar_dur

            # 1. Pre-select next track if we don't have one
            if self.state.next_track is None:
                next_t = _pick_next_track(
                    current, self.library, self.graph, self._recently_played
                )
                if next_t:
                    cue_in = _pick_best_in_cue(next_t)
                    with self._lock:
                        self.state.next_track  = next_t
                        self.state.next_cue_in = cue_in
                    self._request_incoming_prepare(current, next_t, cue_in)
                    print(f"\n  ⏭  Up next: {next_t.title} [{next_t.key}]")

            # 2. Check upcoming OUT cue points
            lookahead_window = bar_dur * LOOKAHEAD_BARS
            upcoming_outs = [
                c for c in current.cue_points
                if c.type == "out"
                and pos < c.timestamp < pos + lookahead_window
                and c.confidence > 0.25
            ]

            if not upcoming_outs:
                # No upcoming cue — check hard cap
                if bars_played >= MAX_DWELL_BARS and self.state.next_track:
                    self._fire_transition("max dwell reached")
                continue

            # 3. Find the best upcoming cue (highest confidence)
            best_out = max(upcoming_outs, key=lambda c: c.confidence)
            time_to_out = best_out.timestamp - pos

            # 4. Decide whether to fire
            should_fire = (
                bars_played >= MIN_DWELL_BARS          # played long enough
                and time_to_out <= bar_dur * 8         # within 8 bars
                and self.state.next_track is not None
            )

            if should_fire:
                with self._lock:
                    self.state.scheduled_out = best_out
                self._fire_transition(
                    f"OUT cue at {_fmt_time(best_out.timestamp)} "
                    f"(conf={best_out.confidence:.2f})"
                )

    def _fire_transition(self, reason: str):
        """Post a TransitionEvent to the producer."""
        if self.state.is_mixing or self._transition_queue is not None:
            return  # already mixing
        if not self.state.next_track or not self.state.next_cue_in:
            return

        print(f"\n  ⚡ Transition fired: {reason}")
        evt = TransitionEvent(
            incoming_track=self.state.next_track,
            cue_in=self.state.next_cue_in,
            cue_out=self.state.scheduled_out,
            n_bars=8,
        )
        with self._transition_lock:
            self._transition_queue = evt

    # ── Audio output ──────────────────────────────────────────────────────────

    def _audio_callback(self, outdata, frames, time_info, status):
        actual = self._ring.read_into(outdata, frames)
        self._record_playback(frames, actual)
        if status:
            self.state.underruns += 1

    def _start_audio_stream(self):
        try:
            self._audio_stream = sd.OutputStream(
                samplerate=SR,
                channels=2,
                dtype='float32',
                blocksize=CHUNK_FRAMES,
                callback=self._audio_callback,
            )
            self._audio_stream.start()
        except Exception as e:
            print(f"  Audio stream error: {e}")
            print("  (Running in headless mode)")

    def _run_headless(self):
        """For file output or testing — drain buffer into output chunks."""
        print(f"  [Headless mode — writing to {self.output_file or 'memory'}]")
        while self.running:
            time.sleep(0.1)
            # Drain the buffer
            while self._ring.available_frames > 0:
                self._pop_buffer(CHUNK_FRAMES)
            if self.max_duration and self.state.total_played >= self.max_duration:
                self.stop()
                break

    def _write_output_file(self):
        if not self._output_chunks:
            return
        import soundfile as sf
        audio = np.concatenate(self._output_chunks, axis=0)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (0.93 / peak)
        sf.write(self.output_file, audio, SR, subtype='PCM_16')
        dur = len(audio) / SR
        mb = os.path.getsize(self.output_file) / 1024 / 1024
        print(f"\n  Written: {self.output_file} ({dur:.1f}s, {mb:.1f} MB)")

    # ── Console UI ────────────────────────────────────────────────────────────

    def _render_ui(self):
        """Minimal in-place status line."""
        t = self.state
        if not t.track:
            return

        pos_str = _fmt_time(t.position)
        dur_str = _fmt_time(t.track.duration)
        buf_s   = self._ring.available_frames / SR

        mixing_str = ""
        if t.is_mixing:
            pct = int(t.mix_progress * 100)
            mixing_str = f"  [MIXING {pct}%]"

        next_str = ""
        if t.next_track:
            next_str = f"  ⏭ {t.next_track.title[:25]}"

        line = (
            f"\r  ▶ {t.track.title[:30]:<30}  "
            f"render={pos_str}/{dur_str}  "
            f"played={_fmt_time(t.playback_position)}  buf={buf_s:.1f}s"
            f"{mixing_str}{next_str}  "
        )
        sys.stdout.write(line)
        sys.stdout.flush()
