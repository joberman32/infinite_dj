"""
Radio mode: an endless, continuously-rendered mix.

A `RadioSession` primes a few seconds of audio so playback can start almost
immediately, then keeps a background thread extending the collage — writing
finalised audio out as fixed-length WAV chunks — until the listener exits.

It leans on `render_collage(..., state=...)`, which is resumable: the same dict
passed back in continues the mix, carrying the uncommitted tail and the sounding
layers across calls. Rendering runs tens of times faster than playback, so the
buffer races ahead of the listener within seconds of starting.
"""

import os
import threading
import time
from typing import List, Optional

import numpy as np
import soundfile as sf

from .mixer import render_collage

# How much audio each render call tries to add, and how much finished audio we
# keep queued ahead of the listener before idling.
BLOCK_SEC = 20.0
CHUNK_SEC = 10.0
DEFAULT_PRIME_SEC = 25.0
DEFAULT_LOOKAHEAD_SEC = 240.0


class RadioSession:
    """One endless mix. Thread-safe for the reads the web server does."""

    def __init__(self, tracks, out_dir: str, job_id: str, params: Optional[dict] = None,
                 prime_sec: float = DEFAULT_PRIME_SEC,
                 lookahead_sec: float = DEFAULT_LOOKAHEAD_SEC):
        if len(tracks) < 2:
            raise ValueError("Select at least 2 tracks for radio.")
        self.tracks = tracks
        self.out_dir = out_dir
        self.job_id = job_id
        self.params = dict(params or {})
        self.prime_sec = prime_sec
        self.lookahead_sec = lookahead_sec

        self._state: dict = {}          # render_collage resume state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.sr = 44100
        self.chunks: List[dict] = []    # {i, start, dur, path}
        self.clips: List[dict] = []     # absolute-time clip metadata for the UI
        self.error: Optional[str] = None
        self._pending = np.zeros((0, 2), dtype=np.float32)   # not yet a full chunk
        self._generated = 0.0           # seconds of audio committed to chunks

        os.makedirs(out_dir, exist_ok=True)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def prime(self):
        """Render just enough to start playing. Blocking, and kept short."""
        while self._generated <= 0.0 and not self._stop.is_set():
            if not self._render_block(self.prime_sec):
                break

    def start(self):
        """Begin the background lookahead."""
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _run(self):
        while not self._stop.is_set():
            # Stay a comfortable distance ahead, then idle rather than
            # rendering (and writing) audio nobody may ever hear.
            if self._generated >= self.lookahead_sec:
                time.sleep(0.5)
                continue
            if not self._render_block(BLOCK_SEC):
                break

    # ── rendering ────────────────────────────────────────────────────────────

    def _render_block(self, seconds: float) -> bool:
        """Extend the mix; returns False if the session should end."""
        try:
            audio, sr, _markers, clips = render_collage(
                self.tracks, target_length_sec=seconds, state=self._state,
                **self.params)
        except Exception as e:                      # surface, don't spin
            with self._lock:
                self.error = str(e)
            return False

        self.sr = sr
        with self._lock:
            self.clips.extend(clips)
        if len(audio):
            self._append(audio)
        elif self._state.get("pos", 0) == 0:
            # produced nothing and has nothing carried — the pool is exhausted
            with self._lock:
                self.error = self.error or "radio stalled: nothing left to place"
            return False
        return True

    def _append(self, audio: np.ndarray):
        """Buffer committed audio and flush it out in fixed-size chunks."""
        self._pending = (audio if not len(self._pending)
                         else np.concatenate([self._pending, audio]))
        n = int(CHUNK_SEC * self.sr)
        while len(self._pending) >= n:
            self._write_chunk(self._pending[:n])
            self._pending = self._pending[n:]

    def _write_chunk(self, block: np.ndarray):
        with self._lock:
            idx = len(self.chunks)
            start = self._generated
        path = os.path.join(self.out_dir, f"{self.job_id}.{idx:05d}.wav")
        sf.write(path, block, self.sr, subtype="PCM_16")
        dur = len(block) / self.sr
        with self._lock:
            self.chunks.append({"i": idx, "start": round(start, 3),
                                "dur": round(dur, 3), "path": path})
            self._generated += dur

    # ── reads for the web layer ──────────────────────────────────────────────

    def state(self) -> dict:
        from .timeline import build_timeline
        with self._lock:
            clips = list(self.clips)
            chunks = [{"i": c["i"], "start": c["start"], "dur": c["dur"]}
                      for c in self.chunks]
            generated = self._generated
            err = self.error
        # Same shape the player already consumes for a finished mix, so the
        # dashboard's labels and crossfade animation work unchanged.
        tl = build_timeline(clips, self.tracks, generated, self.sr)
        return {
            "job": self.job_id,
            "ready": bool(chunks),
            "sr": self.sr,
            "chunk_sec": CHUNK_SEC,
            "generated_sec": round(generated, 3),
            "chunks": chunks,
            "clips": tl["clips"],
            "tracks": tl["tracks"],
            "stopped": self._stop.is_set(),
            "error": err,
        }

    def chunk_path(self, index: int) -> Optional[str]:
        with self._lock:
            if 0 <= index < len(self.chunks):
                return self.chunks[index]["path"]
        return None

    def cleanup(self):
        """Remove this session's chunk files."""
        with self._lock:
            paths = [c["path"] for c in self.chunks]
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
