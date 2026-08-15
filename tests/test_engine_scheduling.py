import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from infinite_dj.engine import (
    GAP_WARN_SECONDS,
    SR,
    AudioRingBuffer,
    StreamEngine,
    TransitionEvent,
    _audible_track_position,
    _build_crossfade_chunk,
    _chunk_level_dbfs,
    _crossfade_progress,
    _transition_start_time,
)
from infinite_dj.mixer import (
    CrossfadeFilterState,
    ShelfCrossfadeState,
    TransitionStyle,
    _blend,
)
from infinite_dj.models import CuePoint, TrackMeta


def track(path: str, downbeats: list[float]) -> TrackMeta:
    return TrackMeta(
        file_path=path,
        title=path,
        duration=300.0,
        bpm=120.0,
        bpm_confidence=1.0,
        beats=[],
        downbeats=downbeats,
        phrases=[],
        key="8A",
        key_name="A minor",
        key_confidence=1.0,
        energy_curve=[],
        sections=[],
        cue_points=[],
        analyzed_at=0.0,
    )


class TransitionSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.current = track("current", [10.0, 12.0, 14.0, 16.0])
        self.incoming = track("incoming", [0.0])
        self.cue_in = CuePoint(0.0, "in", True, 0.5, 1.0)

    def test_uses_the_scheduler_selected_out_cue_exactly(self):
        event = TransitionEvent(
            incoming_track=self.incoming,
            cue_in=self.cue_in,
            cue_out=CuePoint(14.0, "out", True, 0.5, 1.0),
        )

        self.assertEqual(_transition_start_time(event, self.current, 10.5), 14.0)

    def test_late_or_missing_cue_falls_back_to_the_next_downbeat(self):
        late_event = TransitionEvent(
            incoming_track=self.incoming,
            cue_in=self.cue_in,
            cue_out=CuePoint(10.0, "out", True, 0.5, 1.0),
        )
        no_cue_event = TransitionEvent(self.incoming, self.cue_in)

        self.assertEqual(_transition_start_time(late_event, self.current, 10.5), 12.0)
        self.assertEqual(_transition_start_time(no_cue_event, self.current, 10.5), 12.0)

    def test_skip_starts_at_the_current_producer_position(self):
        event = TransitionEvent(
            incoming_track=self.incoming,
            cue_in=self.cue_in,
            trigger_immediately=True,
        )

        self.assertEqual(_transition_start_time(event, self.current, 10.5), 10.5)

    def test_selected_track_is_prepared_before_the_transition_fires(self):
        engine = StreamEngine([self.current, self.incoming])
        incoming_audio = np.zeros((128, 2), dtype=np.float32)
        engine._load_matched = lambda _: incoming_audio

        engine._request_incoming_prepare(self.current, self.incoming, self.cue_in)
        engine._preparation_thread.join(timeout=1)
        event = TransitionEvent(self.incoming, self.cue_in)
        prepared = engine._prepared_for(self.current, event)

        self.assertIsNotNone(prepared)
        self.assertIs(prepared.native_audio, incoming_audio)
        self.assertEqual(prepared.stretched_start_frame, 0)

    def test_track_end_uses_prepared_audio_without_loading_on_producer(self):
        engine = StreamEngine([self.current, self.incoming])
        engine.state.track = self.current
        incoming_audio = np.zeros((128, 2), dtype=np.float32)
        engine._load_matched = lambda _: incoming_audio

        engine._request_incoming_prepare(self.current, self.incoming, self.cue_in)
        engine._preparation_thread.join(timeout=1)
        audio, position = engine._handle_track_end()

        self.assertIs(audio, incoming_audio)
        self.assertEqual(position, 0)
        self.assertIs(engine.state.track, self.incoming)


class RealtimeCrossfadeTests(unittest.TestCase):
    def test_crossfade_progress_stays_scalar_for_the_ui(self):
        phase = np.linspace(0.25, 0.75, 32, dtype=np.float32)

        progress = _crossfade_progress(phase)

        self.assertIsInstance(progress, float)
        self.assertAlmostEqual(progress, 0.75)

    def test_terminal_ui_does_not_crash_if_given_a_phase_ramp(self):
        current = track("current", [])
        engine = StreamEngine([current, track("next", [])])
        engine.state.track = current
        engine.state.is_mixing = True
        engine.state.mix_progress = np.linspace(0.25, 0.75, 32, dtype=np.float32)

        output = StringIO()
        with redirect_stdout(output):
            engine._render_ui()

        self.assertIn("[MIXING 75%]", output.getvalue())

    def _assert_chunked_matches_continuous(self, make_state):
        """Streaming in chunks must equal one continuous render, per topology.

        The engine renders a crossfade in ~93ms producer chunks while the
        offline renderer does it in one call; they have to agree sample-for-
        sample or a live mix and its rendered twin drift apart.
        """
        n = 512
        outgoing = np.ones((n * 2, 2), dtype=np.float32)
        incoming = np.zeros((n * 2, 2), dtype=np.float32)
        phase = np.linspace(0.0, 1.0, n * 2, dtype=np.float32)
        style = TransitionStyle("test", 8)

        continuous = _blend(outgoing, incoming, phase, style=style,
                            filter_state=make_state())
        state = make_state()
        chunked = np.concatenate((
            _blend(outgoing[:n], incoming[:n], phase[:n], style=style,
                   filter_state=state),
            _blend(outgoing[n:], incoming[n:], phase[n:], style=style,
                   filter_state=state),
        ))

        np.testing.assert_allclose(chunked, continuous, rtol=1e-5, atol=1e-5)

    def test_stateful_chunked_blend_matches_continuous_rendering(self):
        self._assert_chunked_matches_continuous(CrossfadeFilterState.create)

    def test_shelf_chunked_blend_matches_continuous_rendering(self):
        self._assert_chunked_matches_continuous(ShelfCrossfadeState.create)

    def test_uneven_chunk_boundaries_do_not_shift_the_shelf_control_grid(self):
        """Coefficient updates ride a global grid, not the chunk boundary.

        The producer's chunk sizes vary (a transition can start mid-chunk), so
        if the control grid were chunk-relative the same crossfade would render
        differently depending on where chunks happened to fall.
        """
        total = 4096
        outgoing = np.ones((total, 2), dtype=np.float32)
        incoming = np.zeros((total, 2), dtype=np.float32)
        phase = np.linspace(0.0, 1.0, total, dtype=np.float32)
        style = TransitionStyle("test", 8)

        continuous = _blend(outgoing, incoming, phase, style=style,
                            filter_state=ShelfCrossfadeState.create())
        state = ShelfCrossfadeState.create()
        parts, i = [], 0
        for size in (700, 1, 333, 1024, 9):          # deliberately unaligned
            j = min(i + size, total)
            parts.append(_blend(outgoing[i:j], incoming[i:j], phase[i:j],
                                style=style, filter_state=state))
            i = j
        parts.append(_blend(outgoing[i:], incoming[i:], phase[i:],
                            style=style, filter_state=state))

        np.testing.assert_allclose(np.concatenate(parts), continuous,
                                   rtol=1e-5, atol=1e-5)

    def test_chunk_crossfade_uses_the_full_phase_ramp(self):
        n = 512
        outgoing = np.zeros((n, 2), dtype=np.float32)
        incoming = np.ones((n, 2), dtype=np.float32)
        phase = np.linspace(0.0, 1.0, n, dtype=np.float32)
        style = TransitionStyle("cut", 0, is_cut=True)

        result = _build_crossfade_chunk(outgoing, incoming, phase, style=style)

        self.assertLess(result[0, 0], result[-1, 0])


class AudioRingBufferTests(unittest.TestCase):
    def test_audible_position_accounts_for_queued_audio(self):
        self.assertEqual(_audible_track_position(12.0, 44100 * 3), 9.0)
        self.assertEqual(_audible_track_position(1.0, 44100 * 3), 0.0)

    def test_wraps_without_dropping_or_reordering_frames(self):
        ring = AudioRingBuffer(5)
        first = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
        self.assertTrue(ring.write(first))

        out = np.empty((3, 2), dtype=np.float32)
        self.assertEqual(ring.read_into(out, 3), 3)
        np.testing.assert_array_equal(out[:, 0], [0, 1, 2])

        second = np.array([[4, 4], [5, 5], [6, 6]], dtype=np.float32)
        self.assertTrue(ring.write(second))
        self.assertEqual(ring.available_frames, 4)

        out = np.empty((4, 2), dtype=np.float32)
        self.assertEqual(ring.read_into(out, 4), 4)
        np.testing.assert_array_equal(out[:, 0], [3, 4, 5, 6])

    def test_underflow_fills_callback_output_with_silence(self):
        ring = AudioRingBuffer(4)
        ring.write(np.ones((2, 2), dtype=np.float32))
        out = np.empty((4, 2), dtype=np.float32)

        self.assertEqual(ring.read_into(out, 4), 2)
        np.testing.assert_array_equal(out[:2], np.ones((2, 2)))
        np.testing.assert_array_equal(out[2:], np.zeros((2, 2)))


class GapDetectionTests(unittest.TestCase):
    """Checks that catch 'one track fades to silence, then a gap remains'."""

    def setUp(self):
        self.current = track("current", [10.0])
        self.incoming = track("incoming", [0.0])
        self.engine = StreamEngine([self.current, self.incoming])
        self.engine.state.track = self.current

    def test_short_shortfall_is_not_reported(self):
        # A few frames short, immediately followed by a full read: well
        # under GAP_WARN_SECONDS, and shouldn't be treated as an audible gap.
        self.engine._record_playback(requested_frames=100, actual_frames=90)
        self.engine._record_playback(requested_frames=100, actual_frames=100)

        self.assertEqual(self.engine.state.gap_events, [])

    def test_sustained_starvation_is_recorded_once_with_its_duration(self):
        chunk = int(GAP_WARN_SECONDS * SR / 2) + 1000  # two calls clears the threshold

        output = StringIO()
        with redirect_stdout(output):
            self.engine._record_playback(requested_frames=chunk, actual_frames=0)
            self.engine._record_playback(requested_frames=chunk, actual_frames=0)
            # The run ends once the buffer is caught up again.
            self.engine._record_playback(requested_frames=chunk, actual_frames=chunk)

        self.assertEqual(len(self.engine.state.gap_events), 1)
        event = self.engine.state.gap_events[0]
        self.assertGreaterEqual(event["duration"], GAP_WARN_SECONDS)
        self.assertEqual(event["track"], "current")
        self.assertIn("GAP", output.getvalue())
        # A second full read shouldn't duplicate the already-closed event.
        self.engine._record_playback(requested_frames=chunk, actual_frames=chunk)
        self.assertEqual(len(self.engine.state.gap_events), 1)

    def test_chunk_level_dbfs_reads_true_silence_as_very_low(self):
        silent = np.zeros((512, 2), dtype=np.float32)
        loud = np.ones((512, 2), dtype=np.float32)

        self.assertLess(_chunk_level_dbfs(silent), -100.0)
        self.assertAlmostEqual(_chunk_level_dbfs(loud), 0.0, places=3)


class ReprepareOnFailureTests(unittest.TestCase):
    """A failed background preparation must be retried, not silently dropped."""

    def setUp(self):
        self.current = track("current", [10.0])
        self.incoming = track("incoming", [0.0])
        self.cue_in = CuePoint(0.0, "in", True, 0.5, 1.0)
        self.engine = StreamEngine([self.current, self.incoming])
        self.engine.state.track = self.current
        self.engine.state.next_track = self.incoming
        self.engine.state.next_cue_in = self.cue_in

    def test_retries_when_nothing_is_prepared_and_no_worker_is_running(self):
        calls = []
        self.engine._request_incoming_prepare = lambda cur, inc, cue: calls.append(
            (cur.file_path, inc.file_path)
        )

        self.engine._maybe_reprepare_next(self.current)

        self.assertEqual(calls, [(self.current.file_path, self.incoming.file_path)])

    def test_does_not_retry_while_a_preparation_is_already_in_flight(self):
        release = threading.Event()

        def slow_load(_track):
            release.wait(timeout=1)
            return np.zeros((64, 2), dtype=np.float32)

        self.engine._load_matched = slow_load
        self.engine._request_incoming_prepare(self.current, self.incoming, self.cue_in)

        calls = []
        real_request = self.engine._request_incoming_prepare
        self.engine._request_incoming_prepare = lambda *a: (calls.append(a), real_request(*a))

        self.engine._maybe_reprepare_next(self.current)
        release.set()
        self.engine._preparation_thread.join(timeout=1)

        self.assertEqual(calls, [])

    def test_does_not_retry_once_the_matching_preparation_is_ready(self):
        self.engine._load_matched = lambda _t: np.zeros((64, 2), dtype=np.float32)
        self.engine._request_incoming_prepare(self.current, self.incoming, self.cue_in)
        self.engine._preparation_thread.join(timeout=1)

        calls = []
        self.engine._request_incoming_prepare = lambda cur, inc, cue: calls.append(1)

        self.engine._maybe_reprepare_next(self.current)

        self.assertEqual(calls, [])


class BlockingFallbackGapTests(unittest.TestCase):
    """The synchronous last-resort path in `_handle_track_end` must self-report."""

    def test_records_a_gap_event_when_the_blocking_load_is_slow(self):
        current = track("current", [10.0])
        incoming = track("incoming", [0.0])
        engine = StreamEngine([current, incoming])
        engine.state.track = current

        def slow_load(_t):
            time.sleep(GAP_WARN_SECONDS + 0.05)
            return np.zeros((64, 2), dtype=np.float32)

        engine._load_matched = slow_load
        output = StringIO()
        with redirect_stdout(output):
            engine._handle_track_end()

        self.assertEqual(len(engine.state.gap_events), 1)
        self.assertEqual(engine.state.gap_events[0]["source"], "blocking_load")
        self.assertIn("GAP RISK", output.getvalue())


if __name__ == "__main__":
    unittest.main()
