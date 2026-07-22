import unittest
import numpy as np

from infinite_dj.engine import (
    AudioRingBuffer,
    StreamEngine,
    TransitionEvent,
    _audible_track_position,
    _build_crossfade_chunk,
    _transition_start_time,
)
from infinite_dj.mixer import CrossfadeFilterState, TransitionStyle, _blend
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
    def test_stateful_chunked_blend_matches_continuous_rendering(self):
        n = 512
        outgoing = np.ones((n * 2, 2), dtype=np.float32)
        incoming = np.zeros((n * 2, 2), dtype=np.float32)
        phase = np.linspace(0.0, 1.0, n * 2, dtype=np.float32)
        style = TransitionStyle("test", 8)

        continuous = _blend(outgoing, incoming, phase, style=style)
        state = CrossfadeFilterState.create()
        chunked = np.concatenate((
            _blend(outgoing[:n], incoming[:n], phase[:n], style=style,
                   filter_state=state),
            _blend(outgoing[n:], incoming[n:], phase[n:], style=style,
                   filter_state=state),
        ))

        np.testing.assert_allclose(chunked, continuous, rtol=1e-5, atol=1e-5)

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


if __name__ == "__main__":
    unittest.main()
