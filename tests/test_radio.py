import tempfile
import unittest

from infinite_dj.radio import RadioSession


class RadioLookaheadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = RadioSession(
            [object(), object()],
            self.tmp.name,
            "test-radio",
            lookahead_sec=240.0,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_consumed_audio_reopens_the_lookahead_window(self):
        self.session._generated = 240.0
        self.assertFalse(self.session._needs_render())

        self.session.update_playback_position(60.0)

        self.assertTrue(self.session._needs_render())
        self.assertEqual(self.session._buffered_ahead(), 180.0)

    def test_playback_heartbeat_is_monotonic_and_clamped_to_generated_audio(self):
        self.session._generated = 240.0

        self.session.update_playback_position(120.0)
        self.session.update_playback_position(90.0)
        self.assertEqual(self.session._played, 120.0)

        self.session.update_playback_position(9999.0)
        self.assertEqual(self.session._played, 240.0)
        self.assertEqual(self.session._buffered_ahead(), 0.0)

    def test_rejects_invalid_playback_positions(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.session.update_playback_position(value)


if __name__ == "__main__":
    unittest.main()
