import sys
import unittest
from unittest import mock

from RealtimeSTT_server import stt_server


class SilenceTimingArgTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, stt_server, "silence_timing", stt_server.silence_timing)

    def _parse(self, *argv):
        with mock.patch.object(sys, "argv", ["stt-server", *argv]):
            return stt_server.parse_arguments()

    def test_silence_timing_flag_adjusts_the_silence_duration(self):
        args = self._parse("-s")
        recorder = mock.Mock(post_speech_silence_duration=None)

        with mock.patch.multiple(
            stt_server, recorder=recorder, global_args=args, prev_text="", audio_queue=mock.Mock()
        ), mock.patch.object(stt_server.asyncio, "run_coroutine_threadsafe"), \
                mock.patch("builtins.print"):
            stt_server.text_detected("I was thinking...", loop=None)

        self.assertEqual(recorder.post_speech_silence_duration, args.mid_sentence_detection_pause)

    def test_silence_timing_is_off_without_the_flag(self):
        self._parse()

        self.assertFalse(stt_server.silence_timing)


if __name__ == "__main__":
    unittest.main()
