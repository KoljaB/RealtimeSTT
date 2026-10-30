import os
import unittest
from unittest import mock

from RealtimeSTT.audio_recorder_client import AudioToTextRecorderClient


class StartServerArgsTests(unittest.TestCase):
    def _server_args(self, **kwargs):
        with mock.patch.object(AudioToTextRecorderClient, "connect", return_value=True):
            client = AudioToTextRecorderClient(use_microphone=False, **kwargs)
        with mock.patch("RealtimeSTT.audio_recorder_client.subprocess") as subprocess_mock, \
                mock.patch.object(os, "name", "posix"):
            client.start_server()
        return subprocess_mock.Popen.call_args.args[0]

    def test_default_language_starts_the_server_with_auto_detection(self):
        args = self._server_args()

        # stt-server defaults to "en", so an empty language has to be sent explicitly
        self.assertIn("--language", args)
        self.assertEqual(args[args.index("--language") + 1], "")

    def test_language_is_passed_to_the_server(self):
        args = self._server_args(language="de")

        self.assertEqual(args[args.index("--language") + 1], "de")

    def test_default_silero_deactivity_detection_is_sent_as_false(self):
        # The client defaults to False (like AudioToTextRecorder), but the server's own
        # default is True, so the client must send the flag explicitly to turn it off.
        args = self._server_args()

        self.assertIn("--no-silero_deactivity_detection", args)
        self.assertNotIn("--silero_deactivity_detection", args)

    def test_silero_deactivity_detection_true_is_sent_explicitly(self):
        args = self._server_args(silero_deactivity_detection=True)

        self.assertIn("--silero_deactivity_detection", args)
        self.assertNotIn("--no-silero_deactivity_detection", args)


if __name__ == "__main__":
    unittest.main()
