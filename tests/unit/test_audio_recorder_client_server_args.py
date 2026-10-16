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


if __name__ == "__main__":
    unittest.main()
