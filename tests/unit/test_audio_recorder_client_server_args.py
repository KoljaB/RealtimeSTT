import ast
import logging
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

from RealtimeSTT.audio_recorder import DEACTIVITY_SILENCE_CONFIRMATION_DURATION
from RealtimeSTT.audio_recorder_client import AudioToTextRecorderClient


class StartServerArgsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server_path = (
            Path(__file__).resolve().parents[2]
            / "RealtimeSTT_server"
            / "stt_server.py"
        )
        module = ast.parse(server_path.read_text(encoding="utf-8"))
        parser = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_arguments"
        )
        # Exercise the real parser without the server's import-time package
        # installation checks and asyncio configuration.
        namespace = {
            "logging": logging,
            "DEACTIVITY_SILENCE_CONFIRMATION_DURATION": (
                DEACTIVITY_SILENCE_CONFIRMATION_DURATION
            ),
        }
        exec(
            compile(ast.Module(body=[parser], type_ignores=[]), str(server_path), "exec"),
            namespace,
        )
        cls._parse_arguments = staticmethod(namespace["parse_arguments"])

    def _parse_server_args(self, args):
        with mock.patch.object(sys, "argv", args), mock.patch("logging.getLogger"):
            return self._parse_arguments()

    def _server_args(self, **kwargs):
        with mock.patch.object(AudioToTextRecorderClient, "connect", return_value=True):
            client = AudioToTextRecorderClient(use_microphone=False, **kwargs)
        with mock.patch("RealtimeSTT.audio_recorder_client.subprocess") as subprocess_mock, \
                mock.patch.object(os, "name", "posix"):
            client.start_server()
        return subprocess_mock.Popen.call_args.args[0]

    def test_server_silero_flags_preserve_default_and_support_both_values(self):
        enable = "--silero_deactivity_detection"
        disable = "--no-silero_deactivity_detection"
        for flags, expected in (
            ([], True),
            ([enable], True),
            ([disable], False),
            ([enable, disable], False),
            ([disable, enable], True),
        ):
            with self.subTest(flags=flags):
                parsed = self._parse_server_args(["stt-server"] + flags)
                self.assertIs(parsed.silero_deactivity_detection, expected)

    def test_default_client_disables_silero_deactivity_detection(self):
        args = self._server_args()

        self.assertIn("--no-silero_deactivity_detection", args)
        self.assertNotIn("--silero_deactivity_detection", args)
        self.assertIs(
            self._parse_server_args(args).silero_deactivity_detection,
            False,
        )

    def test_client_silero_setting_reaches_server(self):
        for enabled, flag in (
            (False, "--no-silero_deactivity_detection"),
            (True, "--silero_deactivity_detection"),
        ):
            with self.subTest(enabled=enabled):
                args = self._server_args(silero_deactivity_detection=enabled)

                self.assertIn(flag, args)
                self.assertIs(
                    self._parse_server_args(args).silero_deactivity_detection,
                    enabled,
                )

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
