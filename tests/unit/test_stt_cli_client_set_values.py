import importlib
import sys
import types
import unittest
from unittest import mock


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.set_calls = []
        self._recording = False  # makes main() leave its loop at once
        self.is_running = False
        _FakeClient.instances.append(self)

    def set_parameter(self, parameter, value):
        self.set_calls.append((parameter, value))

    def shutdown(self):
        pass


class CliSetValueTests(unittest.TestCase):
    def _run_main(self, *argv):
        fake_package = types.ModuleType("RealtimeSTT")
        fake_package.AudioToTextRecorderClient = _FakeClient
        fake_package.AudioInput = object
        _FakeClient.instances.clear()
        with mock.patch.dict(sys.modules, {"RealtimeSTT": fake_package}):
            sys.modules.pop("RealtimeSTT_server.stt_cli_client", None)
            cli = importlib.import_module("RealtimeSTT_server.stt_cli_client")
            with mock.patch.object(cli.sys, "argv", ["stt", *argv]), \
                    mock.patch.object(cli.sys, "stdout"), \
                    mock.patch.object(cli.sys, "stderr"), \
                    mock.patch.object(cli.os, "isatty", return_value=True):
                cli.main()
            sys.modules.pop("RealtimeSTT_server.stt_cli_client", None)
        return _FakeClient.instances[-1].set_calls

    def test_boolean_values_are_sent_as_booleans(self):
        calls = self._run_main(
            "-s", "silero_deactivity_detection", "False",
            "-s", "use_wake_words", "true",
        )
        self.assertEqual(
            calls,
            [("silero_deactivity_detection", False), ("use_wake_words", True)],
        )

    def test_numbers_and_strings_are_unchanged(self):
        calls = self._run_main(
            "-s", "silero_sensitivity", "0.1",
            "-s", "wake_word_activation_delay", "2",
            "-s", "language", "de",
        )
        self.assertEqual(
            calls,
            [
                ("silero_sensitivity", 0.1),
                ("wake_word_activation_delay", 2),
                ("language", "de"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
