import sys
import unittest
from unittest import mock

from RealtimeSTT_server.stt_server import parse_arguments


class SilenceTimingDefaultTests(unittest.TestCase):
    def test_default_matches_its_own_help_text(self):
        # The flag's own help text says "Default is False.", so that's what
        # parse_arguments() must return when the flag isn't passed.
        with mock.patch.object(sys, "argv", ["stt_server.py"]):
            args = parse_arguments()
        self.assertFalse(args.silence_timing)

    def test_flag_still_enables_it(self):
        with mock.patch.object(sys, "argv", ["stt_server.py", "--silence_timing"]):
            args = parse_arguments()
        self.assertTrue(args.silence_timing)


if __name__ == "__main__":
    unittest.main()
