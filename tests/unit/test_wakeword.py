import unittest
from unittest.mock import patch


try:
    from RealtimeSTT.core import wakeword
except ModuleNotFoundError as exc:
    wakeword = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class WakeWordTests(unittest.TestCase):
    def setUp(self):
        if wakeword is None:
            self.skipTest(f"wakeword import failed: {IMPORT_ERROR}")

    def test_bare_wake_words_default_to_porcupine(self):
        self.assertEqual(
            wakeword._normalize_wakeword_backend("", "jarvis"),
            "pvporcupine",
        )

    def test_no_wake_words_keep_backend_empty(self):
        self.assertEqual(
            wakeword._normalize_wakeword_backend("", ""),
            "",
        )

    def test_openwakeword_backend_normalizes_hyphen(self):
        self.assertEqual(
            wakeword._normalize_wakeword_backend("open-wakeword", ""),
            "open_wakeword",
        )

    def test_porcupine_missing_dependency_mentions_extra(self):
        with patch(
            "RealtimeSTT.core.wakeword.import_module",
            side_effect=ModuleNotFoundError("No module named 'pvporcupine'"),
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, r"RealtimeSTT\[porcupine\]"):
                wakeword._load_porcupine_module()

    def test_openwakeword_missing_dependency_mentions_extra(self):
        with patch(
            "RealtimeSTT.core.wakeword.import_module",
            side_effect=ModuleNotFoundError("No module named 'openwakeword'"),
        ):
            with self.assertRaisesRegex(ModuleNotFoundError, r"RealtimeSTT\[openwakeword\]"):
                wakeword._load_openwakeword_modules()


if __name__ == "__main__":
    unittest.main()
