import unittest

from RealtimeSTT.core.text_formatting import (
    find_tail_match_in_text,
    format_number,
    preprocess_output,
)

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
except Exception as exc:  # pragma: no cover - optional runtime deps may be absent
    AudioToTextRecorder = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class TextFormattingTests(unittest.TestCase):
    def test_format_number_keeps_legacy_two_digit_window(self):
        self.assertEqual(format_number(123.456789), "23.45")

    def test_preprocess_output_normalizes_whitespace_and_punctuation(self):
        self.assertEqual(preprocess_output("  hello\t\nworld  "), "Hello world.")

    def test_preprocess_output_handles_empty_text(self):
        self.assertEqual(preprocess_output(" \t\n "), "")

    def test_preprocess_output_respects_uppercase_toggle(self):
        self.assertEqual(
            preprocess_output(
                "hello world",
                ensure_sentence_starting_uppercase=False,
            ),
            "hello world.",
        )

    def test_preprocess_output_respects_period_toggle(self):
        self.assertEqual(
            preprocess_output(
                "hello world",
                ensure_sentence_ends_with_period=False,
            ),
            "Hello world",
        )

    def test_preprocess_output_preview_does_not_add_period(self):
        self.assertEqual(preprocess_output("hello world", preview=True), "Hello world")

    def test_find_tail_match_returns_legacy_match_end_position(self):
        self.assertEqual(
            find_tail_match_in_text("alpha world", "one world two", 5),
            9,
        )

    def test_find_tail_match_uses_rightmost_match(self):
        self.assertEqual(
            find_tail_match_in_text("alpha xyz", "xyz middle xyz", 3),
            14,
        )

    def test_find_tail_match_returns_negative_one_without_match(self):
        self.assertEqual(find_tail_match_in_text("alpha xyz", "no match", 3), -1)

    def test_find_tail_match_returns_negative_one_for_short_inputs(self):
        self.assertEqual(find_tail_match_in_text("ab", "abc", 3), -1)


class AudioToTextRecorderTextFormattingCompatibilityTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"RealtimeSTT import failed: {IMPORT_ERROR}")

    def test_facade_format_number_delegates_to_core_helper(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        self.assertEqual(recorder.format_number(123.456789), "23.45")

    def test_facade_preprocess_output_preserves_recorder_flags(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.ensure_sentence_starting_uppercase = False
        recorder.ensure_sentence_ends_with_period = False

        self.assertEqual(recorder._preprocess_output("  hello\tworld  "), "hello world")

    def test_facade_tail_match_delegates_to_core_helper(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)

        self.assertEqual(
            recorder._find_tail_match_in_text("alpha world", "one world two", 5),
            9,
        )


if __name__ == "__main__":
    unittest.main()
