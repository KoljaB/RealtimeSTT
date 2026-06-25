import threading
import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from RealtimeSTT.core.transcription_api import perform_final_transcription
from RealtimeSTT.transcription_engines import (
    TranscriptionInfo,
    TranscriptionResult,
)


class RecorderStub:
    def __init__(self):
        self.transcription_lock = threading.RLock()
        self.audio = np.ones(8, dtype=np.float32)
        self.transcribe_count = 0
        self.language = "en"
        self.interrupt_stop_event = threading.Event()
        self.was_interrupted = threading.Event()
        self.is_recording = False
        self.state = "recording"
        self.on_vad_detect_stop = None
        self.on_wakeword_detection_end = None
        self.spinner = False
        self.halo = None
        self.allowed_to_early_transcribe = False
        self.detected_language = None
        self.detected_language_probability = 0.0
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.last_transcription_metadata = None
        self.ensure_sentence_starting_uppercase = False
        self.ensure_sentence_ends_with_period = False
        self._current_transcription_force_lowercase_start = True
        self.print_transcription_time = False
        self.main_model_type = "fake"


@unittest.skipIf(np is None, "NumPy is required for transcription API tests")
class PerformFinalTranscriptionTests(unittest.TestCase):
    def _patch_transcription_result(self, response):
        def submit(recorder, *_args, **_kwargs):
            recorder.transcribe_count += 1

        return mock.patch.multiple(
            "RealtimeSTT.core.transcription_api",
            submit_transcription_request=mock.Mock(side_effect=submit),
            receive_transcription_result=mock.Mock(return_value=response),
        )

    def test_force_lowercase_flag_is_consumed_on_success(self):
        recorder = RecorderStub()
        response = (
            "success",
            TranscriptionResult(
                text="And the next keeps going",
                info=TranscriptionInfo(language="en", language_probability=1.0),
            ),
        )

        with self._patch_transcription_result(response):
            text = perform_final_transcription(recorder)

        self.assertEqual(text, "and the next keeps going")
        self.assertFalse(recorder._current_transcription_force_lowercase_start)

    def test_force_lowercase_flag_is_consumed_on_empty_audio(self):
        recorder = RecorderStub()
        recorder.audio = np.array([], dtype=np.float32)

        with mock.patch("builtins.print"):
            text = perform_final_transcription(recorder)

        self.assertEqual(text, "")
        self.assertFalse(recorder._current_transcription_force_lowercase_start)

    def test_force_lowercase_flag_is_consumed_on_transcription_error(self):
        recorder = RecorderStub()

        with self._patch_transcription_result(("error", "boom")), mock.patch(
            "RealtimeSTT.core.transcription_api.logger"
        ):
            with self.assertRaises(Exception):
                perform_final_transcription(recorder)

        self.assertFalse(recorder._current_transcription_force_lowercase_start)


if __name__ == "__main__":
    unittest.main()
