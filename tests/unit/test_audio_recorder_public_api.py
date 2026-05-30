import importlib
import inspect
import logging
from pathlib import Path
import platform
import subprocess
import sys
from typing import Callable, Iterable, List, Optional, Union
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_IMPORTS = {"halo", "numpy", "scipy", "torch", "webrtcvad"}


def import_audio_recorder(testcase):
    try:
        return importlib.import_module("RealtimeSTT.audio_recorder")
    except ModuleNotFoundError as exc:
        if exc.name in OPTIONAL_IMPORTS:
            testcase.skipTest(
                "Audio recorder optional dependency missing: %s" % exc.name
            )
        raise


class AudioRecorderPublicApiTests(unittest.TestCase):
    def test_package_and_module_recorder_imports_share_class(self):
        audio_recorder = import_audio_recorder(self)

        from RealtimeSTT import AudioToTextRecorder as package_recorder
        from RealtimeSTT.audio_recorder import AudioToTextRecorder as module_recorder

        self.assertIs(package_recorder, audio_recorder.AudioToTextRecorder)
        self.assertIs(module_recorder, audio_recorder.AudioToTextRecorder)

    def test_constructor_signature_snapshot(self):
        audio_recorder = import_audio_recorder(self)
        signature = inspect.signature(audio_recorder.AudioToTextRecorder.__init__)
        empty = inspect.Parameter.empty

        expected = [
            ("self", empty, empty),
            ("model", audio_recorder.INIT_MODEL_TRANSCRIPTION, str),
            ("transcription_engine", "faster_whisper", str),
            ("transcription_engine_options", None, Optional[dict]),
            ("download_root", None, str),
            ("language", "", str),
            ("compute_type", "default", str),
            ("input_device_index", None, int),
            ("gpu_device_index", 0, Union[int, List[int]]),
            ("device", "cuda", str),
            ("on_recording_start", None, empty),
            ("on_recording_stop", None, empty),
            ("on_transcription_start", None, empty),
            ("ensure_sentence_starting_uppercase", True, empty),
            ("ensure_sentence_ends_with_period", True, empty),
            ("use_microphone", True, empty),
            ("spinner", True, empty),
            ("level", logging.WARNING, empty),
            ("batch_size", 16, int),
            ("enable_realtime_transcription", False, empty),
            ("use_main_model_for_realtime", False, empty),
            ("realtime_transcription_engine", None, str),
            ("realtime_transcription_engine_options", None, Optional[dict]),
            (
                "realtime_model_type",
                audio_recorder.INIT_MODEL_TRANSCRIPTION_REALTIME,
                empty,
            ),
            (
                "realtime_processing_pause",
                audio_recorder.INIT_REALTIME_PROCESSING_PAUSE,
                empty,
            ),
            (
                "init_realtime_after_seconds",
                audio_recorder.INIT_REALTIME_INITIAL_PAUSE,
                empty,
            ),
            ("on_realtime_transcription_update", None, empty),
            ("on_realtime_transcription_stabilized", None, empty),
            ("realtime_batch_size", 16, int),
            ("silero_sensitivity", audio_recorder.INIT_SILERO_SENSITIVITY, float),
            ("silero_use_onnx", None, Optional[bool]),
            ("silero_deactivity_detection", False, bool),
            ("webrtc_sensitivity", audio_recorder.INIT_WEBRTC_SENSITIVITY, int),
            ("warmup_vad", True, bool),
            (
                "post_speech_silence_duration",
                audio_recorder.INIT_POST_SPEECH_SILENCE_DURATION,
                float,
            ),
            (
                "min_length_of_recording",
                audio_recorder.INIT_MIN_LENGTH_OF_RECORDING,
                float,
            ),
            (
                "min_gap_between_recordings",
                audio_recorder.INIT_MIN_GAP_BETWEEN_RECORDINGS,
                float,
            ),
            (
                "pre_recording_buffer_duration",
                audio_recorder.INIT_PRE_RECORDING_BUFFER_DURATION,
                float,
            ),
            ("pre_recording_buffer_trim_config", None, Optional[dict]),
            ("on_vad_start", None, empty),
            ("on_vad_stop", None, empty),
            ("on_vad_detect_start", None, empty),
            ("on_vad_detect_stop", None, empty),
            ("on_turn_detection_start", None, empty),
            ("on_turn_detection_stop", None, empty),
            ("wakeword_backend", "", str),
            ("openwakeword_model_paths", None, str),
            ("openwakeword_inference_framework", "onnx", str),
            ("wake_words", "", str),
            (
                "wake_words_sensitivity",
                audio_recorder.INIT_WAKE_WORDS_SENSITIVITY,
                float,
            ),
            (
                "wake_word_activation_delay",
                audio_recorder.INIT_WAKE_WORD_ACTIVATION_DELAY,
                float,
            ),
            ("wake_word_timeout", audio_recorder.INIT_WAKE_WORD_TIMEOUT, float),
            (
                "wake_word_buffer_duration",
                audio_recorder.INIT_WAKE_WORD_BUFFER_DURATION,
                float,
            ),
            ("on_wakeword_detected", None, empty),
            ("on_wakeword_timeout", None, empty),
            ("on_wakeword_detection_start", None, empty),
            ("on_wakeword_detection_end", None, empty),
            ("on_recorded_chunk", None, empty),
            ("debug_mode", False, empty),
            (
                "handle_buffer_overflow",
                audio_recorder.INIT_HANDLE_BUFFER_OVERFLOW,
                bool,
            ),
            ("beam_size", 5, int),
            ("beam_size_realtime", 3, int),
            ("buffer_size", audio_recorder.BUFFER_SIZE, int),
            ("sample_rate", audio_recorder.SAMPLE_RATE, int),
            ("initial_prompt", None, Optional[Union[str, Iterable[int]]]),
            ("initial_prompt_realtime", None, Optional[Union[str, Iterable[int]]]),
            ("suppress_tokens", [-1], Optional[List[int]]),
            ("print_transcription_time", False, bool),
            ("early_transcription_on_silence", 0, int),
            ("allowed_latency_limit", audio_recorder.ALLOWED_LATENCY_LIMIT, int),
            ("no_log_file", False, bool),
            ("use_extended_logging", False, bool),
            ("faster_whisper_vad_filter", True, bool),
            ("normalize_audio", False, bool),
            ("start_callback_in_new_thread", False, bool),
            ("realtime_transcription_use_syllable_boundaries", False, bool),
            ("realtime_boundary_detector_sensitivity", 0.6, float),
            (
                "realtime_boundary_followup_delays",
                (0.05, 0.2),
                Optional[Iterable[float]],
            ),
            ("transcription_executor", None, Optional[Callable]),
            ("realtime_transcription_executor", None, Optional[Callable]),
            ("on_realtime_text_stabilization_update", None, empty),
            ("silero_backend", "auto", str),
            ("silero_onnx_model_path", None, Optional[str]),
            ("silero_onnx_threads", 2, int),
            (
                "deactivity_silence_confirmation_duration",
                audio_recorder.DEACTIVITY_SILENCE_CONFIRMATION_DURATION,
                float,
            ),
        ]

        self.assertEqual(list(signature.parameters), [item[0] for item in expected])
        for name, default, annotation in expected:
            parameter = signature.parameters[name]
            self.assertEqual(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            self.assertEqual(parameter.default, default, name)
            self.assertEqual(parameter.annotation, annotation, name)

    def test_public_default_constants_remain_importable(self):
        audio_recorder = import_audio_recorder(self)
        expected = {
            "INIT_MODEL_TRANSCRIPTION": "tiny",
            "INIT_MODEL_TRANSCRIPTION_REALTIME": "tiny",
            "INIT_REALTIME_PROCESSING_PAUSE": 0.2,
            "INIT_REALTIME_INITIAL_PAUSE": 0.2,
            "INIT_SILERO_SENSITIVITY": 0.4,
            "INIT_WEBRTC_SENSITIVITY": 3,
            "INIT_POST_SPEECH_SILENCE_DURATION": 0.6,
            "INIT_MIN_LENGTH_OF_RECORDING": 0.5,
            "INIT_MIN_GAP_BETWEEN_RECORDINGS": 0,
            "INIT_WAKE_WORDS_SENSITIVITY": 0.6,
            "INIT_PRE_RECORDING_BUFFER_DURATION": 1.0,
            "INIT_WAKE_WORD_ACTIVATION_DELAY": 0.0,
            "INIT_WAKE_WORD_TIMEOUT": 5.0,
            "INIT_WAKE_WORD_BUFFER_DURATION": 0.1,
            "ALLOWED_LATENCY_LIMIT": 100,
            "SAMPLE_RATE": 16000,
            "BUFFER_SIZE": 512,
            "DEACTIVITY_SILENCE_CONFIRMATION_DURATION": 0.16,
            "INIT_HANDLE_BUFFER_OVERFLOW": platform.system() != "Darwin",
        }

        for name, value in expected.items():
            self.assertEqual(getattr(audio_recorder, name), value, name)

    def test_package_level_realtime_boundary_exports_remain_supported(self):
        try:
            from RealtimeSTT import (
                RealtimeSpeechBoundaryDetector,
                SpeechBoundaryEvent,
                SpeechBoundaryResult,
            )
            from RealtimeSTT.core.realtime_boundary_detector import (
                RealtimeSpeechBoundaryDetector as core_detector,
                SpeechBoundaryEvent as core_event,
                SpeechBoundaryResult as core_result,
            )
        except ModuleNotFoundError as exc:
            if exc.name in OPTIONAL_IMPORTS:
                self.skipTest(
                    "Realtime boundary optional dependency missing: %s" % exc.name
                )
            raise

        self.assertIs(RealtimeSpeechBoundaryDetector, core_detector)
        self.assertIs(SpeechBoundaryEvent, core_event)
        self.assertIs(SpeechBoundaryResult, core_result)

    def test_compatibility_private_methods_remain_on_recorder(self):
        audio_recorder = import_audio_recorder(self)
        recorder_cls = audio_recorder.AudioToTextRecorder
        expected = [
            "_recording_worker",
            "_realtime_worker",
            "_is_silero_speech",
            "_is_webrtc_speech",
            "_check_voice_activity",
            "_selected_pre_recording_buffer_frames",
            "_set_audio_from_frames",
            "_queue_recorded_audio",
            "_get_next_recorded_audio",
            "_is_voice_active",
            "_run_callback",
            "_set_state",
            "_set_spinner",
            "_preprocess_output",
            "_find_tail_match_in_text",
            "_read_stdout",
            "_audio_data_worker",
        ]

        for name in expected:
            self.assertTrue(callable(getattr(recorder_cls, name, None)), name)

        self.assertIsInstance(
            inspect.getattr_static(recorder_cls, "_audio_data_worker"),
            staticmethod,
        )

    def test_kmp_duplicate_lib_ok_is_true_after_module_import(self):
        code = (
            "import os\n"
            "os.environ.pop('KMP_DUPLICATE_LIB_OK', None)\n"
            "import RealtimeSTT.audio_recorder\n"
            "print(os.environ.get('KMP_DUPLICATE_LIB_OK'))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr or ""
            if "ModuleNotFoundError" in stderr and "No module named" in stderr:
                self.skipTest("Audio recorder import failed: %s" % stderr.strip())
            self.fail(stderr)

        self.assertEqual(result.stdout.strip().splitlines()[-1], "TRUE")


if __name__ == "__main__":
    unittest.main()
