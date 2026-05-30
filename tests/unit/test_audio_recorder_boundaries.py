import ast
import inspect
from pathlib import Path
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_RECORDER_DIR = PROJECT_ROOT / "RealtimeSTT" / "core"


class AudioRecorderBoundaryTests(unittest.TestCase):
    def test_public_recorder_imports_stay_compatible(self):
        from RealtimeSTT import AudioToTextRecorder as package_recorder
        from RealtimeSTT.audio_recorder import (
            AudioToTextRecorder as module_recorder,
            DEACTIVITY_SILENCE_CONFIRMATION_DURATION,
        )

        self.assertIs(package_recorder, module_recorder)
        self.assertEqual(DEACTIVITY_SILENCE_CONFIRMATION_DURATION, 0.16)

    def test_constructor_passes_legacy_init_argument_mapping(self):
        from RealtimeSTT.audio_recorder import AudioToTextRecorder

        def callback(*args):
            return None

        transcription_options = {"temperature": 0.0}
        realtime_options = {"vad": False}
        pre_recording_trim_config = {"enabled": True}
        gpu_devices = [0, 1]
        initial_prompt = [1, 2, 3]
        initial_prompt_realtime = [4, 5, 6]
        suppress_tokens = [-1, 13]
        followup_delays = [0.1, 0.2]
        transcription_executor = object()
        realtime_transcription_executor = object()
        captured = {}

        def fake_initialize_recorder(
            recorder,
            recorder_cls,
            init_args,
            normalize_wakeword_backend,
            load_porcupine_module,
            load_openwakeword_modules,
        ):
            captured["recorder"] = recorder
            captured["recorder_cls"] = recorder_cls
            captured["init_args"] = init_args
            captured["normalize_wakeword_backend"] = normalize_wakeword_backend
            captured["load_porcupine_module"] = load_porcupine_module
            captured["load_openwakeword_modules"] = load_openwakeword_modules

        with mock.patch(
            "RealtimeSTT.audio_recorder.initialize_recorder",
            side_effect=fake_initialize_recorder,
        ):
            recorder = AudioToTextRecorder(
                model="base",
                transcription_engine="fake_engine",
                transcription_engine_options=transcription_options,
                download_root="models",
                language="en",
                compute_type="int8",
                input_device_index=2,
                gpu_device_index=gpu_devices,
                device="cpu",
                on_recording_start=callback,
                on_recording_stop=callback,
                on_transcription_start=callback,
                ensure_sentence_starting_uppercase=False,
                ensure_sentence_ends_with_period=False,
                use_microphone=False,
                spinner=False,
                level=10,
                batch_size=3,
                enable_realtime_transcription=True,
                use_main_model_for_realtime=True,
                realtime_transcription_engine="fake_realtime_engine",
                realtime_transcription_engine_options=realtime_options,
                realtime_model_type="tiny.en",
                realtime_processing_pause=0.01,
                init_realtime_after_seconds=0.02,
                on_realtime_transcription_update=callback,
                on_realtime_transcription_stabilized=callback,
                realtime_batch_size=4,
                silero_sensitivity=0.3,
                silero_use_onnx=True,
                silero_deactivity_detection=True,
                webrtc_sensitivity=2,
                warmup_vad=False,
                post_speech_silence_duration=0.5,
                min_length_of_recording=0.2,
                min_gap_between_recordings=0.1,
                pre_recording_buffer_duration=0.3,
                pre_recording_buffer_trim_config=pre_recording_trim_config,
                on_vad_start=callback,
                on_vad_stop=callback,
                on_vad_detect_start=callback,
                on_vad_detect_stop=callback,
                on_turn_detection_start=callback,
                on_turn_detection_stop=callback,
                wakeword_backend="openwakeword",
                openwakeword_model_paths="wake.onnx",
                openwakeword_inference_framework="onnx",
                wake_words="jarvis",
                wake_words_sensitivity=0.7,
                wake_word_activation_delay=0.4,
                wake_word_timeout=1.5,
                wake_word_buffer_duration=0.25,
                on_wakeword_detected=callback,
                on_wakeword_timeout=callback,
                on_wakeword_detection_start=callback,
                on_wakeword_detection_end=callback,
                on_recorded_chunk=callback,
                debug_mode=True,
                handle_buffer_overflow=False,
                beam_size=6,
                beam_size_realtime=2,
                buffer_size=256,
                sample_rate=8000,
                initial_prompt=initial_prompt,
                initial_prompt_realtime=initial_prompt_realtime,
                suppress_tokens=suppress_tokens,
                print_transcription_time=True,
                early_transcription_on_silence=100,
                allowed_latency_limit=12,
                no_log_file=True,
                use_extended_logging=True,
                faster_whisper_vad_filter=False,
                normalize_audio=True,
                start_callback_in_new_thread=True,
                realtime_transcription_use_syllable_boundaries=True,
                realtime_boundary_detector_sensitivity=0.8,
                realtime_boundary_followup_delays=followup_delays,
                transcription_executor=transcription_executor,
                realtime_transcription_executor=realtime_transcription_executor,
                on_realtime_text_stabilization_update=callback,
                silero_backend="raw_onnx",
                silero_onnx_model_path="silero.onnx",
                silero_onnx_threads=1,
                deactivity_silence_confirmation_duration=0.12,
            )

        init_args = captured["init_args"]
        expected_keys = list(
            inspect.signature(AudioToTextRecorder.__init__).parameters
        )

        self.assertEqual(list(init_args), expected_keys)
        self.assertNotIn("init_args", init_args)
        self.assertIs(captured["recorder"], recorder)
        self.assertIs(captured["recorder_cls"], AudioToTextRecorder)
        self.assertIs(init_args["self"], recorder)
        self.assertIs(init_args["transcription_engine_options"], transcription_options)
        self.assertIs(init_args["realtime_transcription_engine_options"], realtime_options)
        self.assertIs(init_args["pre_recording_buffer_trim_config"], pre_recording_trim_config)
        self.assertIs(init_args["gpu_device_index"], gpu_devices)
        self.assertIs(init_args["initial_prompt"], initial_prompt)
        self.assertIs(init_args["initial_prompt_realtime"], initial_prompt_realtime)
        self.assertIs(init_args["suppress_tokens"], suppress_tokens)
        self.assertIs(init_args["realtime_boundary_followup_delays"], followup_delays)
        self.assertIs(init_args["transcription_executor"], transcription_executor)
        self.assertIs(
            init_args["realtime_transcription_executor"],
            realtime_transcription_executor,
        )
        self.assertIs(init_args["on_recording_start"], callback)
        self.assertTrue(init_args["use_extended_logging"])

    def test_internal_recorder_modules_do_not_import_public_facade(self):
        offenders = []

        for path in sorted(INTERNAL_RECORDER_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "RealtimeSTT.audio_recorder":
                            offenders.append(path.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "RealtimeSTT.audio_recorder":
                        offenders.append(path.name)
                    elif node.module == "RealtimeSTT" and any(
                        alias.name == "audio_recorder" for alias in node.names
                    ):
                        offenders.append(path.name)
                    elif node.level > 0 and node.module == "audio_recorder":
                        offenders.append(path.name)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
