import json
import queue
import threading
import unittest

import numpy as np

from example_fastapi_server.protocol import (
    AudioPacketError,
    decode_audio_packet,
    encode_audio_packet,
    normalize_engine_name,
    parse_json_object,
    require_positive_int,
)
from RealtimeSTT.audio_recorder import AudioToTextRecorder
from RealtimeSTT.transcription_engines import (
    TranscriptionResult,
    get_supported_transcription_engines,
)
from example_fastapi_server.server import (
    ConnectionManager,
    FairInferenceQueue,
    InferenceJob,
    RealtimeSTTService,
    SegmentState,
    ServerSettings,
    parse_args,
    settings_from_args,
)


class FastAPIServerProtocolTests(unittest.TestCase):
    def test_audio_packet_round_trip(self):
        metadata = {
            "sampleRate": 48000,
            "channels": 1,
            "format": "pcm_s16le",
            "frames": 4,
        }
        audio = b"\x01\x00\x02\x00"

        packet = decode_audio_packet(encode_audio_packet(metadata, audio))

        self.assertEqual(packet.metadata, metadata)
        self.assertEqual(packet.audio, audio)

    def test_decode_rejects_invalid_packets(self):
        with self.assertRaisesRegex(AudioPacketError, "missing metadata length"):
            decode_audio_packet(b"\x00\x01")

        with self.assertRaisesRegex(AudioPacketError, "incomplete"):
            decode_audio_packet(b"\x10\x00\x00\x00{}")

        with self.assertRaisesRegex(AudioPacketError, "invalid JSON"):
            decode_audio_packet(b"\x01\x00\x00\x00{")

    def test_require_positive_int(self):
        self.assertEqual(require_positive_int({"sampleRate": 16000}, "sampleRate"), 16000)

        for value in (0, -1, "16000", True, None):
            with self.subTest(value=value):
                with self.assertRaises(AudioPacketError):
                    require_positive_int({"sampleRate": value}, "sampleRate")

    def test_json_object_parser(self):
        self.assertEqual(parse_json_object('{"a": 1}', "--flag"), {"a": 1})
        self.assertIsNone(parse_json_object(None, "--flag"))

        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_json_object("[1, 2]", "--flag")

        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_json_object("{", "--flag")

    def test_settings_normalize_hyphenated_engine_names(self):
        args = parse_args([
            "--engine",
            "cohere-transcribe",
            "--realtime-engine",
            "moonshine-streaming",
            "--engine-options",
            json.dumps({"language": "en"}),
        ])

        settings = settings_from_args(args)

        self.assertEqual(settings.transcription_engine, "cohere_transcribe")
        self.assertEqual(settings.realtime_transcription_engine, "moonshine_streaming")
        self.assertEqual(settings.transcription_engine_options, {"language": "en"})

    def test_parakeet_tuning_profile_sets_latency_oriented_defaults(self):
        args = parse_args([
            "--profile",
            "parakeet-low-latency",
            "--engine",
            "parakeet",
        ])

        settings = settings_from_args(args)

        self.assertEqual(settings.tuning_profile, "parakeet-low-latency")
        self.assertEqual(settings.batch_size, 1)
        self.assertEqual(settings.realtime_batch_size, 1)
        self.assertEqual(settings.realtime_processing_pause, 0.04)
        self.assertEqual(settings.post_speech_silence_duration, 0.45)

    def test_explicit_tuning_flags_override_profile_defaults(self):
        args = parse_args([
            "--profile",
            "parakeet-low-latency",
            "--batch-size",
            "4",
            "--realtime-processing-pause",
            "0.09",
        ])

        settings = settings_from_args(args)

        self.assertEqual(settings.batch_size, 4)
        self.assertEqual(settings.realtime_processing_pause, 0.09)

    def test_normalize_engine_name_handles_none(self):
        self.assertIsNone(normalize_engine_name(None))
        self.assertEqual(normalize_engine_name("Qwen3-ASR"), "qwen3_asr")

    def test_supported_engines_include_kroko_for_fastapi_config(self):
        engines = get_supported_transcription_engines()

        for name in ("kroko_onnx", "kroko", "banafo_kroko"):
            self.assertIn(name, engines)

    def test_segment_ids_replace_realtime_with_final(self):
        state = SegmentState()

        self.assertEqual(state.realtime(), 1)
        self.assertEqual(state.realtime(), 1)
        self.assertEqual(state.final(), 1)
        self.assertEqual(state.realtime(), 2)
        self.assertEqual(state.final(), 2)

    def test_public_settings_hide_backend_option_payloads(self):
        settings = ServerSettings(
            transcription_engine_options={"token": "secret"},
            realtime_transcription_engine_options={"other": "secret"},
        )

        public = settings.public_dict()

        self.assertNotIn("transcription_engine_options", public)
        self.assertNotIn("realtime_transcription_engine_options", public)
        self.assertEqual(public["transcription_engine"], "faster_whisper")

    def test_new_multi_user_settings_parse_from_cli(self):
        args = parse_args([
            "--max-sessions",
            "12",
            "--max-active-speakers",
            "3",
            "--max-realtime-queue-age-ms",
            "900",
            "--max-final-queue-depth-per-session",
            "2",
            "--max-global-inference-queue-depth",
            "17",
            "--use-main-model-for-realtime",
            "--no-model-warmup",
        ])

        settings = settings_from_args(args)

        self.assertEqual(settings.max_sessions, 12)
        self.assertEqual(settings.max_active_speakers, 3)
        self.assertEqual(settings.max_realtime_queue_age_ms, 900)
        self.assertEqual(settings.max_final_queue_depth_per_session, 2)
        self.assertEqual(settings.max_global_inference_queue_depth, 17)
        self.assertTrue(settings.use_main_model_for_realtime)
        self.assertFalse(settings.model_warmup)

    def test_wake_word_settings_parse_from_cli(self):
        args = parse_args([
            "--wakeword-backend",
            "pvporcupine",
            "--wake-words",
            "jarvis",
            "--wake-words-sensitivity",
            "0.7",
            "--wake-word-timeout",
            "3.25",
            "--wake-word-buffer-duration",
            "0.2",
            "--wake-word-followup-window",
            "5",
        ])

        settings = settings_from_args(args)

        self.assertTrue(settings.wake_word_enabled())
        self.assertEqual(settings.wakeword_backend, "pvporcupine")
        self.assertEqual(settings.wake_words, "jarvis")
        self.assertEqual(settings.wake_words_sensitivity, 0.7)
        self.assertEqual(settings.wake_word_timeout, 3.25)
        self.assertEqual(settings.wake_word_buffer_duration, 0.2)
        self.assertEqual(settings.wake_word_followup_window, 5.0)
        self.assertTrue(settings.public_dict()["wake_word_enabled"])

    def test_runtime_settings_update_distinguishes_safe_and_startup_only_fields(self):
        service = RealtimeSTTService(ServerSettings(max_sessions=1), ConnectionManager())

        result = service.update_settings({
            "max_sessions": 3,
            "wake_words": "jarvis",
            "transcription_engine": "moonshine",
            "unknown_setting": 1,
        })

        self.assertEqual(service.settings.max_sessions, 3)
        self.assertEqual(service.settings.wake_words, "jarvis")
        self.assertEqual(result["applied"]["max_sessions"]["appliesTo"], "active_sessions")
        self.assertEqual(result["applied"]["wake_words"]["appliesTo"], "new_sessions")
        self.assertEqual(result["rejected"]["transcription_engine"]["reason"], "startup_only")
        self.assertEqual(result["rejected"]["unknown_setting"]["reason"], "unknown")

    def test_packet_to_server_samples_accepts_valid_pcm_packet(self):
        service = RealtimeSTTService(ServerSettings(audio_queue_size=1), ConnectionManager())
        packet = decode_audio_packet(encode_audio_packet(
            {"sampleRate": 48000, "channels": 1, "format": "pcm_s16le"},
            b"\x01\x00\x02\x00",
        ))

        samples = service.packet_to_server_samples(packet)

        self.assertEqual(samples.dtype.name, "int16")
        self.assertGreaterEqual(samples.size, 1)

    def test_packet_to_server_samples_rejects_bad_pcm_shape(self):
        service = RealtimeSTTService(ServerSettings(), ConnectionManager())

        invalid_packets = [
            ({"sampleRate": 48000, "channels": 1, "format": "float32"}, b"\x00\x00"),
            ({"sampleRate": 48000, "channels": 9, "format": "pcm_s16le"}, b"\x00\x00"),
            ({"sampleRate": 48000, "channels": 2, "format": "pcm_s16le"}, b"\x00\x00"),
        ]

        for metadata, audio in invalid_packets:
            with self.subTest(metadata=metadata, audio=audio):
                packet = decode_audio_packet(encode_audio_packet(metadata, audio))
                with self.assertRaises(AudioPacketError):
                    service.packet_to_server_samples(packet)

    def test_packet_to_server_samples_rejects_frame_count_mismatch(self):
        service = RealtimeSTTService(ServerSettings(), ConnectionManager())
        packet = decode_audio_packet(encode_audio_packet(
            {"sampleRate": 16000, "channels": 1, "format": "pcm_s16le", "frames": 3},
            np.zeros(2, dtype=np.int16).tobytes(),
        ))

        with self.assertRaisesRegex(AudioPacketError, "frames"):
            service.packet_to_server_samples(packet)

    def test_fair_queue_coalesces_realtime_but_preserves_finals(self):
        settings = ServerSettings(max_global_inference_queue_depth=8)
        dropped = []
        queue = FairInferenceQueue("test", settings, lambda job, reason, lane: dropped.append((job, reason)))

        first_realtime = self._job("a", "realtime", sequence=1)
        second_realtime = self._job("a", "realtime", sequence=2)
        final = self._job("a", "final", segment_id=1)

        self.assertTrue(queue.submit(first_realtime).accepted)
        coalesced = queue.submit(second_realtime)
        self.assertTrue(coalesced.accepted)
        self.assertTrue(coalesced.coalesced)
        self.assertTrue(queue.submit(final).accepted)

        first_job = queue.get()
        second_job = queue.get()

        self.assertEqual(first_job.kind, "final")
        self.assertEqual(second_job.kind, "realtime")
        self.assertEqual(second_job.sequence, 2)
        self.assertEqual([(job.sequence, reason) for job, reason in dropped], [(1, "coalesced")])

    def test_fair_queue_round_robins_between_sessions(self):
        queue = FairInferenceQueue("test", ServerSettings(), None)
        self.assertTrue(queue.submit(self._job("a", "final")).accepted)
        self.assertTrue(queue.submit(self._job("a", "final", segment_id=2)).accepted)
        self.assertTrue(queue.submit(self._job("b", "final")).accepted)

        first = queue.get()
        second = queue.get()
        third = queue.get()

        self.assertEqual(first.session_id, "a")
        self.assertEqual(second.session_id, "b")
        self.assertEqual(third.session_id, "a")

    def test_fair_queue_drops_stale_realtime_jobs(self):
        dropped = []
        queue = FairInferenceQueue("test", ServerSettings(), lambda job, reason, lane: dropped.append((job, reason)))
        stale = self._job("a", "realtime", deadline_at=0.001)
        fresh = self._job("b", "realtime", deadline_at=time_now_plus(5))

        self.assertTrue(queue.submit(stale).accepted)
        self.assertTrue(queue.submit(fresh).accepted)

        self.assertEqual(queue.get().session_id, "b")
        self.assertEqual([(job.session_id, reason) for job, reason in dropped], [("a", "stale")])

    def test_recorder_external_executor_preserves_final_transcription_path(self):
        class Executor:
            def __init__(self):
                self.calls = []

            def transcribe(self, audio, language=None, use_prompt=True):
                self.calls.append((audio.copy(), language, use_prompt))
                return TranscriptionResult(text="external executor text")

        executor = Executor()
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.transcription_lock = threading.Lock()
        recorder._uses_external_transcription_executor = True
        recorder.transcription_executor = executor
        recorder._external_transcription_results = queue.Queue()
        recorder._external_transcription_threads = []
        recorder.transcribe_count = 0
        recorder.audio = np.ones(16, dtype=np.float32)
        recorder.language = "en"
        recorder.interrupt_stop_event = threading.Event()
        recorder.was_interrupted = threading.Event()
        recorder._set_state = lambda state: None
        recorder.is_recording = False
        recorder.allowed_to_early_transcribe = True
        recorder.detected_language = None
        recorder.detected_language_probability = 0
        recorder.last_transcription_bytes = None
        recorder.last_transcription_bytes_b64 = None
        recorder.print_transcription_time = False
        recorder.main_model_type = "fake"
        recorder.ensure_sentence_starting_uppercase = False
        recorder.ensure_sentence_ends_with_period = False

        text = recorder.perform_final_transcription(use_prompt=False)

        self.assertEqual(text, "external executor text")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0][1:], ("en", False))

    def _job(self, session_id, kind, segment_id=1, sequence=0, deadline_at=None):
        return InferenceJob(
            request_id=f"{session_id}-{kind}-{segment_id}-{sequence}",
            session_id=session_id,
            kind=kind,
            audio=b"",
            language="en",
            use_prompt=True,
            segment_id=segment_id,
            sequence=sequence,
            generation=0,
            created_at=time_now_plus(0),
            deadline_at=deadline_at,
        )


def time_now_plus(seconds):
    import time

    return time.monotonic() + seconds


if __name__ == "__main__":
    unittest.main()
