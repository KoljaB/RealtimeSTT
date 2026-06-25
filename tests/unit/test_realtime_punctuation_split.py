import logging
import queue
import threading
import time
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
    from RealtimeSTT.core.realtime import (
        _confirm_realtime_punctuation_split_candidate,
        _find_punctuation_split,
        _normalize_realtime_punctuation_split_marks,
        _select_realtime_punctuation_split_hint,
        run_realtime_worker,
    )
    from RealtimeSTT.core.realtime_text_stabilizer import RealtimeTextStabilizer
    from RealtimeSTT.transcription_engines import (
        TranscriptionInfo,
        TranscriptionResult,
    )
except Exception as exc:  # pragma: no cover - optional runtime deps may be absent
    AudioToTextRecorder = None
    run_realtime_worker = None
    RealtimeTextStabilizer = None
    TranscriptionInfo = None
    TranscriptionResult = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def sample_count(frames):
    return sum(len(frame) // 2 for frame in frames)


class FakeRealtimeModel:
    engine_name = "fake_realtime"
    supports_streaming = False

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, language=None, use_prompt=True):
        self.calls += 1
        return TranscriptionResult(
            text="The first clause is stable, And the next keeps going",
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )


class FakeFinalExecutor:
    def __init__(self):
        self.calls = []
        self.plain_calls = 0

    def transcribe(self, audio, language=None, use_prompt=True, **options):
        self.calls.append((int(getattr(audio, "size", 0)), dict(options)))
        info = TranscriptionInfo(language="en", language_probability=1.0)

        if options.get("word_timestamps"):
            return TranscriptionResult(
                text="The first clause is stable, And the next keeps going",
                info=info,
                metadata={
                    "words": [
                        {"word": "The", "start": 0.0, "end": 0.2},
                        {"word": " first", "start": 0.2, "end": 0.4},
                        {"word": " clause", "start": 0.4, "end": 0.6},
                        {"word": " is", "start": 0.6, "end": 0.8},
                        {"word": " stable,", "start": 0.8, "end": 1.0},
                        {"word": " And", "start": 1.0, "end": 1.2},
                        {"word": " the", "start": 1.2, "end": 1.4},
                        {"word": " next", "start": 1.4, "end": 1.6},
                        {"word": " keeps", "start": 1.6, "end": 2.0},
                        {"word": " going", "start": 2.0, "end": 2.4},
                    ],
                },
            )

        self.plain_calls += 1
        if self.plain_calls == 1:
            return TranscriptionResult(text="The first clause is stable,", info=info)
        return TranscriptionResult(text="And the next keeps going", info=info)


class FakeParentPipe:
    def __init__(self):
        self.sent = False

    def send(self, *_args):
        self.sent = True

    def poll(self, timeout=None):
        return True

    def recv(self):
        info = TranscriptionInfo(language="en", language_probability=1.0)
        return "success", TranscriptionResult(text="", info=info, metadata={"words": []})


def result_with_words(*tokens):
    return TranscriptionResult(
        text="".join(token for token, _, _ in tokens),
        info=TranscriptionInfo(language="en", language_probability=1.0),
        metadata={
            "words": [
                {"word": token, "start": start, "end": end}
                for token, start, end in tokens
            ],
        },
    )


class RealtimePunctuationSplitTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"AudioToTextRecorder import failed: {IMPORT_ERROR}")
        if np is None:
            self.skipTest("NumPy is required for realtime punctuation split tests")

    def make_frame(self, samples=1600):
        return (np.ones(samples, dtype=np.int16) * 1000).tobytes()

    def make_recorder_stub(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.enable_realtime_transcription = True
        recorder.is_running = True
        recorder.is_recording = True
        recorder.is_shut_down = False
        recorder.state = "recording"
        recorder.sample_rate = 16000
        recorder.frames = []
        recorder.last_frames = []
        recorder.recorded_audio_queue = queue.Queue()
        recorder.realtime_processing_pause = 0.01
        recorder.init_realtime_after_seconds = 0.0
        recorder.use_main_model_for_realtime = False
        recorder._uses_external_realtime_transcription_executor = False
        recorder.realtime_transcription_executor = None
        recorder.realtime_transcription_model = FakeRealtimeModel()
        recorder.transcription_executor = FakeFinalExecutor()
        recorder._uses_external_transcription_executor = True
        recorder._external_transcription_results = queue.Queue()
        recorder._external_transcription_threads = []
        recorder.transcription_lock = threading.RLock()
        recorder.transcribe_count = 0
        recorder.language = "en"
        recorder.main_model_type = "fake"
        recorder.realtime_model_type = "fake"
        recorder.realtime_punctuation_split_marks = "sentence,comma"
        recorder._realtime_punctuation_split_lock = threading.RLock()
        recorder._force_current_recording_lowercase_start = False
        recorder._current_transcription_force_lowercase_start = False
        recorder.realtime_transcription_count = 0
        recorder.realtime_transcription_success_count = 0
        recorder.realtime_transcription_empty_count = 0
        recorder.realtime_transcription_trigger_counts = {}
        recorder.realtime_observation_sequence = 0
        recorder.realtime_recording_id = 1
        recorder.recording_start_monotonic = time.monotonic() - 3.0
        recorder.recording_start_time = time.time() - 3.0
        recorder.recording_stop_time = 0
        recorder.min_length_of_recording = 0.0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.listen_start = 0
        recorder.start_recording_on_voice_activity = False
        recorder.stop_recording_on_voice_deactivity = False
        recorder.continuous_listening = False
        recorder.use_wake_words = False
        recorder.awaiting_speech_end = False
        recorder.text_storage = []
        recorder.audio = None
        recorder.detected_language = None
        recorder.detected_language_probability = 0.0
        recorder.print_transcription_time = False
        recorder.ensure_sentence_starting_uppercase = False
        recorder.ensure_sentence_ends_with_period = False
        recorder.start_callback_in_new_thread = False
        recorder.spinner = False
        recorder.halo = None
        recorder.wake_words = ""
        recorder.on_recording_stop = None
        recorder.on_transcription_start = None
        recorder.on_vad_detect_start = None
        recorder.on_vad_detect_stop = None
        recorder.on_wakeword_detection_start = None
        recorder.on_wakeword_detection_end = None
        recorder.on_realtime_transcription_update = None
        recorder.on_realtime_transcription_stabilized = None
        recorder.on_realtime_text_stabilization_update = None
        recorder.interrupt_stop_event = threading.Event()
        recorder.was_interrupted = threading.Event()
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.shutdown_event = threading.Event()
        recorder.realtime_text_stabilizer = RealtimeTextStabilizer()
        recorder.realtime_text_stabilizer.reset(
            recorder.realtime_recording_id,
            started_at_monotonic=recorder.recording_start_monotonic,
            started_at_wall_time=recorder.recording_start_time,
        )
        logging.getLogger("realtimestt").setLevel(logging.CRITICAL)
        return recorder

    def run_worker(self, recorder):
        thread = threading.Thread(
            target=run_realtime_worker,
            args=(recorder,),
            daemon=True,
        )
        thread.start()
        return thread

    def stop_worker(self, recorder, thread):
        recorder.is_running = False
        recorder.stop_recording_event.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_stable_punctuation_splits_buffer_and_lowercases_next_final(self):
        recorder = self.make_recorder_stub()
        thread = self.run_worker(recorder)

        try:
            total_input_samples = 0
            for _ in range(8):
                frame = self.make_frame(samples=8000)
                total_input_samples += 8000
                recorder.frames.append(frame)
                time.sleep(0.15)
                if (
                    recorder.recorded_audio_queue.qsize() == 1
                    and recorder._force_current_recording_lowercase_start
                ):
                    break

            self.assertTrue(
                wait_until(
                    lambda: (
                        recorder.recorded_audio_queue.qsize() == 1
                        and recorder._force_current_recording_lowercase_start
                    ),
                    timeout=2.0,
                )
            )
            self.assertFalse(recorder.stop_recording_event.is_set())
            self.stop_worker(recorder, thread)

            with recorder.recorded_audio_queue.mutex:
                queued = recorder.recorded_audio_queue.queue[0]

            self.assertEqual(sample_count(queued["frames"]), 16000)
            self.assertFalse(queued["force_lowercase_start"])
            self.assertEqual(
                sample_count(queued["frames"]) + sample_count(recorder.frames),
                total_input_samples,
            )

            first_text = recorder.text()
            self.assertEqual(first_text, "The first clause is stable,")
            self.assertTrue(recorder.is_recording)

            recorder.stop()
            second_text = recorder.text()
            self.assertEqual(second_text, "and the next keeps going")
            self.assertEqual(
                f"{first_text} {second_text}",
                "The first clause is stable, and the next keeps going",
            )
            self.assertIn(
                True,
                [
                    call_options.get("word_timestamps", False)
                    for _, call_options in recorder.transcription_executor.calls
                ],
            )
        finally:
            recorder.is_recording = False
            if thread.is_alive():
                self.stop_worker(recorder, thread)

    def test_punctuation_split_does_not_block_on_busy_final_transcription(self):
        recorder = self.make_recorder_stub()
        recorder._uses_external_transcription_executor = False
        recorder.parent_transcription_pipe = FakeParentPipe()
        recorder.transcription_lock.acquire()
        thread = self.run_worker(recorder)

        try:
            for _ in range(8):
                recorder.frames.append(self.make_frame(samples=8000))
                time.sleep(0.05)
                if getattr(recorder, "_last_realtime_punctuation_split_attempt_text", ""):
                    break

            self.assertTrue(
                wait_until(
                    lambda: bool(
                        getattr(
                            recorder,
                            "_last_realtime_punctuation_split_attempt_text",
                            "",
                        )
                    ),
                    timeout=1.0,
                )
            )
            recorder.is_running = False
            recorder.stop_recording_event.set()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertFalse(recorder.parent_transcription_pipe.sent)
        finally:
            recorder.transcription_lock.release()
            recorder.is_recording = False
            if thread.is_alive():
                self.stop_worker(recorder, thread)

    def test_display_comma_can_trigger_when_stable_text_lost_punctuation(self):
        event = type("Event", (), {
            "stable_text": "The first clause is stable and the next",
            "display_text": "The first clause is stable, and the next clause continues",
        })()
        hint = _select_realtime_punctuation_split_hint(event)
        self.assertEqual(hint, event.display_text)
        self.assertEqual(
            _find_punctuation_split(
                result_with_words(
                    ("The", 0.0, 0.1),
                    (" first", 0.1, 0.2),
                    (" clause", 0.2, 0.3),
                    (" is", 0.3, 0.4),
                    (" stable", 0.4, 1.0),
                    (" and", 1.0, 1.2),
                    (" the", 1.2, 1.3),
                ),
                hint,
            ),
            (",", 1.0),
        )

    def test_stable_period_does_not_require_known_next_word(self):
        event = type("Event", (), {
            "stable_text": "This first sentence should split now. also the next part continues",
            "display_text": "This first sentence should split now. also the next part continues",
        })()
        hint = _select_realtime_punctuation_split_hint(event)
        self.assertEqual(hint, event.stable_text)
        split = _find_punctuation_split(
            result_with_words(
                ("This", 0.0, 0.1),
                (" first", 0.1, 0.2),
                (" sentence", 0.2, 0.3),
                (" should", 0.3, 0.4),
                (" split", 0.4, 0.5),
                (" now", 0.5, 1.0),
                (" also", 1.2, 1.3),
                (" the", 1.3, 1.4),
            ),
            hint,
        )
        self.assertEqual(split[0], ".")
        self.assertAlmostEqual(split[1], 1.1)

    def test_question_mark_is_supported_as_terminal_split(self):
        split = _find_punctuation_split(
            result_with_words(
                ("Can", 0.0, 0.1),
                (" this", 0.1, 0.2),
                (" question", 0.2, 0.3),
                (" be", 0.3, 0.4),
                (" split", 0.4, 0.5),
                (" correctly", 0.5, 1.0),
                (" the", 1.0, 1.1),
                (" next", 1.1, 1.2),
            ),
            "Can this question be split correctly? The next answer follows",
        )
        self.assertEqual(split, ("?", 1.0))

    def test_decimal_period_and_comma_are_not_split_points(self):
        split = _find_punctuation_split(
            result_with_words(
                ("The", 0.0, 0.1),
                (" value", 0.1, 0.2),
                (" was", 0.2, 0.3),
                (" 3", 0.3, 0.4),
                (".", 0.4, 0.41),
                ("14", 0.41, 0.6),
                (" and", 0.6, 0.7),
            ),
            "The value was 3.14 and it continued",
        )
        self.assertIsNone(split)
        self.assertIsNone(
            _find_punctuation_split(
                result_with_words(
                    ("The", 0.0, 0.1),
                    (" value", 0.1, 0.2),
                    (" was", 0.2, 0.3),
                    (" 3", 0.3, 0.4),
                    (" 14", 0.4, 0.6),
                    (" and", 0.6, 0.7),
                ),
                "The value was 3, 14, and it continued",
            )
        )

    def test_common_abbreviation_period_is_not_a_split_point(self):
        self.assertIsNone(
            _find_punctuation_split(
                result_with_words(
                    ("Dr", 0.0, 0.2),
                    (" Smith", 0.2, 0.5),
                    (" explained", 0.5, 0.8),
                    (" the", 0.8, 0.9),
                ),
                "Dr. Smith explained the result",
            )
        )

    def test_comma_after_abbreviation_context_is_not_a_split_point(self):
        self.assertIsNone(
            _find_punctuation_split(
                result_with_words(
                    ("Dr", 0.0, 0.2),
                    (" Smith", 0.2, 0.5),
                    (" explained", 0.5, 0.8),
                    (" the", 0.8, 0.9),
                    (" result", 0.9, 1.0),
                    (" and", 1.0, 1.1),
                    (" the", 1.1, 1.2),
                    (" audience", 1.2, 1.3),
                ),
                "Dr. Smith explained the result, and the audience continued",
            )
        )

    def test_comma_list_is_not_split_inside_the_list(self):
        self.assertIsNone(
            _find_punctuation_split(
                result_with_words(
                    ("We", 0.0, 0.1),
                    (" need", 0.1, 0.2),
                    (" red", 0.2, 0.3),
                    (" green", 0.3, 0.4),
                    (" and", 0.4, 0.5),
                    (" blue", 0.5, 0.6),
                    (" before", 0.6, 0.7),
                ),
                "We need red, green, and blue before the display continues",
            )
        )

    def test_split_marks_off_disables_punctuation_hints(self):
        event = type("Event", (), {
            "stable_text": "This sentence should not split. even with more text",
        })()

        self.assertEqual(_normalize_realtime_punctuation_split_marks("off"), ())
        self.assertEqual(_select_realtime_punctuation_split_hint(event, "off"), "")

    def test_sentence_marks_do_not_allow_comma_split(self):
        self.assertIsNone(
            _find_punctuation_split(
                result_with_words(
                    ("The", 0.0, 0.1),
                    (" first", 0.1, 0.2),
                    (" clause", 0.2, 0.3),
                    (" is", 0.3, 0.4),
                    (" stable", 0.4, 1.0),
                    (" and", 1.0, 1.2),
                    (" the", 1.2, 1.3),
                ),
                "The first clause is stable, and the next clause continues",
                "sentence",
            )
        )

    def test_sentence_comma_marks_allow_comma_split(self):
        split = _find_punctuation_split(
            result_with_words(
                ("The", 0.0, 0.1),
                (" first", 0.1, 0.2),
                (" clause", 0.2, 0.3),
                (" is", 0.3, 0.4),
                (" stable", 0.4, 1.0),
                (" and", 1.0, 1.2),
                (" the", 1.2, 1.3),
            ),
            "The first clause is stable, and the next clause continues",
            "sentence,comma",
        )
        self.assertEqual(split, (",", 1.0))

    def test_punctuation_split_candidate_needs_repeated_observations(self):
        recorder = type("Recorder", (), {})()
        split_marks = _normalize_realtime_punctuation_split_marks("sentence")
        hint = (".", ("sentence", "should", "split"))

        self.assertFalse(
            _confirm_realtime_punctuation_split_candidate(
                recorder,
                split_marks,
                hint,
            )
        )
        self.assertFalse(
            _confirm_realtime_punctuation_split_candidate(
                recorder,
                split_marks,
                hint,
            )
        )
        self.assertTrue(
            _confirm_realtime_punctuation_split_candidate(
                recorder,
                split_marks,
                hint,
            )
        )

    def test_ellipsis_mark_is_supported(self):
        split = _find_punctuation_split(
            result_with_words(
                ("This", 0.0, 0.1),
                (" trails", 0.1, 0.4),
                (" off", 0.4, 1.0),
                (" and", 1.0, 1.1),
                (" then", 1.1, 1.2),
            ),
            "This trails off... and then continues",
            ("...",),
        )
        self.assertEqual(split, ("...", 1.0))

    def test_dash_marks_are_supported_without_splitting_compounds(self):
        split = _find_punctuation_split(
            result_with_words(
                ("This", 0.0, 0.1),
                (" thought", 0.1, 0.2),
                (" should", 0.2, 0.3),
                (" pause", 0.3, 1.0),
                (" and", 1.0, 1.1),
                (" then", 1.1, 1.2),
            ),
            "This thought should pause \u2014 and then continue",
            "dash",
        )
        self.assertEqual(split, ("\u2014", 1.0))
        self.assertIsNone(
            _find_punctuation_split(
                result_with_words(
                    ("This", 0.0, 0.1),
                    (" real-time", 0.1, 0.5),
                    (" path", 0.5, 0.7),
                    (" continues", 0.7, 1.0),
                ),
                "This real-time path continues with words",
                ("-",),
            )
        )

if __name__ == "__main__":
    unittest.main()
