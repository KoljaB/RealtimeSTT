import threading
import time
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
    from RealtimeSTT.core.realtime import run_realtime_worker
    from RealtimeSTT.core.realtime_merge import StickyRealtimeTranscriptionMerger
    from RealtimeSTT.core.realtime_text_stabilizer import RealtimeTextStabilizer
    from RealtimeSTT.transcription_engines import (
        TranscriptionInfo,
        TranscriptionResult,
    )
except Exception as exc:  # pragma: no cover - import guard for optional deps
    AudioToTextRecorder = None
    run_realtime_worker = None
    RealtimeTextStabilizer = None
    StickyRealtimeTranscriptionMerger = None
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


class FakeStreamingSession:
    def __init__(self):
        self.accepted_sample_counts = []
        self.decode_calls = 0
        self.finished = False
        self.closed = False
        self.total_samples = 0

    def accept_audio(self, audio, sample_rate=None):
        sample_count = int(getattr(audio, "size", 0))
        self.accepted_sample_counts.append(sample_count)
        self.total_samples += sample_count

    def decode(self):
        self.decode_calls += 1

    def get_result(self):
        return TranscriptionResult(
            text="streamed %d" % self.total_samples,
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )

    def finish(self):
        self.finished = True
        return self.get_result()

    def close(self):
        self.closed = True


class FakeStreamingModel:
    engine_name = "fake_streaming"
    supports_streaming = True

    def __init__(self):
        self.sessions = []
        self.transcribe_calls = []

    def create_streaming_session(self, language=None, use_prompt=True):
        session = FakeStreamingSession()
        self.sessions.append(session)
        return session

    def transcribe(self, audio, language=None, use_prompt=True):
        self.transcribe_calls.append(int(getattr(audio, "size", 0)))
        return TranscriptionResult(text="unexpected full-buffer call")


class ScriptedStreamingSession(FakeStreamingSession):
    def __init__(self, texts_by_sample_count):
        super().__init__()
        self.texts_by_sample_count = dict(texts_by_sample_count)

    def get_result(self):
        eligible = [
            sample_count
            for sample_count in self.texts_by_sample_count
            if sample_count <= self.total_samples
        ]
        text = self.texts_by_sample_count[max(eligible)] if eligible else ""
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )


class ScriptedStreamingModel(FakeStreamingModel):
    def __init__(self, texts_by_sample_count, engine_name):
        super().__init__()
        self.texts_by_sample_count = dict(texts_by_sample_count)
        self.engine_name = engine_name

    def create_streaming_session(self, language=None, use_prompt=True):
        session = ScriptedStreamingSession(self.texts_by_sample_count)
        self.sessions.append(session)
        return session


class BlockingStreamingSession(FakeStreamingSession):
    def __init__(self, entered, release, text):
        super().__init__()
        self.entered = entered
        self.release = release
        self.text = text

    def get_result(self):
        self.entered.set()
        self.release.wait(timeout=2.0)
        return TranscriptionResult(
            text=self.text,
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )


class BlockingStreamingModel(FakeStreamingModel):
    def __init__(self, entered, release, text):
        super().__init__()
        self.entered = entered
        self.release = release
        self.text = text

    def create_streaming_session(self, language=None, use_prompt=True):
        session = BlockingStreamingSession(
            self.entered,
            self.release,
            self.text,
        )
        self.sessions.append(session)
        return session


class FakeNonStreamingModel:
    engine_name = "fake_non_streaming"
    supports_streaming = False

    def __init__(self):
        self.transcribe_calls = []

    def transcribe(self, audio, language=None, use_prompt=True):
        sample_count = int(getattr(audio, "size", 0))
        self.transcribe_calls.append(sample_count)
        return TranscriptionResult(
            text="full %d" % sample_count,
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )


class AudioRecorderRealtimeStreamingTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"AudioToTextRecorder import failed: {IMPORT_ERROR}")
        if np is None:
            self.skipTest("NumPy is required for realtime streaming tests")

    def make_recorder_stub(self, model):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.enable_realtime_transcription = True
        recorder.is_running = True
        recorder.is_recording = True
        recorder.realtime_processing_pause = 0.01
        recorder.sample_rate = 16000
        recorder.frames = []
        recorder.last_frames = []
        recorder.realtime_transcription_model = model
        recorder.ultrafast_realtime_transcription_model = None
        recorder.ultrafast_realtime_model_type = None
        recorder.on_ultrafast_transcription_update = None
        recorder.on_merged_realtime_transcription_update = None
        recorder.on_realtime_transcription_merge_update = None
        recorder.realtime_transcription_merger = (
            StickyRealtimeTranscriptionMerger()
        )
        recorder.last_ultrafast_transcription = ""
        recorder.last_merged_realtime_transcription = ""
        recorder.last_realtime_transcription_merge_result = None
        recorder.use_main_model_for_realtime = False
        recorder._uses_external_realtime_transcription_executor = False
        recorder.realtime_transcription_executor = None
        recorder.language = "en"
        recorder.realtime_transcription_count = 0
        recorder.realtime_transcription_success_count = 0
        recorder.realtime_transcription_empty_count = 0
        recorder.realtime_transcription_trigger_counts = {}
        recorder.realtime_observation_sequence = 0
        recorder.realtime_recording_id = 1
        recorder.recording_start_monotonic = time.monotonic()
        recorder.recording_start_time = time.time() - 1.0
        recorder.init_realtime_after_seconds = 0.0
        recorder.realtime_text_stabilizer = RealtimeTextStabilizer()
        recorder.realtime_text_stabilizer.reset(
            recorder.realtime_recording_id,
            started_at_monotonic=recorder.recording_start_monotonic,
            started_at_wall_time=recorder.recording_start_time,
        )
        recorder.text_storage = []
        recorder.realtime_transcription_text = ""
        recorder.realtime_stabilized_text = ""
        recorder.realtime_stabilized_safetext = ""
        recorder.realtime_text_stabilization_event = None
        recorder.realtime_stabilization_accepted_count = 0
        recorder.realtime_stabilization_outlier_count = 0
        recorder.realtime_stabilization_stable_delta_count = 0
        recorder.realtime_transcription_use_syllable_boundaries = False
        recorder.awaiting_speech_end = False
        recorder.on_realtime_text_stabilization_update = None
        recorder.on_realtime_transcription_update = lambda text: None
        recorder.on_realtime_transcription_stabilized = None
        recorder.start_callback_in_new_thread = False
        recorder.ensure_sentence_starting_uppercase = False
        recorder.ensure_sentence_ends_with_period = False
        recorder.realtime_model_type = "fake"
        return recorder

    def run_worker(self, recorder):
        thread = threading.Thread(
            target=run_realtime_worker,
            args=(recorder,),
            daemon=True,
        )
        thread.start()
        return thread

    def make_frame(self, samples=1600):
        return np.arange(samples, dtype=np.int16).tobytes()

    def stop_worker(self, recorder, thread):
        recorder.is_recording = False
        recorder.is_running = False
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_streaming_realtime_model_receives_only_new_frames(self):
        model = FakeStreamingModel()
        recorder = self.make_recorder_stub(model)
        thread = self.run_worker(recorder)

        try:
            recorder.frames.append(self.make_frame())
            self.assertTrue(
                wait_until(
                    lambda: model.sessions
                    and len(model.sessions[0].accepted_sample_counts) >= 1
                )
            )

            recorder.frames.append(self.make_frame())
            self.assertTrue(
                wait_until(
                    lambda: len(model.sessions[0].accepted_sample_counts) >= 2
                )
            )

            recorder.last_frames = list(recorder.frames)
            recorder.is_recording = False
            self.assertTrue(wait_until(lambda: model.sessions[0].finished))
        finally:
            self.stop_worker(recorder, thread)

        session = model.sessions[0]
        self.assertEqual(model.transcribe_calls, [])
        self.assertEqual(session.accepted_sample_counts[:2], [1600, 1600])
        self.assertEqual(session.total_samples, 3200)
        self.assertGreaterEqual(session.decode_calls, 2)

    def test_non_streaming_realtime_model_keeps_full_buffer_fallback(self):
        model = FakeNonStreamingModel()
        recorder = self.make_recorder_stub(model)
        thread = self.run_worker(recorder)

        try:
            recorder.frames.append(self.make_frame())
            self.assertTrue(wait_until(lambda: len(model.transcribe_calls) >= 1))

            recorder.frames.append(self.make_frame())
            self.assertTrue(wait_until(lambda: 3200 in model.transcribe_calls))
        finally:
            self.stop_worker(recorder, thread)

        self.assertEqual(model.transcribe_calls[0], 1600)
        self.assertIn(3200, model.transcribe_calls)

    def test_dual_streaming_models_receive_identical_new_frames(self):
        slow_model = ScriptedStreamingModel(
            {1600: "the quick brown fox"},
            "slow_streaming",
        )
        ultrafast_model = ScriptedStreamingModel(
            {1600: "the quick brown fox jumps"},
            "ultrafast_streaming",
        )
        recorder = self.make_recorder_stub(slow_model)
        recorder.ultrafast_realtime_transcription_model = ultrafast_model
        recorder.ultrafast_realtime_model_type = "ultrafast"
        slow_updates = []
        ultrafast_updates = []
        merged_updates = []
        merge_results = []
        callback_order = []
        recorder.on_realtime_transcription_update = lambda text: (
            slow_updates.append(text),
            callback_order.append("slow"),
        )
        recorder.on_ultrafast_transcription_update = lambda text: (
            ultrafast_updates.append(text),
            callback_order.append("ultrafast"),
        )
        recorder.on_merged_realtime_transcription_update = lambda text: (
            merged_updates.append(text),
            callback_order.append("merged"),
        )
        recorder.on_realtime_transcription_merge_update = merge_results.append
        thread = self.run_worker(recorder)

        try:
            recorder.frames.append(self.make_frame())
            self.assertTrue(wait_until(lambda: bool(merged_updates)))
        finally:
            self.stop_worker(recorder, thread)

        self.assertEqual(
            slow_model.sessions[0].accepted_sample_counts[:1],
            [1600],
        )
        self.assertEqual(
            ultrafast_model.sessions[0].accepted_sample_counts[:1],
            [1600],
        )
        self.assertEqual(slow_updates[-1], "the quick brown fox")
        self.assertEqual(
            ultrafast_updates[-1],
            "the quick brown fox jumps",
        )
        self.assertEqual(
            merged_updates[-1],
            "the quick brown fox jumps",
        )
        self.assertEqual(merge_results[-1].status, "exact")
        self.assertEqual(merge_results[-1].ultrafast_suffix, "jumps")
        self.assertLess(
            callback_order.index("ultrafast"),
            callback_order.index("slow"),
        )
        self.assertLess(
            callback_order.index("slow"),
            callback_order.index("merged"),
        )
        self.assertTrue(slow_model.sessions[0].finished)
        self.assertTrue(slow_model.sessions[0].closed)
        self.assertTrue(ultrafast_model.sessions[0].finished)
        self.assertTrue(ultrafast_model.sessions[0].closed)

    def test_shared_model_uses_two_independent_streaming_sessions(self):
        shared_model = ScriptedStreamingModel(
            {1600: "the quick brown fox"},
            "shared_streaming",
        )
        recorder = self.make_recorder_stub(shared_model)
        recorder.ultrafast_realtime_transcription_model = shared_model
        recorder.ultrafast_realtime_model_type = "shared"
        merged_updates = []
        recorder.on_merged_realtime_transcription_update = (
            merged_updates.append
        )
        thread = self.run_worker(recorder)

        try:
            recorder.frames.append(self.make_frame())
            self.assertTrue(wait_until(lambda: bool(merged_updates)))
        finally:
            self.stop_worker(recorder, thread)

        self.assertEqual(len(shared_model.sessions), 2)
        self.assertIsNot(shared_model.sessions[0], shared_model.sessions[1])
        self.assertEqual(
            [
                session.accepted_sample_counts[:1]
                for session in shared_model.sessions
            ],
            [[1600], [1600]],
        )
        self.assertTrue(all(session.finished for session in shared_model.sessions))
        self.assertTrue(all(session.closed for session in shared_model.sessions))

    def test_ultrafast_callback_preserves_raw_model_text(self):
        slow_model = ScriptedStreamingModel(
            {1600: "raw ultrafast text"},
            "slow_streaming",
        )
        ultrafast_model = ScriptedStreamingModel(
            {1600: "raw ultrafast text"},
            "ultrafast_streaming",
        )
        recorder = self.make_recorder_stub(slow_model)
        recorder.ultrafast_realtime_transcription_model = ultrafast_model
        recorder.ultrafast_realtime_model_type = "ultrafast"
        recorder.ensure_sentence_starting_uppercase = True
        recorder.ensure_sentence_ends_with_period = True
        ultrafast_updates = []
        recorder.on_ultrafast_transcription_update = ultrafast_updates.append
        thread = self.run_worker(recorder)

        try:
            recorder.frames.append(self.make_frame())
            self.assertTrue(wait_until(lambda: bool(ultrafast_updates)))
        finally:
            self.stop_worker(recorder, thread)

        self.assertEqual(ultrafast_updates[0], "raw ultrafast text")

    def test_stale_ultrafast_result_cannot_overwrite_new_recording_state(self):
        entered = threading.Event()
        release = threading.Event()
        slow_model = ScriptedStreamingModel(
            {1600: "the current central text"},
            "slow_streaming",
        )
        ultrafast_model = BlockingStreamingModel(
            entered,
            release,
            "stale ultrafast text",
        )
        recorder = self.make_recorder_stub(slow_model)
        recorder.ultrafast_realtime_transcription_model = ultrafast_model
        recorder.ultrafast_realtime_model_type = "ultrafast"
        ultrafast_updates = []
        merge_results = []
        recorder.on_ultrafast_transcription_update = ultrafast_updates.append
        recorder.on_realtime_transcription_merge_update = merge_results.append
        thread = self.run_worker(recorder)

        recorder.frames.append(self.make_frame())
        self.assertTrue(entered.wait(timeout=2.0))
        recorder.realtime_recording_id = 2
        recorder.realtime_transcription_merger.reset(2)
        recorder.ultrafast_realtime_observation_sequence = 0
        recorder.ultrafast_realtime_transcription_text = "new recording fast"
        recorder.last_ultrafast_transcription = "new recording fast"
        recorder.is_running = False
        release.set()
        thread.join(timeout=2.0)
        recorder.is_recording = False

        self.assertFalse(thread.is_alive())
        self.assertEqual(ultrafast_updates, [])
        self.assertEqual(merge_results, [])
        self.assertEqual(
            recorder.ultrafast_realtime_transcription_text,
            "new recording fast",
        )
        self.assertEqual(
            recorder.last_ultrafast_transcription,
            "new recording fast",
        )
        self.assertEqual(recorder.ultrafast_realtime_observation_sequence, 0)

    def test_failed_fast_alignment_holds_until_slow_text_advances(self):
        slow_model = ScriptedStreamingModel(
            {
                1600: "the quick brown fox",
                3200: "the quick brown fox",
                4800: "the quick brown fox jumps today",
            },
            "slow_streaming",
        )
        ultrafast_model = ScriptedStreamingModel(
            {
                1600: "the quick brown fox jumps",
                3200: "totally unrelated words now",
                4800: "still unrelated output here",
            },
            "ultrafast_streaming",
        )
        recorder = self.make_recorder_stub(slow_model)
        recorder.ultrafast_realtime_transcription_model = ultrafast_model
        recorder.ultrafast_realtime_model_type = "ultrafast"
        merged_updates = []
        merge_results = []
        recorder.on_merged_realtime_transcription_update = merged_updates.append
        recorder.on_realtime_transcription_merge_update = merge_results.append
        thread = self.run_worker(recorder)

        try:
            recorder.frames.append(self.make_frame())
            self.assertTrue(wait_until(lambda: len(merged_updates) == 1))
            self.assertEqual(
                merged_updates[-1],
                "the quick brown fox jumps",
            )

            recorder.frames.append(self.make_frame())
            self.assertTrue(
                wait_until(
                    lambda: ultrafast_model.sessions
                    and ultrafast_model.sessions[0].total_samples >= 3200
                )
            )
            time.sleep(0.05)
            self.assertEqual(
                merged_updates,
                ["the quick brown fox jumps"],
            )
            self.assertTrue(
                any(result.status == "held_no_anchor" for result in merge_results)
            )

            recorder.frames.append(self.make_frame())
            self.assertTrue(wait_until(lambda: len(merged_updates) == 2))
        finally:
            self.stop_worker(recorder, thread)

        self.assertEqual(
            merged_updates,
            [
                "the quick brown fox jumps",
                "the quick brown fox jumps today",
            ],
        )
        self.assertEqual(
            merge_results[-1].status,
            "slow_advanced_no_anchor",
        )


if __name__ == "__main__":
    unittest.main()
