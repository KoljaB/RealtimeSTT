import threading
import time
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
    from RealtimeSTT.realtime_text_stabilizer import RealtimeTextStabilizer
    from RealtimeSTT.transcription_engines import (
        TranscriptionInfo,
        TranscriptionResult,
    )
except Exception as exc:  # pragma: no cover - import guard for optional deps
    AudioToTextRecorder = None
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
        thread = threading.Thread(target=recorder._realtime_worker, daemon=True)
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


if __name__ == "__main__":
    unittest.main()
