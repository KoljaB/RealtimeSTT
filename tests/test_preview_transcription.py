import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from RealtimeSTT.core.preview_transcription import (
    PreviewTranscriptionWorker,
    transcribe_preview,
)
from RealtimeSTT.core.initialization import (
    _initialize_preview_transcription_model,
    _should_share_preview_model,
)
from RealtimeSTT.core.lifecycle import stop_recording
from RealtimeSTT.core.recording import _submit_preview_transcription_at_silence
from RealtimeSTT.core.tail_transcription import append_pcm16_tail
from RealtimeSTT.core.transcription import (
    SharedFinalModelExecutor,
    _SHARED_PREVIEW_REQUEST,
)
from RealtimeSTT.transcription_engines import (
    TranscriptionInfo,
    TranscriptionResult,
)


class FakePreviewModel:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def transcribe(self, audio, language=None, use_prompt=True):
        self.calls.append((audio.copy(), language, use_prompt))
        return TranscriptionResult(
            text=self.text,
            info=TranscriptionInfo(language="en", language_probability=1.0),
        )


class PreviewRecorderStub:
    def __init__(self, model=None):
        self.sample_rate = 16000
        self.language = "en"
        self.preview_transcription_model = model
        self.preview_transcription_executor = None
        self._uses_external_preview_transcription_executor = False
        self.preview_transcription_tail_seconds = 3.0
        self.preview_transcription_min_live_words_for_fuzzy_repair = 3
        self.ensure_sentence_starting_uppercase = False
        self.ensure_sentence_ends_with_period = False
        self.on_preview_transcription_finished = None
        self.start_callback_in_new_thread = False
        self.preview_transcription_queue = queue.Queue()
        self.preview_transcription_stop_event = threading.Event()
        self.preview_transcription_thread = None
        self.last_preview_transcription_result = None
        self.last_preview_transcription = ""


class PreviewTranscriptionTests(unittest.TestCase):
    def test_matching_final_and_preview_config_shares_loaded_model(self):
        recorder = SimpleNamespace(
            enable_preview_transcription=True,
            _uses_external_preview_transcription_executor=False,
            _uses_external_transcription_executor=False,
            preview_model_type="tiny",
            main_model_type="tiny",
            preview_transcription_engine="faster_whisper",
            transcription_engine="faster_whisper",
            preview_transcription_engine_options={"temperature": 0.0},
            transcription_engine_options={"temperature": 0.0},
            preview_transcription_executor=None,
            preview_transcription_model=None,
            preview_transcription_uses_main_model=False,
            shared_preview_transcription_request_queue=queue.Queue(),
            shared_preview_transcription_result_queue=queue.Queue(),
        )

        self.assertTrue(_should_share_preview_model(recorder))

        with mock.patch(
            "RealtimeSTT.core.initialization.create_transcription_engine"
        ) as create_model:
            with mock.patch(
                "RealtimeSTT.core.initialization.start_preview_transcription_worker"
            ) as start_worker:
                _initialize_preview_transcription_model(recorder)

        create_model.assert_not_called()
        start_worker.assert_called_once_with(recorder)
        self.assertTrue(recorder.preview_transcription_uses_main_model)
        self.assertIsInstance(
            recorder.preview_transcription_executor,
            SharedFinalModelExecutor,
        )

    def test_different_preview_model_does_not_share_loaded_model(self):
        recorder = SimpleNamespace(
            enable_preview_transcription=True,
            _uses_external_preview_transcription_executor=False,
            preview_model_type="tiny",
            main_model_type="large-v2",
            preview_transcription_engine="faster_whisper",
            transcription_engine="faster_whisper",
            preview_transcription_engine_options={},
            transcription_engine_options={},
        )

        self.assertFalse(_should_share_preview_model(recorder))

    def test_shared_preview_uses_tagged_final_pipe_and_separate_result_queue(self):
        result_queue = queue.Queue()

        class ParentPipe:
            def __init__(self):
                self.sent = None

            def send(self, request):
                self.sent = request
                result_queue.put(
                    (
                        request[1],
                        "success",
                        TranscriptionResult(
                            text="shared result",
                            info=TranscriptionInfo(
                                language="en",
                                language_probability=1.0,
                            ),
                        ),
                    )
                )

        parent_pipe = ParentPipe()
        recorder = SimpleNamespace(
            parent_transcription_pipe=parent_pipe,
            shared_preview_transcription_result_queue=result_queue,
            shutdown_event=threading.Event(),
            is_shut_down=False,
        )
        dispatch_event = threading.Event()
        executor = SharedFinalModelExecutor(recorder)

        result = executor.transcribe_preview(
            np.ones(5 * 16000, dtype=np.float32),
            language="en",
            dispatch_event=dispatch_event,
        )

        self.assertEqual(result.text, "shared result")
        self.assertTrue(dispatch_event.is_set())
        self.assertEqual(parent_pipe.sent[0], _SHARED_PREVIEW_REQUEST)
        self.assertEqual(parent_pipe.sent[3], "en")

    def test_preview_merges_tail_without_submitting_full_audio(self):
        model = FakePreviewModel(
            "boundary garbage brown fox jumps over the fence"
        )
        recorder = PreviewRecorderStub(model)
        tail_audio = np.ones(3 * 16000, dtype=np.float32)

        result = transcribe_preview(
            recorder,
            tail_audio,
            "The quick brown fox jumps",
            recording_id=7,
        )

        self.assertEqual(
            result.text,
            "The quick brown fox jumps over the fence",
        )
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.recording_id, 7)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0].size, 3 * 16000)

    def test_preview_alignment_failure_returns_live_text_without_full_fallback(self):
        model = FakePreviewModel("unrelated final tail words")
        recorder = PreviewRecorderStub(model)
        tail_audio = np.ones(3 * 16000, dtype=np.float32)

        result = transcribe_preview(
            recorder,
            tail_audio,
            "The original live transcript",
        )

        self.assertEqual(result.status, "alignment_failed")
        self.assertEqual(result.text, "The original live transcript")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0].size, 3 * 16000)

    def test_preview_config_can_disable_short_live_fuzzy_repair(self):
        model = FakePreviewModel("boundary focus on latency first")
        recorder = PreviewRecorderStub(model)
        recorder.preview_transcription_min_live_words_for_fuzzy_repair = 4

        result = transcribe_preview(
            recorder,
            np.ones(3 * 16000, dtype=np.float32),
            "focus on la",
        )

        self.assertEqual(result.status, "alignment_failed")
        self.assertEqual(result.text, "focus on la")

    def test_preview_worker_publishes_structured_result_to_callback(self):
        model = FakePreviewModel("brown fox jumps over the fence")
        recorder = PreviewRecorderStub(model)
        callback_received = []
        callback_event = threading.Event()

        def callback(result):
            callback_received.append(result)
            callback_event.set()

        recorder.on_preview_transcription_finished = callback
        worker = PreviewTranscriptionWorker(recorder)
        recorder.preview_transcription_thread = worker.thread
        worker.start()
        worker.submit(
            np.ones(3 * 16000, dtype=np.float32),
            "The quick brown fox jumps",
            recording_id=11,
        )

        self.assertTrue(callback_event.wait(timeout=1.0))
        worker.stop()

        self.assertEqual(len(callback_received), 1)
        self.assertEqual(callback_received[0].recording_id, 11)
        self.assertEqual(recorder.last_preview_transcription, callback_received[0].text)
        self.assertIs(
            recorder.last_preview_transcription_result,
            callback_received[0],
        )

    def test_preview_worker_rejects_work_when_bounded_queue_is_full(self):
        recorder = PreviewRecorderStub(FakePreviewModel("tail"))
        worker = PreviewTranscriptionWorker(recorder)
        tail_audio = np.ones(16000, dtype=np.float32)

        self.assertTrue(
            worker.submit(tail_audio, "first live text", recording_id=1)
        )
        self.assertTrue(
            worker.submit(tail_audio, "second live text", recording_id=2)
        )
        self.assertFalse(
            worker.submit(tail_audio, "third live text", recording_id=3)
        )
        self.assertEqual(worker.queue.qsize(), 2)
        worker.stop()

    def test_vad_preview_submission_uses_current_live_text_and_tail(self):
        class Worker:
            def __init__(self):
                self.calls = []

            def submit(
                self,
                tail_audio,
                live_text,
                recording_id,
                use_prompt=True,
            ):
                self.calls.append((tail_audio, live_text, recording_id))
                return True

        recorder = PreviewRecorderStub()
        recorder.enable_preview_transcription = True
        recorder.preview_transcription_worker = Worker()
        recorder._preview_transcription_submitted = False
        recorder.realtime_transcription_text = "The exact Live boundary text"
        recorder.realtime_stabilized_text = ""
        recorder.realtime_stabilized_safetext = ""
        recorder.realtime_recording_id = 23
        recorder.active_speech_tail_buffer = bytearray()
        append_pcm16_tail(
            recorder,
            np.arange(12 * 16000, dtype=np.int16).tobytes(),
        )

        self.assertTrue(_submit_preview_transcription_at_silence(recorder))

        self.assertEqual(len(recorder.preview_transcription_worker.calls), 1)
        tail_audio, live_text, recording_id = (
            recorder.preview_transcription_worker.calls[0]
        )
        self.assertEqual(tail_audio.size, 3 * 16000)
        self.assertEqual(live_text, "The exact Live boundary text")
        self.assertEqual(recording_id, 23)

    def test_stop_queues_the_live_snapshot_captured_at_vad(self):
        recorder = SimpleNamespace(
            min_length_of_recording=0.0,
            recording_start_time=0.0,
            _pending_vad_live_text="Live text at confirmed silence",
            realtime_transcription_text="Live text that arrived afterward",
            merged_realtime_transcription_text=(
                "Live text at confirmed silence speculative fast tail"
            ),
            realtime_stabilized_text="",
            frames=[b"\x01\x00" * 32],
            last_frames=[],
            frames_lock=threading.RLock(),
            active_speech_tail_buffer=bytearray(b"\x02\x00" * 16),
            is_recording=True,
            backdate_stop_seconds=0.0,
            backdate_resume_seconds=0.0,
            realtime_text_stabilizer=None,
            is_webrtc_speech_active=False,
            silero_check_time=0.0,
            start_recording_event=threading.Event(),
            stop_recording_event=threading.Event(),
            on_recording_stop=None,
        )

        with mock.patch(
            "RealtimeSTT.core.lifecycle.queue_recorded_audio"
        ) as queue_recorded:
            stop_recording(recorder)

        self.assertEqual(
            queue_recorded.call_args.kwargs["live_text"],
            "Live text at confirmed silence",
        )
        np.testing.assert_array_equal(
            queue_recorded.call_args.kwargs["tail_audio"],
            np.array([2] * 16, dtype=np.float32) / 32768.0,
        )


if __name__ == "__main__":
    unittest.main()
