import threading
import time
import unittest

try:
    from example_fastapi_server.server import (
        FairInferenceQueue,
        InferenceJob,
        LockedStreamingSession,
        QueueSubmitResult,
        SchedulerTranscriptionExecutor,
        ServerSettings,
        SharedEngineWorker,
    )
    from RealtimeSTT.transcription_engines import TranscriptionInfo, TranscriptionResult
except Exception as exc:  # pragma: no cover - optional server dependencies
    FairInferenceQueue = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FakeStreamingSession:
    def __init__(self, engine):
        self.engine = engine
        self.language = "en"
        self.calls = []
        self.closed = False
        self.finished = False

    def accept_audio(self, audio, sample_rate=None):
        self.calls.append(("accept_audio", audio, sample_rate))
        with self.engine.activity_lock:
            self.engine.active_operations += 1
            self.engine.max_active_operations = max(
                self.engine.max_active_operations,
                self.engine.active_operations,
            )
        time.sleep(0.01)
        with self.engine.activity_lock:
            self.engine.active_operations -= 1

    def decode(self):
        self.calls.append(("decode",))

    def get_result(self):
        self.calls.append(("get_result",))
        return TranscriptionResult(text="partial")

    def finish(self):
        self.calls.append(("finish",))
        self.finished = True
        return TranscriptionResult(text="final")

    def input_finished(self):
        self.calls.append(("input_finished",))
        self.finished = True
        return TranscriptionResult(text="final")

    def reset(self):
        self.calls.append(("reset",))

    def close(self):
        self.calls.append(("close",))
        self.closed = True


class FakeStreamingEngine:
    engine_name = "fake-streaming"
    supports_streaming = True

    def __init__(self):
        self.sessions = []
        self.transcribe_calls = []
        self.activity_lock = threading.Lock()
        self.active_operations = 0
        self.max_active_operations = 0
        self.closed = False

    def create_streaming_session(self, language=None, use_prompt=True):
        self.sessions.append((language, use_prompt))
        session = FakeStreamingSession(self)
        return session

    def transcribe(self, audio, language=None, use_prompt=True):
        self.transcribe_calls.append((audio, language, use_prompt))
        return TranscriptionResult(text="ordinary")

    def close(self):
        self.closed = True


@unittest.skipIf(IMPORT_ERROR is not None, "FastAPI server dependencies are not installed")
class FastAPIServerStreamingExecutorTests(unittest.TestCase):
    def make_worker(self, engine=None):
        engine = engine or FakeStreamingEngine()
        settings = ServerSettings(model_warmup=False)
        queue = FairInferenceQueue("realtime", settings)
        worker = SharedEngineWorker(
            "realtime",
            settings,
            queue,
            lambda: engine,
            lambda result: None,
        )
        # Make a loaded worker without starting an inference thread.  This
        # isolates the streaming proxy and lock behavior.
        worker.engine = engine
        worker.ready.set()
        return worker, engine

    def test_worker_preserves_provider_language_metadata(self):
        results = []
        result_ready = threading.Event()

        class MetadataEngine(FakeStreamingEngine):
            def transcribe(self, audio, language=None, use_prompt=True):
                self.transcribe_calls.append((audio, language, use_prompt))
                return TranscriptionResult(
                    text="hallo welt",
                    info=TranscriptionInfo(
                        language="de",
                        language_probability=0.93,
                    ),
                    metadata={"detected_language": "de", "provider": "fake"},
                )

        def capture_result(result):
            results.append(result)
            result_ready.set()

        engine = MetadataEngine()
        settings = ServerSettings(model_warmup=False)
        inference_queue = FairInferenceQueue("final", settings)
        worker = SharedEngineWorker(
            "final",
            settings,
            inference_queue,
            lambda: engine,
            capture_result,
        )
        worker.start()
        try:
            submitted = inference_queue.submit(
                InferenceJob(
                    request_id="metadata-request",
                    session_id="metadata-session",
                    kind="final",
                    audio=[1, 2, 3],
                    language="auto",
                    use_prompt=False,
                    segment_id=1,
                    sequence=1,
                    generation=1,
                    created_at=time.monotonic(),
                )
            )
            self.assertTrue(submitted.accepted)
            self.assertTrue(result_ready.wait(timeout=2.0))
        finally:
            worker.stop()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "hallo welt")
        self.assertEqual(results[0].info.language, "de")
        self.assertAlmostEqual(results[0].info.language_probability, 0.93)
        self.assertEqual(
            results[0].metadata,
            {"detected_language": "de", "provider": "fake"},
        )

    def test_streaming_proxy_forwards_incremental_operations_and_closes(self):
        worker, engine = self.make_worker()
        proxy = worker.create_streaming_session(language="de", use_prompt=False)

        self.assertIsInstance(proxy, LockedStreamingSession)
        proxy.accept_audio([1, 2], sample_rate=16000)
        proxy.decode()
        self.assertEqual(proxy.get_result().text, "partial")
        self.assertEqual(proxy.finish().text, "final")
        proxy.input_finished()
        proxy.reset()
        proxy.close()
        proxy.close()

        self.assertEqual(engine.sessions, [("de", False)])
        names = [call[0] for call in proxy.session.calls]
        self.assertEqual(
            names,
            [
                "accept_audio",
                "decode",
                "get_result",
                "finish",
                "input_finished",
                "reset",
                "close",
            ],
        )
        self.assertNotIn(proxy, worker._streaming_sessions)

    def test_two_session_operations_are_serialized_by_worker_lock(self):
        worker, engine = self.make_worker()
        first = worker.create_streaming_session()
        second = worker.create_streaming_session()
        barrier = threading.Barrier(2)

        def feed(session):
            barrier.wait(timeout=1)
            session.accept_audio([1], sample_rate=16000)

        threads = [
            threading.Thread(target=feed, args=(session,))
            for session in (first, second)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(engine.max_active_operations, 1)
        first.close()
        second.close()

    def test_readiness_and_streaming_support_fail_clearly(self):
        worker, _ = self.make_worker()
        worker.load_error = RuntimeError("bad model")
        with self.assertRaisesRegex(RuntimeError, "failed to load"):
            worker.create_streaming_session()

        worker, engine = self.make_worker()
        engine.supports_streaming = False
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            worker.create_streaming_session()

    def test_final_executor_still_uses_scheduler_and_cannot_create_stream(self):
        calls = []

        class Scheduler:
            def streaming_worker(self, kind):
                calls.append(("streaming_worker", kind))
                raise AssertionError("final executor must not use realtime worker")

        class Service:
            scheduler = Scheduler()

            def transcribe_for_recorder(self, *args):
                calls.append(args)
                return TranscriptionResult(text="queued-final")

        executor = SchedulerTranscriptionExecutor(Service(), "session-a", "final")
        self.assertFalse(executor.supports_streaming)
        self.assertEqual(executor.transcribe([1], language="en").text, "queued-final")
        with self.assertRaisesRegex(RuntimeError, "only for the realtime executor"):
            executor.create_streaming_session()
        self.assertEqual(calls[0][0], "session-a")

    def test_realtime_executor_delegates_to_loaded_worker(self):
        worker, _ = self.make_worker()

        class Scheduler:
            def streaming_worker(self, kind):
                self.kind = kind
                return worker

        class Service:
            scheduler = Scheduler()

        executor = SchedulerTranscriptionExecutor(Service(), "session-a", "realtime")
        self.assertTrue(executor.supports_streaming)
        proxy = executor.create_streaming_session(language="en")
        self.assertIsInstance(proxy, LockedStreamingSession)
        proxy.close()

    def test_realtime_executor_updates_active_stream_language_for_new_turn(self):
        worker, _ = self.make_worker()

        class Scheduler:
            def streaming_worker(self, kind):
                return worker

        class Service:
            scheduler = Scheduler()

        executor = SchedulerTranscriptionExecutor(Service(), "session-a", "realtime")
        proxy = executor.create_streaming_session(language="en")
        executor.set_streaming_language("de")

        self.assertEqual(proxy.session.language, "de")
        self.assertIn(("reset",), proxy.session.calls)
        proxy.close()

    def test_worker_close_closes_streams_and_engine(self):
        worker, engine = self.make_worker()
        proxy = worker.create_streaming_session()
        worker.close_engine()
        self.assertTrue(proxy.closed)
        self.assertTrue(engine.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            proxy.get_result()


if __name__ == "__main__":
    unittest.main()
