import asyncio
import threading
import unittest
from unittest import mock
from types import SimpleNamespace

from RealtimeSTT_server import production_server as production


class _PreviewManager:
    def __init__(self):
        self._loop = object()
        self.events = []
        self.published = threading.Event()

    def set_audio_sequence(self, session_id, sequence):
        return None

    def publish_session(self, session_id, event, authoritative=False):
        self.events.append(dict(event))
        self.published.set()
        delivered = production.concurrent.futures.Future()
        delivered.set_result(True)
        return delivered


class _PreviewService:
    def __init__(self, text="preview text", release=None):
        self.text = text
        self.release = release
        self.started = threading.Event()
        self.calls = []

    def transcribe_turn(self, audio, language, use_prompt):
        self.calls.append((audio.copy(), language, use_prompt))
        self.started.set()
        if self.release is not None:
            self.release.wait(timeout=1.0)
        return SimpleNamespace(text=self.text)


class ProductionPreviewTests(unittest.TestCase):
    @staticmethod
    def _protocol(*, pcm, live_text="", service=None):
        manager = _PreviewManager()
        service = service or _PreviewService()
        protocol = production.ProductionSessionProtocol(
            service,
            manager,
            "session",
            production.ProductionServerSettings(preview_tail_seconds=1.0),
        )
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
            pcm_buffer=bytearray(pcm),
        )
        protocol._last_partial = live_text
        return protocol, service, manager

    @staticmethod
    def _packet(sequence, samples=b"\x01\x00\x02\x00"):
        return production.encode_audio_packet(
            {
                "sampleRate": production.SERVER_SAMPLE_RATE,
                "channels": 1,
                "format": production.PCM_FORMAT,
                "frames": len(samples) // 2,
                "audioSequence": sequence,
            },
            samples,
        )

    @staticmethod
    async def _preview_event(protocol, manager, request_id="request"):
        response = await protocol.preview(
            {
                "type": "preview",
                "turnId": "turn",
                "previewRequestId": request_id,
            }
        )
        assert response is not None
        if not await asyncio.to_thread(manager.published.wait, 1.0):
            raise AssertionError("Preview result was not published")
        return response, manager.events[-1]

    def test_default_preview_transcribes_complete_buffer_even_with_live_text(self):
        manager = _PreviewManager()
        service = _PreviewService(text="recognized overlap and ending")
        protocol = production.ProductionSessionProtocol(
            service,
            manager,
            "session",
            production.ProductionServerSettings(),
        )
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 7)
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
            pcm_buffer=bytearray(pcm),
        )
        protocol._last_partial = "recognized overlap"

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(
            service.calls[0][0].size,
            production.SERVER_SAMPLE_RATE * 7,
        )
        self.assertAlmostEqual(event["inputSeconds"], 7.0)
        self.assertEqual(event["inputScope"], "full_buffer")
        self.assertEqual(event["previewInputCoverage"], "full_turn")
        self.assertEqual(event["status"], "full_buffer")
        self.assertEqual(event["previewModelText"], "recognized overlap and ending")
        self.assertEqual(event["text"], "recognized overlap and ending")

    def test_empty_live_snapshot_transcribes_the_complete_buffer(self):
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 2)
        protocol, service, manager = self._protocol(pcm=pcm)

        response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(response["type"], "previewing")
        self.assertEqual(service.calls[0][0].size, production.SERVER_SAMPLE_RATE * 2)
        self.assertEqual(event["liveText"], "")
        self.assertEqual(event["inputScope"], "full_buffer")
        self.assertAlmostEqual(event["inputSeconds"], 2.0)

    def test_short_empty_preview_retries_once_with_500ms_zero_suffix(self):
        class EmptyThenRecoveredService(_PreviewService):
            def transcribe_turn(self, audio, language, use_prompt):
                call_index = len(self.calls)
                self.calls.append((audio.copy(), language, use_prompt))
                if call_index == 0:
                    return SimpleNamespace(
                        text="",
                        queue_delay=0.001,
                        inference_duration=0.004,
                        total_latency=0.005,
                    )
                return SimpleNamespace(
                    text="recovered preview",
                    queue_delay=0.002,
                    inference_duration=0.005,
                    total_latency=0.007,
                )

        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 2)
        protocol, service, manager = self._protocol(
            pcm=pcm,
            service=EmptyThenRecoveredService(),
        )

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(len(service.calls), 2)
        first_audio = service.calls[0][0]
        retry_audio = service.calls[1][0]
        self.assertEqual(first_audio.size, production.SERVER_SAMPLE_RATE * 2)
        self.assertEqual(
            retry_audio.size,
            first_audio.size + production.SERVER_SAMPLE_RATE // 2,
        )
        self.assertTrue((retry_audio[: first_audio.size] == first_audio).all())
        self.assertTrue((retry_audio[first_audio.size :] == 0.0).all())
        self.assertEqual(event["status"], "full_buffer")
        self.assertEqual(event["text"], "recovered preview")
        self.assertAlmostEqual(event["inputSeconds"], 2.0)
        timing = event["previewTiming"]
        self.assertEqual(timing["asrAttemptCount"], 2)
        self.assertTrue(timing["emptyRetryAttempted"])
        self.assertTrue(timing["emptyRetryRecovered"])
        self.assertEqual(timing["emptyRetrySilenceMs"], 500.0)
        self.assertEqual(timing["asrQueueMs"], 3.0)
        self.assertEqual(timing["asrInferenceMs"], 9.0)
        self.assertEqual(timing["asrTotalMs"], 12.0)
        self.assertEqual(
            [attempt["inputSeconds"] for attempt in timing["asrAttempts"]],
            [2.0, 2.5],
        )

    def test_empty_preview_at_four_seconds_retries_once_with_500ms_silence(self):
        service = _PreviewService(text="")
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 4)
        protocol, service, manager = self._protocol(pcm=pcm, service=service)

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(len(service.calls), 2)
        self.assertEqual(service.calls[0][0].size, production.SERVER_SAMPLE_RATE * 4)
        self.assertEqual(
            service.calls[1][0].size,
            production.SERVER_SAMPLE_RATE * 4 + production.SERVER_SAMPLE_RATE // 2,
        )
        self.assertEqual(event["status"], "empty")
        timing = event["previewTiming"]
        self.assertEqual(timing["asrAttemptCount"], 2)
        self.assertTrue(timing["emptyRetryAttempted"])
        self.assertEqual(timing["emptyRetryReason"], "empty_transcript")
        self.assertEqual(timing["emptyRetrySilenceMs"], 500.0)

    def test_short_empty_preview_retries_only_once_when_retry_is_empty(self):
        service = _PreviewService(text="")
        pcm = b"\x01\x00" * production.SERVER_SAMPLE_RATE
        protocol, service, manager = self._protocol(pcm=pcm, service=service)

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(len(service.calls), 2)
        self.assertEqual(event["status"], "empty")
        timing = event["previewTiming"]
        self.assertEqual(timing["asrAttemptCount"], 2)
        self.assertTrue(timing["emptyRetryAttempted"])
        self.assertFalse(timing["emptyRetryRecovered"])

    def test_nonempty_live_snapshot_still_transcribes_the_complete_buffer(self):
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 2)
        protocol, service, manager = self._protocol(
            pcm=pcm,
            live_text="already recognized",
            service=_PreviewService(text="complete preview transcript"),
        )

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(service.calls[0][0].size, production.SERVER_SAMPLE_RATE * 2)
        self.assertEqual(event["liveText"], "already recognized")
        self.assertEqual(event["inputScope"], "full_buffer")
        self.assertAlmostEqual(event["inputSeconds"], 2.0)
        self.assertEqual(event["status"], "full_buffer")
        self.assertEqual(event["previewModelText"], "complete preview transcript")
        self.assertEqual(event["text"], "complete preview transcript")

    def test_full_buffer_preview_supersedes_regressed_live_suffix(self):
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 2)
        protocol, _service, manager = self._protocol(
            pcm=pcm,
            live_text=(
                "And I feel now maybe I have a kind of nice working "
                "spee to text system spee"
            ),
            service=_PreviewService(
                text=(
                    "And I feel now maybe I have a kind of nice working "
                    "speech to text system."
                )
            ),
        )

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(event["inputScope"], "full_buffer")
        self.assertEqual(event["status"], "full_buffer")
        self.assertEqual(
            event["text"],
            "And I feel now maybe I have a kind of nice working "
            "speech to text system.",
        )

    def test_short_full_buffer_preview_does_not_require_live_tail_alignment(self):
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE // 2)
        protocol, service, manager = self._protocol(
            pcm=pcm,
            live_text="Hey there",
            service=_PreviewService(text="Hey there"),
        )

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(
            service.calls[0][0].size,
            production.SERVER_SAMPLE_RATE // 2,
        )
        self.assertEqual(event["inputScope"], "full_buffer")
        self.assertEqual(event["status"], "full_buffer")
        self.assertEqual(event["text"], "Hey there")

    def test_preview_ignores_merged_ultrafast_suffix(self):
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 2)
        protocol, service, manager = self._protocol(
            pcm=pcm,
            live_text="accurate live anchor",
            service=_PreviewService(text="accurate live anchor preview tail"),
        )
        protocol._last_merged_text = "accurate live anchor speculative fast words"

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(service.calls[0][0].size, production.SERVER_SAMPLE_RATE * 2)
        self.assertEqual(event["liveText"], "accurate live anchor")
        self.assertNotIn("speculative", event["text"])
        self.assertEqual(event["inputScope"], "full_buffer")
        self.assertEqual(event["status"], "full_buffer")

    def test_preview_error_never_substitutes_stale_live_text(self):
        class RaisingService(_PreviewService):
            def transcribe_turn(self, audio, language, use_prompt):
                self.calls.append((audio.copy(), language, use_prompt))
                raise RuntimeError("synthetic Preview failure")

        pcm = b"\x01\x00" * production.SERVER_SAMPLE_RATE
        protocol, _service, manager = self._protocol(
            pcm=pcm,
            live_text="stale live hypothesis",
            service=RaisingService(),
        )

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        self.assertEqual(event["status"], "error")
        self.assertEqual(event["text"], "")
        self.assertEqual(event["previewText"], "")
        self.assertEqual(event["previewModelText"], "")
        self.assertEqual(event["liveText"], "stale live hypothesis")
        self.assertEqual(event["error"]["code"], "preview_transcription_failed")

    def test_preview_freezes_empty_live_snapshot_without_waiting_for_late_live_text(self):
        release = threading.Event()
        service = _PreviewService(release=release)
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 2)
        protocol, service, manager = self._protocol(pcm=pcm, service=service)

        async def scenario():
            response = await protocol.preview(
                {
                    "type": "preview",
                    "turnId": "turn",
                    "previewRequestId": "frozen",
                }
            )
            self.assertEqual(response["type"], "previewing")
            self.assertTrue(await asyncio.to_thread(service.started.wait, 1.0))
            protocol._last_partial = "late live text"
            release.set()
            self.assertTrue(await asyncio.to_thread(manager.published.wait, 1.0))

        asyncio.run(scenario())

        event = manager.events[-1]
        self.assertEqual(service.calls[0][0].size, production.SERVER_SAMPLE_RATE * 2)
        self.assertEqual(event["liveText"], "")
        self.assertEqual(event["inputScope"], "full_buffer")

    def test_preview_only_finalize_uses_full_buffer_when_live_snapshot_is_empty(self):
        pcm = b"\x01\x00" * (production.SERVER_SAMPLE_RATE * 2)
        protocol, service, manager = self._protocol(pcm=pcm)
        protocol.settings.preview_only_transcription = True
        protocol.attach(SimpleNamespace(snapshot=lambda: {}))

        async def scenario():
            response = await protocol.finalize()
            self.assertEqual(response["type"], "finalizing")
            for _ in range(100):
                if len(manager.events) >= 2:
                    break
                await asyncio.sleep(0.001)

        asyncio.run(scenario())

        self.assertEqual(service.calls[0][0].size, production.SERVER_SAMPLE_RATE * 2)
        preview = next(event for event in manager.events if event["type"] == "preview")
        self.assertEqual(preview["inputScope"], "full_buffer")
        self.assertEqual(
            [event["type"] for event in manager.events],
            ["preview", "completion"],
        )

    def test_preview_only_finalize_refreshes_completed_preview_after_new_audio(self):
        service = _PreviewService(text="fresh complete preview")
        protocol, service, manager = self._protocol(pcm=b"", service=service)
        protocol.settings.preview_only_transcription = True
        protocol.attach(SimpleNamespace(snapshot=lambda: {}))

        async def scenario():
            self.assertIsNone(await protocol.audio(self._packet(0)))
            _response, first = await self._preview_event(
                protocol,
                manager,
                request_id="before-resume",
            )
            self.assertEqual(first["previewRequestId"], "before-resume")
            self.assertIsNone(
                await protocol.audio(
                    self._packet(1, b"\x03\x00\x04\x00")
                )
            )

            finalizing = await protocol.finalize()
            self.assertEqual(finalizing["type"], "finalizing")
            self.assertNotEqual(
                finalizing["previewRequestId"],
                "before-resume",
            )
            for _ in range(1000):
                if any(event["type"] == "completion" for event in manager.events):
                    break
                await asyncio.sleep(0.001)

        asyncio.run(scenario())

        self.assertEqual([call[0].size for call in service.calls], [2, 4])
        previews = [event for event in manager.events if event["type"] == "preview"]
        self.assertEqual(len(previews), 2)
        self.assertEqual(previews[-1]["audioRevision"], 2)
        self.assertEqual(previews[-1]["audioPackets"], 2)
        self.assertEqual(previews[-1]["audioFrames"], 4)
        completion = next(
            event for event in manager.events if event["type"] == "completion"
        )
        self.assertEqual(completion["finalCount"], 0)
        self.assertEqual(completion["audioPackets"], 2)

    def test_preview_only_finalize_supersedes_stale_inflight_preview(self):
        class BlockingFirstService(_PreviewService):
            def __init__(self):
                super().__init__()
                self.first_started = threading.Event()
                self.release_first = threading.Event()

            def transcribe_turn(self, audio, language, use_prompt):
                call_index = len(self.calls)
                self.calls.append((audio.copy(), language, use_prompt))
                if call_index == 0:
                    self.first_started.set()
                    if not self.release_first.wait(timeout=1.0):
                        raise AssertionError("stale Preview was not released")
                    return SimpleNamespace(text="stale preview")
                return SimpleNamespace(text="fresh final preview")

        service = BlockingFirstService()
        protocol, service, manager = self._protocol(pcm=b"", service=service)
        protocol.settings.preview_only_transcription = True
        protocol.attach(SimpleNamespace(snapshot=lambda: {}))

        async def scenario():
            try:
                self.assertIsNone(await protocol.audio(self._packet(0)))
                previewing = await protocol.preview(
                    {
                        "type": "preview",
                        "turnId": "turn",
                        "previewRequestId": "inflight-before-resume",
                    }
                )
                self.assertEqual(previewing["type"], "previewing")
                self.assertTrue(
                    await asyncio.to_thread(service.first_started.wait, 1.0)
                )
                self.assertIsNone(
                    await protocol.audio(
                        self._packet(1, b"\x03\x00\x04\x00")
                    )
                )

                finalizing = await protocol.finalize()
                self.assertNotEqual(
                    finalizing["previewRequestId"],
                    "inflight-before-resume",
                )
                for _ in range(1000):
                    if any(
                        event["type"] == "completion" for event in manager.events
                    ):
                        break
                    await asyncio.sleep(0.001)
                self.assertTrue(
                    any(event["type"] == "completion" for event in manager.events)
                )
            finally:
                service.release_first.set()
                for _ in range(1000):
                    with protocol._lock:
                        if not protocol._completion_threads:
                            break
                    await asyncio.sleep(0.001)

        asyncio.run(scenario())

        self.assertEqual([call[0].size for call in service.calls], [2, 4])
        previews = [event for event in manager.events if event["type"] == "preview"]
        self.assertEqual([event["text"] for event in previews], ["fresh final preview"])
        completion = next(
            event for event in manager.events if event["type"] == "completion"
        )
        self.assertEqual(completion["finalCount"], 0)

    def test_preview_admission_is_bounded_and_coalesces_latest_snapshot(self):
        class BlockingService(_PreviewService):
            def __init__(self):
                super().__init__(release=threading.Event())
                self.active = 0
                self.max_active = 0
                self.active_lock = threading.Lock()
            def transcribe_turn(self, audio, language, use_prompt):
                self.calls.append((audio.copy(), language, use_prompt))
                with self.active_lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                self.started.set()
                try:
                    if not self.release.wait(timeout=2.0):
                        raise AssertionError("Preview worker was not released")
                    return SimpleNamespace(text=f"preview {len(self.calls)}")
                finally:
                    with self.active_lock:
                        self.active -= 1
        service = BlockingService()
        protocol, service, manager = self._protocol(
            pcm=b"\x01\x00" * 16,
            service=service,
        )
        async def scenario():
            try:
                first = await protocol.preview(
                    {
                        "type": "preview",
                        "turnId": "turn",
                        "previewRequestId": "preview-0",
                    }
                )
                self.assertEqual(first["type"], "previewing")
                self.assertTrue(await asyncio.to_thread(service.started.wait, 1.0))
                for index in range(1, 32):
                    response = await protocol.preview(
                        {
                            "type": "preview",
                            "turnId": "turn",
                            "previewRequestId": f"preview-{index}",
                        }
                    )
                    self.assertEqual(response["type"], "previewing")
                    with protocol._lock:
                        self.assertLessEqual(len(protocol._completion_threads), 1)
                        self.assertIsNotNone(protocol._preview_pending)
                service.release.set()
                deadline = production.time.monotonic() + 2.0
                while production.time.monotonic() < deadline:
                    with protocol._lock:
                        if not protocol._completion_threads and protocol._preview_pending is None:
                            break
                    await asyncio.sleep(0.005)
                self.assertTrue(manager.published.wait(1.0))
            finally:
                service.release.set()
        asyncio.run(scenario())
        with protocol._lock:
            self.assertEqual(protocol._completion_threads, set())
            self.assertIsNone(protocol._preview_pending)
        self.assertEqual(service.max_active, 1)
        self.assertEqual(len(service.calls), 2)
        previews = [event for event in manager.events if event["type"] == "preview"]
        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0]["previewRequestId"], "preview-31")

    def test_preview_dispatcher_exit_does_not_strand_arriving_request(self):
        service = _PreviewService(text="gap preview")
        protocol, service, manager = self._protocol(pcm=b"", service=service)
        turn = protocol.turn
        request_id = "gap-preview"
        with protocol._lock:
            turn.latest_preview_request_id = request_id
            turn.preview_epoch = 1
            turn.preview_cancelled = threading.Event()
        gap_work = (
            "turn",
            1,
            "en",
            b"",
            "full_buffer",
            "",
            request_id,
            production.time.monotonic(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            "",
            None,
            0,
            1,
            turn.preview_cancelled,
        )
        real_lock = protocol._lock
        callback_owner_states = []
        callback_called = threading.Event()
        class ReleaseHook:
            def __init__(self, lock, callback):
                self.lock = lock
                self.callback = callback
            def __enter__(self):
                self.lock.acquire()
                return self
            def __exit__(self, exc_type, exc_value, traceback):
                self.lock.release()
                callback = self.callback
                self.callback = None
                if callback is not None:
                    callback()
                return False
        def admit_after_dispatch_lock_release():
            callback_owner_states.append(protocol._preview_dispatch_thread)
            callback_called.set()
            protocol._start_preview_worker(*gap_work)
        def run_dispatcher():
            current = threading.current_thread()
            with real_lock:
                protocol._preview_dispatch_thread = current
                protocol._completion_threads.add(current)
            protocol._lock = ReleaseHook(real_lock, admit_after_dispatch_lock_release)
            protocol._run_preview_dispatcher()
        thread = threading.Thread(target=run_dispatcher, daemon=True)
        thread.start()
        try:
            self.assertTrue(manager.published.wait(1.0))
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertTrue(callback_called.is_set())
            self.assertEqual(callback_owner_states, [None])
            with real_lock:
                self.assertIsNone(protocol._preview_dispatch_thread)
                self.assertEqual(protocol._completion_threads, set())
                self.assertIsNone(protocol._preview_pending)
        finally:
            protocol._lock = real_lock
            thread.join(timeout=1.0)

    def test_preview_dispatcher_start_window_keeps_single_owner(self):
        protocol, _service, _manager = self._protocol(pcm=b"")
        work = (None,) * 21
        begin_first_call = threading.Event()
        first_start_entered = threading.Event()
        release_first_start = threading.Event()
        created = []
        starts = []
        errors = []

        class NotYetStartedDispatcher:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(self)

            def is_alive(self):
                return False

            def start(self):
                starts.append(self)
                if len(starts) == 1:
                    first_start_entered.set()
                    release_first_start.wait(timeout=1.0)

        def submit_first_request():
            begin_first_call.wait(timeout=1.0)
            try:
                protocol._start_preview_worker(*work)
            except Exception as exc:
                errors.append(exc)

        caller = threading.Thread(target=submit_first_request, daemon=True)
        caller.start()
        try:
            with mock.patch.object(
                production.threading,
                "Thread",
                NotYetStartedDispatcher,
            ):
                begin_first_call.set()
                self.assertTrue(first_start_entered.wait(timeout=1.0))
                protocol._start_preview_worker(*work)
                release_first_start.set()
                caller.join(timeout=1.0)
        finally:
            release_first_start.set()
            caller.join(timeout=1.0)
            with protocol._lock:
                protocol._preview_dispatch_thread = None
                protocol._preview_pending = None
                protocol._completion_threads.clear()

        self.assertFalse(caller.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(created), 1)
        self.assertEqual(len(starts), 1)

    def test_preview_uses_scheduler_queue_and_inference_timings(self):
        class TimedService(_PreviewService):
            def transcribe_turn(self, audio, language, use_prompt):
                self.calls.append((audio.copy(), language, use_prompt))
                return SimpleNamespace(
                    text=self.text,
                    queue_delay=0.004,
                    inference_duration=0.0125,
                    total_latency=0.0165,
                )

        pcm = b"\x01\x00" * production.SERVER_SAMPLE_RATE
        protocol, _service, manager = self._protocol(
            pcm=pcm,
            service=TimedService(),
        )

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        timing = event["previewTiming"]
        self.assertEqual(timing["asrQueueMs"], 4.0)
        self.assertEqual(timing["asrInferenceMs"], 12.5)
        self.assertEqual(timing["asrTotalMs"], 16.5)

    def test_preview_reports_nonnegative_request_worker_and_asr_timings(self):
        pcm = b"\x01\x00" * production.SERVER_SAMPLE_RATE
        protocol, _service, manager = self._protocol(pcm=pcm)

        _response, event = asyncio.run(self._preview_event(protocol, manager))

        timing = event["previewTiming"]
        self.assertGreaterEqual(timing["requestToWorkerStartMs"], 0.0)
        self.assertIsNone(timing["asrQueueMs"])
        self.assertGreaterEqual(timing["asrInferenceMs"], 0.0)
        self.assertGreaterEqual(timing["asrTotalMs"], timing["asrInferenceMs"])
        self.assertGreaterEqual(
            timing["requestToPublishMs"],
            timing["requestToWorkerStartMs"],
        )


if __name__ == "__main__":
    unittest.main()
