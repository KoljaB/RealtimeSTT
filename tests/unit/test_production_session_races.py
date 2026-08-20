"""Deterministic regression coverage for production-turn race boundaries.

The production WebSocket protocol owns its final job and live callbacks.  These
tests deliberately use event-gated scheduler and transport fakes so they cover
the handoff races without model inference, a real socket, or timing sleeps.
"""

import asyncio
import json
import queue
import threading
import unittest
from types import SimpleNamespace

from RealtimeSTT_server import production_server as production


class _Socket:
    """Small ASGI transport fake that exposes every delivered JSON event."""

    def __init__(self):
        self.messages = []
        self.message_arrived = asyncio.Event()

    async def accept(self):
        return None

    async def send_text(self, payload):
        self.messages.append(json.loads(payload))
        self.message_arrived.set()


class _BlockingSocket(_Socket):
    """Hold outbound delivery while producers continue on worker threads."""

    def __init__(self):
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()
        self.close_code = None

    async def send_text(self, payload):
        self.send_started.set()
        await self.release_send.wait()
        await super().send_text(payload)

    async def close(self, code=None, reason=None):
        del reason
        self.close_code = code
        self.release_send.set()


class _Session:
    """Only the protocol-facing session surface is needed for these tests."""

    def __init__(self):
        self.settings = SimpleNamespace(language="en")
        self.recorder = SimpleNamespace(realtime_transcription_executor=None)
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1

    def snapshot(self):
        return {
            "finalSubmitted": 0,
            "finalCompleted": 0,
            "realtimeCompleted": 0,
        }


class _FinalSubmissionScheduler:
    """Final-lane fake with synchronous, deferred, and rejected outcomes."""

    def __init__(self, service, outcome="deferred"):
        self._service = service
        self._outcome = outcome
        self._jobs = []
        self._jobs_condition = threading.Condition()
        self.cancelled_sessions = []
        self.cancelled_requests = []

    @property
    def jobs(self):
        with self._jobs_condition:
            return list(self._jobs)

    def wait_for_job_count(self, count, timeout=1.0):
        with self._jobs_condition:
            return self._jobs_condition.wait_for(
                lambda: len(self._jobs) >= count,
                timeout=timeout,
            )

    def submit(self, job):
        with self._jobs_condition:
            self._jobs.append(job)
            self._jobs_condition.notify_all()

        if self._outcome == "rejected":
            return SimpleNamespace(accepted=False, reason="synthetic final lane rejection")

        if self._outcome == "synchronous":
            self._service.complete_pending_recorder_transcription(
                SimpleNamespace(request_id=job.request_id, text="synchronous final", error=None)
            )
        elif self._outcome != "never":
            threading.Thread(
                target=self._complete_after_release,
                args=(job,),
                name="test-deferred-final-result",
                daemon=True,
            ).start()
        return SimpleNamespace(accepted=True, reason=None)

    def cancel_session(self, session_id):
        self.cancelled_sessions.append(session_id)

    def cancel_request(self, request_id):
        self.cancelled_requests.append(request_id)
        return True

    def _complete_after_release(self, job):
        self._service.release_final.wait()
        self._service.complete_pending_recorder_transcription(
            SimpleNamespace(request_id=job.request_id, text="deferred final", error=None)
        )


class _GateLiveStream:
    """A live stream whose terminal callback can be released at a chosen point."""

    def __init__(self, finish_text="late old partial"):
        self.accepted_audio = []
        self.finish_started = threading.Event()
        self.release_finish = threading.Event()
        self.finish_text = finish_text
        self.closed = 0
        self.cancelled = 0

    def accept_audio(self, audio, sample_rate=None):
        self.accepted_audio.append((audio.copy(), sample_rate))

    def decode(self):
        return None

    def get_result(self):
        return SimpleNamespace(text="")

    def input_finished(self):
        return None

    def finish(self):
        self.finish_started.set()
        self.release_finish.wait()
        return SimpleNamespace(text=self.finish_text)

    def close(self):
        self.closed += 1

    def cancel(self):
        self.cancelled += 1
        self.release_finish.set()


class _BlockingDecodeLiveStream(_GateLiveStream):
    """Hold decode until cancellation so the bounded input queue stays full."""

    def __init__(self):
        super().__init__(finish_text="")
        self.decode_started = threading.Event()
        self.release_decode = threading.Event()

    def decode(self):
        self.decode_started.set()
        self.release_decode.wait()

    def cancel(self):
        super().cancel()
        self.release_decode.set()


class _BlockingCancelLiveStream(_GateLiveStream):
    """Expose whether stream cancellation blocks the caller thread."""

    def __init__(self):
        super().__init__(finish_text="")
        self.cancel_started = threading.Event()
        self.release_cancel = threading.Event()

    def cancel(self):
        self.cancel_started.set()
        self.release_cancel.wait(1.0)


class _HangingCancelLiveStream(_GateLiveStream):
    """Native cancellation blocks until the test explicitly releases it."""

    def __init__(self):
        super().__init__(finish_text="")
        self.cancel_started = threading.Event()
        self.release_cancel = threading.Event()
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1
        self.cancel_started.set()
        self.release_cancel.wait()


class _ObservedCloseLiveStream(_GateLiveStream):
    """Record whether retirement closes a just-created stream off-loop."""

    def __init__(self):
        super().__init__(finish_text="")
        self.close_called = threading.Event()
        self.close_thread_id = None

    def close(self):
        self.close_thread_id = threading.get_ident()
        super().close()
        self.close_called.set()


class _BlockingCreateLiveStreamFactory:
    """Block only the first stream creation to expose the start/cancel race."""

    def __init__(self):
        self.first_create_started = threading.Event()
        self.release_first_create = threading.Event()
        self._lock = threading.Lock()
        self.streams = []
        self.create_thread_ids = []

    def create(self, language, use_prompt):
        del language, use_prompt
        stream = _ObservedCloseLiveStream()
        with self._lock:
            index = len(self.streams)
            self.streams.append(stream)
            self.create_thread_ids.append(threading.get_ident())
        if index == 0:
            self.first_create_started.set()
            self.release_first_create.wait()
        return stream


class _BlockingAllCreateLiveStreamFactory:
    """Hold every native create call so admission capacity is observable."""

    def __init__(self):
        self._condition = threading.Condition()
        self.release_creates = threading.Event()
        self.streams = []

    def create(self, language, use_prompt):
        del language, use_prompt
        stream = _ObservedCloseLiveStream()
        with self._condition:
            self.streams.append(stream)
            self._condition.notify_all()
        self.release_creates.wait()
        return stream

    def wait_for_creations(self, count, timeout=1.0):
        with self._condition:
            return self._condition.wait_for(lambda: len(self.streams) >= count, timeout)


class _FinalSubmissionService:
    """Subset of ``RealtimeSTTService`` used by ``_service_turn_transcription``."""

    def __init__(self, outcome="deferred", live_stream_factory=None):
        self._pending_recorder_lock = threading.RLock()
        self._pending_recorder_results = {}
        self.failed_pending = []
        self.release_final = threading.Event()
        self.scheduler = _FinalSubmissionScheduler(self, outcome=outcome)
        if live_stream_factory is not None:
            self.scheduler.streaming_worker = lambda kind: SimpleNamespace(
                create_streaming_session=lambda language=None, use_prompt=True: live_stream_factory(
                    language,
                    use_prompt,
                )
            )

    def complete_pending_recorder_transcription(self, result):
        with self._pending_recorder_lock:
            holder = self._pending_recorder_results.get(result.request_id)
        if holder is None:
            return False
        holder["result"] = result
        holder["event"].set()
        return True

    def _pop_pending_recorder_result(self, request_id):
        with self._pending_recorder_lock:
            return self._pending_recorder_results.pop(request_id, None)

    def pending_count(self):
        with self._pending_recorder_lock:
            return len(self._pending_recorder_results)

    def fail_pending_recorder_transcription(self, request_id, reason):
        self.failed_pending.append((request_id, reason))
        return True


class _RejectingDrainQueue:
    """A sealed live queue that deterministically rejects the drain marker."""

    def __init__(self):
        self.put_calls = 0

    def put(self, item, block=True, timeout=None):
        del item, block, timeout
        self.put_calls += 1
        raise queue.Full()

    def put_nowait(self, item):
        del item
        raise queue.Full()


class _ObservedProtocol(production.ProductionSessionProtocol):
    """Expose final-worker completion without changing production behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._final_worker_events = {}
        self._final_worker_events_lock = threading.Lock()

    def final_worker_done(self, turn_id, generation):
        key = (turn_id, generation)
        with self._final_worker_events_lock:
            return self._final_worker_events.setdefault(key, threading.Event())

    def _authoritative_final_worker(
        self,
        turn_id,
        generation,
        language,
        pcm,
        cancelled,
        drain_failure=None,
    ):
        try:
            return super()._authoritative_final_worker(
                turn_id,
                generation,
                language,
                pcm,
                cancelled,
                drain_failure,
            )
        finally:
            self.final_worker_done(turn_id, generation).set()


class _PausedTerminalAdmissionManager(production.OrderedConnectionManager):
    """Pause after admitting a final while the protocol still owns its lock."""

    def __init__(self):
        super().__init__()
        self.terminal_admitted = threading.Event()
        self.release_terminal = threading.Event()

    def publish_session(self, session_id, message, *, authoritative=False):
        delivery = super().publish_session(
            session_id,
            message,
            authoritative=authoritative,
        )
        if message.get("type") in {"final", "error"}:
            self.terminal_admitted.set()
            if not self.release_terminal.wait(1.0):
                raise AssertionError("timed out waiting to release terminal admission")
        return delivery


@unittest.skipIf(
    hasattr(production, "_BACKEND_IMPORT_ERROR"),
    "production server backend dependencies are not installed",
)
class ProductionSessionRaceRegressionTests(unittest.TestCase):
    """Regression tests for the explicit production turn state machine."""

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
    def _settings():
        return production.ProductionServerSettings(
            model_warmup=False,
            finalize_timeout_seconds=1.0,
            audio_queue_size=1,
        )

    async def _open_protocol(self, service, session_id="session", manager=None):
        manager = manager or production.OrderedConnectionManager()
        manager.bind_loop(asyncio.get_running_loop())
        websocket = _Socket()
        await manager.connect(session_id, websocket)
        protocol = _ObservedProtocol(service, manager, session_id, self._settings())
        protocol.attach(_Session())
        return manager, websocket, protocol

    @staticmethod
    async def _close_protocol(manager, protocol, session_id="session"):
        protocol.close()
        await manager.disconnect(session_id)
        manager.clear_session(session_id)

    @staticmethod
    async def _wait_for_turn_messages(websocket, turn_id, count, timeout=1.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            matching = [
                message
                for message in websocket.messages
                if message.get("turnId") == turn_id
            ]
            if len(matching) >= count:
                return matching
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AssertionError(
                    f"Timed out waiting for {count} events for turn {turn_id!r}; "
                    f"received {matching!r}"
                )
            await asyncio.wait_for(websocket.message_arrived.wait(), remaining)
            websocket.message_arrived.clear()

    def _assert_one_terminal_then_completion(self, messages, turn_id, *, failed=False):
        turn_messages = [
            message for message in messages if message.get("turnId") == turn_id
        ]
        self.assertEqual(len(turn_messages), 2, turn_messages)
        self.assertEqual(
            [message["type"] for message in turn_messages],
            ["error", "completion"] if failed else ["final", "completion"],
        )
        self.assertEqual(turn_messages[1]["finalCount"], 1)
        self.assertEqual(turn_messages[1]["turnId"], turn_id)
        return turn_messages

    def test_live_backpressure_capacity_is_chunk_boundary_invariant(self):
        """The queue bound represents audio duration for every packet size."""

        for chunk_ms in (10, 20, 40, 64, 100):
            with self.subTest(chunk_ms=chunk_ms):
                live_queue = production._SampleBoundedQueue(16_000)
                remaining = 16_000
                packet_samples = 16 * chunk_ms
                while remaining:
                    count = min(packet_samples, remaining)
                    live_queue.put_nowait([0] * count)
                    remaining -= count
                self.assertTrue(live_queue.full())
                with self.assertRaises(queue.Full):
                    live_queue.put_nowait([0])

    def test_synchronous_final_result_cannot_beat_registered_job_holder(self):
        """A scheduler callback inside ``submit`` still yields one ordered pair."""

        async def scenario():
            service = _FinalSubmissionService(outcome="synchronous")
            manager, websocket, protocol = await self._open_protocol(service)
            try:
                self.assertEqual(
                    (await protocol.start({"turnId": "sync", "language": "en"}))["type"],
                    "started",
                )
                payload = b"\x01\x00\x02\x00"
                self.assertIsNone(await protocol.audio(self._packet(0, payload)))
                generation = protocol.turn.generation
                self.assertEqual((await protocol.finalize())["type"], "finalizing")
                self.assertEqual(protocol.turn.pcm_buffer, bytearray())

                self.assertTrue(
                    await asyncio.to_thread(
                        protocol.final_worker_done("sync", generation).wait,
                        1.0,
                    )
                )
                messages = await self._wait_for_turn_messages(websocket, "sync", 2)
                terminal, completion = self._assert_one_terminal_then_completion(
                    messages,
                    "sync",
                )
                self.assertEqual(terminal["text"], "synchronous final")
                self.assertEqual(completion["status"], "completed")
                self.assertEqual(len(service.scheduler.jobs), 1)
                self.assertEqual(
                    service.scheduler.jobs[0].audio.tolist(),
                    [1.0 / 32768.0, 2.0 / 32768.0],
                )
                self.assertEqual(service.pending_count(), 0)
            finally:
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_accepted_final_timeout_cancels_scheduler_work(self):
        """A local wait timeout must retire an accepted queued final job."""

        service = _FinalSubmissionService(outcome="never")
        with self.assertRaisesRegex(TimeoutError, "final transcription timed out"):
            production._service_turn_transcription(
                service,
                audio=b"pcm",
                language="en",
                timeout=0.0,
                session_id="timeout-session",
                generation=7,
            )

        self.assertEqual(service.scheduler.cancelled_sessions, [])
        self.assertEqual(
            service.scheduler.cancelled_requests,
            [service.scheduler.jobs[0].request_id],
        )
        self.assertEqual(service.pending_count(), 0)
        self.assertEqual(len(service.scheduler.jobs), 1)
        job = service.scheduler.jobs[0]
        self.assertEqual(job.session_id, "timeout-session")
        self.assertEqual(job.generation, 7)
        self.assertEqual(job.deadline_at, job.created_at)

    def test_cancel_does_not_block_event_loop_on_slow_stream_cancel(self):
        """Native stream cancellation must never run on the ASGI loop."""

        async def scenario():
            stream = _BlockingCancelLiveStream()
            service = _FinalSubmissionService(
                outcome="never",
                live_stream_factory=lambda language, use_prompt: stream,
            )
            manager, _websocket, protocol = await self._open_protocol(service)
            try:
                await protocol.start({"turnId": "slow-cancel", "language": "en"})
                cancelled = await asyncio.wait_for(protocol.cancel(), timeout=0.1)
                self.assertEqual(cancelled["type"], "cancelled")
                self.assertTrue(
                    await asyncio.to_thread(stream.cancel_started.wait, 1.0)
                )
            finally:
                stream.release_cancel.set()
                for thread in list(protocol._live_cancel_threads):
                    await asyncio.to_thread(thread.join, 1.0)
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_stream_creation_is_off_loop_and_stale_result_is_closed(self):
        """A cancelled start cannot install a stream created behind the engine lock."""

        async def scenario():
            factory = _BlockingCreateLiveStreamFactory()
            service = _FinalSubmissionService(
                live_stream_factory=lambda language, use_prompt: factory.create(
                    language,
                    use_prompt,
                )
            )
            manager, _websocket, protocol = await self._open_protocol(service)
            start_task = None
            loop_thread_id = threading.get_ident()
            try:
                start_task = asyncio.create_task(
                    protocol.start({"turnId": "old-start", "language": "en"})
                )
                self.assertTrue(
                    await asyncio.to_thread(factory.first_create_started.wait, 1.0)
                )
                self.assertFalse(start_task.done())

                cancelled = await asyncio.wait_for(protocol.cancel(), timeout=0.1)
                self.assertEqual(cancelled["type"], "cancelled")
                started = await asyncio.wait_for(
                    protocol.start({"turnId": "replacement", "language": "en"}),
                    timeout=1.0,
                )
                self.assertEqual(started["type"], "started")
                self.assertEqual(len(factory.streams), 2)
                replacement_stream = factory.streams[1]
                self.assertIs(protocol.turn.live_stream, replacement_stream)
                self.assertEqual(protocol.turn.phase, "receiving")

                factory.release_first_create.set()
                retired = await asyncio.wait_for(start_task, timeout=1.0)
                self.assertEqual(retired["error"]["code"], "start_cancelled")
                old_stream = factory.streams[0]
                self.assertTrue(
                    await asyncio.to_thread(old_stream.close_called.wait, 1.0)
                )
                self.assertNotEqual(factory.create_thread_ids[0], loop_thread_id)
                self.assertNotEqual(old_stream.close_thread_id, loop_thread_id)
                self.assertEqual(old_stream.closed, 1)
                self.assertIs(protocol.turn.live_stream, replacement_stream)
                self.assertEqual(protocol.turn.phase, "receiving")
            finally:
                factory.release_first_create.set()
                if start_task is not None and not start_task.done():
                    await asyncio.wait_for(start_task, timeout=1.0)
                await self._close_protocol(manager, protocol)
                for thread in list(protocol._live_cancel_threads):
                    await asyncio.to_thread(thread.join, 1.0)

        asyncio.run(scenario())

    def test_cancelled_start_reaps_a_late_native_stream(self):
        """ASGI cancellation cannot orphan a stream that finishes creation later."""

        async def scenario():
            factory = _BlockingCreateLiveStreamFactory()
            service = _FinalSubmissionService(
                live_stream_factory=lambda language, use_prompt: factory.create(
                    language,
                    use_prompt,
                )
            )
            manager, _websocket, protocol = await self._open_protocol(service)
            start_task = asyncio.create_task(
                protocol.start({"turnId": "cancelled-start", "language": "en"})
            )
            try:
                self.assertTrue(
                    await asyncio.to_thread(factory.first_create_started.wait, 1.0)
                )
                start_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await start_task
                self.assertIsNone(protocol.turn)

                factory.release_first_create.set()
                stream = factory.streams[0]
                self.assertTrue(
                    await asyncio.to_thread(stream.close_called.wait, 1.0)
                )
                for thread in list(protocol._live_stream_operation_threads):
                    await asyncio.to_thread(thread.join, 1.0)
                self.assertEqual(protocol._live_stream_operation_threads, set())
                self.assertEqual(stream.closed, 1)
            finally:
                factory.release_first_create.set()
                if not start_task.done():
                    await asyncio.wait_for(start_task, timeout=1.0)
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_live_stream_creation_capacity_has_no_hidden_executor_queue(self):
        """Blocked native creates reject immediately after the hard process limit."""

        async def scenario():
            factory = _BlockingAllCreateLiveStreamFactory()
            service = _FinalSubmissionService(
                live_stream_factory=lambda language, use_prompt: factory.create(
                    language,
                    use_prompt,
                )
            )
            managers = []
            protocols = []
            start_tasks = []
            extra_manager = extra_protocol = None
            try:
                for index in range(production._MAX_LIVE_STREAM_OPERATIONS):
                    manager, _websocket, protocol = await self._open_protocol(
                        service,
                        session_id=f"create-{index}",
                    )
                    managers.append(manager)
                    protocols.append(protocol)
                    start_tasks.append(
                        asyncio.create_task(
                            protocol.start(
                                {"turnId": f"create-{index}", "language": "en"}
                            )
                        )
                    )
                self.assertTrue(
                    await asyncio.to_thread(
                        factory.wait_for_creations,
                        production._MAX_LIVE_STREAM_OPERATIONS,
                        1.0,
                    )
                )
                self.assertEqual(
                    sum(
                        len(protocol._live_stream_operation_threads)
                        for protocol in protocols
                    ),
                    production._MAX_LIVE_STREAM_OPERATIONS,
                )

                extra_manager, _websocket, extra_protocol = await self._open_protocol(
                    service,
                    session_id="create-rejected",
                )
                rejected = await asyncio.wait_for(
                    extra_protocol.start(
                        {"turnId": "create-rejected", "language": "en"}
                    ),
                    timeout=0.1,
                )
                self.assertEqual(rejected["error"]["code"], "start_failed")
                self.assertEqual(
                    len(factory.streams),
                    production._MAX_LIVE_STREAM_OPERATIONS,
                )

                for protocol in protocols:
                    protocol.close()
                factory.release_creates.set()
                for start_task in start_tasks:
                    retired = await asyncio.wait_for(start_task, timeout=1.0)
                    self.assertEqual(retired["error"]["code"], "start_cancelled")
                for stream in factory.streams:
                    self.assertTrue(
                        await asyncio.to_thread(stream.close_called.wait, 1.0)
                    )
                for protocol in protocols:
                    for thread in list(protocol._live_stream_operation_threads):
                        await asyncio.to_thread(thread.join, 1.0)
                    self.assertEqual(protocol._live_stream_operation_threads, set())

                recovered = await asyncio.wait_for(
                    extra_protocol.start(
                        {"turnId": "create-recovered", "language": "en"}
                    ),
                    timeout=1.0,
                )
                self.assertEqual(recovered["type"], "started")
                self.assertEqual((await extra_protocol.cancel())["type"], "cancelled")
            finally:
                factory.release_creates.set()
                for protocol in protocols:
                    protocol.close()
                for start_task in start_tasks:
                    if not start_task.done():
                        await asyncio.wait_for(start_task, timeout=1.0)
                for manager, protocol in zip(managers, protocols):
                    await self._close_protocol(
                        manager,
                        protocol,
                        session_id=protocol.session_id,
                    )
                if extra_manager is not None:
                    await self._close_protocol(
                        extra_manager,
                        extra_protocol,
                        session_id=extra_protocol.session_id,
                    )
                    for thread in list(extra_protocol._live_cancel_threads):
                        await asyncio.to_thread(thread.join, 1.0)

        asyncio.run(scenario())

    def test_audio_queue_size_one_accepts_supported_packets_when_empty(self):
        """The sample queue admits one 64/100-ms packet before backpressure."""

        async def scenario(chunk_ms):
            stream = _BlockingDecodeLiveStream()
            service = _FinalSubmissionService(
                live_stream_factory=lambda language, use_prompt: stream
            )
            manager, _websocket, protocol = await self._open_protocol(service)
            try:
                await protocol.start({"turnId": f"packet-{chunk_ms}", "language": "en"})
                turn = protocol.turn
                sample_count = production.SERVER_SAMPLE_RATE * chunk_ms // 1_000
                self.assertGreaterEqual(turn.live_queue.max_samples, sample_count)
                self.assertIsNone(
                    await protocol.audio(
                        self._packet(0, b"\x01\x00" * sample_count)
                    )
                )
                self.assertTrue(
                    await asyncio.to_thread(stream.decode_started.wait, 1.0)
                )
            finally:
                stream.release_decode.set()
                await self._close_protocol(manager, protocol)
                for thread in list(protocol._live_cancel_threads):
                    await asyncio.to_thread(thread.join, 1.0)

        for chunk_ms in (64, 100):
            with self.subTest(chunk_ms=chunk_ms):
                asyncio.run(scenario(chunk_ms))

    def test_hanging_live_cancellations_are_globally_bounded(self):
        """A hung native cancel never creates an unbounded helper-thread backlog."""

        async def scenario():
            streams = []
            service = _FinalSubmissionService(
                live_stream_factory=lambda language, use_prompt: streams.append(
                    _HangingCancelLiveStream()
                )
                or streams[-1]
            )
            protocols = []
            turns = []
            managers = []
            try:
                for index in range(production._MAX_LIVE_CANCEL_THREADS + 1):
                    manager, _websocket, protocol = await self._open_protocol(
                        service,
                        session_id=f"cancel-{index}",
                    )
                    managers.append(manager)
                    protocols.append(protocol)
                    self.assertEqual(
                        (
                            await protocol.start(
                                {"turnId": f"cancel-{index}", "language": "en"}
                            )
                        )["type"],
                        "started",
                    )
                    turns.append(protocol.turn)
                    cancelled = await asyncio.wait_for(protocol.cancel(), timeout=0.1)
                    self.assertEqual(cancelled["type"], "cancelled")

                for stream in streams[: production._MAX_LIVE_CANCEL_THREADS]:
                    self.assertTrue(
                        await asyncio.to_thread(stream.cancel_started.wait, 1.0)
                    )
                self.assertEqual(
                    sum(stream.cancel_started.is_set() for stream in streams),
                    production._MAX_LIVE_CANCEL_THREADS,
                )
                self.assertEqual(
                    sum(len(protocol._live_cancel_threads) for protocol in protocols),
                    production._MAX_LIVE_CANCEL_THREADS,
                )
                saturated_stream = streams[-1]
                self.assertFalse(saturated_stream.cancel_started.is_set())
                self.assertEqual(saturated_stream.cancel_calls, 0)
                self.assertTrue(
                    await asyncio.to_thread(turns[-1].live_done.wait, 1.0)
                )
                self.assertEqual(saturated_stream.closed, 1)

                # Retiring the same stream twice never starts a second native
                # cancellation attempt while its first one is still hung.
                protocols[0]._stop_live_stream(turns[0])
                self.assertEqual(streams[0].cancel_calls, 1)
            finally:
                for stream in streams:
                    stream.release_cancel.set()
                for protocol in protocols:
                    for thread in list(protocol._live_cancel_threads):
                        await asyncio.to_thread(thread.join, 1.0)
                    self.assertEqual(protocol._live_cancel_threads, set())
                for manager, protocol in zip(managers, protocols):
                    await self._close_protocol(
                        manager,
                        protocol,
                        session_id=protocol.session_id,
                    )

        asyncio.run(scenario())

    def test_double_finalize_with_inflight_job_submits_once_and_completes_once(self):
        """The second finalize loses the phase race without duplicating the job."""

        async def scenario():
            service = _FinalSubmissionService()
            manager, websocket, protocol = await self._open_protocol(service)
            try:
                await protocol.start({"turnId": "inflight", "language": "en"})
                payload = b"\x03\x00\x04\x00"
                self.assertIsNone(await protocol.audio(self._packet(0, payload)))
                generation = protocol.turn.generation

                self.assertEqual((await protocol.finalize())["type"], "finalizing")
                self.assertTrue(
                    await asyncio.to_thread(service.scheduler.wait_for_job_count, 1, 1.0)
                )
                duplicate = await protocol.finalize()
                self.assertEqual(duplicate["error"]["code"], "turn_not_active")
                self.assertEqual(len(service.scheduler.jobs), 1)
                self.assertEqual(websocket.messages, [])

                service.release_final.set()
                self.assertTrue(
                    await asyncio.to_thread(
                        protocol.final_worker_done("inflight", generation).wait,
                        1.0,
                    )
                )
                messages = await self._wait_for_turn_messages(websocket, "inflight", 2)
                terminal, completion = self._assert_one_terminal_then_completion(
                    messages,
                    "inflight",
                )
                self.assertEqual(terminal["text"], "deferred final")
                self.assertEqual(completion["status"], "completed")
                self.assertEqual(
                    service.scheduler.jobs[0].audio.tolist(),
                    [3.0 / 32768.0, 4.0 / 32768.0],
                )
                self.assertEqual(service.pending_count(), 0)
            finally:
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_slow_reader_cannot_accumulate_terminal_turns(self):
        """A turn stays active until its completion reaches the socket."""

        async def scenario():
            service = _FinalSubmissionService(outcome="synchronous")
            manager = production.OrderedConnectionManager(max_pending_events=3)
            manager.bind_loop(asyncio.get_running_loop())
            websocket = _BlockingSocket()
            await manager.connect("session", websocket)
            protocol = _ObservedProtocol(service, manager, "session", self._settings())
            protocol.attach(_Session())
            try:
                await protocol.start({"turnId": "slow-reader", "language": "en"})
                self.assertIsNone(await protocol.audio(self._packet(0)))
                generation = protocol.turn.generation
                await protocol.finalize()
                self.assertTrue(
                    await asyncio.to_thread(
                        protocol.final_worker_done("slow-reader", generation).wait,
                        1.0,
                    )
                )
                await asyncio.wait_for(websocket.send_started.wait(), timeout=1.0)

                for index in range(20):
                    rejected = await protocol.start(
                        {"turnId": f"overflow-{index}", "language": "en"}
                    )
                    self.assertEqual(rejected["error"]["code"], "turn_in_progress")
                with manager._event_lock:
                    pending = len(manager._delivery_states["session"].queue)
                self.assertLessEqual(pending, 1)
                self.assertEqual(protocol.turn.phase, "terminal_result")

                websocket.release_send.set()
                messages = await self._wait_for_turn_messages(
                    websocket, "slow-reader", 2
                )
                self._assert_one_terminal_then_completion(messages, "slow-reader")
                for _ in range(100):
                    if protocol.turn.phase == "completed":
                        break
                    await asyncio.sleep(0.001)
                self.assertEqual(protocol.turn.phase, "completed")
            finally:
                websocket.release_send.set()
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_reset_or_cancel_cannot_split_admitted_terminal_pair(self):
        """Retirement after final admission still yields the old completion."""

        for action in ("reset", "cancel"):
            with self.subTest(action=action):
                async def scenario():
                    service = _FinalSubmissionService(outcome="synchronous")
                    manager = _PausedTerminalAdmissionManager()
                    manager, websocket, protocol = await self._open_protocol(
                        service,
                        manager=manager,
                    )
                    action_started = threading.Event()
                    action_done = threading.Event()
                    action_result = []
                    try:
                        turn_id = f"terminal-{action}"
                        await protocol.start({"turnId": turn_id, "language": "en"})
                        self.assertIsNone(await protocol.audio(self._packet(0)))
                        generation = protocol.turn.generation
                        await protocol.finalize()
                        self.assertTrue(
                            await asyncio.to_thread(manager.terminal_admitted.wait, 1.0)
                        )

                        def retire() -> None:
                            action_started.set()
                            action_result.append(asyncio.run(getattr(protocol, action)()))
                            action_done.set()

                        retire_thread = threading.Thread(target=retire, daemon=True)
                        retire_thread.start()
                        self.assertTrue(
                            await asyncio.to_thread(action_started.wait, 1.0)
                        )
                        self.assertFalse(action_done.is_set())

                        manager.release_terminal.set()
                        self.assertTrue(
                            await asyncio.to_thread(
                                protocol.final_worker_done(turn_id, generation).wait,
                                1.0,
                            )
                        )
                        self.assertTrue(await asyncio.to_thread(action_done.wait, 1.0))
                        messages = await self._wait_for_turn_messages(
                            websocket,
                            turn_id,
                            2,
                        )
                        self._assert_one_terminal_then_completion(messages, turn_id)
                        self.assertEqual(
                            action_result[0]["type"],
                            "reset" if action == "reset" else "cancelled",
                        )
                    finally:
                        manager.release_terminal.set()
                        await self._close_protocol(manager, protocol)

                asyncio.run(scenario())

    def test_reset_loops_cannot_exceed_terminal_reserve_for_slow_reader(self):
        """Reset cannot turn terminal preservation into an unbounded outbox."""

        async def scenario():
            service = _FinalSubmissionService(outcome="synchronous")
            manager = production.OrderedConnectionManager(max_pending_events=3)
            manager.bind_loop(asyncio.get_running_loop())
            websocket = _BlockingSocket()
            await manager.connect("session", websocket)
            protocol = _ObservedProtocol(service, manager, "session", self._settings())
            protocol.attach(_Session())
            maximum_pending = 0
            try:
                for index in range(10):
                    started = await protocol.start(
                        {"turnId": f"reset-loop-{index}", "language": "en"}
                    )
                    self.assertEqual(started["type"], "started")
                    generation = protocol.turn.generation
                    await protocol.finalize()
                    self.assertTrue(
                        await asyncio.to_thread(
                            protocol.final_worker_done(
                                f"reset-loop-{index}", generation
                            ).wait,
                            1.0,
                        )
                    )
                    with manager._event_lock:
                        state = manager._delivery_states.get("session")
                        if state is None or state.closed:
                            break
                        maximum_pending = max(maximum_pending, len(state.queue))
                    await protocol.reset()
                for _ in range(100):
                    if websocket.close_code == 1013:
                        break
                    await asyncio.sleep(0.001)
                self.assertEqual(websocket.close_code, 1013)
                self.assertLessEqual(maximum_pending, manager.max_pending_events + 2)
            finally:
                websocket.release_send.set()
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_live_queue_drain_rejection_has_one_terminal_error_and_completion(self):
        """A full queue cannot hang or submit a final from an unsealed turn."""

        async def scenario():
            service = _FinalSubmissionService()
            manager, websocket, protocol = await self._open_protocol(service)
            try:
                await protocol.start({"turnId": "drain-rejected", "language": "en"})
                self.assertIsNone(await protocol.audio(self._packet(0)))
                generation = protocol.turn.generation
                rejecting_queue = _RejectingDrainQueue()
                protocol.turn.live_queue = rejecting_queue

                self.assertEqual((await protocol.finalize())["type"], "finalizing")
                self.assertTrue(
                    await asyncio.to_thread(
                        protocol.final_worker_done("drain-rejected", generation).wait,
                        1.0,
                    )
                )
                messages = await self._wait_for_turn_messages(
                    websocket,
                    "drain-rejected",
                    2,
                )
                terminal, completion = self._assert_one_terminal_then_completion(
                    messages,
                    "drain-rejected",
                    failed=True,
                )
                self.assertEqual(terminal["error"]["code"], "finalize_failed")
                self.assertEqual(completion["status"], "failed")
                self.assertEqual(rejecting_queue.put_calls, 1)
                self.assertEqual(service.scheduler.jobs, [])
            finally:
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_final_submission_rejection_has_one_terminal_error_and_completion(self):
        """A rejected final-lane submission is terminal, ordered, and leak-free."""

        async def scenario():
            service = _FinalSubmissionService(outcome="rejected")
            manager, websocket, protocol = await self._open_protocol(service)
            try:
                await protocol.start({"turnId": "submission-rejected", "language": "en"})
                self.assertIsNone(await protocol.audio(self._packet(0)))
                generation = protocol.turn.generation
                self.assertEqual((await protocol.finalize())["type"], "finalizing")

                self.assertTrue(
                    await asyncio.to_thread(
                        protocol.final_worker_done("submission-rejected", generation).wait,
                        1.0,
                    )
                )
                messages = await self._wait_for_turn_messages(
                    websocket,
                    "submission-rejected",
                    2,
                )
                terminal, completion = self._assert_one_terminal_then_completion(
                    messages,
                    "submission-rejected",
                    failed=True,
                )
                self.assertEqual(terminal["error"]["code"], "final_transcription_failed")
                self.assertIn("synthetic final lane rejection", terminal["error"]["message"])
                self.assertEqual(completion["status"], "failed")
                self.assertEqual(len(service.scheduler.jobs), 1)
                self.assertEqual(service.pending_count(), 0)
            finally:
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_production_session_handle_propagates_accepted_job_drops(self):
        service = _FinalSubmissionService()
        handle = production.ProductionSessionHandle(service, "session")
        job = SimpleNamespace(request_id="request", kind="final")

        handle.on_job_dropped(job, "shutdown")

        self.assertEqual(
            service.failed_pending,
            [("request", "final transcription was shutdown")],
        )

    def test_cancel_drains_a_full_live_queue_and_stops_the_worker(self):
        """Cancellation cannot strand a worker behind a full bounded queue."""

        async def scenario():
            stream = _BlockingDecodeLiveStream()
            service = _FinalSubmissionService(
                live_stream_factory=lambda language, use_prompt: stream
            )
            manager, _, protocol = await self._open_protocol(service)
            try:
                await protocol.start({"turnId": "full-queue", "language": "en"})
                turn = protocol.turn
                self.assertIsNone(await protocol.audio(self._packet(0)))
                self.assertTrue(await asyncio.to_thread(stream.decode_started.wait, 1.0))
                self.assertIsNone(
                    await protocol.audio(
                        self._packet(1, b"\x01\x00" * turn.live_queue.max_samples)
                    )
                )
                self.assertTrue(turn.live_queue.full())

                self.assertEqual((await protocol.cancel())["type"], "cancelled")
                self.assertTrue(await asyncio.to_thread(turn.live_done.wait, 1.0))
                await asyncio.to_thread(turn.live_thread.join, 1.0)
                self.assertFalse(turn.live_thread.is_alive())
                for _ in range(100):
                    if not protocol._live_cancel_threads:
                        break
                    await asyncio.sleep(0.001)
                self.assertEqual(protocol._live_cancel_threads, set())
                self.assertEqual(stream.cancelled, 1)
                self.assertEqual(stream.closed, 1)
                self.assertEqual(turn.live_queue.unfinished_tasks, 0)
                self.assertEqual(service.scheduler.jobs, [])
            finally:
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_live_drain_timeout_cancels_worker_and_still_emits_terminal_pair(self):
        """A stuck live decoder is retired before the failed completion."""

        async def scenario():
            stream = _BlockingDecodeLiveStream()
            service = _FinalSubmissionService(
                live_stream_factory=lambda language, use_prompt: stream
            )
            manager, websocket, protocol = await self._open_protocol(service)
            protocol.settings.finalize_timeout_seconds = 0.05
            try:
                await protocol.start({"turnId": "drain-timeout", "language": "en"})
                turn = protocol.turn
                self.assertIsNone(await protocol.audio(self._packet(0)))
                self.assertTrue(await asyncio.to_thread(stream.decode_started.wait, 1.0))
                generation = turn.generation

                self.assertEqual((await protocol.finalize())["type"], "finalizing")
                self.assertTrue(
                    await asyncio.to_thread(
                        protocol.final_worker_done("drain-timeout", generation).wait,
                        1.0,
                    )
                )
                messages = await self._wait_for_turn_messages(
                    websocket,
                    "drain-timeout",
                    2,
                )
                terminal, completion = self._assert_one_terminal_then_completion(
                    messages,
                    "drain-timeout",
                    failed=True,
                )
                self.assertEqual(
                    terminal["error"]["code"],
                    "final_transcription_failed",
                )
                self.assertIn("drain timed out", terminal["error"]["message"])
                self.assertEqual(completion["status"], "failed")
                self.assertTrue(await asyncio.to_thread(turn.live_done.wait, 1.0))
                await asyncio.to_thread(turn.live_thread.join, 1.0)
                self.assertFalse(turn.live_thread.is_alive())
                self.assertEqual(service.scheduler.jobs, [])
            finally:
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_completed_final_threads_are_not_retained_across_turns(self):
        """Only active final workers remain referenced during a long session."""

        async def scenario():
            service = _FinalSubmissionService(outcome="synchronous")
            manager, websocket, protocol = await self._open_protocol(service)
            try:
                for index in range(20):
                    turn_id = f"turn-{index}"
                    await protocol.start({"turnId": turn_id, "language": "en"})
                    self.assertIsNone(await protocol.audio(self._packet(0)))
                    generation = protocol.turn.generation
                    self.assertEqual((await protocol.finalize())["type"], "finalizing")
                    self.assertTrue(
                        await asyncio.to_thread(
                            protocol.final_worker_done(turn_id, generation).wait,
                            1.0,
                        )
                    )
                    await self._wait_for_turn_messages(websocket, turn_id, 2)
                    for _ in range(100):
                        if protocol.turn.phase == "completed":
                            break
                        await asyncio.sleep(0.001)
                    self.assertEqual(protocol.turn.phase, "completed")
                    threads = list(protocol._completion_threads)
                    for thread in threads:
                        await asyncio.to_thread(thread.join, 1.0)
                    self.assertEqual(protocol._completion_threads, set())
                self.assertEqual(len(service.scheduler.jobs), 20)
            finally:
                await self._close_protocol(manager, protocol)

        asyncio.run(scenario())

    def test_cancel_or_reset_then_new_turn_fences_late_live_and_final_callbacks(self):
        """Both cancellation paths fence stale callbacks even when IDs are reused."""

        for action in ("cancel", "reset"):
            with self.subTest(action=action):
                asyncio.run(self._exercise_stale_callbacks_after(action))

    async def _exercise_stale_callbacks_after(self, action):
        streams = []

        def create_stream(language, use_prompt):
            self.assertEqual(language, "en")
            self.assertFalse(use_prompt)
            stream = _GateLiveStream(
                finish_text="late old partial" if not streams else "",
            )
            streams.append(stream)
            return stream

        service = _FinalSubmissionService(live_stream_factory=create_stream)
        manager, websocket, protocol = await self._open_protocol(service)
        try:
            old_turn_id = "reused-turn"
            await protocol.start({"turnId": old_turn_id, "language": "en"})
            old_generation = protocol.turn.generation
            old_stream = streams[0]
            self.assertIsNone(await protocol.audio(self._packet(0)))
            self.assertEqual((await protocol.finalize())["type"], "finalizing")
            self.assertTrue(
                await asyncio.to_thread(old_stream.finish_started.wait, 1.0)
            )

            response = await getattr(protocol, action)()
            self.assertEqual(response["type"], "cancelled" if action == "cancel" else "reset")
            self.assertEqual(protocol.session.clear_calls, 1)
            self.assertEqual(
                (await protocol.start({"turnId": old_turn_id, "language": "en"}))["type"],
                "started",
            )
            new_stream = streams[1]

            # Even a live callback released after a replacement turn with the
            # same ID must be cancelled and fenced before final submission.
            old_stream.release_finish.set()
            self.assertTrue(
                await asyncio.to_thread(
                    protocol.final_worker_done(old_turn_id, old_generation).wait,
                    1.0,
                )
            )
            self.assertEqual(
                [
                    message
                    for message in websocket.messages
                    if message.get("turnId") == old_turn_id
                ],
                [],
            )
            self.assertEqual(old_stream.closed, 1)
            self.assertEqual(old_stream.cancelled, 1)
            self.assertEqual(service.scheduler.jobs, [])

            new_generation = protocol.turn.generation
            self.assertIsNone(await protocol.audio(self._packet(0, b"\x05\x00\x06\x00")))
            new_stream.release_finish.set()
            self.assertEqual((await protocol.finalize())["type"], "finalizing")
            self.assertTrue(
                await asyncio.to_thread(service.scheduler.wait_for_job_count, 1, 1.0)
            )
            service.release_final.set()
            self.assertTrue(
                await asyncio.to_thread(
                    protocol.final_worker_done(old_turn_id, new_generation).wait,
                    1.0,
                )
            )
            messages = await self._wait_for_turn_messages(websocket, old_turn_id, 2)
            self._assert_one_terminal_then_completion(messages, old_turn_id)
            self.assertEqual(len(service.scheduler.jobs), 1)
        finally:
            await self._close_protocol(manager, protocol)

    def test_disconnect_reconnect_drops_a_paused_old_completion(self):
        """An admitted terminal pair cannot cross a connection epoch."""

        async def scenario():
            old_turn_id = "old-turn"
            service = _FinalSubmissionService(outcome="synchronous")
            manager = production.OrderedConnectionManager()
            manager.bind_loop(asyncio.get_running_loop())
            old_socket = _BlockingSocket()
            await manager.connect("session", old_socket)
            old_protocol = _ObservedProtocol(service, manager, "session", self._settings())
            old_protocol.attach(_Session())
            old_epoch = manager.connection_epoch("session")
            try:
                await old_protocol.start({"turnId": old_turn_id, "language": "en"})
                old_generation = old_protocol.turn.generation
                self.assertIsNone(await old_protocol.audio(self._packet(0)))
                self.assertEqual((await old_protocol.finalize())["type"], "finalizing")
                self.assertTrue(
                    await asyncio.to_thread(
                        old_protocol.final_worker_done(old_turn_id, old_generation).wait,
                        1.0,
                    )
                )
                await asyncio.wait_for(old_socket.send_started.wait(), timeout=1.0)
                with manager._event_lock:
                    self.assertLessEqual(
                        len(manager._delivery_states["session"].queue),
                        1,
                    )

                old_protocol.close()
                await manager.disconnect("session")
                manager.clear_session("session")
                self.assertNotIn("session", manager._connection_epochs)
                self.assertEqual(manager.connection_epoch("session"), 0)

                new_socket = _Socket()
                await manager.connect("session", new_socket)
                self.assertGreater(manager.connection_epoch("session"), old_epoch)
                new_protocol = _ObservedProtocol(service, manager, "session", self._settings())
                new_protocol.attach(_Session())
                try:
                    await new_protocol.start({"turnId": "new-turn", "language": "en"})
                    self.assertEqual(
                        [
                            message
                            for message in new_socket.messages
                            if message.get("turnId") == old_turn_id
                        ],
                        [],
                    )

                    self.assertIsNone(await new_protocol.audio(self._packet(0)))
                    new_generation = new_protocol.turn.generation
                    self.assertEqual((await new_protocol.finalize())["type"], "finalizing")
                    self.assertTrue(
                        await asyncio.to_thread(service.scheduler.wait_for_job_count, 2, 1.0)
                    )
                    self.assertTrue(
                        await asyncio.to_thread(
                            new_protocol.final_worker_done("new-turn", new_generation).wait,
                            1.0,
                        )
                    )
                    messages = await self._wait_for_turn_messages(new_socket, "new-turn", 2)
                    self._assert_one_terminal_then_completion(messages, "new-turn")
                finally:
                    await self._close_protocol(manager, new_protocol)
            finally:
                old_socket.release_send.set()
                manager.clear_session("session")

        asyncio.run(scenario())

    def test_disconnect_reconnect_fences_a_paused_old_partial(self):
        """A partial tagged with an old transport epoch cannot cross reconnect."""

        async def scenario():
            manager = production.OrderedConnectionManager()
            manager.bind_loop(asyncio.get_running_loop())
            old_socket = _BlockingSocket()
            await manager.connect("session", old_socket)
            old_epoch = manager.connection_epoch("session")
            manager.set_turn("session", "old-turn")
            manager.publish_session(
                "session",
                {"type": "status", "state": "blocked"},
            )
            await asyncio.wait_for(old_socket.send_started.wait(), timeout=1.0)
            manager.publish_session(
                "session",
                {
                    "type": "realtime",
                    "turnId": "old-turn",
                    "text": "old partial",
                    "_connectionEpoch": old_epoch,
                },
            )

            await manager.disconnect("session")
            manager.clear_session("session")
            new_socket = _Socket()
            await manager.connect("session", new_socket)
            self.assertGreater(manager.connection_epoch("session"), old_epoch)
            manager.set_turn("session", "new-turn")

            rejected = manager.publish_session(
                "session",
                {
                    "type": "realtime",
                    "turnId": "old-turn",
                    "text": "late old partial",
                    "_connectionEpoch": old_epoch,
                },
            )
            self.assertIsNone(rejected)
            await asyncio.sleep(0)
            self.assertEqual(new_socket.messages, [])
            old_socket.release_send.set()
            await manager.disconnect("session")
            manager.clear_session("session")

        asyncio.run(scenario())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
