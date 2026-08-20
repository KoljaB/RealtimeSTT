import asyncio
import json
import unittest
import threading
import time
import queue
from types import SimpleNamespace
from unittest import mock

from RealtimeSTT_server import production_server as production


class ProductionServerSettingsTests(unittest.TestCase):
    def test_default_bind_is_loopback_and_capabilities_report_resources(self):
        settings = production.ProductionServerSettings()

        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(production.SERVER_VERSION, "1.0.3")
        capabilities = production.capabilities_for(settings)
        self.assertEqual(capabilities["apiVersion"], "v1")
        self.assertEqual(capabilities["protocolVersion"], "realtimestt.remote.v1")
        self.assertEqual(capabilities["server"]["version"], "1.0.3")
        self.assertEqual(capabilities["audio"]["format"], "pcm_s16le")
        self.assertEqual(capabilities["audio"]["channels"], 1)
        self.assertIn("final", capabilities["models"])
        self.assertIn("live", capabilities["models"])
        self.assertIn("provider", capabilities["models"]["final"])
        self.assertIn("languages", capabilities["models"]["live"])

    def test_non_loopback_requires_bearer_token_and_tls(self):
        with self.assertRaisesRegex(ValueError, "bearer token"):
            production.ProductionServerSettings(host="0.0.0.0")

        with self.assertRaisesRegex(ValueError, "TLS"):
            production.ProductionServerSettings(
                host="0.0.0.0",
                bearer_token="secret",
            )

        settings = production.ProductionServerSettings(
            host="0.0.0.0",
            bearer_token="secret",
            ssl_certfile="server-chain.pem",
            ssl_keyfile="server-key.pem",
        )
        self.assertTrue(production.capabilities_for(settings)["authentication"]["required"])
        self.assertTrue(settings.public_dict()["tls_enabled"])

    def test_tls_cert_and_key_must_be_configured_as_a_pair(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            production.ProductionServerSettings(ssl_certfile="server-chain.pem")

        with self.assertRaisesRegex(ValueError, "provided together"):
            production.ProductionServerSettings(ssl_keyfile="server-key.pem")

    def test_bearer_token_can_come_from_environment_without_public_exposure(self):
        with mock.patch.dict("os.environ", {"REALTIMESTT_SERVER_BEARER_TOKEN": "from-env"}):
            settings = production.ProductionServerSettings()
        self.assertEqual(settings.bearer_token, "from-env")
        self.assertNotIn("from-env", str(settings.public_dict()))

    def test_language_validation_accepts_remote_languages_and_rejects_bad_values(self):
        settings = production.ProductionServerSettings()
        self.assertIsNone(production._language_error("de", settings))
        error = production._language_error("not a language", settings)
        self.assertEqual(error["code"], "unsupported_language")
        self.assertEqual(production._language_error("", settings)["code"], "invalid_language")

    def test_auto_language_is_supported_for_model_side_detection(self):
        settings = production.ProductionServerSettings()

        self.assertIsNone(production._language_error("auto", settings))
        self.assertIn("auto", production.capabilities_for(settings)["languages"])

    def test_auto_detection_uses_provider_language_but_fixed_language_is_preserved(self):
        result = SimpleNamespace(info=SimpleNamespace(language="fr"))

        self.assertEqual(
            production._reported_detected_language(result, "auto"),
            "fr",
        )
        self.assertEqual(
            production._reported_detected_language(result, "de"),
            "de",
        )
        self.assertIsNone(
            production._reported_detected_language(SimpleNamespace(), "auto")
        )


class ProductionServerUtilityTests(unittest.TestCase):
    def test_loopback_detection(self):
        self.assertTrue(production.is_loopback_host("127.0.0.1"))
        self.assertTrue(production.is_loopback_host("::1"))
        self.assertTrue(production.is_loopback_host("localhost"))
        self.assertFalse(production.is_loopback_host("0.0.0.0"))

    def test_structured_error_is_machine_readable(self):
        error = production._structured_error(
            "invalid_audio",
            "bad packet",
            session_id="s1",
            turn_id="t1",
            details={"expected": 1},
        )
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["error"]["code"], "invalid_audio")
        self.assertEqual(error["error"]["details"]["expected"], 1)
        self.assertEqual(error["sessionId"], "s1")
        self.assertEqual(error["turnId"], "t1")

    def test_final_event_barrier_is_one_shot(self):
        barrier = production.FinalEventBarrier()
        final = {"type": "final", "text": "done"}

        self.assertTrue(barrier.resolve(final))
        self.assertFalse(barrier.resolve({"type": "error"}))
        self.assertTrue(barrier.wait(timeout=0.0))
        self.assertEqual(barrier.outcome, final)

    def test_completion_waits_for_pre_finalize_in_flight_final(self):
        """A final counted before stop must still precede completion."""

        async def scenario():
            class Manager:
                def __init__(self):
                    self._loop = asyncio.get_running_loop()
                    self._barriers = {}
                    self._turn_ids = {}
                    self.events = []
                    self.completion = asyncio.Event()

                def set_turn(self, session_id, turn_id):
                    self._turn_ids[session_id] = turn_id

                def register_final_barrier(self, session_id, turn_id):
                    barrier = production.FinalEventBarrier()
                    self._barriers[(session_id, turn_id)] = barrier
                    return barrier

                def unregister_final_barrier(self, session_id, turn_id, barrier):
                    if self._barriers.get((session_id, turn_id)) is barrier:
                        self._barriers.pop((session_id, turn_id), None)

                def publish_session(self, session_id, message):
                    event = dict(message)
                    self.events.append(event)
                    if event.get("type") == "final":
                        barrier = self._barriers.get((session_id, event.get("turnId")))
                        if barrier is not None:
                            barrier.resolve(event)

                async def emit(self, session_id, message):
                    event = dict(message)
                    self.events.append(event)
                    if event.get("type") == "completion":
                        self.completion.set()
                    return True

            class Session:
                def __init__(self, manager):
                    self.manager = manager
                    self.settings = SimpleNamespace(language="en")
                    self.recorder = SimpleNamespace(
                        realtime_transcription_executor=None,
                    )
                    self.final_submitted = 0
                    self.release_final = threading.Event()

                def start_streaming(self):
                    # The final was submitted by the recorder while the turn
                    # was active, before the client issued finalize.  Its
                    # publication is deliberately delayed until after
                    # stop_streaming returns.
                    self.final_submitted = 1

                def stop_streaming(self):
                    def publish_later():
                        self.release_final.wait()
                        self.manager.publish_session(
                            "session",
                            {
                                "type": "final",
                                "turnId": "turn-1",
                                "text": "late final",
                            },
                        )

                    threading.Thread(target=publish_later, daemon=True).start()
                    return False

                def snapshot(self):
                    return {
                        "finalSubmitted": self.final_submitted,
                        "finalCompleted": 1,
                        "realtimeCompleted": 0,
                    }

            manager = Manager()
            session = Session(manager)
            protocol = production.ProductionSessionProtocol(
                SimpleNamespace(),
                manager,
                "session",
                production.ProductionServerSettings(finalize_timeout_seconds=1.0),
            )
            protocol.attach(session)

            started = await protocol.start({"type": "start", "turnId": "turn-1"})
            self.assertEqual(started["type"], "started")
            finalizing = await protocol.finalize()
            self.assertEqual(finalizing["type"], "finalizing")

            # The completion worker must be waiting on the unresolved barrier,
            # not treating the unchanged post-stop counter as no-speech.
            self.assertEqual([event["type"] for event in manager.events], [])
            session.release_final.set()
            await asyncio.wait_for(manager.completion.wait(), timeout=1.0)
            self.assertEqual(
                [event["type"] for event in manager.events],
                ["final", "completion"],
            )
            protocol.close()

        asyncio.run(scenario())

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_recorder_stop_flushes_and_drains_input_before_final_queue(self):
        from example_fastapi_server.server import RecorderBackedRealtimeSession

        calls = []

        class Recorder:
            def flush_audio_input(self):
                calls.append("flush_input")
                return True

            def drain_audio_input(self, timeout=None):
                calls.append(("drain_input", timeout))
                return True

            def flush_buffered_audio(self):
                calls.append("flush_final")
                return False

            def has_pending_recordings(self):
                calls.append("pending_final")
                return False

        session = SimpleNamespace(
            final_submitted=0,
            lock=threading.RLock(),
            streaming=True,
            status="listening",
            recorder=Recorder(),
            settings=SimpleNamespace(finalize_timeout_seconds=2.5),
            service=SimpleNamespace(
                deactivate_speaker=lambda session_id: calls.append(
                    ("deactivate", session_id)
                )
            ),
            session_id="session",
            _trim_recorded_audio_queue=lambda: calls.append("trim_final"),
            publish_status=lambda status: calls.append(("status", status)),
        )

        final_expected = RecorderBackedRealtimeSession.stop_streaming(session)

        self.assertFalse(final_expected)
        self.assertEqual(
            calls,
            [
                "flush_input",
                ("drain_input", 2.5),
                "flush_final",
                "trim_final",
                ("deactivate", "session"),
                ("status", "idle"),
                "pending_final",
            ],
        )

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_recorder_stop_rejects_an_unresolved_input_drain(self):
        from example_fastapi_server.server import RecorderBackedRealtimeSession

        calls = []
        recorder = SimpleNamespace(
            flush_audio_input=lambda: calls.append("flush_input"),
            drain_audio_input=lambda timeout=None: (
                calls.append(("drain_input", timeout)) or False
            ),
            flush_buffered_audio=lambda: calls.append("flush_final"),
        )
        session = SimpleNamespace(
            final_submitted=0,
            lock=threading.RLock(),
            streaming=True,
            status="listening",
            recorder=recorder,
            settings=SimpleNamespace(finalize_timeout_seconds=1.25),
            service=SimpleNamespace(
                deactivate_speaker=lambda session_id: calls.append(
                    ("deactivate", session_id)
                )
            ),
            session_id="session",
            _trim_recorded_audio_queue=lambda: calls.append("trim_final"),
            publish_status=lambda status: calls.append(("status", status)),
        )

        with self.assertRaisesRegex(RuntimeError, "Audio input drain timed out"):
            RecorderBackedRealtimeSession.stop_streaming(session)

        self.assertEqual(
            calls,
            [
                "flush_input",
                ("drain_input", 1.25),
                ("deactivate", "session"),
            ],
        )

    def test_release_service_resources_is_idempotent(self):
        class Engine:
            def __init__(self):
                self.calls = 0

            def close(self):
                self.calls += 1

        class Worker:
            def __init__(self, engine):
                self.engine = engine

        class Service:
            _production_resources_released = False

        service = Service()
        engine = Engine()
        service.scheduler = type("Scheduler", (), {
            "main_worker": Worker(engine),
            "realtime_worker": Worker(engine),
        })()

        production.release_service_resources(service)
        production.release_service_resources(service)
        self.assertEqual(engine.calls, 1)

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_ordered_manager_serializes_event_loop_and_backend_thread_emissions(self):
        class WebSocket:
            def __init__(self):
                self.messages = []

            async def accept(self):
                pass

            async def send_text(self, payload):
                # Give competing producers a chance to enqueue while the
                # session sender is awaiting the transport.
                await asyncio.sleep(0)
                self.messages.append(json.loads(payload))

        async def scenario():
            manager = production.OrderedConnectionManager()
            websocket = WebSocket()
            manager.bind_loop(asyncio.get_running_loop())
            await manager.connect("session", websocket)

            backend_count = 24
            direct_count = 24
            start = threading.Event()

            def publish_backend(index):
                start.wait()
                if index % 2:
                    time.sleep(0.001)
                manager.publish_session(
                    "session", {"type": "backend", "index": index}
                )

            threads = [
                threading.Thread(
                    target=publish_backend,
                    args=(index,),
                )
                for index in range(backend_count)
            ]
            for thread in threads:
                thread.start()
            start.set()

            async def emit_direct(index):
                await asyncio.sleep(0)
                return await manager.emit(
                    "session", {"type": "direct", "index": index}
                )

            direct_tasks = [asyncio.create_task(emit_direct(index)) for index in range(direct_count)]
            self.assertTrue(all(await asyncio.gather(*direct_tasks)))
            await asyncio.gather(*(asyncio.to_thread(thread.join) for thread in threads))
            for _ in range(1000):
                if len(websocket.messages) == backend_count + direct_count:
                    break
                await asyncio.sleep(0.001)

            sequences = [message["eventSequence"] for message in websocket.messages]
            self.assertEqual(
                sequences,
                list(range(1, backend_count + direct_count + 1)),
            )
            self.assertEqual(len(websocket.messages), backend_count + direct_count)

            await manager.disconnect("session")
            manager.clear_session("session")
            self.assertNotIn("session", manager._delivery_states)

        asyncio.run(scenario())

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_ordered_manager_coalesces_pending_partials_and_preserves_terminals(self):
        class WebSocket:
            def __init__(self):
                self.messages = []
                self.first_send = asyncio.Event()
                self.release_first_send = asyncio.Event()

            async def accept(self):
                pass

            async def send_text(self, payload):
                self.messages.append(json.loads(payload))
                if len(self.messages) == 1:
                    self.first_send.set()
                    await self.release_first_send.wait()

        async def scenario():
            manager = production.OrderedConnectionManager(max_pending_events=3)
            websocket = WebSocket()
            manager.bind_loop(asyncio.get_running_loop())
            await manager.connect("session", websocket)

            manager.publish_session("session", {"type": "status", "state": "busy"})
            await websocket.first_send.wait()
            for index in range(100):
                manager.publish_session(
                    "session", {"type": "realtime", "text": f"partial-{index}"}
                )
            manager.publish_session("session", {"type": "final", "text": "final"})
            manager.publish_session(
                "session", {"type": "completion", "status": "completed"}
            )

            with manager._event_lock:
                pending = list(manager._delivery_states["session"].queue)
            self.assertLessEqual(len(pending), 3)
            self.assertEqual(
                sum(item.event.get("type") == "partial" for item in pending),
                1,
            )

            websocket.release_first_send.set()
            for _ in range(100):
                if len(websocket.messages) == 4:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(
                [message["eventSequence"] for message in websocket.messages],
                [1, 2, 3, 4],
            )
            self.assertEqual(websocket.messages[-2]["type"], "final")
            self.assertEqual(websocket.messages[-1]["type"], "completion")
            await manager.disconnect("session")

        asyncio.run(scenario())

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_ordered_manager_does_not_coalesce_partials_across_turns(self):
        class WebSocket:
            def __init__(self):
                self.messages = []
                self.first_send = asyncio.Event()
                self.release_first_send = asyncio.Event()

            async def accept(self):
                pass

            async def send_text(self, payload):
                self.messages.append(json.loads(payload))
                if len(self.messages) == 1:
                    self.first_send.set()
                    await self.release_first_send.wait()

        async def scenario():
            manager = production.OrderedConnectionManager()
            websocket = WebSocket()
            manager.bind_loop(asyncio.get_running_loop())
            await manager.connect("session", websocket)

            manager.set_turn("session", "old-turn")
            manager.publish_session("session", {"type": "status", "state": "busy"})
            await websocket.first_send.wait()
            manager.publish_session("session", {"type": "realtime", "text": "old"})
            manager.set_turn("session", "new-turn")
            manager.publish_session("session", {"type": "realtime", "text": "new"})

            with manager._event_lock:
                pending = list(manager._delivery_states["session"].queue)
            self.assertEqual(
                [item.event.get("turnId") for item in pending],
                ["old-turn", "new-turn"],
            )

            websocket.release_first_send.set()
            for _ in range(100):
                if len(websocket.messages) == 3:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(
                [message["eventSequence"] for message in websocket.messages],
                [1, 2, 3],
            )
            self.assertEqual(
                [message.get("turnId") for message in websocket.messages],
                ["old-turn", "old-turn", "new-turn"],
            )
            await manager.disconnect("session")

        asyncio.run(scenario())


try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional server dependency
    TestClient = None


class _NoopScheduler:
    def __init__(self, settings, result_callback, drop_callback=None, error_callback=None):
        self.settings = settings
        self.result_callback = result_callback

    def start(self):
        pass

    def stop(self):
        pass

    def wait_ready(self, timeout=None):
        return True

    def healthy(self):
        return True

    def submit(self, job):
        return production.QueueSubmitResult(True)

    def cancel_session(self, session_id):
        pass

    def snapshot(self):
        return {"workers": {}, "queues": {}}


class _RawScheduler(_NoopScheduler):
    def submit(self, job):
        def complete():
            now = time.monotonic()
            self.result_callback(
                production.InferenceResult(
                    request_id=job.request_id,
                    session_id=job.session_id,
                    kind=job.kind,
                    segment_id=job.segment_id,
                    sequence=job.sequence,
                    generation=job.generation,
                    text="raw fake transcript",
                    error=None,
                    created_at=job.created_at,
                    started_at=now,
                    completed_at=now,
                    queue_delay=0.0,
                    inference_duration=0.001,
                    total_latency=0.001,
                )
            )

        threading.Thread(target=complete, daemon=True).start()
        return production.QueueSubmitResult(True)


class _NoopRecorder:
    def __init__(self, **kwargs):
        self.on_recording_start = kwargs.get("on_recording_start")
        self.on_recording_stop = kwargs.get("on_recording_stop")
        self.on_transcription_start = kwargs.get("on_transcription_start")
        self.on_vad_start = kwargs.get("on_vad_start")
        self.on_vad_stop = kwargs.get("on_vad_stop")
        self.on_vad_detect_start = kwargs.get("on_vad_detect_start")
        self.on_vad_detect_stop = kwargs.get("on_vad_detect_stop")
        self.on_wakeword_detected = kwargs.get("on_wakeword_detected")
        self.on_wakeword_timeout = kwargs.get("on_wakeword_timeout")
        self.on_wakeword_detection_start = kwargs.get("on_wakeword_detection_start")
        self.on_wakeword_detection_end = kwargs.get("on_wakeword_detection_end")
        self.transcription_executor = kwargs.get("transcription_executor")
        self.realtime_transcription_executor = kwargs.get("realtime_transcription_executor")
        self.realtime_callback = kwargs.get("on_realtime_transcription_update")
        self.is_recording = False
        self.has_audio = False
        self._texts = queue.Queue()

    def feed_audio(self, samples, original_sample_rate=16000):
        self.has_audio = True

    def flush_buffered_audio(self):
        self.has_audio = False
        return False

    def abort(self):
        self.has_audio = False

    def text(self):
        return self._texts.get()

    def shutdown(self):
        self._texts.put(None)


@unittest.skipIf(TestClient is None, "FastAPI test client is not installed")
@unittest.skipIf(hasattr(production, "_BACKEND_IMPORT_ERROR"), "server backend dependencies are not installed")
class ProductionServerAppTests(unittest.TestCase):
    def test_post_admission_handshake_failure_releases_session(self):
        class FailingConnectManager(production.OrderedConnectionManager):
            async def connect(self, session_id, websocket):
                await super().connect(session_id, websocket)
                raise RuntimeError("synthetic initial delivery failure")

        class WebSocket:
            headers = {}

            async def accept(self):
                pass

        settings = production.ProductionServerSettings(
            model_warmup=False,
            idle_timeout_seconds=2.0,
            max_sessions=1,
        )
        with mock.patch.object(
            production,
            "OrderedConnectionManager",
            FailingConnectManager,
        ):
            app = production.create_app(
                settings,
                scheduler_factory=_NoopScheduler,
                recorder_factory=_NoopRecorder,
            )

        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/api/v1/ws/transcribe"
        )

        async def scenario():
            async with app.router.lifespan_context(app):
                await endpoint(WebSocket())

        asyncio.run(scenario())
        service = app.state.realtimestt_service
        self.assertEqual(service.session_count(), 0)
        self.assertEqual(service.manager._connections, {})
        self.assertEqual(service.manager._delivery_states, {})

    def test_openai_shaped_websocket_alias_is_supported(self):
        settings = production.ProductionServerSettings(
            model_warmup=False,
            idle_timeout_seconds=2.0,
        )
        app = production.create_app(
            settings,
            scheduler_factory=_NoopScheduler,
            recorder_factory=_NoopRecorder,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/transcriptions/stream") as websocket:
                hello = websocket.receive_json()
                self.assertEqual(hello["type"], "hello")
                self.assertEqual(hello["apiVersion"], "v1")

    def test_raw_pcm_compatibility_endpoint_uses_shared_final_lane(self):
        settings = production.ProductionServerSettings(
            model_warmup=False,
            finalize_timeout_seconds=2.0,
        )
        app = production.create_app(settings, scheduler_factory=_RawScheduler, recorder_factory=_NoopRecorder)
        self.assertEqual(app.openapi()["info"]["version"], "1.0.3")

        with TestClient(app) as client:
            response = client.post(
                "/transcribe-pcm16?sample_rate=16000&encoding=pcm16&language=de",
                content=b"\x00\x00" * 16,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["text"], "raw fake transcript")
            self.assertEqual(response.json()["detected_language"], "de")

            auto_response = client.post(
                "/transcribe-pcm16?sample_rate=16000&encoding=pcm16&language=auto",
                content=b"\x00\x00" * 16,
            )
            self.assertEqual(auto_response.status_code, 200, auto_response.text)
            self.assertIsNone(auto_response.json()["detected_language"])

    def test_versioned_health_capabilities_and_ws_turn_contract(self):
        settings = production.ProductionServerSettings(
            model_warmup=False,
            idle_timeout_seconds=2.0,
            max_sessions=1,
        )
        app = production.create_app(
            settings,
            scheduler_factory=_NoopScheduler,
            recorder_factory=_NoopRecorder,
        )

        with TestClient(app) as client:
            self.assertEqual(client.get("/api/v1/live").status_code, 200)
            ready = client.get("/api/v1/ready")
            self.assertEqual(ready.status_code, 200)
            capabilities = client.get("/api/v1/capabilities").json()
            self.assertEqual(capabilities["protocolVersion"], "realtimestt.remote.v1")
            self.assertEqual(client.get("/health").json()["status"], "ok")

            with client.websocket_connect("/api/v1/ws/transcribe") as websocket:
                hello = websocket.receive_json()
                self.assertEqual(hello["type"], "hello")
                self.assertEqual(hello["apiVersion"], "v1")

                websocket.send_json({"type": "start", "turnId": "turn-1", "language": "auto"})
                started = self._receive_type(websocket, "started")
                self.assertEqual(started["turnId"], "turn-1")
                self.assertEqual(started["language"], "auto")

                websocket.send_json({"type": "finalize"})
                self.assertEqual(self._receive_type(websocket, "finalizing")["turnId"], "turn-1")
                completion = self._receive_type(websocket, "completion")
                self.assertEqual(completion["status"], "completed")
                self.assertEqual(completion["turnId"], "turn-1")

                websocket.send_json({"type": "start", "turnId": "turn-2", "language": "en"})
                self._receive_type(websocket, "started")
                websocket.send_bytes(
                    production.encode_audio_packet(
                        {
                            "sampleRate": 16000,
                            "channels": 1,
                            "format": "pcm_s16le",
                            "frames": 1,
                            "audioSequence": 2,
                        },
                        b"\x00\x00",
                    )
                )
                error = self._receive_type(websocket, "error")
                self.assertEqual(error["error"]["code"], "audio_sequence_out_of_order")

                websocket.send_json({"type": "reset"})
                reset = self._receive_type(websocket, "reset")
                self.assertEqual(reset["previousTurnId"], "turn-2")

    def _receive_type(self, websocket, event_type, limit=30):
        for _ in range(limit):
            message = websocket.receive_json()
            if message.get("type") == event_type:
                return message
        self.fail(f"Did not receive {event_type!r}")


if __name__ == "__main__":
    unittest.main()
