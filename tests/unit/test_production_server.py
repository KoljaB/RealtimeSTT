import asyncio
import hashlib
import io
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
        self.assertEqual(production._PACKAGE_VERSION_FALLBACK, "1.1.2")
        capabilities = production.capabilities_for(settings)
        self.assertEqual(capabilities["apiVersion"], "v1")
        self.assertEqual(capabilities["protocolVersion"], "realtimestt.remote.v1")
        self.assertEqual(capabilities["server"]["version"], production.SERVER_VERSION)
        self.assertEqual(capabilities["audio"]["format"], "pcm_s16le")
        self.assertEqual(capabilities["audio"]["channels"], 1)
        self.assertEqual(capabilities["audio"]["sampleRates"], [16_000])
        self.assertIn(48_000, capabilities["audio"]["httpSampleRates"])
        self.assertIn("final", capabilities["models"])
        self.assertIn("live", capabilities["models"])
        self.assertIn("provider", capabilities["models"]["final"])
        self.assertIn("languages", capabilities["models"]["live"])
        self.assertIn("resume", capabilities["operations"]["websocket"])
        self.assertEqual(capabilities["resume"]["ackType"], "resume_ack")
        self.assertEqual(capabilities["resume"]["resumeIdField"], "resumeId")
        self.assertEqual(
            capabilities["resume"]["liveProvenance"]["resumeEpochField"],
            "resumeEpoch",
        )
        self.assertEqual(
            capabilities["operations"]["resume"]["liveProvenance"][
                "audioEndSampleExclusiveField"
            ],
            "audioEndSampleExclusive",
        )
        self.assertEqual(
            capabilities["operations"]["resume"]["correlationField"],
            "resumeId",
        )
        self.assertEqual(
            capabilities["resume"]["preview"]["candidateTextField"],
            "candidateText",
        )
        self.assertEqual(
            capabilities["resume"]["preview"]["inputModeField"],
            "candidateInputScope",
        )
        self.assertEqual(
            capabilities["resume"]["preview"]["fullTurnInputMode"],
            "full_turn",
        )
        self.assertEqual(
            capabilities["resume"]["preview"]["inputCoverageField"],
            "previewInputCoverage",
        )
        self.assertEqual(
            capabilities["resume"]["preview"]["fullTurnInputCoverage"],
            "full_turn",
        )
        self.assertEqual(capabilities["previewInputCoverage"], "full_turn")
        self.assertEqual(capabilities["models"]["preview"]["inputCoverage"], "full_turn")
        self.assertEqual(
            capabilities["preview"],
            {
                "inputCoverage": "full_turn",
                "earlyRms": {
                    "supported": True,
                    "requestMode": "early_rms",
                    "firstAttemptSilenceMs": 25.0,
                    "maxAttempts": 1,
                    "emptyRetry": False,
                },
            },
        )
        self.assertNotIn("tailSeconds", capabilities["models"]["preview"])
        self.assertNotIn("previewTailSeconds", capabilities)
        self.assertNotIn("previewTailSeconds", capabilities["limits"])

    def test_preview_only_late_final_capability_is_explicit_and_opt_in(self):
        disabled = production.capabilities_for(
            production.ProductionServerSettings(preview_only_transcription=True)
        )
        self.assertFalse(disabled["finalAsrEnabled"])
        self.assertFalse(disabled["lateFinalAsrEnabled"])
        self.assertEqual(disabled["operations"]["http"], [])

        enabled = production.capabilities_for(
            production.ProductionServerSettings(
                preview_only_transcription=True,
                allow_late_final_transcription=True,
                late_final_max_audio_seconds=30.0,
            )
        )
        self.assertFalse(enabled["finalAsrEnabled"])
        self.assertTrue(enabled["lateFinalAsrEnabled"])
        self.assertEqual(enabled["models"]["lateFinal"]["maxAudioSeconds"], 30.0)
        self.assertEqual(
            enabled["operations"]["http"],
            [
                "transcribe-pcm16?operation="
                + production.LATE_FINAL_OPERATION
            ],
        )

    def test_late_final_audio_limit_is_hard_capped_at_thirty_seconds(self):
        for invalid in (30.001, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "between 0 and 30 seconds"
            ):
                production.ProductionServerSettings(
                    late_final_max_audio_seconds=invalid
                )

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

    def test_cli_rejects_literal_bearer_tokens(self):
        for flag in ("--bearer-token", "--auth-token"):
            with (
                self.subTest(flag=flag),
                mock.patch("sys.stderr", new=io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                production.parse_args([flag, "must-not-enter-process-args"])

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
        self.assertIsNone(
            production._reported_language_probability(
                SimpleNamespace(info=SimpleNamespace(language_probability=0.0))
            )
        )
        self.assertEqual(
            production._reported_language_probability(
                SimpleNamespace(
                    info=SimpleNamespace(
                        language="de", language_probability=0.93
                    )
                )
            ),
            0.93,
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


class ProductionResumeProtocolTests(unittest.TestCase):
    class _Manager:
        def __init__(self):
            self.events = []
            self.resume_acks = []
            self.published = threading.Event()

        def set_audio_sequence(self, session_id, sequence):
            del session_id, sequence

        def publish_session(self, session_id, event, authoritative=False):
            del session_id, authoritative
            self.events.append(dict(event))
            self.published.set()
            delivered = production.concurrent.futures.Future()
            delivered.set_result(True)
            return delivered

        def queue_resume_ack(self, session_id, ack):
            del session_id
            self.resume_acks.append(dict(ack))
            delivered = production.concurrent.futures.Future()
            delivered.set_result(True)
            return delivered

    class _Service:
        def __init__(self):
            self.calls = []

        def transcribe_turn(self, audio, language, use_prompt):
            self.calls.append((audio.copy(), language, use_prompt))
            return SimpleNamespace(text="That is very cool Can you hear me")

    @staticmethod
    def _packet(sequence, samples):
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

    def _protocol(self, manager, service):
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
            expected_audio_sequence=1,
            packet_count=1,
            audio_frames=4,
            audio_seconds=4 / production.SERVER_SAMPLE_RATE,
            audio_revision=1,
            pcm_buffer=bytearray(b"\x01\x00" * 4),
        )
        protocol._last_partial = "That is very cool"
        return protocol

    def test_resume_ack_records_boundary_and_preview_decodes_full_logical_turn(self):
        manager = self._Manager()
        service = self._Service()
        protocol = self._protocol(manager, service)

        async def scenario():
            ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-1",
                    "requestId": "resume-1",
                    "candidateId": "candidate-7",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                    "byteOffset": 8,
                }
            )
            self.assertEqual(ack["type"], "resume_ack")
            self.assertEqual(ack["resumeId"], "resume-1")
            self.assertEqual(ack["requestId"], "resume-1")
            self.assertEqual(ack["candidateStartSample"], 4)
            self.assertEqual(ack["candidateStartByte"], 8)
            self.assertEqual(ack["bufferedBytes"], 8)

            self.assertIsNone(
                await protocol.audio(self._packet(1, b"\x09\x00\x0a\x00" * 2))
            )
            response = await protocol.preview(
                {
                    "type": "preview",
                    "turnId": "turn",
                    "previewRequestId": "preview-1",
                    "resumeId": "resume-1",
                    "candidateId": "candidate-7",
                }
            )
            self.assertEqual(response["type"], "previewing")
            self.assertTrue(await asyncio.to_thread(manager.published.wait, 1.0))

        asyncio.run(scenario())

        self.assertEqual(len(service.calls), 1)
        self.assertEqual(service.calls[0][0].size, 8)
        event = manager.events[-1]
        self.assertEqual(event["inputScope"], "candidate")
        self.assertEqual(event["candidateInputScope"], "full_turn")
        self.assertEqual(event["previewInputCoverage"], "full_turn")
        self.assertEqual(event["resumeId"], "resume-1")
        self.assertEqual(event["resumeEpoch"], 1)
        self.assertEqual(event["resumeRequestId"], "resume-1")
        self.assertEqual(event["inputSampleRange"], {"start": 0, "end": 8})
        self.assertEqual(event["inputByteRange"], {"start": 0, "end": 16})
        self.assertEqual(event["candidateStartSample"], 4)
        self.assertEqual(event["candidateText"], "")
        self.assertEqual(event["candidateOnlyText"], "")
        self.assertEqual(
            event["candidateCumulativeText"],
            "That is very cool Can you hear me",
        )
        self.assertEqual(
            event["cumulativeText"],
            "That is very cool Can you hear me",
        )
        self.assertEqual(event["text"], event["cumulativeText"])

    def test_live_events_keep_immutable_resume_pcm_provenance(self):
        observed_audio_ends = []

        class Merger:
            def observe_slow(
                self,
                text,
                *,
                recording_id,
                audio_end_sample_exclusive=None,
            ):
                del recording_id
                observed_audio_ends.append(audio_end_sample_exclusive)
                return SimpleNamespace(
                    slow_text=text,
                    ultrafast_text="",
                    text=text,
                    ultrafast_suffix="",
                    status="accurate",
                    matched=True,
                    held=False,
                    used_fuzzy_match=False,
                    anchor_length=0,
                    distance=0,
                    slow_generation=1,
                    slow_sequence=1,
                    ultrafast_sequence=0,
                    slow_audio_end_sample_exclusive=audio_end_sample_exclusive,
                    ultrafast_audio_end_sample_exclusive=None,
                    should_publish=True,
                )

        manager = self._Manager()
        protocol = self._protocol(manager, self._Service())
        protocol._realtime_merger = Merger()

        async def scenario():
            ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-1",
                    "candidateId": "candidate-7",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertEqual(ack["resumeEpoch"], 1)

        asyncio.run(scenario())
        protocol._observe_realtime_result(
            "turn",
            1,
            "realtime",
            SimpleNamespace(text="old delayed hypothesis"),
            audio_end_sample_exclusive=4,
        )
        protocol._observe_realtime_result(
            "turn",
            1,
            "realtime",
            SimpleNamespace(text="candidate hypothesis"),
            audio_end_sample_exclusive=6,
        )

        self.assertEqual(observed_audio_ends, [4, 6])
        self.assertEqual(
            [
                (
                    event["resumeEpoch"],
                    event["candidateId"],
                    event["audioEndSampleExclusive"],
                )
                for event in manager.events
            ],
            [(0, None, 4), (1, "candidate-7", 6)],
        )
        self.assertEqual(manager.events[-1]["resumeId"], "resume-1")
        self.assertEqual(manager.events[-1]["candidateStartSample"], 4)

    def test_live_worker_stamps_its_own_pcm_endpoint(self):
        manager = self._Manager()
        protocol = self._protocol(manager, self._Service())
        observed_audio_ends = []

        def observe(_turn_id, _generation, _lane, _result, *, audio_end_sample_exclusive=None):
            observed_audio_ends.append(audio_end_sample_exclusive)

        protocol._observe_realtime_result = observe

        class Stream:
            def accept_audio(self, _samples, *, sample_rate):
                if sample_rate != production.SERVER_SAMPLE_RATE:
                    raise AssertionError("unexpected live stream sample rate")

            def decode(self):
                return None

            def get_result(self):
                return SimpleNamespace(text="partial")

            def input_finished(self):
                return None

            def finish(self):
                return SimpleNamespace(text="finished")

            def close(self):
                return None

        live_queue = queue.Queue()
        live_queue.put([0.0] * 4)
        live_queue.put([0.0] * 2)
        live_queue.put(None)
        done = threading.Event()
        protocol._live_worker(
            "turn",
            1,
            "realtime",
            live_queue,
            Stream(),
            done,
            threading.Event(),
        )

        self.assertTrue(done.is_set())
        self.assertEqual(observed_audio_ends, [4, 6, 6])

    def test_resume_keeps_one_existing_live_stream_per_lane(self):
        manager = self._Manager()
        service = self._Service()
        protocol = self._protocol(manager, service)
        accurate_stream = object()
        ultrafast_stream = object()
        protocol.turn.live_stream = accurate_stream
        protocol.turn.ultrafast_live_stream = ultrafast_stream

        async def scenario():
            ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-one-stream",
                    "candidateId": "candidate-one-stream",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertEqual(ack["accepted"], True)

        asyncio.run(scenario())
        self.assertIs(protocol.turn.live_stream, accurate_stream)
        self.assertIs(protocol.turn.ultrafast_live_stream, ultrafast_stream)
        self.assertEqual(service.calls, [])
        self.assertEqual(protocol._completion_threads, set())

    def test_resume_prevents_queued_preview_from_asr_admission(self):
        manager = self._Manager()
        service = self._Service()
        protocol = self._protocol(manager, service)
        queued_workers = []

        def hold_worker(*args):
            queued_workers.append(args)

        async def scenario():
            with mock.patch.object(protocol, "_start_preview_worker", side_effect=hold_worker):
                response = await protocol.preview(
                    {
                        "type": "preview",
                        "turnId": "turn",
                        "previewRequestId": "preview-before-resume",
                    }
                )
                self.assertEqual(response["type"], "previewing")
                ack = await protocol.resume(
                    {
                        "type": "resume",
                        "turnId": "turn",
                        "resumeId": "resume-queued",
                        "candidateId": "candidate-queued",
                        "audioSequence": 1,
                        "sampleOffset": 4,
                    }
                )
                self.assertEqual(ack["accepted"], True)

        asyncio.run(scenario())
        self.assertEqual(len(queued_workers), 1)
        protocol._run_preview_worker(*queued_workers[0])
        self.assertEqual(service.calls, [])
        self.assertEqual(manager.events, [])

    def test_resume_does_not_wait_for_an_already_blocked_preview_worker(self):
        class BlockingService(self._Service):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()

            def transcribe_turn(self, audio, language, use_prompt):
                self.calls.append((audio.copy(), language, use_prompt))
                self.entered.set()
                if not self.release.wait(1.0):
                    raise TimeoutError("blocked preview was not released")
                return SimpleNamespace(text="old preview")

        manager = self._Manager()
        service = BlockingService()
        protocol = self._protocol(manager, service)

        async def scenario():
            response = await protocol.preview(
                {
                    "type": "preview",
                    "turnId": "turn",
                    "previewRequestId": "preview-blocked",
                }
            )
            self.assertEqual(response["type"], "previewing")
            self.assertTrue(await asyncio.to_thread(service.entered.wait, 1.0))
            started = time.monotonic()
            ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-after-blocked-preview",
                    "candidateId": "candidate-after-blocked-preview",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(ack["accepted"], True)

        asyncio.run(scenario())
        service.release.set()
        deadline = time.monotonic() + 1.0
        while protocol._completion_threads and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(manager.events, [])
    def test_resume_rejects_wrong_boundary_with_correlated_error(self):
        manager = self._Manager()
        protocol = self._protocol(manager, self._Service())

        async def scenario():
            error = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "requestId": "resume-bad",
                    "sampleOffset": 3,
                }
            )
            self.assertEqual(error["type"], "error")
            self.assertEqual(error["requestId"], "resume-bad")
            self.assertEqual(error["resumeId"], "resume-bad")
            self.assertEqual(error["error"]["code"], "sample_offset_mismatch")

        asyncio.run(scenario())

    def test_resume_rejects_conflicting_correlation_ids_without_mutating_turn(self):
        manager = self._Manager()
        protocol = self._protocol(manager, self._Service())

        async def scenario():
            error = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-canonical",
                    "requestId": "resume-conflict",
                    "candidateId": "candidate-should-not-exist",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertEqual(error["type"], "error")
            self.assertEqual(error["error"]["code"], "resume_correlation_mismatch")
            self.assertEqual(error["resumeId"], "resume-canonical")
            self.assertEqual(error["requestId"], "resume-canonical")
            self.assertEqual(
                error["error"]["details"],
                {
                    "resumeId": "resume-canonical",
                    "requestId": "resume-conflict",
                },
            )

        asyncio.run(scenario())
        self.assertEqual(protocol.turn.resume_count, 0)
        self.assertIsNone(protocol.turn.candidate_id)

    def test_resume_does_not_mutate_when_ack_reservation_fails(self):
        class RejectingManager(self._Manager):
            def queue_resume_ack(self, session_id, ack):
                del session_id
                self.resume_acks.append(dict(ack))
                return None

        manager = RejectingManager()
        protocol = self._protocol(manager, self._Service())

        async def scenario():
            error = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-rejected",
                    "candidateId": "candidate-rejected",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertEqual(error["type"], "error")
            self.assertEqual(error["error"]["code"], "resume_ack_unavailable")
            self.assertEqual(error["resumeId"], "resume-rejected")

        asyncio.run(scenario())
        self.assertEqual(protocol.turn.resume_count, 0)
        self.assertEqual(protocol.turn.resume_epoch, 0)
        self.assertEqual(protocol.turn.resume_provenance, [])
        self.assertIsNone(protocol.turn.candidate_id)
        self.assertEqual(protocol.turn.candidate_base_text, "")
        self.assertIsNone(protocol.turn.last_resume_request_id)

        self.assertIsNone(protocol.turn.last_resume_ack)
        self.assertIsNone(protocol.turn.last_resume_ack_delivery)
    def test_resume_provenance_history_is_bounded(self):
        manager = self._Manager()
        protocol = self._protocol(manager, self._Service())
        total = production._MAX_RESUME_PROVENANCE + 10
        async def scenario():
            for index in range(total):
                request_id = f"resume-{index + 1}"
                ack = await protocol.resume(
                    {
                        "type": "resume",
                        "turnId": "turn",
                        "resumeId": request_id,
                        "candidateId": f"candidate-{index + 1}",
                        "audioSequence": 1,
                        "sampleOffset": 4,
                    }
                )
                self.assertTrue(ack["accepted"])
        asyncio.run(scenario())
        self.assertEqual(len(protocol.turn.resume_provenance), production._MAX_RESUME_PROVENANCE)
        self.assertEqual(protocol.turn.resume_provenance[0][2], "resume-11")
        self.assertEqual(protocol.turn.resume_provenance[-1][2], f"resume-{total}")

    def test_resume_preserves_live_dedupe_anchor(self):
        manager = self._Manager()
        protocol = self._protocol(manager, self._Service())
        protocol._observe_realtime_result(
            "turn",
            1,
            "realtime",
            SimpleNamespace(text="That is very cool"),
        )
        manager.events.clear()

        async def scenario():
            ack = await protocol.resume(
                {"type": "resume", "turnId": "turn", "requestId": "resume-2"}
            )
            self.assertEqual(ack["accepted"], True)

        asyncio.run(scenario())
        protocol._observe_realtime_result(
            "turn",
            1,
            "realtime",
            SimpleNamespace(text="That is very cool"),
        )
        self.assertEqual(manager.events, [])
        protocol._observe_realtime_result(
            "turn",
            1,
            "realtime",
            SimpleNamespace(text="That is very cool again"),
        )
        self.assertEqual([event["type"] for event in manager.events], ["realtime"])

    def test_preview_publication_fences_reused_candidate_by_resume_id(self):
        manager = self._Manager()
        protocol = self._protocol(manager, self._Service())

        async def scenario():
            first = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-old",
                    "requestId": "resume-old",
                    "candidateId": "candidate-reused",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertEqual(first["accepted"], True)
            second = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-new",
                    "requestId": "resume-new",
                    "candidateId": "candidate-reused",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertEqual(second["accepted"], True)

        asyncio.run(scenario())
        stale_payload = {
            "type": "preview",
            "turnId": "turn",
            "candidateId": "candidate-reused",
            "resumeId": "resume-old",
        }
        protocol._publish_preview_result(
            "turn", 1, "preview-old", 1, stale_payload, "exact", None
        )
        self.assertEqual(manager.events, [])

        current_payload = {**stale_payload, "resumeId": "resume-new"}
        protocol._publish_preview_result(
            "turn", 1, "preview-new", 1, current_payload, "exact", None
        )
        self.assertEqual(manager.events, [current_payload])

    def test_final_event_barrier_is_one_shot(self):
        barrier = production.FinalEventBarrier()
        final = {"type": "final", "text": "done"}

        self.assertTrue(barrier.resolve(final))
        self.assertFalse(barrier.resolve({"type": "error"}))
        self.assertTrue(barrier.wait(timeout=0.0))
        self.assertEqual(barrier.outcome, final)

    def test_finalizing_is_admitted_before_authoritative_terminal_pair(self):
        """The command acknowledgement owns the FIFO before a fast Final."""

        async def scenario():
            class Manager:
                def __init__(self):
                    self._loop = asyncio.get_running_loop()
                    self._barriers = {}
                    self._turn_ids = {}
                    self.events = []
                    self.completion = asyncio.Event()
                    self.suppressed = set()

                def set_turn(self, session_id, turn_id):
                    self._turn_ids[session_id] = turn_id

                def set_audio_sequence(self, session_id, sequence):
                    pass

                def register_final_barrier(self, session_id, turn_id):
                    barrier = production.FinalEventBarrier()
                    self._barriers[(session_id, turn_id)] = barrier
                    return barrier

                def unregister_final_barrier(self, session_id, turn_id, barrier):
                    if self._barriers.get((session_id, turn_id)) is barrier:
                        self._barriers.pop((session_id, turn_id), None)

                def suppress_type(self, session_id, message_type, enabled=True):
                    if enabled:
                        self.suppressed.add((session_id, message_type))
                    else:
                        self.suppressed.discard((session_id, message_type))

                def publish_session(self, session_id, message, authoritative=False):
                    completion = production.concurrent.futures.Future()
                    event = dict(message)
                    if not authoritative and event.get("type") in {"final", "error"}:
                        completion.set_result(False)
                        return completion
                    self.events.append(event)
                    if event.get("type") == "final":
                        barrier = self._barriers.get((session_id, event.get("turnId")))
                        if barrier is not None:
                            barrier.resolve(event)
                    if event.get("type") == "completion":
                        self.completion.set()
                    completion.set_result(True)
                    return completion

                async def emit(self, session_id, message):
                    event = dict(message)
                    self.events.append(event)
                    if event.get("type") == "completion":
                        self.completion.set()
                    return True

            class Session:
                def __init__(self):
                    self.settings = SimpleNamespace(language="en")
                    self.recorder = SimpleNamespace(
                        realtime_transcription_executor=None,
                    )
                    self.ingested = []

                def start_streaming(self):
                    pass

                def drain_streaming_audio(self):
                    pass

                def ingest_audio_packet(self, packet):
                    self.ingested.append(packet.audio)
                    return True, None

                def snapshot(self):
                    return {
                        "finalSubmitted": 1,
                        "finalCompleted": 1,
                        "realtimeCompleted": 0,
                    }

            class Service:
                def __init__(self):
                    self.release_final = threading.Event()
                    self.calls = []

                def transcribe_turn(self, audio, language, use_prompt):
                    self.calls.append((audio.copy(), language, use_prompt))
                    self.release_final.wait()
                    return SimpleNamespace(text="authoritative final")

            manager = Manager()
            session = Session()
            service = Service()
            protocol = production.ProductionSessionProtocol(
                service,
                manager,
                "session",
                production.ProductionServerSettings(finalize_timeout_seconds=1.0),
            )
            protocol.attach(session)

            started = await protocol.start({"type": "start", "turnId": "turn-1"})
            self.assertEqual(started["type"], "started")
            packet = production.encode_audio_packet(
                {
                    "sampleRate": 16_000,
                    "channels": 1,
                    "format": production.PCM_FORMAT,
                    "frames": 2,
                    "audioSequence": 0,
                },
                b"\x01\x00\x02\x00",
            )
            self.assertIsNone(await protocol.audio(packet))
            scheduled_callbacks = []

            def capture_call_soon(callback, *args, **kwargs):
                scheduled_callbacks.append((callback, args))
                return mock.Mock()

            with mock.patch.object(
                asyncio.get_running_loop(),
                "call_soon",
                side_effect=capture_call_soon,
            ):
                finalizing = await protocol.finalize()
            self.assertEqual(finalizing["type"], "finalizing")
            self.assertEqual(len(scheduled_callbacks), 1)
            self.assertEqual([event["type"] for event in manager.events], [])
            await manager.emit("session", finalizing)
            service.release_final.set()
            callback, callback_args = scheduled_callbacks[0]
            callback(*callback_args)
            await asyncio.wait_for(manager.completion.wait(), timeout=1.0)
            self.assertEqual(
                [event["type"] for event in manager.events],
                ["finalizing", "final", "completion"],
            )
            self.assertEqual(manager.events[1]["text"], "authoritative final")
            self.assertEqual(len(service.calls), 1)
            self.assertFalse(service.calls[0][2])
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
    def test_ordered_manager_keeps_resume_ack_before_candidate_partial(self):
        """A post-Resume partial must not steal the pre-Ack partial's slot."""

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
            # Hold one status send so a single pending partial completely
            # fills the ordinary budget. Resume must still reserve ACK plus
            # its first candidate partial in their bounded FIFO slots.
            manager = production.OrderedConnectionManager(max_pending_events=1)
            websocket = WebSocket()
            manager.bind_loop(asyncio.get_running_loop())
            await manager.connect("session", websocket)
            manager.set_turn("session", "turn")
            protocol = self._protocol(manager, self._Service())

            status = asyncio.create_task(
                manager.emit("session", {"type": "status", "state": "busy"})
            )
            await websocket.first_send.wait()
            manager.publish_session(
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "old partial",
                    "resumeEpoch": 0,
                    "candidateId": None,
                    "resumeId": None,
                    "audioEndSampleExclusive": 4,
                },
            )
            with manager._event_lock:
                self.assertEqual(len(manager._delivery_states["session"].queue), 1)

            ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-1",
                    "requestId": "resume-1",
                    "candidateId": "candidate-1",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            # resume() must reserve the ordered ACK while it still owns the
            # protocol state. A live worker that wakes immediately afterwards
            # then sees the candidate only behind that barrier.
            ack_delivery = protocol.take_queued_resume_ack_delivery(ack)
            self.assertIsNotNone(ack_delivery)
            for _ in range(100):
                with manager._event_lock:
                    queued_types = [
                        item.event.get("type")
                        for item in manager._delivery_states["session"].queue
                    ]
                if production.RESUME_ACK_TYPE in queued_types:
                    break
                await asyncio.sleep(0)
            self.assertIn(production.RESUME_ACK_TYPE, queued_types)

            candidate_provenance = protocol._live_resume_provenance(
                protocol.turn,
                6,
            )
            candidate_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "candidate partial",
                    **candidate_provenance,
                },
            )
            self.assertIsNotNone(candidate_delivery)
            candidate_update_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "candidate partial updated",
                    **candidate_provenance,
                },
            )
            self.assertIs(candidate_update_delivery, candidate_delivery)
            # A late callback from the old, cumulative stream must not
            # replace the post-ACK candidate slot. A client ignores this
            # stale provenance, so replacing the candidate would make
            # Live ASR appear to die immediately after Resume.
            stale_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "late old partial",
                    "resumeEpoch": 0,
                    "candidateId": None,
                    "resumeId": None,
                    "audioEndSampleExclusive": 8,
                },
            )
            self.assertIsNone(stale_delivery)

            with manager._event_lock:
                pending = list(manager._delivery_states["session"].queue)
            self.assertEqual(
                [item.event.get("type") for item in pending],
                ["partial", production.RESUME_ACK_TYPE, "partial"],
            )
            self.assertLessEqual(len(pending), manager.max_pending_events + 2)
            queued_ack = pending[1].event
            queued_candidate = pending[2].event
            self.assertGreater(
                queued_candidate["eventSequence"],
                queued_ack["eventSequence"],
            )
            self.assertEqual(queued_candidate["resumeEpoch"], ack["resumeEpoch"])
            self.assertEqual(queued_candidate["candidateId"], ack["candidateId"])
            self.assertEqual(queued_candidate["resumeId"], ack["resumeId"])
            self.assertEqual(queued_candidate["audioEndSampleExclusive"], 6)
            self.assertEqual(queued_candidate["text"], "candidate partial updated")

            websocket.release_first_send.set()
            self.assertTrue(await status)
            self.assertTrue(await asyncio.wrap_future(ack_delivery))
            self.assertTrue(await asyncio.wrap_future(candidate_delivery))
            self.assertEqual(
                [message["type"] for message in websocket.messages],
                ["status", "partial", production.RESUME_ACK_TYPE, "partial"],
            )
            ack_index = next(
                index
                for index, message in enumerate(websocket.messages)
                if message["type"] == production.RESUME_ACK_TYPE
            )
            wire_candidate = websocket.messages[ack_index + 1]
            self.assertEqual(wire_candidate["text"], "candidate partial updated")
            self.assertGreater(
                wire_candidate["eventSequence"],
                websocket.messages[ack_index]["eventSequence"],
            )
            self.assertEqual(wire_candidate["resumeEpoch"], ack["resumeEpoch"])
            self.assertEqual(wire_candidate["candidateId"], ack["candidateId"])
            self.assertEqual(wire_candidate["resumeId"], ack["resumeId"])
            with manager._event_lock:
                self.assertEqual(
                    manager._delivery_states["session"].resume_ack_pending,
                    0,
                )
            await manager.disconnect("session")
            protocol.close()

        asyncio.run(scenario())

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_ordered_manager_rejects_stale_partial_before_candidate_after_resume(self):
        """A stale pre-Resume partial cannot consume the candidate reserve."""

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
            manager = production.OrderedConnectionManager(max_pending_events=1)
            websocket = WebSocket()
            manager.bind_loop(asyncio.get_running_loop())
            await manager.connect("session", websocket)
            manager.set_turn("session", "turn")
            protocol = self._protocol(manager, self._Service())

            status = asyncio.create_task(
                manager.emit("session", {"type": "status", "state": "busy"})
            )
            await websocket.first_send.wait()
            old_partial = manager.publish_session(
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "old partial",
                    "resumeEpoch": 0,
                    "candidateId": None,
                    "resumeId": None,
                    "audioEndSampleExclusive": 4,
                },
            )
            self.assertIsNotNone(old_partial)
            ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-1",
                    "requestId": "resume-1",
                    "candidateId": "candidate-1",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            ack_delivery = protocol.take_queued_resume_ack_delivery(ack)
            self.assertIsNotNone(ack_delivery)

            # The stale callback arrives before the first legitimate
            # candidate update. It must not take the one bounded reserve.
            stale_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "late old partial",
                    "resumeEpoch": 0,
                    "candidateId": None,
                    "resumeId": None,
                    "audioEndSampleExclusive": 6,
                },
            )
            self.assertIsNone(stale_delivery)

            candidate_provenance = protocol._live_resume_provenance(
                protocol.turn,
                6,
            )
            candidate_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "candidate partial",
                    **candidate_provenance,
                },
            )
            self.assertIsNotNone(candidate_delivery)
            with manager._event_lock:
                pending = list(manager._delivery_states["session"].queue)
            self.assertEqual(
                [item.event["eventSequence"] for item in pending],
                [2, 3, 4],
            )
            self.assertEqual(
                [item.event["type"] for item in pending],
                ["partial", production.RESUME_ACK_TYPE, "partial"],
            )
            self.assertEqual(pending[-1].event["text"], "candidate partial")
            self.assertEqual(pending[-1].event["resumeId"], "resume-1")

            websocket.release_first_send.set()
            self.assertTrue(await status)
            self.assertTrue(await asyncio.wrap_future(old_partial))
            self.assertTrue(await asyncio.wrap_future(ack_delivery))
            self.assertTrue(await asyncio.wrap_future(candidate_delivery))
            self.assertEqual(
                [message["type"] for message in websocket.messages],
                ["status", "partial", production.RESUME_ACK_TYPE, "partial"],
            )
            self.assertEqual(websocket.messages[-1]["text"], "candidate partial")
            self.assertEqual(websocket.messages[-1]["resumeId"], "resume-1")
            await manager.disconnect("session")
            protocol.close()

        asyncio.run(scenario())

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_ordered_manager_rejects_prior_resume_partial_after_newer_resume(self):
        """A delayed candidate from Resume N cannot replace Resume N+1."""

        class WebSocket:
            def __init__(self):
                self.messages = []
                self.second_ack_started = asyncio.Event()
                self.release_second_ack = asyncio.Event()

            async def accept(self):
                pass

            async def send_text(self, payload):
                message = json.loads(payload)
                self.messages.append(message)
                if (
                    message.get("type") == production.RESUME_ACK_TYPE
                    and message.get("resumeId") == "resume-2"
                ):
                    self.second_ack_started.set()
                    await self.release_second_ack.wait()

        async def scenario():
            manager = production.OrderedConnectionManager(max_pending_events=1)
            websocket = WebSocket()
            manager.bind_loop(asyncio.get_running_loop())
            await manager.connect("session", websocket)
            manager.set_turn("session", "turn")
            protocol = self._protocol(manager, self._Service())

            first_ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-1",
                    "requestId": "resume-1",
                    "candidateId": "candidate-1",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            first_ack_delivery = protocol.take_queued_resume_ack_delivery(first_ack)
            self.assertIsNotNone(first_ack_delivery)
            self.assertTrue(await asyncio.wrap_future(first_ack_delivery))
            first_candidate_provenance = protocol._live_resume_provenance(
                protocol.turn,
                6,
            )
            first_candidate_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "first candidate",
                    **first_candidate_provenance,
                },
            )
            self.assertIsNotNone(first_candidate_delivery)
            self.assertTrue(await asyncio.wrap_future(first_candidate_delivery))

            second_ack = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "resume-2",
                    "requestId": "resume-2",
                    "candidateId": "candidate-2",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            second_ack_delivery = protocol.take_queued_resume_ack_delivery(second_ack)
            self.assertIsNotNone(second_ack_delivery)
            await asyncio.wait_for(websocket.second_ack_started.wait(), timeout=1.0)

            second_candidate_provenance = protocol._live_resume_provenance(
                protocol.turn,
                8,
            )
            second_candidate_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "second candidate",
                    **second_candidate_provenance,
                },
            )
            self.assertIsNotNone(second_candidate_delivery)
            stale_delivery = await asyncio.to_thread(
                manager.publish_session,
                "session",
                {
                    "type": "realtime",
                    "turnId": "turn",
                    "text": "late first candidate",
                    **first_candidate_provenance,
                },
            )
            self.assertIsNone(stale_delivery)
            with manager._event_lock:
                pending = list(manager._delivery_states["session"].queue)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].event["text"], "second candidate")
            self.assertEqual(pending[0].event["resumeEpoch"], second_ack["resumeEpoch"])
            self.assertEqual(pending[0].event["candidateId"], "candidate-2")
            self.assertEqual(pending[0].event["resumeId"], "resume-2")

            websocket.release_second_ack.set()
            self.assertTrue(await asyncio.wrap_future(second_ack_delivery))
            self.assertTrue(await asyncio.wrap_future(second_candidate_delivery))
            second_ack_index = next(
                index
                for index, message in enumerate(websocket.messages)
                if (
                    message.get("type") == production.RESUME_ACK_TYPE
                    and message.get("resumeId") == "resume-2"
                )
            )
            self.assertEqual(
                websocket.messages[second_ack_index + 1]["text"],
                "second candidate",
            )
            self.assertEqual(
                websocket.messages[second_ack_index + 1]["resumeId"],
                "resume-2",
            )
            self.assertNotIn(
                "late first candidate",
                [message.get("text") for message in websocket.messages],
            )
            await manager.disconnect("session")
            protocol.close()

        asyncio.run(scenario())

    @unittest.skipIf(
        hasattr(production, "_BACKEND_IMPORT_ERROR"),
        "server backend dependencies are not installed",
    )
    def test_resume_rejects_without_mutating_when_ack_reserve_is_exhausted(self):
        """A full ordinary queue may admit one ACK, but not a second Resume."""

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
            manager = production.OrderedConnectionManager(max_pending_events=1)
            websocket = WebSocket()
            manager.bind_loop(asyncio.get_running_loop())
            await manager.connect("session", websocket)
            manager.set_turn("session", "turn")
            protocol = self._protocol(manager, self._Service())

            status = asyncio.create_task(
                manager.emit("session", {"type": "status", "state": "busy"})
            )
            await websocket.first_send.wait()
            old_partial = manager.publish_session(
                "session",
                {"type": "realtime", "turnId": "turn", "text": "old partial"},
            )
            existing_ack = manager.queue_resume_ack(
                "session",
                {
                    "type": production.RESUME_ACK_TYPE,
                    "turnId": "turn",
                    "resumeId": "already-reserved",
                    "requestId": "already-reserved",
                    "candidateId": "already-reserved",
                    "resumeEpoch": 1,
                },
            )
            self.assertIsNotNone(old_partial)
            self.assertIsNotNone(existing_ack)

            error = await protocol.resume(
                {
                    "type": "resume",
                    "turnId": "turn",
                    "resumeId": "must-not-mutate",
                    "candidateId": "candidate-must-not-exist",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                }
            )
            self.assertEqual(error["type"], "error")
            self.assertEqual(error["error"]["code"], "resume_ack_unavailable")
            self.assertEqual(protocol.turn.resume_count, 0)
            self.assertEqual(protocol.turn.resume_provenance, [])
            self.assertIsNone(protocol.turn.candidate_id)
            self.assertIsNone(protocol.turn.last_resume_request_id)
            self.assertIsNone(protocol.turn.last_resume_ack)

            with manager._event_lock:
                pending = list(manager._delivery_states["session"].queue)
                self.assertEqual(
                    [item.event["eventSequence"] for item in pending],
                    [2, 3],
                )
                self.assertEqual(
                    [item.event["type"] for item in pending],
                    ["partial", production.RESUME_ACK_TYPE],
                )

            websocket.release_first_send.set()
            self.assertTrue(await status)
            self.assertTrue(await asyncio.wrap_future(old_partial))
            self.assertTrue(await asyncio.wrap_future(existing_ack))
            self.assertEqual(
                [message["eventSequence"] for message in websocket.messages],
                [1, 2, 3],
            )
            with manager._event_lock:
                state = manager._delivery_states["session"]
                self.assertEqual(state.resume_ack_pending, 0)
                self.assertIsNone(state.resume_candidate_reserve_key)
                self.assertFalse(state.resume_candidate_partial_reserved)
            await manager.disconnect("session")
            protocol.close()

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
            rejected = manager.publish_session(
                "session",
                {
                    "type": "realtime",
                    "turnId": "old-turn",
                    "text": "late old",
                },
            )
            self.assertIsNone(rejected)

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
                    info=SimpleNamespace(language="fr", language_probability=0.91),
                )
            )

        threading.Thread(target=complete, daemon=True).start()
        return production.QueueSubmitResult(True)


class _HashScheduler(_NoopScheduler):
    def submit(self, job):
        def complete():
            now = time.monotonic()
            text = hashlib.sha256(job.audio.tobytes()).hexdigest()
            self.result_callback(
                production.InferenceResult(
                    request_id=job.request_id,
                    session_id=job.session_id,
                    kind=job.kind,
                    segment_id=job.segment_id,
                    sequence=job.sequence,
                    generation=job.generation,
                    text=text,
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


class _FakeLiveStream:
    def __init__(self):
        self.accepted = []
        self.decode_calls = 0
        self.finished = 0
        self.closed = 0

    def accept_audio(self, audio, sample_rate=None):
        self.accepted.append((audio.copy(), sample_rate))

    def decode(self):
        self.decode_calls += 1

    def get_result(self):
        return SimpleNamespace(text="same partial")

    def input_finished(self):
        self.finished += 1

    def finish(self):
        return SimpleNamespace(text="same partial")

    def close(self):
        self.closed += 1


class _StreamingHashScheduler(_HashScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streams = []

    def streaming_worker(self, kind):
        self.kind = kind
        scheduler = self

        class Worker:
            def create_streaming_session(self, language=None, use_prompt=True):
                stream = _FakeLiveStream()
                stream.language = language
                stream.use_prompt = use_prompt
                scheduler.streams.append(stream)
                return stream

        return Worker()


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
    def test_empty_silent_and_short_turns_terminate_once(self):
        import numpy as np

        cases = {
            "empty": np.array([], dtype=np.int16),
            "silence-100ms": np.zeros(1_600, dtype=np.int16),
            "voiced-100ms": np.tile(np.array([4_000, -4_000], dtype=np.int16), 800),
            "silence-608ms": np.zeros(9_728, dtype=np.int16),
        }
        for name, samples in cases.items():
            with self.subTest(name=name):
                app = production.create_app(
                    production.ProductionServerSettings(
                        model_warmup=False,
                        finalize_timeout_seconds=2.0,
                    ),
                    scheduler_factory=_HashScheduler,
                    recorder_factory=_NoopRecorder,
                )
                with TestClient(app) as client:
                    with client.websocket_connect("/api/v1/ws/transcribe") as websocket:
                        websocket.receive_json()
                        websocket.send_json({"type": "start", "turnId": name, "language": "en"})
                        self._receive_type(websocket, "started")
                        if samples.size:
                            websocket.send_bytes(
                                production.encode_audio_packet(
                                    {
                                        "sampleRate": 16_000,
                                        "channels": 1,
                                        "format": production.PCM_FORMAT,
                                        "frames": int(samples.size),
                                        "audioSequence": 0,
                                    },
                                    samples.tobytes(),
                                )
                            )
                        websocket.send_json({"type": "finalize"})
                        events = []
                        while True:
                            event = websocket.receive_json()
                            events.append(event)
                            if event.get("type") == "completion":
                                break

                terminals = [
                    event
                    for event in events
                    if event.get("type") == "final"
                    or (
                        event.get("type") == "error"
                        and event.get("where") == "final"
                    )
                ]
                self.assertEqual(len(terminals), 1)
                self.assertEqual(sum(event.get("type") == "completion" for event in events), 1)
                self.assertLess(events.index(terminals[0]), len(events) - 1)
                if name == "empty":
                    self.assertEqual(terminals[0]["status"], "no_speech")

    def test_live_turn_uses_one_stream_and_only_new_frames(self):
        import numpy as np

        settings = production.ProductionServerSettings(
            model_warmup=False,
            finalize_timeout_seconds=2.0,
        )
        app = production.create_app(
            settings,
            scheduler_factory=_StreamingHashScheduler,
            recorder_factory=_NoopRecorder,
        )
        chunks = [np.arange(320, dtype=np.int16) + offset for offset in (0, 500, 1000)]

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/transcribe") as websocket:
                websocket.receive_json()
                websocket.send_json({"type": "start", "turnId": "live-turn", "language": "en"})
                self._receive_type(websocket, "started")
                for sequence, chunk in enumerate(chunks):
                    websocket.send_bytes(
                        production.encode_audio_packet(
                            {
                                "sampleRate": 16_000,
                                "channels": 1,
                                "format": production.PCM_FORMAT,
                                "frames": int(chunk.size),
                                "audioSequence": sequence,
                            },
                            chunk.tobytes(),
                        )
                    )
                websocket.send_json({"type": "finalize"})
                events = []
                while True:
                    event = websocket.receive_json()
                    events.append(event)
                    if event.get("type") == "completion":
                        break

            scheduler = app.state.realtimestt_service.scheduler
            self.assertEqual(len(scheduler.streams), 1)
            stream = scheduler.streams[0]
            self.assertEqual(stream.language, "en")
            self.assertFalse(stream.use_prompt)
            self.assertEqual(stream.decode_calls, len(chunks))
            self.assertEqual(stream.finished, 1)
            self.assertEqual(stream.closed, 1)
            for (actual, sample_rate), expected in zip(stream.accepted, chunks):
                self.assertEqual(sample_rate, 16_000)
                np.testing.assert_array_equal(
                    actual,
                    expected.astype(np.float32) / 32768.0,
                )
            partials = [event for event in events if event.get("type") == "partial"]
            self.assertEqual(len(partials), 1)
            completion = next(event for event in events if event.get("type") == "completion")
            self.assertEqual(completion["finalCount"], 1)
            self.assertEqual(completion["partialCount"], 1)
            self.assertEqual(completion["stageTelemetry"]["decodeCalls"], len(chunks))

    def test_websocket_final_matches_http_and_is_single_for_all_chunk_sizes(self):
        import numpy as np

        settings = production.ProductionServerSettings(
            model_warmup=False,
            finalize_timeout_seconds=2.0,
        )
        source = np.arange(16_000 // 2, dtype=np.int16)

        for chunk_ms in (10, 20, 40, 64, 100):
            with self.subTest(chunk_ms=chunk_ms):
                app = production.create_app(
                    settings,
                    scheduler_factory=_HashScheduler,
                    recorder_factory=_NoopRecorder,
                )
                with TestClient(app) as client:
                    expected = client.post(
                        "/transcribe-pcm16?sample_rate=16000&encoding=pcm16&language=en",
                        content=source.tobytes(),
                    ).json()["text"]
                    with client.websocket_connect("/api/v1/ws/transcribe") as websocket:
                        websocket.receive_json()
                        websocket.send_json(
                            {"type": "start", "turnId": f"turn-{chunk_ms}", "language": "en"}
                        )
                        self._receive_type(websocket, "started")
                        chunk_frames = 16_000 * chunk_ms // 1_000
                        for sequence, start in enumerate(range(0, source.size, chunk_frames)):
                            chunk = source[start : start + chunk_frames]
                            websocket.send_bytes(
                                production.encode_audio_packet(
                                    {
                                        "sampleRate": 16_000,
                                        "channels": 1,
                                        "format": production.PCM_FORMAT,
                                        "frames": int(chunk.size),
                                        "audioSequence": sequence,
                                    },
                                    chunk.tobytes(),
                                )
                            )
                        websocket.send_json({"type": "finalize"})
                        events = []
                        while True:
                            event = websocket.receive_json()
                            events.append(event)
                            if event.get("type") == "completion":
                                break

                finals = [event for event in events if event.get("type") == "final"]
                completions = [event for event in events if event.get("type") == "completion"]
                self.assertEqual(len(finals), 1)
                self.assertEqual(len(completions), 1)
                self.assertEqual(finals[0]["text"], expected)
                self.assertLess(events.index(finals[0]), events.index(completions[0]))

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

    def test_resume_ack_delivery_does_not_block_websocket_receive_loop(self):
        class BlockingAckManager(production.OrderedConnectionManager):
            def __init__(self):
                super().__init__()
                self.ack_send_started = asyncio.Event()
                self.release_ack_send = asyncio.Event()
                self.resume_ack_rejected = asyncio.Event()
                self.second_audio_processed = asyncio.Event()
                self.audio_sequences = []

            def queue_resume_ack(self, session_id, message):
                delivery = super().queue_resume_ack(session_id, message)
                if delivery is None:
                    self.resume_ack_rejected.set()
                return delivery

            def set_audio_sequence(self, session_id, sequence):
                super().set_audio_sequence(session_id, sequence)
                self.audio_sequences.append(sequence)
                if sequence == 1:
                    self.second_audio_processed.set()

            async def send(self, session_id, message):
                if message.get("type") == production.RESUME_ACK_TYPE:
                    self.ack_send_started.set()
                    await self.release_ack_send.wait()
                return await super().send(session_id, message)

        class WebSocket:
            headers = {}

            def __init__(self):
                self.inbound = asyncio.Queue()
                self.messages = []

            async def accept(self):
                pass

            async def receive(self):
                return await self.inbound.get()

            async def send_text(self, payload):
                self.messages.append(json.loads(payload))

        with mock.patch.object(
            production,
            "OrderedConnectionManager",
            BlockingAckManager,
        ):
            app = production.create_app(
                production.ProductionServerSettings(
                    model_warmup=False,
                    idle_timeout_seconds=2.0,
                ),
                scheduler_factory=_NoopScheduler,
                recorder_factory=_NoopRecorder,
            )

        manager = app.state.realtimestt_service.manager
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/api/v1/ws/transcribe"
        )

        def packet(sequence):
            return production.encode_audio_packet(
                {
                    "sampleRate": production.SERVER_SAMPLE_RATE,
                    "channels": 1,
                    "format": production.PCM_FORMAT,
                    "frames": 4,
                    "audioSequence": sequence,
                },
                b"\x01\x00" * 4,
            )

        async def wait_until(predicate, timeout=1.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.001)
            self.fail("timed out waiting for deterministic websocket progress")

        async def scenario():
            websocket = WebSocket()
            async with app.router.lifespan_context(app):
                handler = asyncio.create_task(endpoint(websocket))
                try:
                    await wait_until(
                        lambda: any(
                            message.get("type") == "hello"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "start",
                                    "turnId": "turn",
                                    "language": "en",
                                }
                            ),
                        }
                    )
                    await wait_until(
                        lambda: any(
                            message.get("type") == "started"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put(
                        {"type": "websocket.receive", "bytes": packet(0)}
                    )
                    await wait_until(lambda: 0 in manager.audio_sequences)
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "resume",
                                    "turnId": "turn",
                                    "resumeId": "resume-blocked-send",
                                    "candidateId": "candidate-blocked-send",
                                    "audioSequence": 1,
                                    "sampleOffset": 4,
                                }
                            ),
                        }
                    )
                    await asyncio.wait_for(manager.ack_send_started.wait(), timeout=1.0)
                    # A second Resume cannot reserve another ACK while the
                    # first one owns the transport. Its structured rejection
                    # must queue behind that ACK without stalling receive().
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "resume",
                                    "turnId": "turn",
                                    "resumeId": "resume-while-blocked",
                                    "candidateId": "candidate-while-blocked",
                                    "audioSequence": 1,
                                    "sampleOffset": 4,
                                }
                            ),
                        }
                    )
                    await asyncio.wait_for(manager.resume_ack_rejected.wait(), timeout=1.0)
                    await websocket.inbound.put(
                        {"type": "websocket.receive", "bytes": packet(1)}
                    )
                    # The old handler awaited the ACK's transport Future, so
                    # this exact assertion timed out while send_text was held.
                    await asyncio.wait_for(
                        manager.second_audio_processed.wait(), timeout=0.25
                    )
                    self.assertFalse(
                        any(
                            message.get("type") == production.RESUME_ACK_TYPE
                            for message in websocket.messages
                        )
                    )
                finally:
                    manager.release_ack_send.set()
                    await wait_until(
                        lambda: any(
                            message.get("type") == production.RESUME_ACK_TYPE
                            for message in websocket.messages
                        )
                    )
                    await wait_until(
                        lambda: any(
                            message.get("type") == "error"
                            and message.get("code") == "resume_ack_unavailable"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put({"type": "websocket.disconnect"})
                    await asyncio.wait_for(handler, timeout=1.0)

            self.assertEqual(manager.audio_sequences, [0, 1])
            self.assertEqual(
                sum(
                    message.get("type") == production.RESUME_ACK_TYPE
                    for message in websocket.messages
                ),
                1,
            )
            self.assertEqual(
                [
                    message.get("code")
                    for message in websocket.messages
                    if message.get("type") == "error"
                ],
                ["resume_ack_unavailable"],
            )

        asyncio.run(scenario())

    def test_generic_resume_error_delivery_does_not_block_websocket_receive_loop(self):
        """A correlated Resume validation error must not stall PCM ingress."""

        class BlockingResumeErrorManager(production.OrderedConnectionManager):
            def __init__(self):
                super().__init__()
                self.error_send_started = asyncio.Event()
                self.release_error_send = asyncio.Event()
                self.second_audio_processed = asyncio.Event()
                self.audio_sequences = []

            def set_audio_sequence(self, session_id, sequence):
                super().set_audio_sequence(session_id, sequence)
                self.audio_sequences.append(sequence)
                if sequence == 1:
                    self.second_audio_processed.set()

            async def send(self, session_id, message):
                if (
                    message.get("type") == "error"
                    and message.get("code") == "resume_correlation_mismatch"
                ):
                    self.error_send_started.set()
                    await self.release_error_send.wait()
                return await super().send(session_id, message)

        class WebSocket:
            headers = {}

            def __init__(self):
                self.inbound = asyncio.Queue()
                self.messages = []

            async def accept(self):
                pass

            async def receive(self):
                return await self.inbound.get()

            async def send_text(self, payload):
                self.messages.append(json.loads(payload))

        with mock.patch.object(
            production,
            "OrderedConnectionManager",
            BlockingResumeErrorManager,
        ):
            app = production.create_app(
                production.ProductionServerSettings(
                    model_warmup=False,
                    idle_timeout_seconds=2.0,
                ),
                scheduler_factory=_NoopScheduler,
                recorder_factory=_NoopRecorder,
            )

        manager = app.state.realtimestt_service.manager
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/api/v1/ws/transcribe"
        )

        def packet(sequence):
            return production.encode_audio_packet(
                {
                    "sampleRate": production.SERVER_SAMPLE_RATE,
                    "channels": 1,
                    "format": production.PCM_FORMAT,
                    "frames": 4,
                    "audioSequence": sequence,
                },
                b"\x01\x00" * 4,
            )

        async def wait_until(predicate, timeout=1.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.001)
            self.fail("timed out waiting for deterministic websocket progress")

        async def scenario():
            websocket = WebSocket()
            async with app.router.lifespan_context(app):
                handler = asyncio.create_task(endpoint(websocket))
                try:
                    await wait_until(
                        lambda: any(
                            message.get("type") == "hello"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "start",
                                    "turnId": "turn",
                                    "language": "en",
                                }
                            ),
                        }
                    )
                    await wait_until(
                        lambda: any(
                            message.get("type") == "started"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put(
                        {"type": "websocket.receive", "bytes": packet(0)}
                    )
                    await wait_until(lambda: 0 in manager.audio_sequences)
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "resume",
                                    "turnId": "turn",
                                    "resumeId": "resume-first",
                                    "requestId": "resume-second",
                                    "candidateId": "candidate",
                                    "audioSequence": 1,
                                    "sampleOffset": 4,
                                }
                            ),
                        }
                    )
                    await asyncio.wait_for(manager.error_send_started.wait(), timeout=1.0)
                    await websocket.inbound.put(
                        {"type": "websocket.receive", "bytes": packet(1)}
                    )
                    # Before this fix handler-level manager.emit() waited for
                    # the blocked error transport and this timed out.
                    await asyncio.wait_for(
                        manager.second_audio_processed.wait(), timeout=0.25
                    )
                    self.assertFalse(
                        any(
                            message.get("type") == "error"
                            and message.get("code") == "resume_correlation_mismatch"
                            for message in websocket.messages
                        )
                    )
                finally:
                    manager.release_error_send.set()
                    await wait_until(
                        lambda: any(
                            message.get("type") == "error"
                            and message.get("code") == "resume_correlation_mismatch"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put({"type": "websocket.disconnect"})
                    await asyncio.wait_for(handler, timeout=1.0)

            self.assertEqual(manager.audio_sequences, [0, 1])
            errors = [
                message
                for message in websocket.messages
                if message.get("type") == "error"
            ]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["code"], "resume_correlation_mismatch")
            self.assertEqual(errors[0]["resumeId"], "resume-first")
            self.assertEqual(errors[0]["requestId"], "resume-first")

        asyncio.run(scenario())

    def test_ping_queued_after_blocked_resume_error_keeps_receive_loop_live(self):
        """An advisory pong cannot reintroduce Resume-error receive stalls."""

        class BlockingResumeErrorManager(production.OrderedConnectionManager):
            def __init__(self):
                super().__init__()
                self.error_send_started = asyncio.Event()
                self.release_error_send = asyncio.Event()
                self.pong_enqueued = asyncio.Event()
                self.pong_enqueue_count = 0
                self.second_audio_processed = asyncio.Event()
                self.audio_sequences = []

            def _enqueue(
                self,
                session_id,
                message,
                *,
                respect_suppression=False,
            ):
                delivery = super()._enqueue(
                    session_id,
                    message,
                    respect_suppression=respect_suppression,
                )
                if message.get("type") == "pong" and delivery is not None:
                    self.pong_enqueue_count += 1
                    self.pong_enqueued.set()
                return delivery

            def set_audio_sequence(self, session_id, sequence):
                super().set_audio_sequence(session_id, sequence)
                self.audio_sequences.append(sequence)
                if sequence == 1:
                    self.second_audio_processed.set()

            async def send(self, session_id, message):
                if (
                    message.get("type") == "error"
                    and message.get("code") == "resume_correlation_mismatch"
                ):
                    self.error_send_started.set()
                    await self.release_error_send.wait()
                return await super().send(session_id, message)

        class WebSocket:
            headers = {}

            def __init__(self):
                self.inbound = asyncio.Queue()
                self.messages = []
                self.disconnect_received = asyncio.Event()

            async def accept(self):
                pass

            async def receive(self):
                message = await self.inbound.get()
                if message.get("type") == "websocket.disconnect":
                    self.disconnect_received.set()
                return message

            async def send_text(self, payload):
                self.messages.append(json.loads(payload))

        with mock.patch.object(
            production,
            "OrderedConnectionManager",
            BlockingResumeErrorManager,
        ):
            app = production.create_app(
                production.ProductionServerSettings(
                    model_warmup=False,
                    idle_timeout_seconds=2.0,
                ),
                scheduler_factory=_NoopScheduler,
                recorder_factory=_NoopRecorder,
            )

        manager = app.state.realtimestt_service.manager
        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/api/v1/ws/transcribe"
        )

        def packet(sequence):
            return production.encode_audio_packet(
                {
                    "sampleRate": production.SERVER_SAMPLE_RATE,
                    "channels": 1,
                    "format": production.PCM_FORMAT,
                    "frames": 4,
                    "audioSequence": sequence,
                },
                b"\x01\x00" * 4,
            )

        async def wait_until(predicate, timeout=1.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.001)
            self.fail("timed out waiting for deterministic websocket progress")

        async def scenario():
            websocket = WebSocket()
            async with app.router.lifespan_context(app):
                handler = asyncio.create_task(endpoint(websocket))
                try:
                    await wait_until(
                        lambda: any(
                            message.get("type") == "hello"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "start",
                                    "turnId": "turn",
                                    "language": "en",
                                }
                            ),
                        }
                    )
                    await wait_until(
                        lambda: any(
                            message.get("type") == "started"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put(
                        {"type": "websocket.receive", "bytes": packet(0)}
                    )
                    await wait_until(lambda: 0 in manager.audio_sequences)
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps(
                                {
                                    "type": "resume",
                                    "turnId": "turn",
                                    "resumeId": "resume-first",
                                    "requestId": "resume-second",
                                    "candidateId": "candidate",
                                    "audioSequence": 1,
                                    "sampleOffset": 4,
                                }
                            ),
                        }
                    )
                    await asyncio.wait_for(manager.error_send_started.wait(), timeout=1.0)
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps({"type": "ping"}),
                        }
                    )
                    await asyncio.wait_for(manager.pong_enqueued.wait(), timeout=1.0)
                    with manager._event_lock:
                        pending_types = [
                            item.event.get("type")
                            for item in manager._delivery_states[
                                next(iter(manager._delivery_states))
                            ].queue
                        ]
                    self.assertEqual(pending_types, ["pong"])
                    await websocket.inbound.put(
                        {"type": "websocket.receive", "bytes": packet(1)}
                    )
                    # Before this fix the handler awaited manager.emit(pong),
                    # so neither this PCM packet nor the peer disconnect was
                    # received until the blocked error transport released.
                    await asyncio.wait_for(
                        manager.second_audio_processed.wait(), timeout=0.25
                    )
                    await websocket.inbound.put({"type": "websocket.disconnect"})
                    await asyncio.wait_for(
                        websocket.disconnect_received.wait(), timeout=0.25
                    )
                    await asyncio.wait_for(handler, timeout=1.0)
                finally:
                    manager.release_error_send.set()
                    if not handler.done():
                        await websocket.inbound.put({"type": "websocket.disconnect"})
                        await asyncio.wait_for(handler, timeout=1.0)

            self.assertEqual(manager.audio_sequences, [0, 1])
            self.assertEqual(manager.pong_enqueue_count, 1)
            self.assertTrue(websocket.disconnect_received.is_set())
            self.assertEqual(
                [
                    message.get("type")
                    for message in websocket.messages
                    if message.get("type") == "pong"
                ],
                [],
            )

        asyncio.run(scenario())

    def test_resume_ack_rejection_fails_closed_when_error_cannot_queue(self):
        """A saturated ACK lane must not leave a Resume caller silently idle."""

        class RejectingErrorManager(production.OrderedConnectionManager):
            def publish_session(self, session_id, message, *, authoritative=False):
                if (
                    message.get("type") == "error"
                    and message.get("code") == "resume_ack_unavailable"
                ):
                    return None
                return super().publish_session(
                    session_id,
                    message,
                    authoritative=authoritative,
                )

        class WebSocket:
            headers = {}

            def __init__(self):
                self.inbound = asyncio.Queue()
                self.messages = []
                self.closed = []

            async def accept(self):
                pass

            async def receive(self):
                return await self.inbound.get()

            async def send_text(self, payload):
                self.messages.append(json.loads(payload))

            async def close(self, code=None, reason=None):
                self.closed.append((code, reason))

        with mock.patch.object(
            production,
            "OrderedConnectionManager",
            RejectingErrorManager,
        ):
            app = production.create_app(
                production.ProductionServerSettings(
                    model_warmup=False,
                    idle_timeout_seconds=2.0,
                ),
                scheduler_factory=_NoopScheduler,
                recorder_factory=_NoopRecorder,
            )

        endpoint = next(
            route.endpoint
            for route in app.routes
            if getattr(route, "path", None) == "/api/v1/ws/transcribe"
        )

        async def reject_resume(_protocol, _payload):
            return {
                "type": "error",
                "code": "resume_ack_unavailable",
                "message": "synthetic queue exhaustion",
            }

        async def wait_until(predicate, timeout=1.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.001)
            self.fail("timed out waiting for deterministic websocket progress")

        async def scenario():
            websocket = WebSocket()
            async with app.router.lifespan_context(app):
                with mock.patch.object(
                    production.ProductionSessionProtocol,
                    "resume",
                    new=reject_resume,
                ):
                    handler = asyncio.create_task(endpoint(websocket))
                    await wait_until(
                        lambda: any(
                            message.get("type") == "hello"
                            for message in websocket.messages
                        )
                    )
                    await websocket.inbound.put(
                        {
                            "type": "websocket.receive",
                            "text": json.dumps({"type": "resume"}),
                        }
                    )
                    await asyncio.wait_for(handler, timeout=1.0)

            self.assertEqual(websocket.closed, [(1013, "outbound backpressure")])
            self.assertFalse(
                any(message.get("type") == "error" for message in websocket.messages)
            )

        asyncio.run(scenario())

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
        self.assertEqual(app.openapi()["info"]["version"], production.SERVER_VERSION)

        with TestClient(app) as client:
            response = client.post(
                "/transcribe-pcm16?sample_rate=16000&encoding=pcm16&language=de",
                content=b"\x00\x00" * 16,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["text"], "raw fake transcript")
            self.assertEqual(response.json()["detected_language"], "de")
            self.assertEqual(response.json()["language_probability"], 0.91)

            auto_response = client.post(
                "/transcribe-pcm16?sample_rate=16000&encoding=pcm16&language=auto",
                content=b"\x00\x00" * 16,
            )
            self.assertEqual(auto_response.status_code, 200, auto_response.text)
            self.assertEqual(auto_response.json()["detected_language"], "fr")
            self.assertEqual(auto_response.json()["language_probability"], 0.91)

    def test_preview_only_http_allows_only_opted_in_bounded_late_final(self):
        disabled_app = production.create_app(
            production.ProductionServerSettings(
                model_warmup=False,
                preview_only_transcription=True,
                finalize_timeout_seconds=2.0,
            ),
            scheduler_factory=_RawScheduler,
            recorder_factory=_NoopRecorder,
        )
        late_query = (
            "/transcribe-pcm16?sample_rate=16000&encoding=pcm16&language=en"
            f"&operation={production.LATE_FINAL_OPERATION}"
        )
        with TestClient(disabled_app) as client:
            disabled = client.post(late_query, content=b"\x00\x00" * 16)
            self.assertEqual(disabled.status_code, 409, disabled.text)
            self.assertEqual(
                disabled.json()["error"]["code"],
                "late_final_asr_disabled",
            )

        enabled_app = production.create_app(
            production.ProductionServerSettings(
                model_warmup=False,
                preview_only_transcription=True,
                allow_late_final_transcription=True,
                late_final_max_audio_seconds=0.001,
                finalize_timeout_seconds=2.0,
            ),
            scheduler_factory=_RawScheduler,
            recorder_factory=_NoopRecorder,
        )
        with TestClient(enabled_app) as client:
            ordinary = client.post(
                "/transcribe-pcm16?sample_rate=16000&encoding=pcm16&language=en",
                content=b"\x00\x00" * 16,
            )
            self.assertEqual(ordinary.status_code, 409, ordinary.text)
            self.assertEqual(ordinary.json()["error"]["code"], "final_asr_disabled")

            late = client.post(late_query, content=b"\x00\x00" * 16)
            self.assertEqual(late.status_code, 200, late.text)
            self.assertEqual(late.json()["text"], "raw fake transcript")

            too_long = client.post(late_query, content=b"\x00\x00" * 17)
            self.assertEqual(too_long.status_code, 413, too_long.text)
            self.assertEqual(too_long.json()["error"]["code"], "audio_size_limit")

    def test_versioned_health_capabilities_and_ws_turn_contract(self):
        settings = production.ProductionServerSettings(
            model_warmup=False,
            max_sessions=1,
            idle_timeout_seconds=2.0,
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

    def test_websocket_resume_preview_contract_and_legacy_resume(self):
        settings = production.ProductionServerSettings(
            model_warmup=False,
            idle_timeout_seconds=2.0,
        )
        app = production.create_app(
            settings,
            scheduler_factory=_RawScheduler,
            recorder_factory=_NoopRecorder,
        )

        def send_audio(websocket, sequence, samples):
            websocket.send_bytes(
                production.encode_audio_packet(
                    {
                        "sampleRate": 16_000,
                        "channels": 1,
                        "format": production.PCM_FORMAT,
                        "frames": len(samples) // 2,
                        "audioSequence": sequence,
                    },
                    samples,
                )
            )

        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/transcribe") as websocket:
                hello = websocket.receive_json()
                self.assertEqual(hello["type"], "hello")
                self.assertEqual(
                    hello["capabilities"]["operations"]["resume"]["correlationField"],
                    "resumeId",
                )
                websocket.send_json(
                    {"type": "start", "turnId": "candidate-turn", "language": "en"}
                )
                self._receive_type(websocket, "started")
                send_audio(websocket, 0, b"\x01\x00" * 4)

                first_resume = {
                    "type": "resume",
                    "turnId": "candidate-turn",
                    "resumeId": "resume-old",
                    "requestId": "resume-old",
                    "candidateId": "candidate-reused",
                    "audioSequence": 1,
                    "sampleOffset": 4,
                    "byteOffset": 8,
                }
                websocket.send_json(first_resume)
                first_ack = self._receive_type(websocket, "resume_ack")
                self.assertEqual(first_ack["resumeId"], "resume-old")
                self.assertEqual(first_ack["requestId"], "resume-old")

                send_audio(websocket, 1, b"\x02\x00" * 4)
                websocket.send_json(
                    {
                        **first_resume,
                        "resumeId": "resume-new",
                        "requestId": "resume-new",
                        "audioSequence": 2,
                        "sampleOffset": 8,
                        "byteOffset": 16,
                    }
                )
                second_ack = self._receive_type(websocket, "resume_ack")
                self.assertEqual(second_ack["resumeId"], "resume-new")
                self.assertEqual(second_ack["candidateId"], "candidate-reused")

                send_audio(websocket, 2, b"\x03\x00" * 4)
                websocket.send_json(
                    {
                        "type": "preview",
                        "turnId": "candidate-turn",
                        "previewRequestId": "preview-new",
                        "resumeId": "resume-new",
                        "candidateId": "candidate-reused",
                    }
                )
                preview = self._receive_type(websocket, "preview")
                self.assertEqual(preview["candidateId"], "candidate-reused")
                self.assertEqual(preview["resumeId"], "resume-new")
                self.assertEqual(preview["candidateText"], "")
                self.assertEqual(preview["candidateOnlyText"], "")
                self.assertEqual(preview["resumeEpoch"], 2)
                self.assertEqual(preview["candidateInputScope"], "full_turn")
                self.assertEqual(preview["previewInputCoverage"], "full_turn")
                self.assertEqual(preview["candidateCumulativeText"], "raw fake transcript")
                self.assertEqual(preview["inputScope"], "candidate")
                self.assertEqual(preview["inputSampleRange"], {"start": 0, "end": 12})

                # Old clients still send only their logical turn identifier.
                websocket.send_json({"type": "resume", "turnId": "candidate-turn"})
                legacy_ack = self._receive_type(websocket, "resume_ack")
                self.assertTrue(legacy_ack["accepted"])
                self.assertEqual(legacy_ack["resumeId"], legacy_ack["requestId"])

    def _receive_type(self, websocket, event_type, limit=30):
        for _ in range(limit):
            message = websocket.receive_json()
            if message.get("type") == event_type:
                return message
        self.fail(f"Did not receive {event_type!r}")


if __name__ == "__main__":
    unittest.main()
