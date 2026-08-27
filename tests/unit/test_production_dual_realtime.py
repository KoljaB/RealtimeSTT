import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from RealtimeSTT_server import production_server as production


class _Manager:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def set_turn(self, _session_id, _turn_id):
        pass

    def set_audio_sequence(self, _session_id, _sequence):
        pass

    def connection_epoch(self, _session_id):
        return 1

    def publish_session(self, _session_id, event, authoritative=False):
        del authoritative
        with self._lock:
            self.events.append(dict(event))
        delivered = production.concurrent.futures.Future()
        delivered.set_result(True)
        return delivered


class _LiveStream:
    def __init__(self, text):
        self.text = text
        self.accepted = []
        self.closed = 0

    def accept_audio(self, audio, sample_rate=None):
        self.accepted.append((audio.copy(), sample_rate))

    def decode(self):
        pass

    def get_result(self):
        return SimpleNamespace(text=self.text)

    def input_finished(self):
        pass

    def finish(self):
        return SimpleNamespace(text=self.text)

    def close(self):
        self.closed += 1


class _BlockingOrderManager(_Manager):
    def __init__(self):
        super().__init__()
        self.raw_publish_started = threading.Event()
        self.release_raw_publish = threading.Event()

    def publish_session(self, session_id, event, authoritative=False):
        if (
            event.get("type") == "ultrafast"
            and not self.raw_publish_started.is_set()
        ):
            self.raw_publish_started.set()
            if not self.release_raw_publish.wait(timeout=2.0):
                raise AssertionError("raw publication was not released")
        return super().publish_session(session_id, event, authoritative)


class _MutatingLiveStream(_LiveStream):
    def __init__(self, text, mutate):
        super().__init__(text)
        self.mutate = mutate
        self.received = []

    def accept_audio(self, audio, sample_rate=None):
        self.received.append(audio)
        if self.mutate and audio.size:
            audio[0] = -1.0
        super().accept_audio(audio, sample_rate)


class _MutatingDualScheduler:
    def __init__(self):
        self.streams = {}

    def streaming_worker(self, kind):
        scheduler = self

        class Worker:
            def create_streaming_session(self, language=None, use_prompt=True):
                del language, use_prompt
                text = (
                    "we should ship the feature"
                    if kind == "realtime"
                    else "we should ship the feature today"
                )
                stream = _MutatingLiveStream(text, mutate=kind == "realtime")
                scheduler.streams[kind] = stream
                return stream

        return Worker()


class _RecordingSentinelQueue:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []

    def put(self, item, block=True, timeout=None):
        self.calls.append((item, block, timeout))
        if self.delay:
            time.sleep(self.delay)

class _DualScheduler:
    def __init__(self):
        self.streams = {}

    def streaming_worker(self, kind):
        scheduler = self

        class Worker:
            def create_streaming_session(self, language=None, use_prompt=True):
                del language, use_prompt
                text = (
                    "we should ship the feature"
                    if kind == "realtime"
                    else "we should ship the feature today now"
                )
                stream = _LiveStream(text)
                scheduler.streams[kind] = stream
                return stream

        return Worker()


class ProductionDualRealtimeTests(unittest.TestCase):
    def test_live_workers_apply_their_lane_specific_cpu_affinity(self):
        settings = production.ProductionServerSettings(
            realtime_cpu_affinity=(24,),
            ultrafast_realtime_cpu_affinity=(25, 26),
        )
        protocol = production.ProductionSessionProtocol(
            SimpleNamespace(scheduler=SimpleNamespace()),
            _Manager(),
            "session",
            settings,
        )

        for lane, expected_affinity in (
            ("realtime", (24,)),
            ("ultrafast", (25, 26)),
        ):
            with self.subTest(lane=lane):
                live_queue = production.queue.Queue()
                live_queue.put(production._LIVE_CANCEL)
                live_done = threading.Event()
                stream = _LiveStream("")

                with mock.patch.object(
                    production,
                    "set_current_thread_cpu_affinity",
                ) as set_affinity:
                    protocol._live_worker(
                        "turn",
                        1,
                        lane,
                        live_queue,
                        stream,
                        live_done,
                        threading.Event(),
                    )

                set_affinity.assert_called_once_with(
                    f"{lane} live",
                    expected_affinity,
                )
                self.assertTrue(live_done.is_set())
                self.assertEqual(stream.closed, 1)

    def test_unexpected_live_worker_exit_emits_degradation_error(self):
        class FailingStream:
            def accept_audio(self, audio, sample_rate=None):
                del audio, sample_rate
            def decode(self):
                raise RuntimeError("synthetic live decoder failure")
            def close(self):
                return None
        manager = _Manager()
        protocol = production.ProductionSessionProtocol(
            SimpleNamespace(scheduler=SimpleNamespace()),
            manager,
            "session",
            production.ProductionServerSettings(),
        )
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
        )
        live_queue = production.queue.Queue()
        live_queue.put([0.0, 0.0, 0.0, 0.0])
        live_done = threading.Event()
        live_cancelled = threading.Event()
        protocol._live_worker(
            "turn",
            1,
            "realtime",
            live_queue,
            FailingStream(),
            live_done,
            live_cancelled,
        )
        self.assertTrue(live_done.is_set())
        failure = next(event for event in manager.events if event["type"] == "error")
        self.assertEqual(failure["error"]["code"], "live_lane_failed")
        self.assertEqual(failure["lane"], "realtime")
        self.assertTrue(failure["degraded"])
        self.assertEqual(protocol.turn.telemetry["liveLaneStatus"]["realtime"], "failed")
        self.assertEqual(protocol.turn.phase, "receiving")

    def test_capabilities_describe_optional_ultrafast_lane_and_merge_contract(self):
        settings = production.ProductionServerSettings(
            ultrafast_realtime_model_type="ultrafast-80ms",
            ultrafast_realtime_transcription_engine="sherpa_onnx",
            ultrafast_realtime_max_tail_words=5,
        )

        capabilities = production.capabilities_for(settings)

        self.assertTrue(capabilities["mergedRealtime"]["enabled"])
        self.assertEqual(capabilities["mergedRealtime"]["maxTailWords"], 5)
        self.assertEqual(
            capabilities["mergedRealtime"]["ultrafastEventType"],
            "ultrafast",
        )
        self.assertEqual(
            capabilities["mergedRealtime"]["ultrafastTextField"],
            "ultrafastText",
        )
        self.assertEqual(
            capabilities["models"]["live"]["ultrafast"]["model"],
            "ultrafast-80ms",
        )

    def test_ultrafast_tail_limit_rejects_fractional_values(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            production.ProductionServerSettings(
                ultrafast_realtime_max_tail_words=2.5,
            )
    def test_production_cli_maps_optional_ultrafast_lane(self):
        args = production.parse_args(
            [
                "--ultrafast-realtime-model",
                "ultrafast-80ms",
                "--ultrafast-realtime-engine",
                "sherpa_onnx_nemotron",
                "--ultrafast-realtime-engine-options",
                '{"provider":"cpu","num_threads":2}',
                "--ultrafast-realtime-max-tail-words",
                "4",
            ]
        )

        settings = production.settings_from_args(args)

        self.assertEqual(
            settings.ultrafast_realtime_model_type,
            "ultrafast-80ms",
        )
        self.assertEqual(
            settings.ultrafast_realtime_transcription_engine,
            "sherpa_onnx_nemotron",
        )
        self.assertEqual(
            settings.ultrafast_realtime_transcription_engine_options,
            {"provider": "cpu", "num_threads": 2},
        )
        self.assertEqual(settings.ultrafast_realtime_max_tail_words, 4)

    def test_dual_lanes_receive_identical_audio_and_publish_provenance(self):
        import numpy as np

        scheduler = _DualScheduler()
        manager = _Manager()
        service = SimpleNamespace(scheduler=scheduler)
        protocol = production.ProductionSessionProtocol(
            service,
            manager,
            "session",
            production.ProductionServerSettings(
                ultrafast_realtime_model_type="ultrafast-80ms",
            ),
        )

        async def scenario():
            started = await protocol.start(
                {"type": "start", "turnId": "turn", "language": "en"}
            )
            self.assertEqual(started["type"], "started")
            samples = np.arange(320, dtype=np.int16)
            error = await protocol.audio(
                production.encode_audio_packet(
                    {
                        "sampleRate": production.SERVER_SAMPLE_RATE,
                        "channels": 1,
                        "format": production.PCM_FORMAT,
                        "frames": int(samples.size),
                        "audioSequence": 0,
                    },
                    samples.tobytes(),
                )
            )
            self.assertIsNone(error)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if any(
                    event.get("ultrafastSuffix") == "today now"
                    for event in manager.events
                ):
                    break
                await asyncio.sleep(0.005)
            return samples

        samples = asyncio.run(scenario())
        protocol.close()

        self.assertEqual(set(scheduler.streams), {"realtime", "ultrafast"})
        slow = scheduler.streams["realtime"].accepted[0]
        fast = scheduler.streams["ultrafast"].accepted[0]
        self.assertEqual(slow[1], production.SERVER_SAMPLE_RATE)
        self.assertEqual(fast[1], production.SERVER_SAMPLE_RATE)
        np.testing.assert_array_equal(slow[0], fast[0])
        np.testing.assert_array_equal(
            slow[0],
            samples.astype(np.float32) / 32768.0,
        )
        merged = next(
            event
            for event in manager.events
            if event.get("type") == "realtime"
            and event.get("ultrafastSuffix") == "today now"
        )
        self.assertEqual(merged["text"], "we should ship the feature")
        self.assertEqual(merged["accurateText"], "we should ship the feature")
        self.assertEqual(
            merged["mergedText"],
            "we should ship the feature today now",
        )

    def test_raw_publication_cannot_be_overtaken_by_slow_partial(self):
        manager = _BlockingOrderManager()
        protocol = production.ProductionSessionProtocol(
            SimpleNamespace(scheduler=SimpleNamespace()),
            manager,
            "session",
            production.ProductionServerSettings(
                ultrafast_realtime_model_type="ultrafast-80ms",
            ),
        )
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
        )
        errors = []
        slow_finished = threading.Event()

        def observe_fast():
            try:
                protocol._observe_realtime_result(
                    "turn",
                    1,
                    "ultrafast",
                    SimpleNamespace(text="we should ship today"),
                )
            except BaseException as exc:
                errors.append(exc)

        def observe_slow():
            try:
                protocol._observe_realtime_result(
                    "turn",
                    1,
                    "realtime",
                    SimpleNamespace(text="we should ship"),
                )
            except BaseException as exc:
                errors.append(exc)
            finally:
                slow_finished.set()

        fast_thread = threading.Thread(target=observe_fast)
        fast_thread.start()
        self.assertTrue(manager.raw_publish_started.wait(timeout=1.0))

        slow_thread = threading.Thread(target=observe_slow)
        slow_thread.start()
        self.assertFalse(slow_finished.wait(timeout=0.05))

        manager.release_raw_publish.set()
        fast_thread.join(timeout=1.0)
        slow_thread.join(timeout=1.0)

        self.assertFalse(fast_thread.is_alive())
        self.assertFalse(slow_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            [event["type"] for event in manager.events],
            ["ultrafast", "realtime"],
        )

    def test_each_realtime_lane_receives_an_independent_audio_array(self):
        import numpy as np

        scheduler = _MutatingDualScheduler()
        manager = _Manager()
        protocol = production.ProductionSessionProtocol(
            SimpleNamespace(scheduler=scheduler),
            manager,
            "session",
            production.ProductionServerSettings(
                ultrafast_realtime_model_type="ultrafast-80ms",
            ),
        )

        async def scenario():
            started = await protocol.start(
                {"type": "start", "turnId": "turn", "language": "en"}
            )
            self.assertEqual(started["type"], "started")
            samples = np.arange(320, dtype=np.int16)
            error = await protocol.audio(
                production.encode_audio_packet(
                    {
                        "sampleRate": production.SERVER_SAMPLE_RATE,
                        "channels": 1,
                        "format": production.PCM_FORMAT,
                        "frames": int(samples.size),
                        "audioSequence": 0,
                    },
                    samples.tobytes(),
                )
            )
            self.assertIsNone(error)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if all(
                    scheduler.streams.get(kind)
                    and scheduler.streams[kind].received
                    for kind in ("realtime", "ultrafast")
                ):
                    break
                await asyncio.sleep(0.005)
            return samples

        samples = asyncio.run(scenario())
        self.assertTrue(scheduler.streams["realtime"].received)
        self.assertTrue(scheduler.streams["ultrafast"].received)
        slow_audio = scheduler.streams["realtime"].received[0]
        fast_audio = scheduler.streams["ultrafast"].received[0]
        protocol.close()

        self.assertIsNot(slow_audio, fast_audio)
        np.testing.assert_array_equal(
            fast_audio,
            samples.astype(np.float32) / 32768.0,
        )

    def test_finalize_uses_one_shared_deadline_for_both_queue_sentinels(self):
        manager = _Manager()
        settings = production.ProductionServerSettings(
            finalize_timeout_seconds=0.1,
        )
        protocol = production.ProductionSessionProtocol(
            SimpleNamespace(scheduler=SimpleNamespace()),
            manager,
            "session",
            settings,
        )
        first_queue = _RecordingSentinelQueue(delay=0.02)
        second_queue = _RecordingSentinelQueue()
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
            live_queue=first_queue,
            ultrafast_live_queue=second_queue,
        )
        protocol._run_authoritative_final_worker = lambda *args: None

        response = asyncio.run(protocol.finalize())
        self.assertEqual(response["type"], "finalizing")
        thread = next(iter(protocol._completion_threads))
        thread.join(timeout=1.0)

        self.assertEqual(first_queue.calls[0][0], None)
        self.assertEqual(second_queue.calls[0][0], None)
        first_timeout = first_queue.calls[0][2]
        second_timeout = second_queue.calls[0][2]
        self.assertGreater(first_timeout, second_timeout)
        self.assertGreater(first_timeout - second_timeout, 0.005)
    def test_raw_ultrafast_event_precedes_slow_and_suppresses_duplicates(self):
        manager = _Manager()
        protocol = production.ProductionSessionProtocol(
            SimpleNamespace(scheduler=SimpleNamespace()),
            manager,
            "session",
            production.ProductionServerSettings(
                ultrafast_realtime_model_type="ultrafast-80ms",
            ),
        )
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
        )

        protocol._observe_realtime_result(
            "turn",
            1,
            "ultrafast",
            SimpleNamespace(text="we should ship today"),
        )
        protocol._observe_realtime_result(
            "turn",
            1,
            "ultrafast",
            SimpleNamespace(text="we should ship today"),
        )

        raw_events = [
            event for event in manager.events if event.get("type") == "ultrafast"
        ]
        self.assertEqual(len(raw_events), 1)
        raw = raw_events[0]
        self.assertEqual(raw["ultrafastText"], "we should ship today")
        self.assertEqual(raw["text"], "we should ship today")
        self.assertEqual(raw["slowText"], "")
        self.assertEqual(raw["mergedText"], "")
        self.assertEqual(raw["ultrafastSuffix"], "")
        self.assertEqual(raw["mergeStatus"], "waiting_for_slow")

        protocol._observe_realtime_result(
            "turn",
            1,
            "realtime",
            SimpleNamespace(text="we should ship"),
        )

        partial = next(
            event for event in manager.events if event.get("type") == "realtime"
        )
        self.assertEqual(partial["ultrafastText"], "we should ship today")
        self.assertEqual(partial["slowText"], "we should ship")
        self.assertEqual(partial["mergedText"], "we should ship today")


if __name__ == "__main__":
    unittest.main()
