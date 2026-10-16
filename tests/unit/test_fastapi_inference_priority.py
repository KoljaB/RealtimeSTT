import threading
import time
import unittest

from example_fastapi_server.server import (
    InferenceJob, InferenceScheduler, ServerSettings, parse_args, settings_from_args,
)
from tests.unit.test_fastapi_server_streaming_executor import FakeStreamingEngine


class FastAPIInferencePriorityTests(unittest.TestCase):
    def make_scheduler(self, *, enabled=True, shared_main=False):
        results = {}
        condition = threading.Condition()

        def capture(result):
            with condition:
                results[result.request_id] = result
                condition.notify_all()

        settings = ServerSettings(
            model_warmup=False, defer_live_during_final=enabled,
            use_main_model_for_realtime=shared_main,
            ultrafast_realtime_model_type="test-ultrafast",
        )
        scheduler = InferenceScheduler(settings, capture)
        engines = {}
        for lane in ("main", "realtime", "ultrafast"):
            engine = FakeStreamingEngine()
            engines[lane] = engine
            worker = getattr(scheduler, lane + "_worker")
            if worker is not None:
                worker.engine_factory = lambda engine=engine: engine
        return scheduler, engines, results, condition

    def job(self, request_id, kind, audio):
        return InferenceJob(
            request_id=request_id, session_id=request_id, kind=kind,
            audio=audio, language="en", use_prompt=False, segment_id=1,
            sequence=1, generation=1, created_at=time.monotonic(),
        )

    def test_shared_main_realtime_jobs_are_live_operations(self):
        scheduler, engines, results, condition = self.make_scheduler(shared_main=True)
        snapshots = []
        transcribe = engines["main"].transcribe

        def capture_gate(*args, **kwargs):
            snapshots.append(scheduler.inference_gate.snapshot())
            return transcribe(*args, **kwargs)

        engines["main"].transcribe = capture_gate
        scheduler.start()
        try:
            self.assertTrue(scheduler.wait_ready(2))
            self.assertTrue(scheduler.submit(self.job("live", "realtime", [1])).accepted)
            with condition:
                self.assertTrue(condition.wait_for(lambda: "live" in results, 1))
            self.assertIsNone(results["live"].error)
            self.assertEqual(results["live"].text, "ordinary")
            self.assertFalse(snapshots[0]["finalActive"])
            self.assertEqual(snapshots[0]["activeLiveOperations"], 1)
            self.assertTrue(scheduler.submit(self.job("final", "final", [2])).accepted)
            with condition:
                self.assertTrue(condition.wait_for(lambda: "final" in results, 1))
            self.assertTrue(snapshots[1]["finalActive"])
            self.assertEqual(snapshots[1]["activeLiveOperations"], 0)
        finally:
            scheduler.stop()

    def test_shared_main_streaming_operations_are_live_operations(self):
        scheduler, _, _, _ = self.make_scheduler(shared_main=True)
        scheduler.start()
        try:
            self.assertTrue(scheduler.wait_ready(2))
            proxy = scheduler.streaming_worker("realtime").create_streaming_session()
            snapshots = []
            proxy.session.decode = lambda: snapshots.append(scheduler.inference_gate.snapshot())
            proxy.decode()
            self.assertFalse(snapshots[0]["finalActive"])
            self.assertEqual(snapshots[0]["activeLiveOperations"], 1)
            proxy.close()
        finally:
            scheduler.stop()

    def test_flag_is_opt_in_and_all_workers_share_one_gate(self):
        self.assertFalse(settings_from_args(parse_args([])).defer_live_during_final)
        self.assertTrue(
            settings_from_args(parse_args(["--defer-live-during-final"])).defer_live_during_final
        )
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                scheduler, _, _, _ = self.make_scheduler(enabled=enabled)
                for lane in ("main", "realtime", "ultrafast"):
                    self.assertIs(
                        getattr(scheduler, lane + "_worker").inference_gate,
                        scheduler.inference_gate,
                    )
                if enabled:
                    self.assertIsNotNone(scheduler.inference_gate)
                    self.assertEqual(scheduler.snapshot()["liveInferenceDeferral"]["drops"], 0)
                else:
                    self.assertIsNone(scheduler.inference_gate)
                    self.assertEqual(
                        scheduler.snapshot()["liveInferenceDeferral"], {"enabled": False},
                    )

    def test_final_defers_live_job_and_cancelled_waiter_does_not_reach_engine(self):
        scheduler, engines, results, condition = self.make_scheduler()
        main_entered, release_main, live_claimed = (
            threading.Event() for _ in range(3)
        )
        original_transcribe = engines["main"].transcribe

        def transcribe(*args, **kwargs):
            main_entered.set()
            if not release_main.wait(3):
                raise RuntimeError("test did not release main inference")
            return original_transcribe(*args, **kwargs)

        engines["main"].transcribe = transcribe
        worker = scheduler.realtime_worker
        original_claim = worker._claim_job

        def claim(request_id):
            result = original_claim(request_id)
            live_claimed.set()
            return result

        worker._claim_job = claim
        scheduler.start()
        try:
            self.assertTrue(scheduler.wait_ready(2))
            self.assertTrue(scheduler.healthy())
            self.assertTrue(scheduler.submit(self.job("main", "final", [1])).accepted)
            self.assertTrue(main_entered.wait(1))
            self.assertTrue(scheduler.submit(self.job("cancelled", "realtime", [2])).accepted)
            self.assertTrue(live_claimed.wait(1))
            with condition:
                self.assertFalse(condition.wait_for(lambda: "cancelled" in results, .05))
            self.assertEqual(engines["realtime"].transcribe_calls, [])
            self.assertTrue(worker.cancel_request("cancelled"))
            self.assertTrue(scheduler.submit(self.job("successor", "realtime", [3])).accepted)
            release_main.set()
            with condition:
                self.assertTrue(condition.wait_for(lambda: len(results) == 3, 2))
            self.assertEqual(results["cancelled"].error, "cancelled")
            self.assertEqual(results["cancelled"].metadata, {"cancelled": True})
            self.assertEqual(results["successor"].text, "ordinary")
            self.assertIsNone(results["successor"].error)
            self.assertEqual(engines["realtime"].transcribe_calls, [([3], "en", False)])
            snapshot = scheduler.snapshot()["liveInferenceDeferral"]
            self.assertGreaterEqual(snapshot["liveDelays"], 1)
            self.assertEqual(snapshot["drops"], 0)
            self.assertFalse(snapshot["finalActive"])
        finally:
            release_main.set()
            scheduler.stop()

    def test_scheduler_stop_releases_deferred_job_before_final_finishes(self):
        scheduler, engines, results, condition = self.make_scheduler()
        main_entered, release_main, live_claimed = (threading.Event() for _ in range(3))
        transcribe = engines["main"].transcribe

        def block_final(*args, **kwargs):
            main_entered.set()
            if not release_main.wait(3):
                raise RuntimeError("test did not release final inference")
            return transcribe(*args, **kwargs)

        engines["main"].transcribe = block_final
        original_claim = scheduler.realtime_worker._claim_job

        def claim(request_id):
            result = original_claim(request_id)
            live_claimed.set()
            return result

        scheduler.realtime_worker._claim_job = claim
        scheduler.start()
        stopper = threading.Thread(target=scheduler.stop, daemon=True)
        try:
            self.assertTrue(scheduler.wait_ready(2))
            self.assertTrue(scheduler.submit(self.job("main", "final", [1])).accepted)
            self.assertTrue(main_entered.wait(1))
            self.assertTrue(scheduler.submit(self.job("live", "realtime", [2])).accepted)
            self.assertTrue(live_claimed.wait(1))
            stopper.start()
            scheduler.realtime_worker.thread.join(1)
            self.assertFalse(scheduler.realtime_worker.thread.is_alive())
            self.assertEqual(engines["realtime"].transcribe_calls, [])
            with condition:
                self.assertTrue(condition.wait_for(lambda: "live" in results, 1))
            self.assertIsNotNone(results["live"].error)
        finally:
            release_main.set()
            if stopper.ident is None:
                scheduler.stop()
            else:
                stopper.join(2)
        self.assertFalse(stopper.is_alive())

    def test_stop_releases_deferred_stream_without_decoding(self):
        scheduler, _, _, _ = self.make_scheduler()
        scheduler.start()
        attempted, done = threading.Event(), threading.Event()
        errors = []
        thread = None
        try:
            self.assertTrue(scheduler.wait_ready(2))
            proxy = scheduler.streaming_worker("realtime").create_streaming_session()

            def decode():
                attempted.set()
                try:
                    proxy.decode()
                except RuntimeError as exc:
                    errors.append(str(exc))
                finally:
                    done.set()

            with scheduler.inference_gate.enter("main"):
                thread = threading.Thread(target=decode, daemon=True)
                thread.start()
                self.assertTrue(attempted.wait(1))
                self.assertFalse(done.wait(.03))
                scheduler.stop()
                self.assertTrue(done.wait(1))
                self.assertTrue(errors)
                self.assertNotIn(("decode",), proxy.session.calls)
        finally:
            if thread is not None:
                thread.join(1)
            scheduler.stop()

    def test_streaming_operations_wait_and_preserve_audio_on_both_live_lanes(self):
        scheduler, _, _, _ = self.make_scheduler()
        scheduler.start()
        threads, errors = [], []
        try:
            self.assertTrue(scheduler.wait_ready(2))
            for lane in ("realtime", "ultrafast"):
                with self.subTest(lane=lane):
                    proxy = scheduler.streaming_worker(lane).create_streaming_session()
                    for method in ("accept_audio", "decode", "finish", "input_finished"):
                        with self.subTest(method=method):
                            attempted, finished = threading.Event(), threading.Event()
                            previous_calls = list(proxy.session.calls)

                            def invoke():
                                attempted.set()
                                try:
                                    if method == "accept_audio":
                                        proxy.accept_audio([1, 2, 3], sample_rate=16000)
                                    else:
                                        getattr(proxy, method)()
                                except Exception as exc:
                                    errors.append(exc)
                                finally:
                                    finished.set()

                            with scheduler.inference_gate.enter("main"):
                                thread = threading.Thread(target=invoke)
                                threads.append(thread)
                                thread.start()
                                self.assertTrue(attempted.wait(1))
                                self.assertFalse(finished.wait(.03))
                                self.assertEqual(proxy.session.calls, previous_calls)
                            thread.join(1)
                            self.assertFalse(thread.is_alive())
                            self.assertEqual(errors, [])
                            self.assertEqual(proxy.session.calls[-1][0], method)
                    self.assertEqual(
                        proxy.session.calls[0], ("accept_audio", [1, 2, 3], 16000),
                    )
                    proxy.close()
            self.assertEqual(scheduler.snapshot()["liveInferenceDeferral"]["drops"], 0)
        finally:
            for thread in threads:
                thread.join(1)
            scheduler.stop()


if __name__ == "__main__":
    unittest.main()
