import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from RealtimeSTT.core import shutdown as shutdown_module


class FakeEvent:
    def __init__(self):
        self.set_calls = 0

    def set(self):
        self.set_calls += 1


class FakeWorker:
    def __init__(
            self,
            *,
            alive=False,
            stop_on_join=True,
            stop_on_terminate=True,
            stop_on_kill=True,
    ):
        self.alive = alive
        self.stop_on_join = stop_on_join
        self.stop_on_terminate = stop_on_terminate
        self.stop_on_kill = stop_on_kill
        self.join_timeouts = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self.stop_on_join:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        if self.stop_on_terminate:
            self.alive = False

    def kill(self):
        self.kill_calls += 1
        if self.stop_on_kill:
            self.alive = False


class OrderedWorker(FakeWorker):
    def __init__(self, order, label):
        super().__init__()
        self.order = order
        self.label = label

    def join(self, timeout=None):
        self.order.append(self.label)
        super().join(timeout=timeout)


class OrderedCloseable:
    def __init__(self, order, label):
        self.order = order
        self.label = label
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        self.order.append(self.label)


def make_recorder(
        *,
        use_microphone=False,
        recording_thread=None,
        realtime_thread=None,
        reader_process=None,
        transcript_process=None,
):
    return SimpleNamespace(
        shutdown_lock=threading.Lock(),
        is_shut_down=False,
        continuous_listening=True,
        start_recording_event=FakeEvent(),
        stop_recording_event=FakeEvent(),
        shutdown_event=FakeEvent(),
        is_recording=True,
        is_running=True,
        recording_thread=recording_thread,
        realtime_thread=realtime_thread,
        use_microphone=SimpleNamespace(value=use_microphone),
        reader_process=reader_process,
        transcript_process=transcript_process,
        parent_transcription_pipe=None,
        enable_realtime_transcription=False,
        realtime_transcription_model=None,
    )


class ShutdownTests(unittest.TestCase):
    def test_recording_and_realtime_threads_use_bounded_joins(self):
        recording_thread = FakeWorker()
        realtime_thread = FakeWorker()
        recorder = make_recorder(
            recording_thread=recording_thread,
            realtime_thread=realtime_thread,
        )

        with mock.patch.object(shutdown_module.gc, "collect"):
            shutdown_module.shutdown_recorder(recorder)

        self.assertEqual(
            recording_thread.join_timeouts,
            [shutdown_module._WORKER_JOIN_TIMEOUT],
        )
        self.assertEqual(
            realtime_thread.join_timeouts,
            [shutdown_module._WORKER_JOIN_TIMEOUT],
        )
        self.assertTrue(recorder.is_shut_down)
        self.assertFalse(recorder.is_running)
        self.assertEqual(recorder.shutdown_event.set_calls, 1)

    def test_nonterminating_process_is_forcefully_stopped_and_reaped(self):
        reader_process = FakeWorker(
            alive=True,
            stop_on_join=False,
            stop_on_terminate=True,
        )
        transcript_process = FakeWorker(
            alive=True,
            stop_on_join=False,
            stop_on_terminate=True,
        )
        recorder = make_recorder(
            use_microphone=True,
            recording_thread=FakeWorker(),
            realtime_thread=FakeWorker(),
            reader_process=reader_process,
            transcript_process=transcript_process,
        )

        with mock.patch.object(shutdown_module.gc, "collect"):
            with self.assertLogs("realtimestt", level="WARNING") as logs:
                shutdown_module.shutdown_recorder(recorder)

        for process in (reader_process, transcript_process):
            self.assertEqual(
                process.join_timeouts,
                [
                    shutdown_module._WORKER_JOIN_TIMEOUT,
                    shutdown_module._FORCEFUL_JOIN_TIMEOUT,
                ],
            )
            self.assertEqual(process.terminate_calls, 1)
            self.assertFalse(process.is_alive())

        self.assertIn(
            "WARNING:realtimestt:Reader process did not terminate in time. "
            "Terminating forcefully.",
            logs.output,
        )
        self.assertIn(
            "WARNING:realtimestt:Transcript process did not terminate in time. "
            "Terminating forcefully.",
            logs.output,
        )

    def test_process_kill_is_used_when_terminate_does_not_stop_worker(self):
        transcript_process = FakeWorker(
            alive=True,
            stop_on_join=False,
            stop_on_terminate=False,
            stop_on_kill=True,
        )
        recorder = make_recorder(
            recording_thread=FakeWorker(),
            realtime_thread=FakeWorker(),
            transcript_process=transcript_process,
        )

        with mock.patch.object(shutdown_module.gc, "collect"):
            with self.assertLogs("realtimestt", level="WARNING"):
                shutdown_module.shutdown_recorder(recorder)

        self.assertEqual(
            transcript_process.join_timeouts,
            [
                shutdown_module._WORKER_JOIN_TIMEOUT,
                shutdown_module._FORCEFUL_JOIN_TIMEOUT,
                shutdown_module._FORCEFUL_JOIN_TIMEOUT,
            ],
        )
        self.assertEqual(transcript_process.terminate_calls, 1)
        self.assertEqual(transcript_process.kill_calls, 1)
        self.assertFalse(transcript_process.is_alive())

    def test_stuck_thread_is_reported_while_other_cleanup_continues(self):
        recording_thread = FakeWorker(alive=True, stop_on_join=False)
        realtime_thread = FakeWorker(alive=True, stop_on_join=False)
        transcript_process = FakeWorker(
            alive=True,
            stop_on_join=False,
            stop_on_terminate=True,
        )
        recorder = make_recorder(
            recording_thread=recording_thread,
            realtime_thread=realtime_thread,
            transcript_process=transcript_process,
        )

        with mock.patch.object(shutdown_module.gc, "collect"):
            with self.assertLogs("realtimestt", level="ERROR") as logs:
                shutdown_module.shutdown_recorder(recorder)

        self.assertTrue(recorder.is_shut_down)
        self.assertEqual(transcript_process.terminate_calls, 1)
        self.assertTrue(
            any("Recording thread did not stop within" in message
                for message in logs.output)
        )
        self.assertTrue(
            any("Realtime thread did not stop within" in message
                for message in logs.output)
        )

    def test_models_close_before_realtime_and_preview_threads_join(self):
        order = []
        realtime_model = OrderedCloseable(order, "realtime-close")
        ultrafast_model = OrderedCloseable(order, "ultrafast-close")
        preview_model = OrderedCloseable(order, "preview-close")
        recorder = make_recorder(
            realtime_thread=OrderedWorker(order, "realtime-join"),
        )
        recorder.enable_realtime_transcription = True
        recorder.realtime_transcription_model = realtime_model
        recorder.ultrafast_realtime_transcription_model = ultrafast_model
        recorder.preview_transcription_model = preview_model
        recorder.preview_transcription_thread = OrderedWorker(
            order,
            "preview-join",
        )

        with mock.patch.object(shutdown_module.gc, "collect"):
            shutdown_module.shutdown_recorder(recorder)

        self.assertEqual(
            order,
            [
                "realtime-close",
                "ultrafast-close",
                "preview-close",
                "realtime-join",
                "preview-join",
            ],
        )
        self.assertEqual(realtime_model.close_calls, 1)
        self.assertEqual(ultrafast_model.close_calls, 1)
        self.assertEqual(preview_model.close_calls, 1)
        self.assertIsNone(recorder.realtime_transcription_model)
        self.assertIsNone(recorder.ultrafast_realtime_transcription_model)
        self.assertIsNone(recorder.preview_transcription_model)

    def test_shared_realtime_model_is_closed_only_once(self):
        order = []
        shared_model = OrderedCloseable(order, "shared-close")
        recorder = make_recorder()
        recorder.enable_realtime_transcription = True
        recorder.realtime_transcription_model = shared_model
        recorder.ultrafast_realtime_transcription_model = shared_model

        with mock.patch.object(shutdown_module.gc, "collect"):
            shutdown_module.shutdown_recorder(recorder)

        self.assertEqual(order, ["shared-close"])
        self.assertEqual(shared_model.close_calls, 1)
        self.assertIsNone(recorder.realtime_transcription_model)
        self.assertIsNone(recorder.ultrafast_realtime_transcription_model)

    def test_shutdown_remains_idempotent_after_worker_cleanup(self):
        recording_thread = FakeWorker()
        realtime_thread = FakeWorker()
        recorder = make_recorder(
            recording_thread=recording_thread,
            realtime_thread=realtime_thread,
        )

        with mock.patch.object(shutdown_module.gc, "collect"):
            shutdown_module.shutdown_recorder(recorder)
            shutdown_module.shutdown_recorder(recorder)

        self.assertEqual(
            recording_thread.join_timeouts,
            [shutdown_module._WORKER_JOIN_TIMEOUT],
        )
        self.assertEqual(
            realtime_thread.join_timeouts,
            [shutdown_module._WORKER_JOIN_TIMEOUT],
        )


if __name__ == "__main__":
    unittest.main()
