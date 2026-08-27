import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from RealtimeSTT.core import transcription as transcription_module
from RealtimeSTT.core.transcription import TranscriptionWorker


class FakePipe:
    def __init__(self):
        self.closed = False
        self.sent = []

    def poll(self, timeout=0):
        time.sleep(min(float(timeout or 0), 0.01))
        return False

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


class BlockingEngine:
    def __init__(self, warmup_error=None):
        self.warmup_error = warmup_error
        self.entered = threading.Event()
        self.closed = threading.Event()
        self.close_calls = 0
        self._close_lock = threading.Lock()

    def warmup(self, audio):
        if self.warmup_error is not None:
            raise self.warmup_error

    def transcribe(self, audio, **options):
        self.entered.set()
        if not self.closed.wait(2.0):
            raise RuntimeError("test inference was not cancelled")
        raise RuntimeError("aborted")

    def close(self):
        with self._close_lock:
            self.close_calls += 1
            self.closed.set()


class TranscriptionWorkerShutdownTests(unittest.TestCase):
    def make_worker(self, shutdown_event=None):
        return TranscriptionWorker(
            FakePipe(),
            FakePipe(),
            "transcribe_cpp",
            {},
            "model.gguf",
            None,
            "default",
            0,
            "cuda",
            threading.Event(),
            shutdown_event or threading.Event(),
            threading.Event(),
            1,
            None,
            None,
            1,
            False,
            False,
        )

    def test_shutdown_event_cancels_inflight_engine_and_closes_once(self):
        shutdown_event = threading.Event()
        worker = self.make_worker(shutdown_event)
        worker.queue.put((np.zeros(160, dtype=np.float32), "en", False))
        engine = BlockingEngine()

        with patch.object(
            transcription_module,
            "create_transcription_engine",
            return_value=engine,
        ), patch.object(
            transcription_module.sf,
            "read",
            return_value=(np.zeros(160, dtype=np.float32), 16000),
        ), patch.object(
            transcription_module.logging,
            "error",
        ) as log_error:
            thread = threading.Thread(target=worker.run)
            thread.start()
            self.assertTrue(worker.ready_event.wait(0.5))
            self.assertTrue(engine.entered.wait(0.5))

            shutdown_event.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(engine.close_calls, 1)
        self.assertTrue(worker.conn.closed)
        self.assertTrue(worker.stdout_pipe.closed)
        self.assertEqual(worker.conn.sent, [])
        log_error.assert_not_called()

    def test_warmup_failure_still_closes_initialized_engine(self):
        worker = self.make_worker()
        engine = BlockingEngine(warmup_error=RuntimeError("warmup failed"))

        with patch.object(
            transcription_module,
            "create_transcription_engine",
            return_value=engine,
        ), patch.object(
            transcription_module.sf,
            "read",
            return_value=(np.zeros(160, dtype=np.float32), 16000),
        ):
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                worker.run()

        self.assertEqual(engine.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
