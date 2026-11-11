import gc
import multiprocessing as mp
import weakref
import threading
import time
import unittest

from RealtimeSTT.core.safepipe import ParentPipe


class FalsyError(Exception):
    def __bool__(self):
        return False


class FailingPipe:
    def __init__(self, error, release=None):
        self.error = error
        self.release = release
        self.entered = threading.Event()
        self.closed = False

    def send(self, data):
        self.entered.set()
        if self.release is not None and not self.release.wait(2):
            raise AssertionError("test did not release the pipe")
        raise self.error

    def recv(self):
        raise self.error

    def poll(self, timeout=0):
        raise self.error

    def close(self):
        self.closed = True


class SafePipeTests(unittest.TestCase):
    def start_call(self, callback):
        outcome = []
        def invoke():
            try:
                outcome.append((True, callback()))
            except Exception as error:
                outcome.append((False, error))
        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        return thread, outcome

    def assert_failed(self, call, error_type):
        thread, outcome = call
        thread.join(1)
        self.assertFalse(thread.is_alive(), "pipe caller remained blocked")
        self.assertFalse(outcome[0][0])
        self.assertIsInstance(outcome[0][1], error_type)

    def test_real_pipe_preserves_messages_and_close_behavior(self):
        raw_parent, child = mp.Pipe()
        parent = ParentPipe(raw_parent)
        self.addCleanup(child.close)
        self.addCleanup(parent.close)
        parent.send({"request": 1})
        self.assertEqual(child.recv(), {"request": 1})
        self.assertFalse(parent.poll(0))
        child.send(("success", b"audio"))
        self.assertTrue(parent.poll(1))
        self.assertEqual(parent.recv(), ("success", b"audio"))
        child.send(ValueError("exception object is valid payload"))
        payload = parent.recv()
        self.assertIsInstance(payload, ValueError)
        self.assertEqual(str(payload), "exception object is valid payload")
        parent.close()
        parent.close()
        self.assertIsNone(parent.send("closed"))
        self.assertIsNone(parent.recv())
        self.assertFalse(parent.poll())

    def test_transport_failure_reaches_current_and_future_callers(self):
        for operation in ("send", "recv", "poll"):
            with self.subTest(operation=operation):
                parent = ParentPipe(FailingPipe(BrokenPipeError("peer gone")))
                self.addCleanup(parent.close)
                callback = lambda: parent.send("request") if operation == "send" else getattr(parent, operation)()
                self.assert_failed(self.start_call(callback), BrokenPipeError)
                self.assert_failed(self.start_call(lambda: parent.send("next")), BrokenPipeError)
                self.assert_failed(self.start_call(parent.recv), BrokenPipeError)
                self.assert_failed(self.start_call(parent.poll), BrokenPipeError)

    def test_failure_wakes_requests_already_queued_behind_failed_send(self):
        release = threading.Event()
        pipe = FailingPipe(OSError("transport failed"), release)
        parent = ParentPipe(pipe)
        first = self.start_call(lambda: parent.send("first"))
        self.addCleanup(parent.close)
        self.addCleanup(release.set)
        self.assertTrue(pipe.entered.wait(1))
        second = self.start_call(parent.recv)
        deadline = time.monotonic() + 1
        while parent._request_queue.qsize() == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertGreater(parent._request_queue.qsize(), 0)
        release.set()
        self.assert_failed(first, OSError)
        self.assert_failed(second, OSError)

    def test_failure_type_is_preserved_for_false_valued_exceptions(self):
        parent = ParentPipe(FailingPipe(FalsyError("false-valued transport failure")))
        self.addCleanup(parent.close)
        with self.assertLogs("RealtimeSTT.safepipe", level="ERROR"):
            self.assert_failed(self.start_call(parent.recv), FalsyError)
        parent._worker_thread.join(1)
        later = self.start_call(parent.recv)
        self.assert_failed(later, BrokenPipeError)
        self.assertIn("FalsyError: false-valued transport failure", str(later[1][0][1]))

    def test_each_caller_gets_an_independent_exception(self):
        error = BrokenPipeError("peer gone")
        parent = ParentPipe(FailingPipe(error))
        self.addCleanup(parent.close)
        received = []
        for _ in range(5):
            call = self.start_call(parent.recv)
            self.assert_failed(call, BrokenPipeError)
            received.append(call[1][0][1])
        self.assertEqual(len({id(item) for item in received}), len(received))
        self.assertIs(received[0], error)
        for item in received[1:]:
            self.assertIsNot(item, error)
            self.assertIn("peer gone", str(item))
            self.assertIsNone(item.__cause__)

    def test_unexpected_worker_error_is_raised_instead_of_returned_as_data(self):
        parent = ParentPipe(FailingPipe(ValueError("invalid transport state")))
        self.addCleanup(parent.close)
        with self.assertLogs("RealtimeSTT.safepipe", level="ERROR"):
            self.assert_failed(self.start_call(parent.recv), ValueError)
        self.assert_failed(self.start_call(parent.recv), BrokenPipeError)

    def test_failure_does_not_copy_custom_exceptions(self):
        class UncopyableError(Exception):
            def __copy__(self):
                raise RuntimeError("copy disabled")
        parent = ParentPipe(FailingPipe(UncopyableError("original failure")))
        self.addCleanup(parent.close)
        with self.assertLogs("RealtimeSTT.safepipe", level="ERROR"):
            self.assert_failed(self.start_call(parent.recv), UncopyableError)
        later = self.start_call(parent.recv)
        self.assert_failed(later, BrokenPipeError)
        self.assertIn("UncopyableError: original failure", str(later[1][0][1]))

    def test_unprintable_exception_still_wakes_callers(self):
        class BadStringError(Exception):
            def __str__(self):
                raise RuntimeError("string conversion failed")
        parent = ParentPipe(FailingPipe(BadStringError()))
        self.addCleanup(parent.close)
        with self.assertLogs("RealtimeSTT.safepipe", level="ERROR"):
            self.assert_failed(self.start_call(parent.recv), BadStringError)
        later = self.start_call(parent.recv)
        self.assert_failed(later, BrokenPipeError)
        self.assertIn("BadStringError", str(later[1][0][1]))

    def test_closed_failed_wrapper_does_not_retain_request_payload(self):
        class Payload:
            pass
        class Pipe:
            def send(self, data):
                raise BrokenPipeError("peer gone")
            def close(self):
                pass
        parent = ParentPipe(Pipe())
        payload = Payload()
        reference = weakref.ref(payload)
        with self.assertRaises(BrokenPipeError):
            parent.send(payload)
        parent.close()
        del payload
        gc.collect()
        self.assertIsNone(reference())


if __name__ == "__main__":
    unittest.main()
