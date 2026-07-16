import unittest
from unittest import mock

from RealtimeSTT.core.runtime import read_stdout_pipe


class FakeShutdownEvent:
    def __init__(self, values):
        self.values = list(values)

    def is_set(self):
        if self.values:
            return self.values.pop(0)
        return True


class RuntimeStdoutReaderTests(unittest.TestCase):
    def test_read_stdout_pipe_logs_received_worker_message(self):
        class Pipe:
            def poll(self, timeout):
                self.timeout = timeout
                return True

            def recv(self):
                return "worker ready"

        pipe = Pipe()
        recorder = type("Recorder", (), {})()
        recorder.shutdown_event = FakeShutdownEvent([False, True])
        recorder.parent_stdout_pipe = pipe

        with mock.patch("RealtimeSTT.core.runtime.time.sleep") as sleep:
            with self.assertLogs("realtimestt", level="DEBUG") as logs:
                read_stdout_pipe(recorder)

        self.assertEqual(pipe.timeout, 0.1)
        sleep.assert_called_once_with(0.1)
        self.assertIn(
            "DEBUG:realtimestt:Receive from stdout pipe",
            logs.output,
        )
        self.assertIn("INFO:realtimestt:worker ready", logs.output)

    def test_read_stdout_pipe_ignores_closed_pipe_errors(self):
        class Pipe:
            def __init__(self):
                self.calls = 0

            def poll(self, timeout):
                self.calls += 1
                raise BrokenPipeError()

        pipe = Pipe()
        recorder = type("Recorder", (), {})()
        recorder.shutdown_event = FakeShutdownEvent([False, True])
        recorder.parent_stdout_pipe = pipe

        with mock.patch("RealtimeSTT.core.runtime.time.sleep") as sleep:
            read_stdout_pipe(recorder)

        self.assertEqual(pipe.calls, 1)
        sleep.assert_called_once_with(0.1)

    def test_read_stdout_pipe_logs_keyboard_interrupt_and_exits(self):
        class Pipe:
            def poll(self, timeout):
                raise KeyboardInterrupt()

        recorder = type("Recorder", (), {})()
        recorder.shutdown_event = FakeShutdownEvent([False])
        recorder.parent_stdout_pipe = Pipe()

        with mock.patch("RealtimeSTT.core.runtime.time.sleep") as sleep:
            with self.assertLogs("realtimestt", level="INFO") as logs:
                read_stdout_pipe(recorder)

        sleep.assert_not_called()
        self.assertEqual(
            logs.output,
            [
                "INFO:realtimestt:"
                "KeyboardInterrupt in read from stdout detected, exiting..."
            ],
        )

    def test_read_stdout_pipe_logs_unexpected_error_and_traceback(self):
        class Pipe:
            def poll(self, timeout):
                raise RuntimeError("boom")

        recorder = type("Recorder", (), {})()
        recorder.shutdown_event = FakeShutdownEvent([False])
        recorder.parent_stdout_pipe = Pipe()

        with mock.patch("RealtimeSTT.core.runtime.time.sleep") as sleep:
            with self.assertLogs("realtimestt", level="ERROR") as logs:
                read_stdout_pipe(recorder)

        sleep.assert_not_called()
        self.assertIn(
            "ERROR:realtimestt:Unexpected error in read from stdout: boom",
            logs.output[0],
        )
        self.assertIn("RuntimeError: boom", logs.output[1])


if __name__ == "__main__":
    unittest.main()
