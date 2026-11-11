"""Delay live decoder operations while a CPU final request owns inference.

Audio queues are untouched. Both live lanes may run together when no final
request is pending. Main inference does not wait for an already running live
operation: independent engines may finish concurrently. Only new live work
is deferred, avoiding head-of-line blocking by a long live decode.
"""
from contextlib import contextmanager
import threading
import time


class InferencePriorityGate:
    def __init__(self):
        self._condition = threading.Condition()
        self._live = 0
        self._final = False
        self._waiting_final = 0
        self.live_delays = 0
        self.live_delay_seconds = 0.0

    @contextmanager
    def enter(self, lane, stop_event=None):
        def stopping():
            return stop_event is not None and stop_event.is_set()

        final = lane == 'main'
        started = time.monotonic()
        with self._condition:
            if final:
                self._waiting_final += 1
                try:
                    self._condition.wait_for(lambda: stopping() or not self._final)
                    if stopping():
                        raise RuntimeError("inference worker is stopping")
                    self._final = True
                finally:
                    self._waiting_final -= 1
                    self._condition.notify_all()
            else:
                delayed = self._final or self._waiting_final > 0
                self._condition.wait_for(
                    lambda: stopping() or (not self._final and self._waiting_final == 0)
                )
                if stopping():
                    raise RuntimeError("inference worker is stopping")
                self._live += 1
                if delayed:
                    self.live_delays += 1
                    self.live_delay_seconds += time.monotonic() - started
        try:
            yield
        finally:
            with self._condition:
                if final:
                    self._final = False
                else:
                    self._live -= 1
                self._condition.notify_all()

    def wake_waiters(self):
        with self._condition:
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {'finalActive': self._final, 'waitingFinal': self._waiting_final,
                    'activeLiveOperations': self._live, 'liveDelays': self.live_delays,
                    'liveDelayMs': round(self.live_delay_seconds * 1000, 3), 'drops': 0}
