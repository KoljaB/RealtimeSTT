"""
Thread-safe parent-side wrappers for multiprocessing pipes.
"""

import sys
import multiprocessing as mp
import queue
import threading
import time
import logging

logger = logging.getLogger("RealtimeSTT.safepipe")

try:
    # Multiprocessing start method is process-global; leave existing settings intact.
    if sys.platform.startswith('linux') or sys.platform == 'darwin':  # For Linux or macOS
        mp.set_start_method("spawn")
    elif mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
except RuntimeError as e:
    logger.debug("Start method has already been set. Details: %s", e)


class ParentPipe:
    """
    Serializes parent-pipe operations through a dedicated worker thread.
    """
    def __init__(self, parent_synthesize_pipe):
        """
        Creates a thread-safe parent wrapper for a multiprocessing pipe.
        """
        self.name = "ParentPipe"
        self._pipe = parent_synthesize_pipe
        self._closed = False
        self._failure = None
        self._state_lock = threading.Lock()

        self._request_queue = queue.Queue()
        self._stop_event = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._pipe_worker,
            name=f"{self.name}_Worker",
            daemon=True
        )
        self._worker_thread.start()

    def _enqueue_request(self, request):
        """Atomically admit work or report the worker's terminal failure."""
        with self._state_lock:
            if self._failure is not None:
                raise BrokenPipeError(self._failure)
            if self._closed:
                return False
            self._request_queue.put(request)
        return True

    def _finish_requests(self, error, current=None):
        """Wake every admitted caller before the worker exits."""
        with self._state_lock:
            if self._failure is None:
                try:
                    detail = str(error)
                except Exception:
                    detail = "<unprintable exception>"
                self._failure = f"Pipe worker stopped after {type(error).__name__}: {detail}"
            if current is not None:
                current["result_queue"].put((False, error))
            while True:
                try:
                    request = self._request_queue.get_nowait()
                except queue.Empty:
                    break
                request["result_queue"].put((False, BrokenPipeError(self._failure)))

    @staticmethod
    def _receive_result(result_queue, timeout=None):
        succeeded, value = result_queue.get(timeout=timeout)
        if not succeeded:
            raise value
        return value

    def _pipe_worker(self):
        """
        Executes serialized pipe operations until closed.
        """
        while not self._stop_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if request["type"] == "CLOSE":
                break

            try:
                if request["type"] == "SEND":
                    data = request["data"]
                    logger.debug("[%s] Worker: sending => %s", self.name, data)
                    self._pipe.send(data)
                    request["result_queue"].put((True, None))

                elif request["type"] == "RECV":
                    logger.debug("[%s] Worker: receiving...", self.name)
                    data = self._pipe.recv()
                    request["result_queue"].put((True, data))

                elif request["type"] == "POLL":
                    timeout = request.get("timeout", 0.0)
                    logger.debug("[%s] Worker: poll() with timeout: %s", self.name, timeout)
                    result = self._pipe.poll(timeout)
                    request["result_queue"].put((True, result))

            except (EOFError, BrokenPipeError, OSError) as e:
                # Notify the waiting caller when the child end closes first.
                logger.debug("[%s] Worker: pipe closed or error occurred (%s). Shutting down.", self.name, e)
                self._finish_requests(e, request)
                break

            except Exception as e:
                logger.exception("[%s] Worker: unexpected error.", self.name)
                self._finish_requests(e, request)
                break

        self._finish_requests(EOFError("Pipe worker has stopped"))
        logger.debug("[%s] Worker: stopping.", self.name)
        try:
            self._pipe.close()
        except Exception as e:
            logger.debug("[%s] Worker: error during pipe close: %s", self.name, e)

    def send(self, data):
        """
        Sends data through the worker thread.
        """
        if self._closed:
            logger.debug("[%s] send() called but pipe is already closed", self.name)
            return
        logger.debug("[%s] send() requested with: %s", self.name, data)
        result_queue = queue.Queue()
        request = {
            "type": "SEND",
            "data": data,
            "result_queue": result_queue
        }
        if not self._enqueue_request(request):
            return
        self._receive_result(result_queue)  # Wait until sending completes.
        logger.debug("[%s] send() completed", self.name)

    def recv(self):
        """
        Receives data through the worker thread.
        """
        if self._closed:
            logger.debug("[%s] recv() called but pipe is already closed", self.name)
            return None
        logger.debug("[%s] recv() requested", self.name)
        result_queue = queue.Queue()
        request = {
            "type": "RECV",
            "result_queue": result_queue
        }
        if not self._enqueue_request(request):
            return None
        data = self._receive_result(result_queue)

        # Log a preview for huge byte blobs.
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], bytes):
            data_preview = (data[0], f"<{len(data[1])} bytes>")
        else:
            data_preview = data
        logger.debug("[%s] recv() returning => %s", self.name, data_preview)
        return data

    def poll(self, timeout=0.0):
        """
        Checks whether data is available through the worker thread.
        """
        if self._closed:
            return False
        logger.debug("[%s] poll() requested with timeout: %s", self.name, timeout)
        result_queue = queue.Queue()
        request = {
            "type": "POLL",
            "timeout": timeout,
            "result_queue": result_queue
        }
        if not self._enqueue_request(request):
            return False
        try:
            # Use a slightly longer timeout to give the worker a chance.
            result = self._receive_result(result_queue, timeout=timeout + 0.1)
        except queue.Empty:
            result = False
        logger.debug("[%s] poll() returning => %s", self.name, result)
        return result

    def close(self):
        """
        Closes the pipe and stops the worker thread.
        """
        with self._state_lock:
            if self._closed:
                return
            logger.debug("[%s] close() called", self.name)
            self._closed = True
            stop_request = {"type": "CLOSE", "result_queue": queue.Queue()}
            self._request_queue.put(stop_request)
            self._stop_event.set()
        self._worker_thread.join()
        logger.debug("[%s] closed", self.name)


def SafePipe(debug=False):
    """
    Returns a thread-safe parent pipe and a raw child pipe.
    """
    parent_synthesize_pipe, child_synthesize_pipe = mp.Pipe()
    parent_pipe = ParentPipe(parent_synthesize_pipe)
    return parent_pipe, child_synthesize_pipe


def child_process_code(child_end):
    """
    Runs a small child-process pipe example.
    """
    for i in range(3):
        msg = child_end.recv()
        logger.debug("[Child] got: %s", msg)
        child_end.send(f"ACK: {msg}")
    child_end.close()


if __name__ == "__main__":
    parent_pipe, child_pipe = SafePipe()

    p = mp.Process(target=child_process_code, args=(child_pipe,))
    p.start()

    stop_polling_event = threading.Event()

    def sender_thread(n):
        """
        Sends one example message and waits for an acknowledgement.
        """
        try:
            parent_pipe.send(f"hello_from_thread_{n}")
        except Exception as e:
            logger.debug("[sender_thread_%s] send exception: %s", n, e)
            return

        # Polling mirrors callers that must tolerate early child shutdown.
        for _ in range(10):
            try:
                if parent_pipe.poll(0.1):
                    reply = parent_pipe.recv()
                    logger.debug("[sender_thread_%s] got: %s", n, reply)
                    break
                else:
                    logger.debug("[sender_thread_%s] no data yet...", n)
            except (OSError, EOFError, BrokenPipeError) as e:
                logger.debug("[sender_thread_%s] poll/recv exception: %s. Exiting thread.", n, e)
                break

            # Allow exit if a shutdown is signaled.
            if stop_polling_event.is_set():
                logger.debug("[sender_thread_%s] stop event set. Exiting thread.", n)
                break

    threads = []
    for i in range(3):
        t = threading.Thread(target=sender_thread, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    stop_polling_event.set()
    parent_pipe.close()
    p.join()
