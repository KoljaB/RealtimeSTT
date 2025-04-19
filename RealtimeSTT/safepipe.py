import sys
import multiprocessing as mp
import queue
import threading
import time
import logging

# Configure logging. Adjust level and formatting as needed.
# logging.basicConfig(level=logging.DEBUG,
#                     format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    # Only set the start method if it hasn't been set already.
    if sys.platform.startswith('linux') or sys.platform == 'darwin':  # For Linux or macOS
        mp.set_start_method("spawn")
    elif mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
except RuntimeError as e:
    logger.debug("Start method has already been set. Details: %s", e)


class ParentPipe:
    """
    A thread-safe wrapper around the 'parent end' of a multiprocessing pipe.
    All actual pipe operations happen in a dedicated worker thread, so it's safe
    for multiple threads to call send(), recv(), or poll() on the same ParentPipe
    without interfering.
    """
    def __init__(self, parent_synthesize_pipe):
        self.name = "ParentPipe"
        self._pipe = parent_synthesize_pipe  # The raw pipe.
        self._closed = False  # A flag to mark if close() has been called.

        # The request queue for sending operations to the worker.
        self._request_queue = queue.Queue()

        # This event signals the worker thread to stop.
        self._stop_event = threading.Event()

        # Worker thread that executes actual .send(), .recv(), .poll() calls.
        self._worker_thread = threading.Thread(
            target=self._pipe_worker,
            name=f"{self.name}_Worker",
            daemon=True
        )
        self._worker_thread.start()

    def _pipe_worker(self):
        while not self._stop_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if request["type"] == "CLOSE":
                # Exit worker loop on CLOSE request.
                break

            try:
                if request["type"] == "SEND":
                    data = request["data"]
                    logger.debug("[%s] Worker: sending => %s", self.name, data)
                    self._pipe.send(data)
                    request["result_queue"].put(None)

                elif request["type"] == "RECV":
                    logger.debug("[%s] Worker: receiving...", self.name)
                    data = self._pipe.recv()
                    request["result_queue"].put(data)

                elif request["type"] == "POLL":
                    timeout = request.get("timeout", 0.0)
                    logger.debug("[%s] Worker: poll() with timeout: %s", self.name, timeout)
                    result = self._pipe.poll(timeout)
                    request["result_queue"].put(result)

            except (EOFError, BrokenPipeError, OSError) as e:
                # When the other end has closed or an error occurs,
                # log and notify the waiting thread.
                logger.debug("[%s] Worker: pipe closed or error occurred (%s). Shutting down.", self.name, e)
                request["result_queue"].put(None)
                break

            except Exception as e:
                logger.exception("[%s] Worker: unexpected error.", self.name)
                request["result_queue"].put(e)
                break

        logger.debug("[%s] Worker: stopping.", self.name)
        try:
            self._pipe.close()
        except Exception as e:
            logger.debug("[%s] Worker: error during pipe close: %s", self.name, e)

    def send(self, data):
        """
        Synchronously asks the worker thread to perform .send().
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
        self._request_queue.put(request)
        result_queue.get()  # Wait until sending completes.
        logger.debug("[%s] send() completed", self.name)

    def recv(self):
        """
        Synchronously asks the worker to perform .recv() and returns the data.
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
        self._request_queue.put(request)
        data = result_queue.get()

        # Log a preview for huge byte blobs.
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], bytes):
            data_preview = (data[0], f"<{len(data[1])} bytes>")
        else:
            data_preview = data
        logger.debug("[%s] recv() returning => %s", self.name, data_preview)
        return data

    def poll(self, timeout=0.0):
        """
        Synchronously checks whether data is available.
        Returns True if data is ready, or False otherwise.
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
        self._request_queue.put(request)
        try:
            # Use a slightly longer timeout to give the worker a chance.
            result = result_queue.get(timeout=timeout + 0.1)
        except queue.Empty:
            result = False
        logger.debug("[%s] poll() returning => %s", self.name, result)
        return result

    def close(self):
        """
        Closes the pipe and stops the worker thread. The _closed flag makes
        sure no further operations are attempted.
        """
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
    Returns a pair: (thread-safe parent pipe, raw child pipe).
    """
    parent_synthesize_pipe, child_synthesize_pipe = mp.Pipe()
    parent_pipe = ParentPipe(parent_synthesize_pipe)
    return parent_pipe, child_synthesize_pipe


def child_process_code(child_end):
    """
    Example child process code that receives messages, logs them,
    sends acknowledgements, and then closes.
    """
    for i in range(3):
        msg = child_end.recv()
        logger.debug("[Child] got: %s", msg)
        child_end.send(f"ACK: {msg}")
    child_end.close()


if __name__ == "__main__":
    parent_pipe, child_pipe = SafePipe()

    # Create child process with the child_process_code function.
    p = mp.Process(target=child_process_code, args=(child_pipe,))
    p.start()

    # Event to signal sender threads to stop if needed.
    stop_polling_event = threading.Event()

    def sender_thread(n):
        try:
            parent_pipe.send(f"hello_from_thread_{n}")
        except Exception as e:
            logger.debug("[sender_thread_%s] send exception: %s", n, e)
            return

        # Use a poll loop with error handling.
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

    # Signal shutdown to any polling threads, then close the pipe.
    stop_polling_event.set()
    parent_pipe.close()
    p.join()
