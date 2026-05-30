"""Internal runtime helpers for recorder worker plumbing."""

import logging
import platform
import threading
import time
import traceback

import torch.multiprocessing as mp


logger = logging.getLogger("realtimestt")


def start_recorder_worker(target=None, args=()):
    """
    Starts a recorder worker with the historical platform split.

    Args:
    - target: Worker callable to start.
    - args: Positional arguments passed to the worker.
    """
    if (platform.system() == 'Linux'):
        thread = threading.Thread(target=target, args=args)
        thread.deamon = True
        thread.start()
        return thread
    else:
        thread = mp.Process(target=target, args=args)
        thread.start()
        return thread


def read_stdout_pipe(recorder):
    """
    Forwards child-process stdout messages through the recorder logger.
    """
    while not recorder.shutdown_event.is_set():
        try:
            if recorder.parent_stdout_pipe.poll(0.1):
                logger.debug("Receive from stdout pipe")
                message = recorder.parent_stdout_pipe.recv()
                logger.info(message)
        except (BrokenPipeError, EOFError, OSError):
            # Shutdown can close the pipe before this reader notices.
            pass
        except KeyboardInterrupt:  # handle manual interruption (Ctrl+C)
            logger.info("KeyboardInterrupt in read from stdout detected, exiting...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in read from stdout: {e}", exc_info=True)
            logger.error(traceback.format_exc())  # Log the full traceback here
            break
        time.sleep(0.1)
