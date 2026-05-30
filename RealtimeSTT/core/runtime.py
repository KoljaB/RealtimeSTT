"""Internal runtime helpers for recorder worker plumbing."""

import logging
import time
import traceback


logger = logging.getLogger("realtimestt")


def read_stdout_pipe(recorder):
    while not recorder.shutdown_event.is_set():
        try:
            if recorder.parent_stdout_pipe.poll(0.1):
                logger.debug("Receive from stdout pipe")
                message = recorder.parent_stdout_pipe.recv()
                logger.info(message)
        except (BrokenPipeError, EOFError, OSError):
            # The pipe probably has been closed, so we ignore the error
            pass
        except KeyboardInterrupt:  # handle manual interruption (Ctrl+C)
            logger.info("KeyboardInterrupt in read from stdout detected, exiting...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in read from stdout: {e}", exc_info=True)
            logger.error(traceback.format_exc())  # Log the full traceback here
            break
        time.sleep(0.1)
