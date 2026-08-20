"""
Internal recorder shutdown helpers.
"""

import gc
import logging


logger = logging.getLogger("realtimestt")


# Worker shutdown must never depend on a worker eventually observing the
# shutdown event.  Keep the normal grace period aligned with the historical
# process joins, and use a shorter second wait after forceful termination.
_WORKER_JOIN_TIMEOUT = 10
_FORCEFUL_JOIN_TIMEOUT = 1


def _worker_is_alive(worker, worker_name):
    """
    Returns whether a worker is still alive, treating inspection failures as
    active workers so cleanup failures are reported instead of hidden.
    """
    try:
        return worker.is_alive()
    except Exception:
        logger.error(
            "Could not determine whether %s stopped during shutdown.",
            worker_name,
            exc_info=True,
        )
        return True


def _join_worker(worker, worker_name, timeout):
    """
    Joins a worker for a bounded period and reports join/inspection failures.
    """
    try:
        worker.join(timeout=timeout)
    except Exception:
        logger.error(
            "Could not join %s during shutdown.",
            worker_name,
            exc_info=True,
        )
        return False

    if _worker_is_alive(worker, worker_name):
        return False

    return True


def _finish_thread(thread, thread_name):
    """
    Waits for a recorder thread without allowing shutdown to hang forever.

    Python does not provide a safe way to kill an arbitrary thread.  A thread
    that ignores the shutdown event is therefore reported explicitly while
    the rest of the cleanup continues.
    """
    if thread is None:
        return

    if _join_worker(thread, thread_name, _WORKER_JOIN_TIMEOUT):
        return

    logger.error(
        "%s did not stop within %s seconds during shutdown; "
        "it remains active because Python threads cannot be terminated safely.",
        thread_name,
        _WORKER_JOIN_TIMEOUT,
    )


def _finish_process(process, process_name):
    """
    Waits for a process and forcefully terminates it if needed.

    On Linux the runtime may represent a worker as a thread, so termination
    is optional.  In that case the bounded join still prevents shutdown from
    hanging and the missing forceful operation is logged.
    """
    if process is None:
        return

    if _join_worker(process, process_name, _WORKER_JOIN_TIMEOUT):
        return

    logger.warning(
        "%s did not terminate in time. Terminating forcefully.",
        process_name,
    )

    terminate = getattr(process, "terminate", None)
    kill = getattr(process, "kill", None)
    termination_succeeded = False
    if not callable(terminate):
        if callable(kill):
            logger.warning(
                "%s does not expose terminate(); trying kill() directly.",
                process_name,
            )
        else:
            logger.error(
                "%s is still active after the shutdown timeout and does not "
                "support forceful termination.",
                process_name,
            )
    else:
        try:
            terminate()
            termination_succeeded = True
        except Exception:
            logger.error(
                "Could not terminate %s forcefully during shutdown.",
                process_name,
                exc_info=True,
            )

    if (termination_succeeded
            and _join_worker(process, process_name, _FORCEFUL_JOIN_TIMEOUT)):
        return

    if callable(kill):
        logger.warning(
            "%s remained active after the first forceful cleanup attempt; "
            "killing forcefully.",
            process_name,
        )
        try:
            kill()
        except Exception:
            logger.error(
                "Could not kill %s after forceful termination failed.",
                process_name,
                exc_info=True,
            )
        else:
            if _join_worker(process, process_name, _FORCEFUL_JOIN_TIMEOUT):
                return

    logger.error(
        "%s is still active after forceful shutdown cleanup.",
        process_name,
    )


def _close_resource(resource, resource_name):
    """Closes a shutdown resource while keeping later cleanup best effort."""
    if resource is None:
        return

    try:
        resource.close()
    except Exception:
        logger.error(
            "Could not close %s during shutdown.",
            resource_name,
            exc_info=True,
        )


def shutdown_recorder(recorder):
    """
    Stops worker threads, subprocesses, pipes, and realtime resources.
    """
    with recorder.shutdown_lock:
        if recorder.is_shut_down:
            return

        print("\033[91mRealtimeSTT shutting down\033[0m")

        # Wake wait_audio() and text() callers before worker teardown.
        recorder.is_shut_down = True
        recorder.continuous_listening = False
        recorder.start_recording_event.set()
        recorder.stop_recording_event.set()

        recorder.shutdown_event.set()
        recorder.is_recording = False
        recorder.is_running = False

        logger.debug('Finishing recording thread')
        _finish_thread(
            getattr(recorder, "recording_thread", None),
            "Recording thread",
        )

        logger.debug('Terminating reader process')

        # Give the reader loop time to flush and close its device handle.
        use_microphone = getattr(recorder, "use_microphone", False)
        if getattr(use_microphone, "value", use_microphone):
            _finish_process(
                getattr(recorder, "reader_process", None),
                "Reader process",
            )

        logger.debug('Terminating transcription process')
        _finish_process(
            getattr(recorder, "transcript_process", None),
            "Transcript process",
        )

        _close_resource(
            getattr(recorder, "parent_transcription_pipe", None),
            "parent transcription pipe",
        )

        logger.debug('Finishing realtime thread')
        _finish_thread(
            getattr(recorder, "realtime_thread", None),
            "Realtime thread",
        )

        if getattr(recorder, "enable_realtime_transcription", False):
            if getattr(recorder, "realtime_transcription_model", None):
                del recorder.realtime_transcription_model
                recorder.realtime_transcription_model = None
        gc.collect()
