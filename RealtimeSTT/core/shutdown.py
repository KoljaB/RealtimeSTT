"""Internal recorder shutdown helpers."""

import gc
import logging


logger = logging.getLogger("realtimestt")


def shutdown_recorder(recorder):
    with recorder.shutdown_lock:
        if recorder.is_shut_down:
            return

        print("\033[91mRealtimeSTT shutting down\033[0m")

        # Force wait_audio() and text() to exit
        recorder.is_shut_down = True
        recorder.continuous_listening = False
        recorder.start_recording_event.set()
        recorder.stop_recording_event.set()

        recorder.shutdown_event.set()
        recorder.is_recording = False
        recorder.is_running = False

        logger.debug('Finishing recording thread')
        if recorder.recording_thread:
            recorder.recording_thread.join()

        logger.debug('Terminating reader process')

        # Give it some time to finish the loop and cleanup.
        if recorder.use_microphone.value:
            recorder.reader_process.join(timeout=10)

            if recorder.reader_process.is_alive():
                logger.warning("Reader process did not terminate "
                                "in time. Terminating forcefully."
                                )
                recorder.reader_process.terminate()

        logger.debug('Terminating transcription process')
        if recorder.transcript_process:
            recorder.transcript_process.join(timeout=10)

        if recorder.transcript_process and recorder.transcript_process.is_alive():
            logger.warning("Transcript process did not terminate "
                            "in time. Terminating forcefully."
                            )
            recorder.transcript_process.terminate()

        if recorder.parent_transcription_pipe:
            recorder.parent_transcription_pipe.close()

        logger.debug('Finishing realtime thread')
        if recorder.realtime_thread:
            recorder.realtime_thread.join()

        if recorder.enable_realtime_transcription:
            if recorder.realtime_transcription_model:
                del recorder.realtime_transcription_model
                recorder.realtime_transcription_model = None
        gc.collect()
