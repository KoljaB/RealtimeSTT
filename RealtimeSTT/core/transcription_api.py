"""
Internal public transcription API helpers.
"""

import base64
import copy
import logging
import threading
import time

from .state import set_recorder_state
from .text_formatting import preprocess_output
from .transcription import (
    receive_transcription_result,
    submit_transcription_request,
)


logger = logging.getLogger("realtimestt")


def text(recorder, on_transcription_finished=None):
    """
    Waits for audio and returns the final transcription text.
    """
    recorder.interrupt_stop_event.clear()
    recorder.was_interrupted.clear()
    try:
        recorder.wait_audio()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt in text() method")
        recorder.shutdown()
        raise  # Re-raise the exception after cleanup

    if recorder.is_shut_down or recorder.interrupt_stop_event.is_set():
        if recorder.interrupt_stop_event.is_set():
            recorder.was_interrupted.set()
        return ""

    if on_transcription_finished:
        threading.Thread(target=on_transcription_finished,
                        args=(recorder.transcribe(),)).start()
    else:
        return recorder.transcribe()


def transcribe(recorder):
    """
    Starts final transcription for the recorder's current audio.
    """
    audio_copy = copy.deepcopy(recorder.audio)
    set_recorder_state(recorder, "transcribing")
    if recorder.on_transcription_start:
        abort_value = recorder.on_transcription_start(audio_copy)
        if not abort_value:
            return recorder.perform_final_transcription(audio_copy)
        return None
    else:
        return recorder.perform_final_transcription(audio_copy)


def _set_state_after_transcription(recorder):
    """
    Restores recorder state after final transcription completes.
    """
    if recorder.is_recording:
        set_recorder_state(recorder, "recording")
    else:
        set_recorder_state(recorder, "inactive")


def perform_final_transcription(recorder, audio_bytes=None, use_prompt=True):
    """
    Runs final transcription and formats the resulting text.
    """
    start_time = 0
    with recorder.transcription_lock:
        if audio_bytes is None:
            audio_bytes = copy.deepcopy(recorder.audio)

        if audio_bytes is None or len(audio_bytes) == 0:
            print("No audio data available for transcription")
            #logger.info("No audio data available for transcription")
            return ""

        try:
            if recorder.transcribe_count == 0:
                logger.debug("Adding transcription request, no early transcription started")
                start_time = time.time()
                submit_transcription_request(
                    recorder,
                    audio_bytes,
                    recorder.language,
                    use_prompt,
                )

            while recorder.transcribe_count > 0:
                logger.debug(F"Receive from parent_transcription_pipe after sendiung transcription request, transcribe_count: {recorder.transcribe_count}")
                response = receive_transcription_result(recorder, timeout=0.1)
                if response is None:
                    if recorder.interrupt_stop_event.is_set():
                        recorder.was_interrupted.set()
                        _set_state_after_transcription(recorder)
                        return ""
                    continue
                status, result = response
                recorder.transcribe_count -= 1

            recorder.allowed_to_early_transcribe = True
            _set_state_after_transcription(recorder)
            if status == 'success':
                recorder.detected_language = (
                    result.info.language if result.info.language_probability > 0 else None
                )
                recorder.detected_language_probability = result.info.language_probability
                recorder.last_transcription_bytes = copy.deepcopy(audio_bytes)
                recorder.last_transcription_bytes_b64 = base64.b64encode(recorder.last_transcription_bytes.tobytes()).decode('utf-8')
                recorder.last_transcription_metadata = getattr(result, "metadata", None)
                transcription = preprocess_output(
                    result.text,
                    ensure_sentence_starting_uppercase=(
                        recorder.ensure_sentence_starting_uppercase
                    ),
                    ensure_sentence_ends_with_period=(
                        recorder.ensure_sentence_ends_with_period
                    ),
                )
                end_time = time.time()
                transcription_time = end_time - start_time

                if start_time:
                    if recorder.print_transcription_time:
                        print(f"Model {recorder.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                    else:
                        logger.debug(f"Model {recorder.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                return "" if recorder.interrupt_stop_event.is_set() else transcription
            else:
                logger.error(f"Transcription error: {result}")
                raise Exception(result)
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}", exc_info=True)
            raise e
