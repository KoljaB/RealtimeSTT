"""Recording worker loop for :class:`AudioToTextRecorder`."""

import datetime
import logging
import queue
import struct
import time

import numpy as np

from .transcription import submit_transcription_request
from .state import run_callback, set_recorder_state
from .voice_activity import (
    append_to_pre_recording_buffer,
    check_voice_activity,
    clear_pre_recording_buffer,
    is_silero_speech,
    is_voice_active,
    is_webrtc_speech,
    reset_silero_vad_state,
    selected_pre_recording_buffer_frames,
)
from .wakeword import process_wakeword

logger = logging.getLogger("realtimestt")

INT16_MAX_ABS_VALUE = 32768.0


def run_recording_worker(recorder):
    """
    The main worker method which constantly monitors the audio
    input for voice activity and accordingly starts/stops the recording.
    """

    self = recorder

    if self.use_extended_logging:
        logger.debug('Debug: Entering try block')

    last_inner_try_time = 0
    try:
        if self.use_extended_logging:
            logger.debug('Debug: Initializing variables')
        time_since_last_buffer_message = 0
        was_recording = False
        delay_was_passed = False
        wakeword_detected_time = None
        wakeword_samples_to_remove = None
        self.allowed_to_early_transcribe = True

        if self.use_extended_logging:
            logger.debug('Debug: Starting main loop')
        # Continuously monitor audio for voice activity
        while self.is_running:

            # if self.use_extended_logging:
            #     logger.debug('Debug: Entering inner try block')
            if last_inner_try_time:
                last_processing_time = time.time() - last_inner_try_time
                if last_processing_time > 0.1:
                    if self.use_extended_logging:
                        logger.warning('### WARNING: PROCESSING TOOK TOO LONG')
            last_inner_try_time = time.time()
            try:
                # if self.use_extended_logging:
                #     logger.debug('Debug: Trying to get data from audio queue')
                try:
                    data = self.audio_queue.get(timeout=0.01)
                    self.last_words_buffer.append(data)
                except queue.Empty:
                    # if self.use_extended_logging:
                    #     logger.debug('Debug: Queue is empty, checking if still running')
                    if not self.is_running:
                        if self.use_extended_logging:
                            logger.debug('Debug: Not running, breaking loop')
                        break
                    # if self.use_extended_logging:
                    #     logger.debug('Debug: Continuing to next iteration')
                    continue

                if self.use_extended_logging:
                    logger.debug('Debug: Checking for on_recorded_chunk callback')
                if self.on_recorded_chunk:
                    if self.use_extended_logging:
                        logger.debug('Debug: Calling on_recorded_chunk')
                    run_callback(self, self.on_recorded_chunk, data)

                if self.use_extended_logging:
                    logger.debug('Debug: Checking if handle_buffer_overflow is True')
                if self.handle_buffer_overflow:
                    if self.use_extended_logging:
                        logger.debug('Debug: Handling buffer overflow')
                    # Handle queue overflow
                    if (self.audio_queue.qsize() >
                            self.allowed_latency_limit):
                        if self.use_extended_logging:
                            logger.debug('Debug: Queue size exceeds limit, logging warnings')
                        logger.warning("Audio queue size exceeds "
                                        "latency limit. Current size: "
                                        f"{self.audio_queue.qsize()}. "
                                        "Discarding old audio chunks."
                                        )

                    if self.use_extended_logging:
                        logger.debug('Debug: Discarding old chunks if necessary')
                    while (self.audio_queue.qsize() >
                            self.allowed_latency_limit):

                        data = self.audio_queue.get()

            except BrokenPipeError:
                logger.error("BrokenPipeError _recording_worker", exc_info=True)
                self.is_running = False
                break

            if self.use_extended_logging:
                logger.debug('Debug: Updating time_since_last_buffer_message')
            # Feed the extracted data to the audio_queue
            if time_since_last_buffer_message:
                time_passed = time.time() - time_since_last_buffer_message
                if time_passed > 1:
                    if self.use_extended_logging:
                        logger.debug("_recording_worker processing audio data")
                    time_since_last_buffer_message = time.time()
            else:
                time_since_last_buffer_message = time.time()

            if self.use_extended_logging:
                logger.debug('Debug: Initializing failed_stop_attempt')
            failed_stop_attempt = False

            if self.use_extended_logging:
                logger.debug('Debug: Checking if not recording')
            if not self.is_recording:
                if self.use_extended_logging:
                    logger.debug('Debug: Handling not recording state')
                # Handle not recording state
                time_since_listen_start = (time.time() - self.listen_start
                                        if self.listen_start else 0)

                wake_word_activation_delay_passed = (
                    time_since_listen_start >
                    self.wake_word_activation_delay
                )

                if self.use_extended_logging:
                    logger.debug('Debug: Handling wake-word timeout callback')
                # Handle wake-word timeout callback
                if wake_word_activation_delay_passed \
                        and not delay_was_passed:

                    if self.use_wake_words and self.wake_word_activation_delay:
                        if self.on_wakeword_timeout:
                            if self.use_extended_logging:
                                logger.debug('Debug: Calling on_wakeword_timeout')
                            run_callback(self, self.on_wakeword_timeout)
                delay_was_passed = wake_word_activation_delay_passed

                if self.use_extended_logging:
                    logger.debug('Debug: Setting state and spinner text')
                # Set state and spinner text
                if not self.recording_stop_time:
                    if self.use_wake_words \
                            and wake_word_activation_delay_passed \
                            and not self.wakeword_detected:
                        if self.use_extended_logging:
                            logger.debug('Debug: Setting state to "wakeword"')
                        set_recorder_state(self, "wakeword")
                    else:
                        if self.listen_start:
                            if self.use_extended_logging:
                                logger.debug('Debug: Setting state to "listening"')
                            set_recorder_state(self, "listening")
                        else:
                            if self.use_extended_logging:
                                logger.debug('Debug: Setting state to "inactive"')
                            set_recorder_state(self, "inactive")

                if self.use_extended_logging:
                    logger.debug('Debug: Checking wake word conditions')
                if self.use_wake_words and wake_word_activation_delay_passed:
                    try:
                        if self.use_extended_logging:
                            logger.debug('Debug: Processing wakeword')
                        wakeword_index = process_wakeword(self, data)

                    except struct.error:
                        logger.error("Error unpacking audio data "
                                    "for wake word processing.", exc_info=True)
                        continue

                    except Exception as e:
                        logger.error(f"Wake word processing error: {e}", exc_info=True)
                        continue

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking if wake word detected')
                    # If a wake word is detected
                    if wakeword_index >= 0:
                        if self.use_extended_logging:
                            logger.debug('Debug: Wake word detected, updating variables')
                        self.wake_word_detect_time = time.time()
                        wakeword_detected_time = time.time()
                        wakeword_samples_to_remove = int(self.sample_rate * self.wake_word_buffer_duration)
                        self.wakeword_detected = True
                        if self.on_wakeword_detected:
                            if self.use_extended_logging:
                                logger.debug('Debug: Calling on_wakeword_detected')
                            run_callback(self, self.on_wakeword_detected)

                if self.use_extended_logging:
                    logger.debug('Debug: Checking voice activity conditions')
                # Check for voice activity to
                # trigger the start of recording
                if ((not self.use_wake_words
                    or not wake_word_activation_delay_passed)
                        and self.start_recording_on_voice_activity) \
                        or self.wakeword_detected:

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking if voice is active')

                    if is_voice_active(self):

                        if self.on_vad_start:
                           run_callback(self, self.on_vad_start)

                        if self.use_extended_logging:
                            logger.debug('Debug: Voice activity detected')
                        logger.info("voice activity detected")

                        if self.use_extended_logging:
                            logger.debug('Debug: Starting recording')
                        pre_recording_frames = selected_pre_recording_buffer_frames(self)
                        self.start()

                        self.start_recording_on_voice_activity = False

                        if self.use_extended_logging:
                            logger.debug('Debug: Adding buffered audio to frames')
                        # Add the buffered audio
                        # to the recording frames
                        self.frames.extend(pre_recording_frames)
                        clear_pre_recording_buffer(self)

                        if self.use_extended_logging:
                            logger.debug('Debug: Resetting Silero VAD model states')
                        reset_silero_vad_state(self)
                    else:
                        if self.use_extended_logging:
                            logger.debug('Debug: Checking voice activity')
                        data_copy = data[:]
                        check_voice_activity(self, data_copy)

                if self.use_extended_logging:
                    logger.debug('Debug: Resetting speech_end_silence_start')

                if self.speech_end_silence_start != 0:
                    self.speech_end_silence_start = 0
                    if self.on_turn_detection_stop:
                        if self.use_extended_logging:
                            logger.debug('Debug: Calling on_turn_detection_stop')
                        run_callback(self, self.on_turn_detection_stop)

            else:
                if self.use_extended_logging:
                    logger.debug('Debug: Handling recording state')
                # If we are currently recording
                if wakeword_samples_to_remove and wakeword_samples_to_remove > 0:
                    if self.use_extended_logging:
                        logger.debug('Debug: Removing wakeword samples')
                    # Remove samples from the beginning of self.frames
                    samples_removed = 0
                    while wakeword_samples_to_remove > 0 and self.frames:
                        frame = self.frames[0]
                        frame_samples = len(frame) // 2  # Assuming 16-bit audio
                        if wakeword_samples_to_remove >= frame_samples:
                            self.frames.pop(0)
                            samples_removed += frame_samples
                            wakeword_samples_to_remove -= frame_samples
                        else:
                            self.frames[0] = frame[wakeword_samples_to_remove * 2:]
                            samples_removed += wakeword_samples_to_remove
                            samples_to_remove = 0

                    wakeword_samples_to_remove = 0

                if self.use_extended_logging:
                    logger.debug('Debug: Checking if stop_recording_on_voice_deactivity is True')
                # Stop the recording if silence is detected after speech
                if self.stop_recording_on_voice_deactivity:
                    if self.use_extended_logging:
                        logger.debug('Debug: Determining if speech is detected')
                    is_speech = (
                        is_silero_speech(self, data) if self.silero_deactivity_detection
                        else is_webrtc_speech(self, data)
                    )
                    if is_speech:
                        self.speech_end_silence_candidate_start = 0
                    elif not self.speech_end_silence_start:
                        now = time.time()
                        if not self.speech_end_silence_candidate_start:
                            self.speech_end_silence_candidate_start = now
                        if (
                            now - self.speech_end_silence_candidate_start
                            < self.deactivity_silence_confirmation_duration
                        ):
                            is_speech = True

                    if self.use_extended_logging:
                        logger.debug('Debug: Formatting speech_end_silence_start')
                    if not self.speech_end_silence_start:
                        str_speech_end_silence_start = "0"
                    else:
                        str_speech_end_silence_start = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]
                    if self.use_extended_logging:
                        logger.debug(f"is_speech: {is_speech}, str_speech_end_silence_start: {str_speech_end_silence_start}")

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking if speech is not detected')
                    if not is_speech:
                        if self.use_extended_logging:
                            logger.debug('Debug: Handling voice deactivity')
                        # Voice deactivity was detected, so we start
                        # measuring silence time before stopping recording
                        if self.speech_end_silence_start == 0 and \
                            (time.time() - self.recording_start_time > self.min_length_of_recording):

                            self.speech_end_silence_start = time.time()
                            self.speech_end_silence_candidate_start = 0
                            self.awaiting_speech_end = True
                            if self.on_turn_detection_start:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Calling on_turn_detection_start')

                                run_callback(self, self.on_turn_detection_start)

                        if self.use_extended_logging:
                            logger.debug('Debug: Checking early transcription conditions')
                        if self.speech_end_silence_start and self.early_transcription_on_silence and len(self.frames) > 0 and \
                            (time.time() - self.speech_end_silence_start > self.early_transcription_on_silence) and \
                            self.allowed_to_early_transcribe:
                                if self.use_extended_logging:
                                    logger.debug("Debug:Adding early transcription request")
                                audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                                audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

                                if self.use_extended_logging:
                                    logger.debug("Debug: early transcription request submit")
                                submit_transcription_request(
                                    self,
                                    audio,
                                    self.language,
                                    True,
                                )
                                if self.use_extended_logging:
                                    logger.debug("Debug: early transcription request submit return")
                                self.allowed_to_early_transcribe = False

                    else:
                        self.awaiting_speech_end = False
                        if self.use_extended_logging:
                            logger.debug('Debug: Handling speech detection')
                        if self.speech_end_silence_start:
                            if self.use_extended_logging:
                                logger.info("Resetting self.speech_end_silence_start")

                            if self.speech_end_silence_start != 0:
                                self.speech_end_silence_start = 0
                                if self.on_turn_detection_stop:
                                    if self.use_extended_logging:
                                        logger.debug('Debug: Calling on_turn_detection_stop')
                                    run_callback(self, self.on_turn_detection_stop)

                            self.allowed_to_early_transcribe = True

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking if silence duration exceeds threshold')
                    # Wait for silence to stop recording after speech
                    if self.speech_end_silence_start and time.time() - \
                            self.speech_end_silence_start >= \
                            self.post_speech_silence_duration:

                        if self.on_vad_stop:
                            run_callback(self, self.on_vad_stop)

                        if self.use_extended_logging:
                            logger.debug('Debug: Formatting silence start time')
                        # Get time in desired format (HH:MM:SS.nnn)
                        silence_start_time = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]

                        if self.use_extended_logging:
                            logger.debug('Debug: Calculating time difference')
                        # Calculate time difference
                        time_diff = time.time() - self.speech_end_silence_start

                        if self.use_extended_logging:
                            logger.debug('Debug: Logging voice deactivity detection')
                            logger.info(f"voice deactivity detected at {silence_start_time}, "
                                    f"time since silence start: {time_diff:.3f} seconds")

                            logger.debug('Debug: Appending data to frames and stopping recording')
                        self.frames.append(data)
                        self.stop()
                        if not self.is_recording:
                            if self.speech_end_silence_start != 0:
                                self.speech_end_silence_start = 0
                                if self.on_turn_detection_stop:
                                    if self.use_extended_logging:
                                        logger.debug('Debug: Calling on_turn_detection_stop')
                                    run_callback(self, self.on_turn_detection_stop)

                            if self.use_extended_logging:
                                logger.debug('Debug: Handling non-wake word scenario')
                        else:
                            if self.use_extended_logging:
                                logger.debug('Debug: Setting failed_stop_attempt to True')
                            failed_stop_attempt = True

                        self.awaiting_speech_end = False

            if self.use_extended_logging:
                logger.debug('Debug: Checking if recording stopped')
            if not self.is_recording and was_recording:
                if self.use_extended_logging:
                    logger.debug('Debug: Resetting after stopping recording')
                # Reset after stopping recording to ensure clean state
                if self.continuous_listening:
                    self.start_recording_on_voice_activity = True
                    self.stop_recording_on_voice_deactivity = True
                else:
                    self.stop_recording_on_voice_deactivity = False
                clear_pre_recording_buffer(self)

            if self.use_extended_logging:
                logger.debug('Debug: Checking Silero time')
            if time.time() - self.silero_check_time > 0.1:
                self.silero_check_time = 0

            if self.use_extended_logging:
                logger.debug('Debug: Handling wake word timeout')
            # Handle wake word timeout (waited to long initiating
            # speech after wake word detection)
            if self.wake_word_detect_time and time.time() - \
                    self.wake_word_detect_time > self.wake_word_timeout:

                self.wake_word_detect_time = 0
                if self.wakeword_detected and self.on_wakeword_timeout:
                    if self.use_extended_logging:
                        logger.debug('Debug: Calling on_wakeword_timeout')
                    run_callback(self, self.on_wakeword_timeout)
                self.wakeword_detected = False

            if self.use_extended_logging:
                logger.debug('Debug: Updating was_recording')
            was_recording = self.is_recording

            if self.use_extended_logging:
                logger.debug('Debug: Checking if recording and not failed stop attempt')
            if self.is_recording and not failed_stop_attempt:
                if self.use_extended_logging:
                    logger.debug('Debug: Appending data to frames')
                self.frames.append(data)

            if self.use_extended_logging:
                logger.debug('Debug: Checking if not recording or speech end silence start')
            if not self.is_recording or self.speech_end_silence_start:
                if self.use_extended_logging:
                    logger.debug('Debug: Appending data to audio buffer')
                append_to_pre_recording_buffer(self, data)

    except Exception as e:
        logger.debug('Debug: Caught exception in main try block')
        if not self.interrupt_stop_event.is_set():
            logger.error(f"Unhandled exeption in _recording_worker: {e}", exc_info=True)
            raise

    if self.use_extended_logging:
        logger.debug('Debug: Exiting _recording_worker method')
