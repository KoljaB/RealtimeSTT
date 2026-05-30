"""Realtime transcription worker loop for :class:`AudioToTextRecorder`."""

import logging
import time

import numpy as np

from .realtime_boundary_detector import RealtimeSpeechBoundaryDetector
from .realtime_text_stabilizer import (
    RealtimeTextObservation,
    RealtimeTextStabilizer,
)
from .state import run_callback
from .text_formatting import preprocess_output
from .transcription import call_transcription_executor


logger = logging.getLogger("realtimestt")

TIME_SLEEP = 0.02
INT16_MAX_ABS_VALUE = 32768.0


def run_realtime_worker(recorder):
    """
    Performs real-time transcription if the feature is enabled.

    This worker is intentionally defensive:
    - realtime transcription must never crash the recorder
    - empty/None buffers are skipped
    - frame buffers are snapshotted before transcription
    - model/pipe errors are logged and skipped
    """

    self = recorder

    logger.debug("Starting realtime worker")

    if not self.enable_realtime_transcription:
        logger.debug("Realtime transcription disabled; realtime worker exits")
        return

    def _sleep_briefly():
        time.sleep(0.001)

    def _safe_get_realtime_pause():
        pause = getattr(self, "realtime_processing_pause", 0.2)
        try:
            return max(0.001, float(pause))
        except Exception:
            return 0.2

    def _safe_get_realtime_fallback_pause():
        pause = getattr(self, "realtime_processing_pause", 0.2)
        try:
            return float(pause)
        except Exception:
            return 0.2

    def _safe_get_sample_rate():
        for attr_name in (
            "sample_rate",
            "input_device_sample_rate",
            "input_device_samplerate",
            "device_sample_rate",
        ):
            value = getattr(self, attr_name, None)
            if value:
                try:
                    return int(value)
                except Exception:
                    pass

        return 16000

    def _snapshot_frames():
        frames = getattr(self, "frames", None)

        if not frames:
            return None

        # Use a frame lock if the class provides one.
        # Different RealtimeSTT versions may name this differently,
        # so keep this optional.
        frame_lock = (
            getattr(self, "frames_lock", None)
            or getattr(self, "frame_lock", None)
            or getattr(self, "audio_lock", None)
        )

        try:
            if frame_lock:
                with frame_lock:
                    return tuple(self.frames)
            return tuple(self.frames)

        except Exception as e:
            logger.debug(f"Could not snapshot realtime frames: {e}", exc_info=True)
            return None

    def _frames_to_audio_array(frames_snapshot, enforce_min_samples=True):
        if not frames_snapshot:
            return None

        valid_frames = []

        for frame in frames_snapshot:
            if frame is None:
                continue

            try:
                if len(frame) == 0:
                    continue
            except Exception:
                pass

            valid_frames.append(frame)

        if not valid_frames:
            return None

        try:
            raw_audio = b"".join(valid_frames)
        except Exception as e:
            logger.debug(f"Could not join realtime audio frames: {e}", exc_info=True)
            return None

        if not raw_audio:
            return None

        # int16 audio needs pairs of bytes.
        # Drop a trailing broken byte if one somehow appears.
        if len(raw_audio) % 2:
            raw_audio = raw_audio[:-1]

        if not raw_audio:
            return None

        try:
            audio_array = np.frombuffer(raw_audio, dtype=np.int16)
        except Exception as e:
            logger.debug(f"Could not convert realtime buffer to int16 array: {e}", exc_info=True)
            return None

        if audio_array is None or audio_array.size == 0:
            return None

        if enforce_min_samples:
            sample_rate = _safe_get_sample_rate()

            # Avoid sending tiny initial buffers into Whisper.
            # 50 ms is enough to avoid startup races without adding real latency.
            min_samples = max(1, int(sample_rate * 0.05))

            if audio_array.size < min_samples:
                logger.debug(
                    "Skipping realtime transcription because buffer is too small: "
                    f"{audio_array.size} samples < {min_samples} samples"
                )
                return None

        logger.debug(f"Current realtime buffer size: {audio_array.size}")

        try:
            audio_array = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
        except Exception as e:
            logger.debug(f"Could not normalize realtime audio: {e}", exc_info=True)
            return None

        if audio_array is None or audio_array.size == 0:
            return None

        return audio_array

    def _count_frame_samples(frames_snapshot):
        sample_count = 0
        for frame in frames_snapshot or ():
            if frame is None:
                continue
            try:
                sample_count += len(frame) // 2
            except Exception:
                continue
        return sample_count

    def _extract_text_and_language(transcription_result):
        if transcription_result is None:
            return "", None, 0.0

        text = getattr(transcription_result, "text", "") or ""

        info = getattr(transcription_result, "info", None)
        language = None
        language_probability = 0.0

        if info is not None:
            language_probability = getattr(info, "language_probability", 0.0) or 0.0

            if language_probability > 0:
                language = getattr(info, "language", None)

        return text, language, language_probability

    def _transcribe_with_main_model(audio_array):
        try:
            if self._uses_external_transcription_executor:
                return call_transcription_executor(
                    self.transcription_executor,
                    audio_array,
                    self.language,
                    True,
                )

            with self.transcription_lock:
                self.parent_transcription_pipe.send(
                    (audio_array, self.language, True)
                )

                if not self.parent_transcription_pipe.poll(timeout=5):
                    logger.warning("Realtime transcription timed out")
                    return None

                logger.debug(
                    "Receive from realtime worker after transcription request "
                    "to main model"
                )

                status, result = self.parent_transcription_pipe.recv()

                if status != "success":
                    logger.error(f"Realtime transcription error: {result}")
                    return None

                return result

        except Exception as e:
            logger.error(f"Error in realtime transcription with main model: {e}", exc_info=True)
            return None

    def _transcribe_with_realtime_model(audio_array):
        if self._uses_external_realtime_transcription_executor:
            try:
                return call_transcription_executor(
                    self.realtime_transcription_executor,
                    audio_array,
                    self.language,
                    True,
                )
            except Exception as e:
                logger.warning(f"Realtime transcription skipped: {e}", exc_info=True)
                return None

        model = getattr(self, "realtime_transcription_model", None)

        if model is None:
            logger.warning("Realtime transcription model is None; skipping")
            return None

        try:
            return model.transcribe(
                audio_array,
                language=self.language if self.language else None,
                use_prompt=True,
            )

        except Exception as e:
            logger.warning(f"Realtime transcription skipped: {e}", exc_info=True)
            return None

    streaming_session = None
    streaming_session_recording_id = None
    streaming_session_frame_count = 0

    def _streaming_realtime_target():
        if self.use_main_model_for_realtime:
            return None

        if self._uses_external_realtime_transcription_executor:
            target = getattr(self, "realtime_transcription_executor", None)
        else:
            target = getattr(self, "realtime_transcription_model", None)

        if target is None:
            return None

        if not getattr(target, "supports_streaming", False):
            return None

        if not hasattr(target, "create_streaming_session"):
            return None

        return target

    def _close_streaming_session():
        nonlocal streaming_session
        nonlocal streaming_session_recording_id
        nonlocal streaming_session_frame_count

        if streaming_session is not None and hasattr(streaming_session, "close"):
            try:
                streaming_session.close()
            except Exception as e:
                logger.debug(
                    f"Could not close realtime streaming session: {e}",
                    exc_info=True,
                )

        streaming_session = None
        streaming_session_recording_id = None
        streaming_session_frame_count = 0

    def _create_streaming_session(target):
        try:
            return target.create_streaming_session(
                language=self.language if self.language else None,
                use_prompt=True,
            )
        except TypeError:
            return target.create_streaming_session()

    def _ensure_streaming_session(recording_id):
        nonlocal streaming_session
        nonlocal streaming_session_recording_id
        nonlocal streaming_session_frame_count

        target = _streaming_realtime_target()
        if target is None:
            _close_streaming_session()
            return None

        if (
            streaming_session is None
            or streaming_session_recording_id != recording_id
        ):
            if streaming_session is not None:
                try:
                    previous_frames = tuple(getattr(self, "last_frames", None) or ())
                except Exception:
                    previous_frames = None
                _finish_streaming_session(previous_frames)
            else:
                _close_streaming_session()

            try:
                streaming_session = _create_streaming_session(target)
            except Exception as e:
                logger.warning(
                    f"Realtime streaming session creation failed: {e}",
                    exc_info=True,
                )
                streaming_session = None
                return None

            streaming_session_recording_id = recording_id
            streaming_session_frame_count = 0

        return streaming_session

    def _finish_streaming_session(frames_snapshot=None):
        nonlocal streaming_session_frame_count

        if streaming_session is None:
            return None

        try:
            if frames_snapshot:
                frame_count = len(frames_snapshot)
                if frame_count >= streaming_session_frame_count:
                    remaining_frames = frames_snapshot[streaming_session_frame_count:frame_count]
                    audio_array = _frames_to_audio_array(
                        remaining_frames,
                        enforce_min_samples=False,
                    )
                    if audio_array is not None:
                        streaming_session.accept_audio(
                            audio_array,
                            sample_rate=_safe_get_sample_rate(),
                        )
                        streaming_session_frame_count = frame_count

            return streaming_session.finish()
        except Exception as e:
            logger.debug(
                f"Could not finish realtime streaming session: {e}",
                exc_info=True,
            )
            return None
        finally:
            _close_streaming_session()

    def _transcribe_with_realtime_streaming_model(
        frames_snapshot,
        sample_rate,
        recording_id,
    ):
        nonlocal streaming_session_frame_count

        session = _ensure_streaming_session(recording_id)
        if session is None:
            return None

        frame_count = len(frames_snapshot or ())
        if frame_count < streaming_session_frame_count:
            _close_streaming_session()
            session = _ensure_streaming_session(recording_id)
            if session is None:
                return None

        new_frames = frames_snapshot[streaming_session_frame_count:frame_count]
        audio_array = _frames_to_audio_array(
            new_frames,
            enforce_min_samples=False,
        )

        if audio_array is None:
            logger.debug("Skipping realtime streaming decode because no new audio is available")
            return None

        try:
            session.accept_audio(audio_array, sample_rate=sample_rate)
            session.decode()
            streaming_session_frame_count = frame_count
            return session.get_result()
        except Exception as e:
            logger.warning(
                f"Realtime streaming transcription skipped: {e}",
                exc_info=True,
            )
            _close_streaming_session()
            return None

    def _safe_realtime_callback(callback, *args):
        try:
            run_callback(self, callback, *args)
        except Exception as e:
            logger.error(f"Realtime callback failed: {e}", exc_info=True)

    def _publish_realtime_text(
        realtime_text,
        sequence,
        trigger_reason,
        frame_count,
        sample_count,
        sample_rate,
        recording_id,
        recording_started_at_monotonic,
        recording_start_time,
        created_at_monotonic,
        completed_at_monotonic,
        completed_at_wall_time,
        detected_language,
        detected_language_probability,
    ):
        raw_text = "" if realtime_text is None else str(realtime_text)

        if recording_start_time is None:
            return

        if not self.is_recording:
            return

        publish_allowed = (
            completed_at_wall_time - recording_start_time
            > self.init_realtime_after_seconds
        )

        realtime_text_stabilizer = getattr(
            self,
            "realtime_text_stabilizer",
            None,
        )
        if realtime_text_stabilizer is None:
            realtime_text_stabilizer = RealtimeTextStabilizer()
            self.realtime_text_stabilizer = realtime_text_stabilizer

        observation = RealtimeTextObservation(
            recording_id=recording_id,
            sequence=sequence,
            raw_text=raw_text,
            audio_start_sample=0,
            audio_end_sample_exclusive=sample_count,
            sample_rate=sample_rate,
            created_at_monotonic=created_at_monotonic,
            completed_at_monotonic=completed_at_monotonic,
            recording_started_at_monotonic=recording_started_at_monotonic,
            recording_started_at_wall_time=recording_start_time,
            received_at_wall_time=completed_at_wall_time,
            trigger_reason=trigger_reason,
            language=detected_language,
            language_probability=detected_language_probability,
            engine_name=getattr(
                getattr(self, "realtime_transcription_model", None),
                "engine_name",
                None,
            ),
            model_name=getattr(self, "realtime_model_type", None),
            frame_count=frame_count,
            sample_count=sample_count,
            publish_allowed=publish_allowed,
            awaiting_speech_end=getattr(self, "awaiting_speech_end", False),
        )
        event = realtime_text_stabilizer.observe(observation)
        self.realtime_text_stabilization_event = event

        if event.accepted:
            self.realtime_stabilization_accepted_count = (
                getattr(self, "realtime_stabilization_accepted_count", 0)
                + 1
            )
        if event.is_outlier:
            self.realtime_stabilization_outlier_count = (
                getattr(self, "realtime_stabilization_outlier_count", 0)
                + 1
            )
        if event.stable_delta and event.should_publish:
            self.realtime_stabilization_stable_delta_count = (
                getattr(self, "realtime_stabilization_stable_delta_count", 0)
                + 1
            )

        if raw_text.strip():
            self.realtime_transcription_text = raw_text.strip()

        if event.accepted and raw_text.strip():
            self.text_storage.append(self.realtime_transcription_text)

        self.realtime_stabilized_text = event.stable_text
        self.realtime_stabilized_safetext = event.stable_text

        if not raw_text.strip() or not publish_allowed:
            return

        structured_callback = getattr(
            self,
            "on_realtime_text_stabilization_update",
            None,
        )
        if structured_callback:
            _safe_realtime_callback(structured_callback, event)

        stabilized_display_text = event.display_text or raw_text.strip()
        _safe_realtime_callback(
            self._on_realtime_transcription_stabilized,
            preprocess_output(
                stabilized_display_text,
                preview=True,
                ensure_sentence_starting_uppercase=(
                    self.ensure_sentence_starting_uppercase
                ),
                ensure_sentence_ends_with_period=(
                    self.ensure_sentence_ends_with_period
                ),
            ),
        )

        _safe_realtime_callback(
            self._on_realtime_transcription_update,
            preprocess_output(
                raw_text.strip(),
                preview=True,
                ensure_sentence_starting_uppercase=(
                    self.ensure_sentence_starting_uppercase
                ),
                ensure_sentence_ends_with_period=(
                    self.ensure_sentence_ends_with_period
                ),
            ),
        )

    last_transcription_time = time.time()

    def _run_realtime_transcription(trigger_reason):
        nonlocal last_transcription_time

        last_transcription_time = time.time()

        frames_snapshot = _snapshot_frames()
        sample_rate = _safe_get_sample_rate()
        recording_id = getattr(self, "realtime_recording_id", 0)
        streaming_target = _streaming_realtime_target()
        created_at_monotonic = time.monotonic()

        if streaming_target is not None:
            if not frames_snapshot:
                logger.debug("Skipping realtime streaming decode because audio buffer is empty")
                return False

            frame_count = len(frames_snapshot or ())
            sample_count = _count_frame_samples(frames_snapshot)
            transcription_result = _transcribe_with_realtime_streaming_model(
                frames_snapshot,
                sample_rate,
                recording_id,
            )
            if transcription_result is None:
                return False
        else:
            audio_array = _frames_to_audio_array(frames_snapshot)

            if audio_array is None:
                logger.debug("Skipping realtime transcription because audio buffer is empty")
                return False

            sample_count = int(audio_array.size)
            frame_count = len(frames_snapshot or ())

            if self.use_main_model_for_realtime:
                transcription_result = _transcribe_with_main_model(audio_array)
            else:
                transcription_result = _transcribe_with_realtime_model(audio_array)

        self.realtime_transcription_count += 1
        self.realtime_transcription_trigger_counts[trigger_reason] = (
            self.realtime_transcription_trigger_counts.get(trigger_reason, 0)
            + 1
        )

        self.realtime_observation_sequence = (
            getattr(self, "realtime_observation_sequence", 0) + 1
        )
        observation_sequence = self.realtime_observation_sequence
        recording_started_at_monotonic = getattr(
            self,
            "recording_start_monotonic",
            None,
        )
        recording_start_time = getattr(self, "recording_start_time", None)

        completed_at_monotonic = time.monotonic()
        completed_at_wall_time = time.time()

        realtime_text, detected_language, detected_language_probability = (
            _extract_text_and_language(transcription_result)
        )

        self.detected_realtime_language = detected_language
        self.detected_realtime_language_probability = detected_language_probability

        if not realtime_text:
            self.realtime_transcription_empty_count += 1
            logger.debug("Realtime transcription returned empty text")
            _publish_realtime_text(
                realtime_text,
                observation_sequence,
                trigger_reason,
                frame_count,
                sample_count,
                sample_rate,
                recording_id,
                recording_started_at_monotonic,
                recording_start_time,
                created_at_monotonic,
                completed_at_monotonic,
                completed_at_wall_time,
                detected_language,
                detected_language_probability,
            )
            return False

        self.realtime_transcription_success_count += 1
        logger.debug(f"Realtime text detected ({trigger_reason}): {realtime_text}")

        _publish_realtime_text(
            realtime_text,
            observation_sequence,
            trigger_reason,
            frame_count,
            sample_count,
            sample_rate,
            recording_id,
            recording_started_at_monotonic,
            recording_start_time,
            created_at_monotonic,
            completed_at_monotonic,
            completed_at_wall_time,
            detected_language,
            detected_language_probability,
        )
        return True

    use_syllable_boundaries = bool(
        getattr(self, "realtime_transcription_use_syllable_boundaries", False)
    )
    boundary_detector = None
    boundary_detector_frame_count = 0
    boundary_followup_deadlines = []
    boundary_recording_start_time = None

    def _get_boundary_followup_offsets():
        delays = getattr(
            self,
            "realtime_boundary_followup_delays",
            (0.05, 0.2),
        )

        if delays is None:
            return []

        if isinstance(delays, (int, float)):
            delays = [delays]

        offsets = []

        try:
            for delay in delays:
                try:
                    delay = float(delay)
                except Exception:
                    continue

                if delay < 0:
                    continue

                offsets.append(delay)
        except TypeError:
            return []

        return sorted(set(offsets))

    def _reset_boundary_scheduler():
        nonlocal boundary_detector
        nonlocal boundary_detector_frame_count
        nonlocal boundary_followup_deadlines

        sensitivity = getattr(self, "realtime_boundary_detector_sensitivity", 0.6)

        try:
            sensitivity = float(sensitivity)
        except Exception:
            sensitivity = 0.6

        boundary_detector = RealtimeSpeechBoundaryDetector(
            sample_rate=_safe_get_sample_rate(),
            sensitivity=sensitivity,
        )
        boundary_detector_frame_count = 0
        boundary_followup_deadlines = []

    def _process_new_boundary_frames(frames_snapshot):
        nonlocal boundary_detector_frame_count

        if boundary_detector is None:
            _reset_boundary_scheduler()

        if not frames_snapshot:
            boundary_detector_frame_count = 0
            return False

        frame_count = len(frames_snapshot)

        if frame_count < boundary_detector_frame_count:
            _reset_boundary_scheduler()
            boundary_detector_frame_count = 0

        new_frames = frames_snapshot[boundary_detector_frame_count:frame_count]
        boundary_detector_frame_count = frame_count

        if not new_frames:
            return False

        boundary_detected = False

        for frame in new_frames:
            try:
                result = boundary_detector.process_bytes(frame)
            except Exception as e:
                logger.debug(
                    f"Could not process realtime boundary frame: {e}",
                    exc_info=True,
                )
                continue

            if result.boundary_detected:
                boundary_detected = True

        return boundary_detected

    def _run_syllable_boundary_scheduler():
        nonlocal boundary_followup_deadlines
        nonlocal boundary_recording_start_time

        recording_start_time = getattr(self, "recording_start_time", None)

        if recording_start_time != boundary_recording_start_time:
            boundary_recording_start_time = recording_start_time
            _reset_boundary_scheduler()

        frames_snapshot = _snapshot_frames()
        boundary_detected = _process_new_boundary_frames(frames_snapshot)
        now = time.time()

        if boundary_detected:
            boundary_followup_deadlines = [
                now + offset for offset in _get_boundary_followup_offsets()
            ]
            return _run_realtime_transcription("syllable-boundary")

        due_followup = any(
            deadline <= now for deadline in boundary_followup_deadlines
        )

        if due_followup:
            # Coalesce all expired follow-ups into one current-buffer pass.
            boundary_followup_deadlines = [
                deadline for deadline in boundary_followup_deadlines
                if deadline > now
            ]
            return _run_realtime_transcription("syllable-boundary-followup")

        fallback_pause = _safe_get_realtime_fallback_pause()

        if fallback_pause > 0 and now - last_transcription_time >= fallback_pause:
            return _run_realtime_transcription("syllable-boundary-fallback")

        return False

    while self.is_running:
        try:
            if not self.is_recording:
                if streaming_session is not None:
                    try:
                        finished_frames = tuple(getattr(self, "last_frames", None) or ())
                    except Exception:
                        finished_frames = None
                    if not finished_frames:
                        finished_frames = _snapshot_frames()
                    _finish_streaming_session(finished_frames)

                # Important:
                # Reset timer while idle so the worker does not instantly
                # transcribe an empty startup buffer when recording begins.
                last_transcription_time = time.time()
                if use_syllable_boundaries:
                    boundary_recording_start_time = None
                    boundary_followup_deadlines = []
                time.sleep(TIME_SLEEP)
                continue

            if use_syllable_boundaries:
                if self.awaiting_speech_end:
                    _sleep_briefly()
                    continue

                _run_syllable_boundary_scheduler()
                _sleep_briefly()
                continue

            realtime_processing_pause = _safe_get_realtime_pause()

            while time.time() - last_transcription_time < realtime_processing_pause:
                _sleep_briefly()

                if not self.is_running or not self.is_recording:
                    break

            if not self.is_running:
                break

            if not self.is_recording:
                continue

            if self.awaiting_speech_end:
                _sleep_briefly()
                continue

            _run_realtime_transcription("timer")

        except Exception as e:
            # Realtime transcription is a convenience feature.
            # It must never kill the recorder/session.
            logger.error(f"Unhandled exception in _realtime_worker loop: {e}", exc_info=True)
            time.sleep(TIME_SLEEP)

    if streaming_session is not None:
        _finish_streaming_session(_snapshot_frames())

    logger.debug("Realtime worker stopped")
