"""
Realtime transcription worker loop for :class:`AudioToTextRecorder`.
"""

import logging
import inspect
import re
import threading
import time

import numpy as np

from .realtime_boundary_detector import RealtimeSpeechBoundaryDetector
from .realtime_text_stabilizer import (
    RealtimeTextObservation,
    RealtimeTextStabilizer,
)
from .realtime_callbacks import (
    publish_realtime_transcription_stabilized,
    publish_realtime_transcription_update,
)
from .recording_buffers import get_frames_lock, queue_recorded_audio, snapshot_frames
from .state import run_callback
from .text_formatting import preprocess_output
from .transcription import call_transcription_executor


logger = logging.getLogger("realtimestt")

TIME_SLEEP = 0.02
INT16_MAX_ABS_VALUE = 32768.0
_PUNCTUATION_SPLIT_REQUIRED_OBSERVATIONS = 3
_PUNCTUATION_SPLIT_MARK_PRESETS = {
    "off": (),
    "none": (),
    "false": (),
    "sentence": (".", "?", "!"),
    "terminal": (".", "?", "!"),
    "comma": (",",),
    "ellipsis": ("...",),
    "dash": ("\u2014", "\u2013", "-"),
    "all": (".", "?", "!", ",", "...", "\u2014", "\u2013", "-"),
}
_SUPPORTED_PUNCTUATION_SPLIT_MARKS = {
    ",",
    ".",
    "?",
    "!",
    "...",
    "\u2014",
    "\u2013",
    "-",
}
_ABBREVIATIONS_BEFORE_PERIOD = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
}


def _normalize_realtime_punctuation_split_marks(marks="sentence,comma"):
    if marks is None:
        return ()
    if isinstance(marks, str):
        value = marks.strip().casefold()
        if not value:
            return ()
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        try:
            parts = list(marks)
        except TypeError as exc:
            raise TypeError(
                "realtime_punctuation_split_marks must be a string or iterable"
            ) from exc

    normalized = []
    for part in parts:
        if not isinstance(part, str):
            raise TypeError("punctuation split marks must be strings")
        part = part.strip()
        if not part:
            continue
        preset = _PUNCTUATION_SPLIT_MARK_PRESETS.get(part.casefold())
        values = preset if preset is not None else (part,)
        for value in values:
            if value not in _SUPPORTED_PUNCTUATION_SPLIT_MARKS:
                raise ValueError(f"Unsupported punctuation split mark: {value!r}")
            if value not in normalized:
                normalized.append(value)

    order = {value: index for index, value in enumerate(normalized)}
    return tuple(sorted(normalized, key=lambda value: (-len(value), order[value])))


def _normalized_words(text):
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _last_strong_punctuation_index(text, before_index):
    return max(text.rfind(".", 0, before_index), text.rfind("?", 0, before_index), text.rfind("!", 0, before_index))


def _is_digit_punctuation(text, index):
    left = index - 1
    right = index + 1
    while left >= 0 and text[left].isspace():
        left -= 1
    while right < len(text) and text[right].isspace():
        right += 1
    return left >= 0 and right < len(text) and text[left].isdigit() and text[right].isdigit()


def _is_compound_hyphen(text, index):
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isalnum()
        and text[index + 1].isalnum()
    )


def _has_abbreviation_period(text):
    folded = (text or "").casefold()
    return any(
        re.search(rf"\b{re.escape(abbreviation)}\.\s", folded)
        for abbreviation in _ABBREVIATIONS_BEFORE_PERIOD
    )


def _iter_punctuation_split_candidates(text, marks):
    text = text or ""
    stripped_length = len(text.rstrip())
    for index in range(len(text)):
        for punctuation in marks:
            if not text.startswith(punctuation, index):
                continue
            end_index = index + len(punctuation)
            if end_index >= stripped_length:
                continue
            yield index, punctuation, end_index
            break


def _punctuation_split_hint(text, marks="sentence,comma"):
    marks = _normalize_realtime_punctuation_split_marks(marks)
    if not marks:
        return None

    candidates = []
    for index, punctuation, end_index in _iter_punctuation_split_candidates(text, marks):
        if punctuation in {".", ",", "-"} and _is_digit_punctuation(text, index):
            continue
        if punctuation == "-" and _is_compound_hyphen(text, index):
            continue

        before_words = _normalized_words(text[:index])
        after_words = _normalized_words(text[end_index:])
        if not before_words or len(after_words) < 2:
            continue

        boundary = _last_strong_punctuation_index(text, index) + 1
        segment_before = text[boundary:index]
        segment_words = _normalized_words(segment_before)
        if punctuation == "." and before_words[-1] in _ABBREVIATIONS_BEFORE_PERIOD:
            continue
        if punctuation in ".?!" and len(segment_words) < 3:
            continue
        if punctuation == "..." and len(segment_words) < 2:
            continue
        if punctuation in {"-", "\u2014", "\u2013"} and len(segment_words) < 3:
            continue
        if punctuation == ",":
            if len(segment_words) < 4 or "," in segment_before:
                continue
            if _has_abbreviation_period(text[:index]):
                continue
            if (
                after_words[0] in {"and", "or"}
                and after_words[1] not in {"a", "an", "the", "this", "that", "these", "those", "it", "there", "we", "i"}
            ):
                continue

        candidates.append((punctuation, tuple(before_words[-3:])))

    return candidates[-1] if candidates else None


def _select_realtime_punctuation_split_hint(event, marks="sentence,comma"):
    for name in ("stable_text", "consensus_display_text", "display_text", "raw_observation_text"):
        text = getattr(event, name, "") or ""
        if _punctuation_split_hint(text, marks):
            return text
    return ""


def _find_word_end_for_suffix(words, suffix):
    normalized = [_normalized_words(str(word.get("word", ""))) for word in words]
    flat = [items[-1] if items else "" for items in normalized]
    suffix = tuple(word for word in suffix if word)
    if not suffix:
        return None

    for size in range(min(3, len(suffix)), 0, -1):
        needle = suffix[-size:]
        for index in range(len(flat) - size, -1, -1):
            if tuple(flat[index:index + size]) == needle:
                end_time = words[index + size - 1].get("end")
                if end_time is None:
                    return None
                end_time = float(end_time)
                if index + size < len(words):
                    next_start = words[index + size].get("start")
                    if next_start is not None and float(next_start) > end_time:
                        return (end_time + float(next_start)) / 2
                return end_time
    return None


def _find_punctuation_split(
    transcription_result,
    hint_text=None,
    marks="sentence,comma",
):
    metadata = getattr(transcription_result, "metadata", None) or {}
    words = metadata.get("words") or []
    if not words:
        return None

    hint = _punctuation_split_hint(hint_text, marks)
    if hint:
        punctuation, suffix = hint
        end_time = _find_word_end_for_suffix(words, suffix)
        if end_time is not None:
            return punctuation, end_time

    text = "".join(str(word.get("word", "")) for word in words)
    hint = _punctuation_split_hint(text, marks)
    if not hint:
        return None

    punctuation, suffix = hint
    end_time = _find_word_end_for_suffix(words, suffix)
    if end_time is None:
        return None
    return punctuation, end_time


def _get_realtime_punctuation_split_lock(recorder):
    lock = getattr(recorder, "_realtime_punctuation_split_lock", None)
    if lock is None:
        lock = threading.RLock()
        recorder._realtime_punctuation_split_lock = lock
    return lock


def _clear_realtime_punctuation_split_candidate(recorder):
    with _get_realtime_punctuation_split_lock(recorder):
        recorder._realtime_punctuation_split_candidate = None


def _confirm_realtime_punctuation_split_candidate(recorder, split_marks, hint):
    key = (split_marks, hint)
    with _get_realtime_punctuation_split_lock(recorder):
        candidate = getattr(recorder, "_realtime_punctuation_split_candidate", None)
        if candidate and candidate[0] == key:
            count = candidate[1] + 1
        else:
            count = 1

        recorder._realtime_punctuation_split_candidate = (key, count)
        return count >= _PUNCTUATION_SPLIT_REQUIRED_OBSERVATIONS


def run_realtime_worker(recorder):
    """
    Runs realtime transcription when the feature is enabled.

    The worker skips empty buffers, snapshots frame buffers before
    transcription, logs model or pipe errors, and never stops the recorder.
    """

    self = recorder

    logger.debug("Starting realtime worker")

    if not self.enable_realtime_transcription:
        logger.debug("Realtime transcription disabled; realtime worker exits")
        return

    def _sleep_briefly():
        """
        Sleeps for the realtime worker polling interval.
        """

        time.sleep(0.001)

    def _safe_get_realtime_pause():
        """
        Returns the configured realtime processing pause.
        """

        pause = getattr(self, "realtime_processing_pause", 0.2)
        try:
            return max(0.001, float(pause))
        except Exception:
            return 0.2

    def _safe_get_realtime_fallback_pause():
        """
        Returns the fallback realtime processing pause.
        """

        pause = getattr(self, "realtime_processing_pause", 0.2)
        try:
            return float(pause)
        except Exception:
            return 0.2

    def _safe_get_sample_rate():
        """
        Returns the recorder sample rate with a safe fallback.
        """

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
        """
        Copies buffered realtime frames under the recorder lock.
        """

        try:
            frames = snapshot_frames(self)
            return frames if frames else None

        except Exception as e:
            logger.debug(f"Could not snapshot realtime frames: {e}", exc_info=True)
            return None

    def _frames_to_audio_array(frames_snapshot, enforce_min_samples=True):
        """
        Converts captured frames into a float audio array.
        """

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
        """
        Counts samples in a frame snapshot.
        """

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
        """
        Extracts text and language metadata from a transcription result.
        """

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

    def _callable_accepts_word_timestamps(callback):
        """
        Returns whether an external executor accepts word timestamp requests.
        """

        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return False

        for parameter in signature.parameters.values():
            if parameter.kind == parameter.VAR_KEYWORD:
                return True
            if parameter.name == "word_timestamps":
                return True
        return False

    def _main_transcription_supports_word_timestamps():
        """
        Returns whether the main transcription path can return word timings.
        """

        if self._uses_external_transcription_executor:
            executor = getattr(self, "transcription_executor", None)
            callback = getattr(executor, "transcribe", executor)
            return _callable_accepts_word_timestamps(callback)

        engine_name = (
            getattr(self, "transcription_engine", None)
            or "faster_whisper"
        )
        engine_name = str(engine_name).strip().lower().replace("-", "_")
        return engine_name == "faster_whisper"

    def _log_word_timestamp_skip_once():
        """
        Logs one debug message when punctuation splitting lacks word timings.
        """

        if getattr(self, "_realtime_word_timestamp_skip_logged", False):
            return
        self._realtime_word_timestamp_skip_logged = True
        logger.debug(
            "Skipping realtime punctuation split because the main "
            "transcription engine does not expose word timestamps"
        )

    def _transcribe_with_main_model(audio_array):
        """
        Runs realtime transcription through the main model.
        """

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

    def _transcribe_with_main_model_word_timestamps(audio_array):
        """
        Runs final-model transcription with word timestamps.
        """

        try:
            if not _main_transcription_supports_word_timestamps():
                _log_word_timestamp_skip_once()
                return None

            if self._uses_external_transcription_executor:
                return call_transcription_executor(
                    self.transcription_executor,
                    audio_array,
                    self.language,
                    True,
                    word_timestamps=True,
                )

            if not self.transcription_lock.acquire(blocking=False):
                logger.debug("Skipping realtime punctuation split because final transcription is busy")
                return None
            try:
                self.parent_transcription_pipe.send(
                    (audio_array, self.language, True, {"word_timestamps": True})
                )

                if not self.parent_transcription_pipe.poll(timeout=10):
                    logger.warning("Realtime punctuation split transcription timed out")
                    return None

                status, result = self.parent_transcription_pipe.recv()

                if status != "success":
                    logger.error(f"Realtime punctuation split transcription error: {result}")
                    return None

                return result
            finally:
                self.transcription_lock.release()

        except Exception as e:
            logger.error(f"Error in realtime punctuation split transcription: {e}", exc_info=True)
            return None

    def _transcribe_with_realtime_model(audio_array):
        """
        Runs realtime transcription through the realtime model.
        """

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
        """
        Selects the active realtime streaming model target.
        """

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
        """
        Closes the active realtime streaming session.
        """

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
        """
        Creates a realtime streaming session for a target model.
        """

        try:
            return target.create_streaming_session(
                language=self.language if self.language else None,
                use_prompt=True,
            )
        except TypeError:
            return target.create_streaming_session()

    def _ensure_streaming_session(recording_id):
        """
        Ensures a streaming session exists for the recording.
        """

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
                    previous_frames = snapshot_frames(self, "last_frames")
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
        """
        Finishes the active realtime streaming session.
        """

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
        """
        Runs realtime transcription through the streaming model.
        """

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
        """
        Invokes a realtime callback without breaking the worker.
        """

        try:
            run_callback(self, callback, *args)
        except Exception as e:
            logger.error(f"Realtime callback failed: {e}", exc_info=True)

    def _lowercase_first_text(text):
        """
        Lowercases the first recognized character.
        """

        return text[:1].lower() + text[1:] if text else text

    def _split_frames_at_sample(frames, split_sample):
        """
        Splits 16-bit PCM frame bytes at a sample offset.
        """

        left, right = [], []
        remaining = int(split_sample)
        for frame in frames:
            samples = len(frame) // 2
            if remaining >= samples:
                left.append(frame)
                remaining -= samples
            elif remaining <= 0:
                right.append(frame)
            else:
                byte_index = max(0, min(len(frame), remaining * 2))
                left.append(frame[:byte_index])
                right.append(frame[byte_index:])
                remaining = 0
        return left, right

    def _reset_after_punctuation_split(right_frames, punctuation, sample_rate):
        """
        Starts a new realtime segment from the right-side audio remainder.
        """

        remaining_seconds = _count_frame_samples(right_frames) / float(sample_rate)
        with get_frames_lock(self):
            self.frames = list(right_frames)
            self.last_frames = []
        self.text_storage = []
        self.realtime_transcription_text = ""
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.realtime_observation_sequence = 0
        self.realtime_recording_id = getattr(self, "realtime_recording_id", 0) + 1
        self.recording_start_monotonic = time.monotonic() - remaining_seconds
        self.recording_start_time = time.time() - remaining_seconds
        self._force_current_recording_lowercase_start = punctuation == ","
        self._last_realtime_punctuation_split_attempt_text = ""
        _clear_realtime_punctuation_split_candidate(self)
        self.realtime_text_stabilizer.reset(
            self.realtime_recording_id,
            started_at_monotonic=self.recording_start_monotonic,
            started_at_wall_time=self.recording_start_time,
        )
        _close_streaming_session()

    def _maybe_split_on_stable_punctuation(event, frames_snapshot, sample_rate):
        """
        Splits the active recording when stable punctuation has a timestamp.
        """

        split_marks = _normalize_realtime_punctuation_split_marks(
            getattr(self, "realtime_punctuation_split_marks", "off")
        )
        if not split_marks:
            _clear_realtime_punctuation_split_candidate(self)
            return False
        hint_text = _select_realtime_punctuation_split_hint(event, split_marks)
        if not hint_text:
            _clear_realtime_punctuation_split_candidate(self)
            return False
        hint = _punctuation_split_hint(hint_text, split_marks)
        if not hint:
            _clear_realtime_punctuation_split_candidate(self)
            return False
        sample_count = _count_frame_samples(frames_snapshot)
        if sample_count < sample_rate * 2:
            return False
        attempt_key = (split_marks, hint, sample_count // sample_rate)
        with _get_realtime_punctuation_split_lock(self):
            if getattr(self, "_realtime_punctuation_split_busy", False):
                return False
            if not _confirm_realtime_punctuation_split_candidate(
                self,
                split_marks,
                hint,
            ):
                return False
            if attempt_key == getattr(
                self,
                "_last_realtime_punctuation_split_attempt_text",
                "",
            ):
                return False
            self._last_realtime_punctuation_split_attempt_text = attempt_key
            self._realtime_punctuation_split_busy = True

        def split_in_background():
            try:
                audio_array = _frames_to_audio_array(frames_snapshot)
                if audio_array is None:
                    return
                result = _transcribe_with_main_model_word_timestamps(audio_array)
                split = _find_punctuation_split(result, hint_text, split_marks)
                if split is None:
                    return

                punctuation, split_time = split
                split_sample = int(split_time * sample_rate)
                if split_sample <= 0:
                    return

                current_frames = list(_snapshot_frames() or frames_snapshot)
                current_sample_count = _count_frame_samples(current_frames)
                if split_sample >= current_sample_count:
                    return
                left_frames, right_frames = _split_frames_at_sample(current_frames, split_sample)
                if not left_frames or not right_frames:
                    return

                queue_recorded_audio(
                    self,
                    left_frames,
                    force_lowercase_start=getattr(
                        self,
                        "_force_current_recording_lowercase_start",
                        False,
                    ),
                )
                _reset_after_punctuation_split(right_frames, punctuation, sample_rate)
            finally:
                with _get_realtime_punctuation_split_lock(self):
                    self._realtime_punctuation_split_busy = False

        threading.Thread(
            target=split_in_background,
            daemon=True,
            name="RealtimeSTTPunctuationSplit",
        ).start()
        return True

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
        frames_snapshot,
    ):
        """
        Publishes realtime text with timing and language metadata.
        """

        raw_text = "" if realtime_text is None else str(realtime_text)
        force_lowercase_start = getattr(
            self,
            "_force_current_recording_lowercase_start",
            False,
        )
        if force_lowercase_start:
            raw_text = _lowercase_first_text(raw_text)

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

        if event.accepted and _maybe_split_on_stable_punctuation(
            event,
            frames_snapshot,
            sample_rate,
        ):
            return

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
            publish_realtime_transcription_stabilized,
            self,
            preprocess_output(
                stabilized_display_text,
                preview=True,
                ensure_sentence_starting_uppercase=(
                    self.ensure_sentence_starting_uppercase
                    and not force_lowercase_start
                ),
                ensure_sentence_ends_with_period=(
                    self.ensure_sentence_ends_with_period
                ),
            ),
        )

        _safe_realtime_callback(
            publish_realtime_transcription_update,
            self,
            preprocess_output(
                raw_text.strip(),
                preview=True,
                ensure_sentence_starting_uppercase=(
                    self.ensure_sentence_starting_uppercase
                    and not force_lowercase_start
                ),
                ensure_sentence_ends_with_period=(
                    self.ensure_sentence_ends_with_period
                ),
            ),
        )

    last_transcription_time = time.time()

    def _run_realtime_transcription(trigger_reason):
        """
        Runs one realtime transcription pass for buffered audio.
        """

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
                frames_snapshot,
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
            frames_snapshot,
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
        """
        Returns follow-up delays for syllable boundary checks.
        """

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
        """
        Resets realtime boundary scheduling state.
        """

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
        """
        Processes newly captured frames for boundary detection.
        """

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
        """
        Runs follow-up realtime passes at syllable boundaries.
        """

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
                        finished_frames = snapshot_frames(self, "last_frames")
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
