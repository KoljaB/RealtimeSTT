"""
Internal recorder-level voice activity and pre-roll helpers.
"""

import logging
import threading
import time

import numpy as np
from scipy import signal
import webrtcvad

from ..preroll import PrerollFrameMetadata, select_preroll_frames


logger = logging.getLogger("realtimestt")

SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0


class _BColors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def silero_vad_probability(recorder, audio_chunk):
    result = recorder.silero_vad_model(audio_chunk, SAMPLE_RATE)
    if isinstance(result, (float, int)):
        return float(result)
    if hasattr(result, "item"):
        return float(result.item())
    return float(np.asarray(result).reshape(-1)[0])


def reset_silero_vad_state(recorder):
    """
    Reset Silero's recurrent state and the recorder-side Silero flag.

    Silero VAD keeps hidden state between chunk calls. That is useful while
    evaluating one continuous stream, but it must not leak across warmup,
    listening attempts, or completed recordings.
    """
    recorder._silero_vad_generation = (
        getattr(recorder, "_silero_vad_generation", 0) + 1
    )
    reset_states = getattr(
        getattr(recorder, "silero_vad_model", None),
        "reset_states",
        None,
    )
    if reset_states:
        try:
            lock = getattr(recorder, "silero_vad_lock", None)
            if lock is None:
                reset_states()
            else:
                with lock:
                    reset_states()
        except Exception:
            logger.debug("Silero VAD state reset skipped", exc_info=True)
    recorder.is_silero_speech_active = False


def warmup_voice_activity_detectors(recorder):
    """
    Prime VAD runtimes without changing recorder state.

    The first Silero invocation can otherwise pay lazy Torch/JIT setup
    costs on the first user speech chunk. That delays voice activation and
    therefore delays the first realtime transcription, even when the ASR
    model workers are already warmed.
    """
    try:
        frame_samples = int(16000 * 0.01)
        silence_frame = np.zeros(frame_samples, dtype=np.int16).tobytes()
        recorder.webrtc_vad_model.is_speech(silence_frame, 16000)
    except Exception:
        logger.debug("WebRTC VAD warmup skipped", exc_info=True)

    try:
        sample_count = max(1, int(getattr(recorder, "buffer_size", BUFFER_SIZE)))
        t = np.arange(sample_count, dtype=np.float32) / float(SAMPLE_RATE)
        # A quiet tone exercises the model path without marking the
        # recorder as actively recording or speech-active.
        tone = (0.03 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        silence = np.zeros(sample_count, dtype=np.float32)

        silero_vad_probability(recorder, tone)
        silero_vad_probability(recorder, silence)

        reset_silero_vad_state(recorder)
    except Exception:
        logger.debug("Silero VAD warmup skipped", exc_info=True)

    recorder.silero_working = False
    recorder.is_silero_speech_active = False
    recorder.is_webrtc_speech_active = False
    recorder.last_webrtc_speech_time = 0


def is_silero_speech(recorder, chunk, generation=None):
    """
    Returns true if speech is detected in the provided audio data

    Args:
        data (bytes): raw bytes of audio data (1024 raw bytes with
        16000 sample rate and 16 bits per sample)
    """
    if generation is None:
        generation = getattr(recorder, "_silero_vad_generation", 0)

    recorder.silero_working = True
    try:
        if generation != getattr(recorder, "_silero_vad_generation", 0):
            return False

        if recorder.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, recorder.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        if generation != getattr(recorder, "_silero_vad_generation", 0):
            return False

        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        lock = getattr(recorder, "silero_vad_lock", None)
        if lock is None:
            vad_prob = silero_vad_probability(recorder, audio_chunk)
        else:
            with lock:
                if generation != getattr(recorder, "_silero_vad_generation", 0):
                    return False
                vad_prob = silero_vad_probability(recorder, audio_chunk)

        if generation != getattr(recorder, "_silero_vad_generation", 0):
            return False

        is_silero_speech_active = vad_prob > (1 - recorder.silero_sensitivity)
        if is_silero_speech_active:
            if not recorder.is_silero_speech_active and recorder.use_extended_logging:
                logger.info(f"{_BColors.OKGREEN}Silero VAD detected speech{_BColors.ENDC}")
        elif recorder.is_silero_speech_active and recorder.use_extended_logging:
            logger.info(f"{_BColors.WARNING}Silero VAD detected silence{_BColors.ENDC}")
        recorder.is_silero_speech_active = is_silero_speech_active
        return is_silero_speech_active
    finally:
        recorder.silero_working = False


def is_webrtc_speech(recorder, chunk, all_frames_must_be_true=False):
    """
    Returns true if speech is detected in the provided audio data

    Args:
        data (bytes): raw bytes of audio data (1024 raw bytes with
        16000 sample rate and 16 bits per sample)
    """
    speech_str = f"{_BColors.OKGREEN}WebRTC VAD detected speech{_BColors.ENDC}"
    silence_str = f"{_BColors.WARNING}WebRTC VAD detected silence{_BColors.ENDC}"
    if recorder.sample_rate != 16000:
        pcm_data = np.frombuffer(chunk, dtype=np.int16)
        data_16000 = signal.resample_poly(
            pcm_data, 16000, recorder.sample_rate)
        chunk = data_16000.astype(np.int16).tobytes()

    # Number of audio frames per millisecond
    frame_length = int(16000 * 0.01)  # for 10ms frame
    num_frames = int(len(chunk) / (2 * frame_length))
    speech_frames = 0

    for i in range(num_frames):
        start_byte = i * frame_length * 2
        end_byte = start_byte + frame_length * 2
        frame = chunk[start_byte:end_byte]
        if recorder.webrtc_vad_model.is_speech(frame, 16000):
            speech_frames += 1
            if not all_frames_must_be_true:
                if recorder.debug_mode:
                    logger.info(f"Speech detected in frame {i + 1}"
                          f" of {num_frames}")
                if not recorder.is_webrtc_speech_active and recorder.use_extended_logging:
                    logger.info(speech_str)
                recorder.is_webrtc_speech_active = True
                recorder.last_webrtc_speech_time = time.time()
                return True
    if all_frames_must_be_true:
        if recorder.debug_mode and speech_frames == num_frames:
            logger.info(f"Speech detected in {speech_frames} of "
                  f"{num_frames} frames")
        elif recorder.debug_mode:
            logger.info(f"Speech not detected in all {num_frames} frames")
        speech_detected = speech_frames == num_frames
        if speech_detected and not recorder.is_webrtc_speech_active and recorder.use_extended_logging:
            logger.info(speech_str)
        elif not speech_detected and recorder.is_webrtc_speech_active and recorder.use_extended_logging:
            logger.info(silence_str)
        recorder.is_webrtc_speech_active = speech_detected
        return speech_detected
    else:
        if recorder.debug_mode:
            logger.info(f"Speech not detected in any of {num_frames} frames")
        if recorder.is_webrtc_speech_active and recorder.use_extended_logging:
            logger.info(silence_str)
        recorder.is_webrtc_speech_active = False
        return False


def check_voice_activity(recorder, data, thread_factory=None):
    """
    Initiate check if voice is active based on the provided data.

    Args:
        data: The audio data to be checked for voice activity.
    """
    if thread_factory is None:
        thread_factory = threading.Thread

    was_webrtc_speech_active = recorder.is_webrtc_speech_active
    recorder._is_webrtc_speech(data)

    # First quick performing check for voice activity using WebRTC
    if recorder.is_webrtc_speech_active:

        if not recorder.silero_working:
            if not was_webrtc_speech_active:
                reset_silero_vad_state(recorder)
            recorder.silero_working = True
            silero_generation = getattr(recorder, "_silero_vad_generation", 0)

            # Run the intensive check in a separate thread
            thread_factory(
                target=recorder._is_silero_speech,
                args=(data, silero_generation)).start()


def pre_recording_buffer_trim_enabled(recorder):
    config = getattr(recorder, "pre_recording_buffer_trim_config", None) or {}
    return bool(config.get("enabled", False))


def append_to_pre_recording_buffer(recorder, data):
    recorder.audio_buffer.append(data)
    metadata_buffer = getattr(recorder, "audio_buffer_metadata", None)
    if metadata_buffer is not None:
        metadata_buffer.append(preroll_frame_metadata(recorder, data))


def clear_pre_recording_buffer(recorder):
    recorder.audio_buffer.clear()
    metadata_buffer = getattr(recorder, "audio_buffer_metadata", None)
    if metadata_buffer is not None:
        metadata_buffer.clear()


def selected_pre_recording_buffer_frames(recorder):
    frames = list(recorder.audio_buffer)
    recorder._pending_preroll_selection = None
    if not frames:
        return frames

    if not pre_recording_buffer_trim_enabled(recorder):
        return frames

    metadata = list(getattr(recorder, "audio_buffer_metadata", ()))
    if len(metadata) != len(frames):
        metadata = [metadata_for_frame_without_vad(recorder, frame) for frame in frames]

    config = getattr(recorder, "pre_recording_buffer_trim_config", None) or {}
    selection = select_preroll_frames(
        metadata,
        int(getattr(recorder, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE),
        min_silence_ms=config.get("min_silence_ms", 200.0),
        guard_ms=config.get("guard_ms", 160.0),
        max_gap_ms=config.get("max_gap_ms", 80.0),
        min_included_ms=config.get("min_included_ms", 600.0),
        energy_silence_rms=config.get("energy_silence_rms"),
        noise_floor_multiplier=config.get("noise_floor_multiplier", 2.5),
        energy_margin_rms=config.get("energy_margin_rms", 25.0),
    )
    selection.diagnostics.update(webrtc_replay_preroll_diagnostics(recorder, frames))
    recorder._pending_preroll_selection = selection
    return frames[selection.start_index:]


def preroll_frame_metadata(recorder, data):
    sample_count = max(0, len(data) // 2)
    rms = frame_rms(data)
    webrtc_is_speech = bool(getattr(recorder, "is_webrtc_speech_active", False))
    silero_is_speech = bool(getattr(recorder, "is_silero_speech_active", False))
    is_speech = webrtc_is_speech or silero_is_speech
    return PrerollFrameMetadata(
        sample_count=sample_count,
        is_speech=is_speech,
        rms=rms,
        webrtc_is_speech=webrtc_is_speech,
        silero_is_speech=silero_is_speech,
    )


def metadata_for_frame_without_vad(recorder, frame):
    return PrerollFrameMetadata(
        sample_count=max(0, len(frame) // 2),
        is_speech=None,
        rms=frame_rms(frame),
    )


def frame_rms(data):
    if not data:
        return None
    try:
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return None
        audio = samples.astype(np.float32)
        return float(np.sqrt(np.mean(audio * audio)))
    except Exception:
        logger.debug("Could not calculate pre-roll frame RMS", exc_info=True)
        return None


def webrtc_replay_preroll_diagnostics(recorder, frames):
    speech_sample_count = 0
    analyzed_sample_count = 0
    frame_length = int(16000 * 0.01)
    sample_rate = int(getattr(recorder, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)

    try:
        vad_model = webrtcvad.Vad(int(getattr(recorder, "webrtc_sensitivity", 3)))
        for chunk in list(frames or ()):
            replay_chunk = chunk
            if sample_rate != 16000:
                pcm_data = np.frombuffer(replay_chunk, dtype=np.int16)
                data_16000 = signal.resample_poly(
                    pcm_data,
                    16000,
                    sample_rate,
                )
                replay_chunk = data_16000.astype(np.int16).tobytes()

            num_frames = int(len(replay_chunk) / (2 * frame_length))
            for index in range(num_frames):
                start_byte = index * frame_length * 2
                end_byte = start_byte + frame_length * 2
                frame = replay_chunk[start_byte:end_byte]
                analyzed_sample_count += frame_length
                if vad_model.is_speech(frame, 16000):
                    speech_sample_count += frame_length
    except Exception as exc:
        logger.debug("Could not replay WebRTC VAD over pre-roll", exc_info=True)
        return {"webrtcReplayError": str(exc)}

    return {
        "webrtcReplaySpeechSampleCount": speech_sample_count,
        "webrtcReplayAnalyzedSampleCount": analyzed_sample_count,
        "webrtcReplaySpeechSeconds": speech_sample_count / 16000.0,
        "webrtcReplayAnalyzedSeconds": analyzed_sample_count / 16000.0,
    }


def is_voice_active(recorder):
    """
    Determine if voice is active.

    Returns:
        bool: True if voice is active, False otherwise.
    """
    webrtc_speech_recent = (
        time.time() - getattr(recorder, "last_webrtc_speech_time", 0) <= 1.0
    )
    return (
        (recorder.is_webrtc_speech_active or webrtc_speech_recent)
        and recorder.is_silero_speech_active
    )
