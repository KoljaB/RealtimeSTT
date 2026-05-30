"""Public recorder facade for RealtimeSTT.

This module preserves the stable recorder API while delegating subsystem
implementation details to ``RealtimeSTT.core``.
"""

# Standard library imports.
import base64
import copy
import gc
import logging
import os
import platform
import threading
import time
from typing import Callable, Iterable, List, Optional, Union

# Third-party imports.
import torch.multiprocessing as mp
from scipy.signal import resample
import numpy as np

# Internal imports.
from .core.audio_input_worker import run_audio_data_worker
from .core.realtime_text_stabilizer import RealtimeTextStabilizer
from .core.initialization import initialize_recorder
from .core.recorder_config import build_recorder_init_args
from .core.realtime import run_realtime_worker
from .core.recording import run_recording_worker
from .core.recording_buffers import (
    clear_audio_queue as clear_recorder_audio_queue,
    flush_buffered_audio as flush_recorder_buffered_audio,
    get_next_recorded_audio,
    has_pending_recordings as recorder_has_pending_recordings,
    queue_recorded_audio,
    set_audio_from_frames,
)
from .core.runtime import read_stdout_pipe
from .core.state import run_callback, set_recorder_state
from .core.state import set_spinner
from .core.text_formatting import (
    find_tail_match_in_text,
    format_number,
    preprocess_output,
)
from .core.transcription import (
    TranscriptionWorker,
    receive_transcription_result,
    submit_transcription_request,
)
from .core.wakeword import (
    _load_openwakeword_modules,
    _load_porcupine_module,
    _normalize_wakeword_backend,
)
from .core.voice_activity import (
    check_voice_activity,
    is_silero_speech,
    is_voice_active,
    is_webrtc_speech,
    reset_silero_vad_state,
    selected_pre_recording_buffer_frames,
)


# Compatibility environment setup.
#
# Some downstream combinations of Torch, OpenMP, and audio/model runtimes rely on
# this import-time setting. Keep it early and behavior-compatible; it is not a
# general runtime recommendation for application code.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Logger setup.
logger = logging.getLogger("realtimestt")
logger.propagate = False


# Public constructor defaults and compatibility constants.
INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_REALTIME_INITIAL_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 100
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
DEACTIVITY_SILENCE_CONFIRMATION_DURATION = 0.16

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True


# Internal recorder constants.
TIME_SLEEP = 0.02
INT16_MAX_ABS_VALUE = 32768.0


# Console color constants used by debug speech detection output.
class bcolors:
    OKGREEN = '\033[92m'  # Green for active speech detection
    WARNING = '\033[93m'  # Yellow for silence detection
    ENDC = '\033[0m'      # Reset to default color


class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    configured transcription engine.
    """

    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 transcription_engine: str = "faster_whisper",
                 transcription_engine_options: Optional[dict] = None,
                 download_root: str = None, 
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 level=logging.WARNING,
                 batch_size: int = 16,

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 use_main_model_for_realtime=False,
                 realtime_transcription_engine: str = None,
                 realtime_transcription_engine_options: Optional[dict] = None,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 init_realtime_after_seconds=INIT_REALTIME_INITIAL_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,
                 realtime_batch_size: int = 16,

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: Optional[bool] = None,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 warmup_vad: bool = True,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                     INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 pre_recording_buffer_trim_config: Optional[dict] = None,
                 on_vad_start=None,
                 on_vad_stop=None,
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,
                 on_turn_detection_start=None,
                 on_turn_detection_stop=None,

                 # Wake word parameters
                 wakeword_backend: str = "",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = (
                    INIT_WAKE_WORD_ACTIVATION_DELAY
                 ),
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
                 initial_prompt_realtime: Optional[Union[str, Iterable[int]]] = None,
                 suppress_tokens: Optional[List[int]] = [-1],
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logging: bool = False,
                 faster_whisper_vad_filter: bool = True,
                 normalize_audio: bool = False,
                 start_callback_in_new_thread: bool = False,
                 realtime_transcription_use_syllable_boundaries: bool = False,
                 realtime_boundary_detector_sensitivity: float = 0.6,
                 realtime_boundary_followup_delays: Optional[Iterable[float]] = (
                     0.05,
                     0.2,
                 ),
                 transcription_executor: Optional[Callable] = None,
                 realtime_transcription_executor: Optional[Callable] = None,
                 on_realtime_text_stabilization_update=None,
                 silero_backend: str = "auto",
                 silero_onnx_model_path: Optional[str] = None,
                 silero_onnx_threads: int = 2,
                 deactivity_silence_confirmation_duration: float = (
                     DEACTIVITY_SILENCE_CONFIRMATION_DURATION
                 ),
                 ):
        """
        Initializes an audio recorder and  transcription
        and wake word detection.

        Args:
        Main transcription and output:
        - model (str, default="tiny"): Specifies the size of the transcription
            model to use or the path to a converted model directory.
            Valid options are 'tiny', 'tiny.en', 'base', 'base.en',
            'small', 'small.en', 'medium', 'medium.en', 'large-v1',
            'large-v2'.
            If a specific size is provided, the model is downloaded
            from the Hugging Face Hub.
        - transcription_engine (str, default="faster_whisper"): Transcription
            backend to use for the main model.
        - transcription_engine_options (dict, default=None): Optional
            backend-specific options for the main transcription engine.
        - download_root (str, default=None): Specifies the root path were the Whisper models 
          are downloaded to. When empty, the default is used. 
        - language (str, default=""): Language code for speech-to-text engine.
            If not specified, the model will attempt to detect the language
            automatically.
        - compute_type (str, default="default"): Specifies the type of
            computation to be used for transcription.
            See https://opennmt.net/CTranslate2/quantization.html.
        - input_device_index (int, default=0): The index of the audio input
            device to use.
        - gpu_device_index (int, default=0): Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of
            IDs (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can
            run in parallel when transcribe() is called from multiple Python
            threads
        - device (str, default="cuda"): Device for model to use. Can either be 
            "cuda" or "cpu".
        - on_recording_start (callable, default=None): Callback function to be
            called when recording of audio to be transcripted starts.
        - on_recording_stop (callable, default=None): Callback function to be
            called when recording of audio to be transcripted stops.
        - on_transcription_start (callable, default=None): Callback function
            to be called when transcription of audio to text starts.
        - ensure_sentence_starting_uppercase (bool, default=True): Ensures
            that every sentence detected by the algorithm starts with an
            uppercase letter.
        - ensure_sentence_ends_with_period (bool, default=True): Ensures that
            every sentence that doesn't end with punctuation such as "?", "!"
            ends with a period
        - use_microphone (bool, default=True): Specifies whether to use the
            microphone as the audio input source. If set to False, the
            audio input source will be the audio data sent through the
            feed_audio() method.
        - spinner (bool, default=True): Show spinner animation with current
            state.
        - level (int, default=logging.WARNING): Logging level.
        - batch_size (int, default=16): Batch size for the main transcription

        Realtime transcription:
        - enable_realtime_transcription (bool, default=False): Enables or
            disables real-time transcription of audio. When set to True, the
            audio will be transcribed continuously as it is being recorded.
        - use_main_model_for_realtime (str, default=False):
            If True, use the main transcription model for both regular and
            real-time transcription. If False, use a separate model specified
            by realtime_model_type for real-time transcription.
            Using a single model can save memory and potentially improve
            performance, but may not be optimized for real-time processing.
            Using separate models allows for a smaller, faster model for
            real-time transcription while keeping a more accurate model for
            final transcription.
        - realtime_transcription_engine (str, default=None): Backend to use for
            the realtime model. If None, the main transcription engine is used.
        - realtime_transcription_engine_options (dict, default=None): Optional
            backend-specific options for the realtime transcription engine. If
            None, transcription_engine_options is reused.
        - realtime_model_type (str, default="tiny"): Specifies the machine
            learning model to be used for real-time transcription. Valid
            options include 'tiny', 'tiny.en', 'base', 'base.en', 'small',
            'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
        - realtime_processing_pause (float, default=0.1): Specifies the time
            interval in seconds after a chunk of audio gets transcribed. Lower
            values will result in more "real-time" (frequent) transcription
            updates but may increase computational load. When
            realtime_transcription_use_syllable_boundaries is True, this is
            used as a forced fallback interval if no syllable boundary is
            detected. Set to 0 or lower to disable the fallback interval.
        - init_realtime_after_seconds (float, default=0.2): Specifies the 
            initial waiting time after the recording was initiated before
            yielding the first realtime transcription
        - on_realtime_transcription_update = A callback function that is
            triggered whenever there's an update in the real-time
            transcription. The function is called with the newly transcribed
            text as its argument.
        - on_realtime_transcription_stabilized = A callback function that is
            triggered when the transcribed text stabilizes in quality. The
            stabilized text is generally more accurate but may arrive with a
            slight delay compared to the regular real-time updates.
        - on_realtime_text_stabilization_update = A callback function that is
            triggered with a structured realtime stabilization event.
        - realtime_batch_size (int, default=16): Batch size for the real-time
            transcription model.

        Voice activity and turn detection:
        - silero_sensitivity (float, default=SILERO_SENSITIVITY): Sensitivity
            for the Silero Voice Activity Detection model ranging from 0
            (least sensitive) to 1 (most sensitive). Default is 0.5.
        - silero_use_onnx (bool, default=None): Legacy Silero backend switch.
            If True, keeps the previous torch.hub ONNX path. If False, keeps
            the previous torch.hub PyTorch path. If None, silero_backend
            controls backend selection and defaults to the fastest accurate
            option available.
        - silero_backend (str, default="auto"): Silero VAD runtime backend.
            "auto" prefers raw CPU ONNX Runtime with
            silero_vad_op18_ifless.onnx, then raw silero_vad.onnx, then
            PyTorch CPU. Other options include "legacy", "pytorch_cpu",
            "pytorch_cuda", "official_onnx", "raw_onnx", and
            "raw_onnx_ifless". CUDA remains available but is not automatic,
            because launch overhead is slower than CPU ONNX for single-stream
            32 ms chunks.
        - silero_onnx_model_path (str, default=None): Optional explicit ONNX
            model path for "official_onnx", "raw_onnx", or
            "raw_onnx_ifless" backends.
        - silero_onnx_threads (int, default=2): ONNX Runtime intra-op thread
            count for raw ONNX Silero backends. Inter-op threads stay at 1.
        - silero_deactivity_detection (bool, default=False): Enables the Silero
            model for end-of-speech detection. More robust against background
            noise. Utilizes additional GPU resources but improves accuracy in
            noisy environments. When False, uses the default WebRTC VAD,
            which is more sensitive but may continue recording longer due
            to background sounds.
        - webrtc_sensitivity (int, default=WEBRTC_SENSITIVITY): Sensitivity
            for the WebRTC Voice Activity Detection engine ranging from 0
            (least aggressive / most sensitive) to 3 (most aggressive,
            least sensitive). Default is 3.
        - post_speech_silence_duration (float, default=0.2): Duration in
            seconds of silence that must follow speech before the recording
            is considered to be completed. This ensures that any brief
            pauses during speech don't prematurely end the recording.
        - deactivity_silence_confirmation_duration (float, default=0.16):
            Duration in seconds that VAD silence must persist before it is
            accepted as confirmed end-of-speech silence.
        - min_gap_between_recordings (float, default=1.0): Specifies the
            minimum time interval in seconds that should exist between the
            end of one recording session and the beginning of another to
            prevent rapid consecutive recordings.
        - min_length_of_recording (float, default=1.0): Specifies the minimum
            duration in seconds that a recording session should last to ensure
            meaningful audio capture, preventing excessively short or
            fragmented recordings.
        - pre_recording_buffer_duration (float, default=0.2): Duration in
            seconds for the audio buffer to maintain pre-roll audio
            (compensates speech activity detection latency)
        - pre_recording_buffer_trim_config (dict, default=None): Optional
            conservative pre-roll trimming settings. Set ``enabled`` to True
            to trim only clear dead air before speech while preserving a
            configured minimum pre-roll tail.
        - on_vad_start (callable, default=None): Callback function to be called
            when the system detected the start of voice activity presence.
        - on_vad_stop (callable, default=None): Callback function to be called
            when the system detected the stop (end) of voice activity presence.
        - on_vad_detect_start (callable, default=None): Callback function to
            be called when the system listens for voice activity. This is not
            called when VAD actually happens (use on_vad_start for this), but
            when the system starts listening for it.
        - on_vad_detect_stop (callable, default=None): Callback function to be
            called when the system stops listening for voice activity. This is
            not called when VAD actually stops (use on_vad_stop for this), but
            when the system stops listening for it.
        - on_turn_detection_start (callable, default=None): Callback function
            to be called when the system starts to listen for a turn of speech.
        - on_turn_detection_stop (callable, default=None): Callback function to
            be called when the system stops listening for a turn of speech.

        Wake word detection:
        - wakeword_backend (str, default=""): Specifies the backend library to
            use for wake word detection. Supported options include 'pvporcupine'
            for using the Porcupine wake word engine or 'oww' for using the
            OpenWakeWord engine. If wake_words is set and wakeword_backend is
            empty, Porcupine is selected for backward compatibility.
        - openwakeword_model_paths (str, default=None): Comma-separated paths
            to model files for the openwakeword library. These paths point to
            custom models that can be used for wake word detection when the
            openwakeword library is selected as the wakeword_backend.
        - openwakeword_inference_framework (str, default="onnx"): Specifies
            the inference framework to use with the openwakeword library.
            Can be either 'onnx' for Open Neural Network Exchange format 
            or 'tflite' for TensorFlow Lite.
        - wake_words (str, default=""): Comma-separated string of wake words to
            initiate recording when using the 'pvporcupine' wakeword backend.
            Supported wake words include: 'alexa', 'americano', 'blueberry',
            'bumblebee', 'computer', 'grapefruits', 'grasshopper', 'hey google',
            'hey siri', 'jarvis', 'ok google', 'picovoice', 'porcupine',
            'terminator'. For the 'openwakeword' backend, wake words are
            automatically extracted from the provided model files, so specifying
            them here is not necessary.
        - wake_words_sensitivity (float, default=0.5): Sensitivity for wake
            word detection, ranging from 0 (least sensitive) to 1 (most
            sensitive). Default is 0.5.
        - wake_word_activation_delay (float, default=0): Duration in seconds
            after the start of monitoring before the system switches to wake
            word activation if no voice is initially detected. If set to
            zero, the system uses wake word activation immediately.
        - wake_word_timeout (float, default=5): Duration in seconds after a
            wake word is recognized. If no subsequent voice activity is
            detected within this window, the system transitions back to an
            inactive state, awaiting the next wake word or voice activation.
        - wake_word_buffer_duration (float, default=0.1): Duration in seconds
            to buffer audio data during wake word detection. This helps in
            cutting out the wake word from the recording buffer so it does not
            falsely get detected along with the following spoken text, ensuring
            cleaner and more accurate transcription start triggers.
            Increase this if parts of the wake word get detected as text.
        - on_wakeword_detected (callable, default=None): Callback function to
            be called when a wake word is detected.
        - on_wakeword_timeout (callable, default=None): Callback function to
            be called when the system goes back to an inactive state after when
            no speech was detected after wake word activation
        - on_wakeword_detection_start (callable, default=None): Callback
             function to be called when the system starts to listen for wake
             words
        - on_wakeword_detection_end (callable, default=None): Callback
            function to be called when the system stops to listen for
            wake words (e.g. because of timeout or wake word detected)

        Audio input, decoding, and transcription tuning:
        - on_recorded_chunk (callable, default=None): Callback function to be
            called when a chunk of audio is recorded. The function is called
            with the recorded audio chunk as its argument.
        - debug_mode (bool, default=False): If set to True, the system will
            print additional debug information to the console.
        - handle_buffer_overflow (bool, default=True): If set to True, the system
            will log a warning when an input overflow occurs during recording and
            remove the data from the buffer.
        - beam_size (int, default=5): The beam size to use for beam search
            decoding.
        - beam_size_realtime (int, default=3): The beam size to use for beam
            search decoding in the real-time transcription model.
        - buffer_size (int, default=512): The buffer size to use for audio
            recording. Changing this may break functionality.
        - sample_rate (int, default=16000): The sample rate to use for audio
            recording. Changing this will very probably functionality (as the
            WebRTC VAD model is very sensitive towards the sample rate).
        - initial_prompt (str or iterable of int, default=None): Initial
            prompt to be fed to the main transcription model.
        - initial_prompt_realtime (str or iterable of int, default=None):
            Initial prompt to be fed to the real-time transcription model.
        - suppress_tokens (list of int, default=[-1]): Tokens to be suppressed
            from the transcription output.
        - print_transcription_time (bool, default=False): Logs processing time
            of main model transcription 
        - early_transcription_on_silence (int, default=0): If set, the
            system will transcribe audio faster when silence is detected.
            Transcription will start after the specified milliseconds, so 
            keep this value lower than post_speech_silence_duration. 
            Ideally around post_speech_silence_duration minus the estimated
            transcription time with the main model.
            If silence lasts longer than post_speech_silence_duration, the 
            recording is stopped, and the transcription is submitted. If 
            voice activity resumes within this period, the transcription 
            is discarded. Results in faster final transcriptions to the cost
            of additional GPU load due to some unnecessary final transcriptions.
        - allowed_latency_limit (int, default=100): Maximal amount of chunks
            that can be unprocessed in queue before discarding chunks.

        Logging, VAD warmup, normalization, and callback execution:
        - no_log_file (bool, default=False): Skips writing of debug log file.
        - use_extended_logging (bool, default=False): Writes extensive
            log messages for the recording worker, that processes the audio
            chunks.
        - faster_whisper_vad_filter (bool, default=True): If set to True,
            the system will additionally use the VAD filter from the faster_whisper library
            for voice activity detection. This filter is more robust against
            background noise but requires additional GPU resources.
        - warmup_vad (bool, default=True): If set to True, performs a tiny
            warm-up pass through WebRTC and Silero VAD during initialization so
            the first real audio chunk does not pay lazy model/runtime setup
            costs.
        - normalize_audio (bool, default=False): If set to True, the system will
            normalize the audio to a specific range before processing. This can
            help improve the quality of the transcription.
        - start_callback_in_new_thread (bool, default=False): If set to True,
            the callback functions will be executed in a
            new thread. This can help improve performance by allowing the
            callback to run concurrently with other operations.

        Realtime boundary scheduling:
        - realtime_transcription_use_syllable_boundaries (bool, default=False):
            If set to True, realtime transcription is scheduled from a cheap
            vowel/syllable-boundary detector instead of the fixed
            realtime_processing_pause timer.
        - realtime_boundary_detector_sensitivity (float, default=0.6):
            Sensitivity for the acoustic syllable-boundary detector, ranging
            from 0 (conservative) to 1 (eager).
        - realtime_boundary_followup_delays (iterable of float, default=(0.05, 0.2)):
            If using syllable-boundary scheduling, force additional realtime
            transcriptions after each detected boundary at these offsets in
            seconds. Empty iterable or None disables follow-up transcriptions.

        Raises:
            Exception: Errors related to initializing transcription
            model, wake word detection, or audio recording.
        """

        init_args = build_recorder_init_args(
            self,
            model=model,
            transcription_engine=transcription_engine,
            transcription_engine_options=transcription_engine_options,
            download_root=download_root,
            language=language,
            compute_type=compute_type,
            input_device_index=input_device_index,
            gpu_device_index=gpu_device_index,
            device=device,
            on_recording_start=on_recording_start,
            on_recording_stop=on_recording_stop,
            on_transcription_start=on_transcription_start,
            ensure_sentence_starting_uppercase=ensure_sentence_starting_uppercase,
            ensure_sentence_ends_with_period=ensure_sentence_ends_with_period,
            use_microphone=use_microphone,
            spinner=spinner,
            level=level,
            batch_size=batch_size,
            enable_realtime_transcription=enable_realtime_transcription,
            use_main_model_for_realtime=use_main_model_for_realtime,
            realtime_transcription_engine=realtime_transcription_engine,
            realtime_transcription_engine_options=(
                realtime_transcription_engine_options
            ),
            realtime_model_type=realtime_model_type,
            realtime_processing_pause=realtime_processing_pause,
            init_realtime_after_seconds=init_realtime_after_seconds,
            on_realtime_transcription_update=on_realtime_transcription_update,
            on_realtime_transcription_stabilized=(
                on_realtime_transcription_stabilized
            ),
            realtime_batch_size=realtime_batch_size,
            silero_sensitivity=silero_sensitivity,
            silero_use_onnx=silero_use_onnx,
            silero_deactivity_detection=silero_deactivity_detection,
            webrtc_sensitivity=webrtc_sensitivity,
            warmup_vad=warmup_vad,
            post_speech_silence_duration=post_speech_silence_duration,
            min_length_of_recording=min_length_of_recording,
            min_gap_between_recordings=min_gap_between_recordings,
            pre_recording_buffer_duration=pre_recording_buffer_duration,
            pre_recording_buffer_trim_config=pre_recording_buffer_trim_config,
            on_vad_start=on_vad_start,
            on_vad_stop=on_vad_stop,
            on_vad_detect_start=on_vad_detect_start,
            on_vad_detect_stop=on_vad_detect_stop,
            on_turn_detection_start=on_turn_detection_start,
            on_turn_detection_stop=on_turn_detection_stop,
            wakeword_backend=wakeword_backend,
            openwakeword_model_paths=openwakeword_model_paths,
            openwakeword_inference_framework=openwakeword_inference_framework,
            wake_words=wake_words,
            wake_words_sensitivity=wake_words_sensitivity,
            wake_word_activation_delay=wake_word_activation_delay,
            wake_word_timeout=wake_word_timeout,
            wake_word_buffer_duration=wake_word_buffer_duration,
            on_wakeword_detected=on_wakeword_detected,
            on_wakeword_timeout=on_wakeword_timeout,
            on_wakeword_detection_start=on_wakeword_detection_start,
            on_wakeword_detection_end=on_wakeword_detection_end,
            on_recorded_chunk=on_recorded_chunk,
            debug_mode=debug_mode,
            handle_buffer_overflow=handle_buffer_overflow,
            beam_size=beam_size,
            beam_size_realtime=beam_size_realtime,
            buffer_size=buffer_size,
            sample_rate=sample_rate,
            initial_prompt=initial_prompt,
            initial_prompt_realtime=initial_prompt_realtime,
            suppress_tokens=suppress_tokens,
            print_transcription_time=print_transcription_time,
            early_transcription_on_silence=early_transcription_on_silence,
            allowed_latency_limit=allowed_latency_limit,
            no_log_file=no_log_file,
            use_extended_logging=use_extended_logging,
            faster_whisper_vad_filter=faster_whisper_vad_filter,
            normalize_audio=normalize_audio,
            start_callback_in_new_thread=start_callback_in_new_thread,
            realtime_transcription_use_syllable_boundaries=(
                realtime_transcription_use_syllable_boundaries
            ),
            realtime_boundary_detector_sensitivity=(
                realtime_boundary_detector_sensitivity
            ),
            realtime_boundary_followup_delays=realtime_boundary_followup_delays,
            transcription_executor=transcription_executor,
            realtime_transcription_executor=realtime_transcription_executor,
            on_realtime_text_stabilization_update=(
                on_realtime_text_stabilization_update
            ),
            silero_backend=silero_backend,
            silero_onnx_model_path=silero_onnx_model_path,
            silero_onnx_threads=silero_onnx_threads,
            deactivity_silence_confirmation_duration=(
                deactivity_silence_confirmation_duration
            ),
        )

        initialize_recorder(
            self,
            AudioToTextRecorder,
            init_args,
            normalize_wakeword_backend=_normalize_wakeword_backend,
            load_porcupine_module=_load_porcupine_module,
            load_openwakeword_modules=_load_openwakeword_modules,
        )
                   

    # Public lifecycle API.

    def start(self, frames = None):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval
        # between stopping and starting recording
        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logger.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            self._pending_preroll_selection = None
            self.last_preroll_selection = None
            return self

        logger.info("recording started")
        set_recorder_state(self, "recording")
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.realtime_observation_sequence = 0
        self.realtime_recording_id = getattr(self, "realtime_recording_id", 0) + 1
        self.recording_start_monotonic = time.monotonic()
        self.last_preroll_selection = getattr(
            self,
            "_pending_preroll_selection",
            None,
        )
        self._pending_preroll_selection = None
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = []
        if frames:
            self.frames = frames

        self.recording_start_time = time.time()
        self.speech_end_silence_candidate_start = 0
        realtime_text_stabilizer = getattr(
            self,
            "realtime_text_stabilizer",
            None,
        )
        if realtime_text_stabilizer is None:
            realtime_text_stabilizer = RealtimeTextStabilizer()
            self.realtime_text_stabilizer = realtime_text_stabilizer
        realtime_text_stabilizer.reset(
            self.realtime_recording_id,
            started_at_monotonic=self.recording_start_monotonic,
            started_at_wall_time=self.recording_start_time,
        )
        reset_silero_vad_state(self)
        self.is_recording = True
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            run_callback(self, self.on_recording_start)

        return self

    def stop(self,
             backdate_stop_seconds: float = 0.0,
             backdate_resume_seconds: float = 0.0,
        ):
        """
        Stops recording audio.

        Args:
        - backdate_stop_seconds (float, default="0.0"): Specifies the number of
            seconds to backdate the stop time. This is useful when the stop
            command is issued after the actual stop time.
        - backdate_resume_seconds (float, default="0.0"): Specifies the number
            of seconds to backdate the time relistening is initiated.
        """

        # Ensure there's a minimum interval
        # between starting and stopping recording
        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            logger.info("Attempted to stop recording "
                         "too soon after starting."
                         )
            return self

        logger.info("recording stopped")
        stopped_frames = copy.deepcopy(self.frames)
        self.last_frames = copy.deepcopy(stopped_frames)
        self.backdate_stop_seconds = backdate_stop_seconds
        self.backdate_resume_seconds = backdate_resume_seconds
        queue_recorded_audio(
            self,
            stopped_frames,
            backdate_stop_seconds,
            backdate_resume_seconds,
        )
        self.frames = []
        self.is_recording = False
        self.recording_stop_time = time.time()
        realtime_text_stabilizer = getattr(
            self,
            "realtime_text_stabilizer",
            None,
        )
        if realtime_text_stabilizer is not None:
            realtime_text_stabilizer.finalize()
        reset_silero_vad_state(self)
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        self.last_recording_start_time = self.recording_start_time
        self.last_recording_stop_time = self.recording_stop_time

        if self.on_recording_stop:
            run_callback(self, self.on_recording_stop)

        return self

    def listen(self):
        """
        Puts recorder in immediate "listen" state.
        This is the state after a wake word detection, for example.
        The recorder now "listens" for voice activation.
        Once voice is detected we enter "recording" state.
        """
        self.listen_start = time.time()
        set_recorder_state(self, "listening")
        reset_silero_vad_state(self)
        self.start_recording_on_voice_activity = True

    def text(self,
             on_transcription_finished=None,
             ):
        """
        Transcribes audio captured by this class instance
        using the configured transcription engine.

        - Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        - Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        - Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously, and
              the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if not callback is set):
            str: The transcription of the recorded audio
        """
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in text() method")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        if on_transcription_finished:
            threading.Thread(target=on_transcription_finished,
                            args=(self.transcribe(),)).start()
        else:
            return self.transcribe()

    def transcribe(self):
        """
        Transcribes audio captured by this class instance using the
        configured transcription engine.

        Automatically starts recording upon voice activity if not manually
          started using `recorder.start()`.
        Automatically stops recording upon voice deactivity if not manually
          stopped with `recorder.stop()`.
        Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): Callback function
              to be executed when transcription is ready.
            If provided, transcription will be performed asynchronously,
              and the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously,
              and the result will be returned.

        Returns (if no callback is set):
            str: The transcription of the recorded audio.

        Raises:
            Exception: If there is an error during the transcription process.
        """
        audio_copy = copy.deepcopy(self.audio)
        set_recorder_state(self, "transcribing")
        if self.on_transcription_start:
            abort_value = self.on_transcription_start(audio_copy)
            if not abort_value:
                return self.perform_final_transcription(audio_copy)
            return None
        else:
            return self.perform_final_transcription(audio_copy)

    def perform_final_transcription(self, audio_bytes=None, use_prompt=True):
        start_time = 0
        with self.transcription_lock:
            if audio_bytes is None:
                audio_bytes = copy.deepcopy(self.audio)

            if audio_bytes is None or len(audio_bytes) == 0:
                print("No audio data available for transcription")
                #logger.info("No audio data available for transcription")
                return ""

            try:
                if self.transcribe_count == 0:
                    logger.debug("Adding transcription request, no early transcription started")
                    start_time = time.time()  # Start timing
                    submit_transcription_request(
                        self,
                        audio_bytes,
                        self.language,
                        use_prompt,
                    )

                while self.transcribe_count > 0:
                    logger.debug(F"Receive from parent_transcription_pipe after sendiung transcription request, transcribe_count: {self.transcribe_count}")
                    response = receive_transcription_result(self, timeout=0.1)
                    if response is None: # check if transcription done
                        if self.interrupt_stop_event.is_set(): # check if interrupted
                            self.was_interrupted.set()
                            self._set_state_after_transcription()
                            return "" # return empty string if interrupted
                        continue
                    status, result = response
                    self.transcribe_count -= 1

                self.allowed_to_early_transcribe = True
                self._set_state_after_transcription()
                if status == 'success':
                    self.detected_language = (
                        result.info.language if result.info.language_probability > 0 else None
                    )
                    self.detected_language_probability = result.info.language_probability
                    self.last_transcription_bytes = copy.deepcopy(audio_bytes)
                    self.last_transcription_bytes_b64 = base64.b64encode(self.last_transcription_bytes.tobytes()).decode('utf-8')
                    self.last_transcription_metadata = getattr(result, "metadata", None)
                    transcription = preprocess_output(
                        result.text,
                        ensure_sentence_starting_uppercase=(
                            self.ensure_sentence_starting_uppercase
                        ),
                        ensure_sentence_ends_with_period=(
                            self.ensure_sentence_ends_with_period
                        ),
                    )
                    end_time = time.time()  # End timing
                    transcription_time = end_time - start_time

                    if start_time:
                        if self.print_transcription_time:
                            print(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                        else:
                            logger.debug(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                    return "" if self.interrupt_stop_event.is_set() else transcription # if interrupted return empty string
                else:
                    logger.error(f"Transcription error: {result}")
                    raise Exception(result)
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}", exc_info=True)
                raise e

    def wait_audio(self):
        """
        Waits for the start and completion of the audio recording process.

        This method is responsible for:
        - Waiting for voice activity to begin recording if not yet started.
        - Waiting for voice inactivity to complete the recording.
        - Setting the audio buffer from the recorded frames.
        - Resetting recording-related attributes.

        Side effects:
        - Updates the state of the instance.
        - Modifies the audio attribute to contain the processed audio data.
        """

        armed_for_voice_activity = False

        try:
            logger.info("Setting listen time")
            if self.listen_start == 0:
                self.listen_start = time.time()

            queued_recording = get_next_recorded_audio(self)

            # If not yet started recording, wait for voice activity to initiate.
            if queued_recording is None and not self.is_recording and not self.frames:
                set_recorder_state(self, "listening")
                reset_silero_vad_state(self)
                self.start_recording_on_voice_activity = True
                armed_for_voice_activity = True

                # Wait until recording starts
                logger.debug('Waiting for recording start')
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02):
                        break

            # If recording is ongoing, wait for voice inactivity
            # to finish recording.
            if queued_recording is None and self.is_recording:
                self.stop_recording_on_voice_deactivity = True

                # Wait until recording stops
                logger.debug('Waiting for recording stop')
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout=0.02)):
                        break

            if queued_recording is None:
                queued_recording = get_next_recorded_audio(self)

            if queued_recording is not None:
                frames = queued_recording["frames"]
                backdate_stop_seconds = queued_recording["backdate_stop_seconds"]
                backdate_resume_seconds = queued_recording["backdate_resume_seconds"]
            else:
                frames = self.frames
                if len(frames) == 0:
                    frames = self.last_frames
                backdate_stop_seconds = self.backdate_stop_seconds
                backdate_resume_seconds = self.backdate_resume_seconds

            frames_to_read = set_audio_from_frames(
                self,
                frames,
                backdate_stop_seconds,
                backdate_resume_seconds,
            )

            if not self.is_recording:
                self.frames.clear()
                self.last_frames.clear()
                self.frames.extend(frames_to_read)

            # Reset backdating parameters
            self.backdate_stop_seconds = 0.0
            self.backdate_resume_seconds = 0.0

            self.listen_start = 0

            if not self.is_recording:
                set_recorder_state(self, "inactive")

            if (
                    armed_for_voice_activity
                    and not self.use_wake_words
                    and not self.interrupt_stop_event.is_set()
                    and not self.is_shut_down):
                self.continuous_listening = True
                reset_silero_vad_state(self)
                self.start_recording_on_voice_activity = True
                self.stop_recording_on_voice_deactivity = True

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

    def feed_audio(self, chunk, original_sample_rate=16000):
        """
        Feed an audio chunk into the processing pipeline. Chunks are
        accumulated until the buffer size is reached, and then the accumulated
        data is fed into the audio_queue.
        """
        # Check if the buffer attribute exists, if not, initialize it
        if not hasattr(self, 'buffer'):
            self.buffer = bytearray()

        # Check if input is a NumPy array
        if isinstance(chunk, np.ndarray):
            # Handle stereo to mono conversion if necessary
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # Resample to 16000 Hz if necessary
            if original_sample_rate != 16000:
                num_samples = int(len(chunk) * 16000 / original_sample_rate)
                chunk = resample(chunk, num_samples)

            # Ensure data type is int16
            chunk = chunk.astype(np.int16)

            # Convert the NumPy array to bytes
            chunk = chunk.tobytes()

        # Append the chunk to the buffer
        self.buffer += chunk
        buf_size = 2 * self.buffer_size  # silero complains if too short

        # Check if the buffer has reached or exceeded the buffer_size
        while len(self.buffer) >= buf_size:
            # Extract self.buffer_size amount of data from the buffer
            to_process = self.buffer[:buf_size]
            self.buffer = self.buffer[buf_size:]

            # Feed the extracted data to the audio_queue
            self.audio_queue.put(to_process)

    def shutdown(self):
        """
        Safely shuts down the audio recording by stopping the
        recording worker and closing the audio stream.
        """

        with self.shutdown_lock:
            if self.is_shut_down:
                return

            print("\033[91mRealtimeSTT shutting down\033[0m")

            # Force wait_audio() and text() to exit
            self.is_shut_down = True
            self.continuous_listening = False
            self.start_recording_event.set()
            self.stop_recording_event.set()

            self.shutdown_event.set()
            self.is_recording = False
            self.is_running = False

            logger.debug('Finishing recording thread')
            if self.recording_thread:
                self.recording_thread.join()

            logger.debug('Terminating reader process')

            # Give it some time to finish the loop and cleanup.
            if self.use_microphone.value:
                self.reader_process.join(timeout=10)

                if self.reader_process.is_alive():
                    logger.warning("Reader process did not terminate "
                                    "in time. Terminating forcefully."
                                    )
                    self.reader_process.terminate()

            logger.debug('Terminating transcription process')
            if self.transcript_process:
                self.transcript_process.join(timeout=10)

            if self.transcript_process and self.transcript_process.is_alive():
                logger.warning("Transcript process did not terminate "
                                "in time. Terminating forcefully."
                                )
                self.transcript_process.terminate()

            if self.parent_transcription_pipe:
                self.parent_transcription_pipe.close()

            logger.debug('Finishing realtime thread')
            if self.realtime_thread:
                self.realtime_thread.join()

            if self.enable_realtime_transcription:
                if self.realtime_transcription_model:
                    del self.realtime_transcription_model
                    self.realtime_transcription_model = None
            gc.collect()

    # Public compatibility utilities.

    def wakeup(self):
        """
        If in wake work modus, wake up as if a wake word was spoken.
        """
        self.listen_start = time.time()

    def abort(self):
        state = self.state
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.interrupt_stop_event.set()
        if self.state != "inactive": # if inactive, was_interrupted will never be set
            self.was_interrupted.wait()
            set_recorder_state(self, "transcribing")
        self.was_interrupted.clear()
        if self.is_recording: # if recording, make sure to stop the recorder
            self.stop()

    def set_microphone(self, microphone_on=True):
        """
        Set the microphone on or off.
        """
        logger.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

    def clear_audio_queue(self):
        """
        Safely empties the audio queue to ensure no remaining audio 
        fragments get processed e.g. after waking up the recorder.
        """
        return clear_recorder_audio_queue(self)

    def has_pending_recordings(self):
        return recorder_has_pending_recordings(self)

    def flush_buffered_audio(self, min_abs_level=50):
        return flush_recorder_buffered_audio(self, min_abs_level)

    def format_number(self, num):
        return format_number(num)

    # Context manager API.

    def __enter__(self):
        """
        Method to setup the context manager protocol.

        This enables the instance to be used in a `with` statement, ensuring
        proper resource management. When the `with` block is entered, this
        method is automatically called.

        Returns:
            self: The current instance of the class.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Method to define behavior when the context manager protocol exits.

        This is called when exiting the `with` block and ensures that any
        necessary cleanup or resource release processes are executed, such as
        shutting down the system properly.

        Args:
            exc_type (Exception or None): The type of the exception that
              caused the context to be exited, if any.
            exc_value (Exception or None): The exception instance that caused
              the context to be exited, if any.
            traceback (Traceback or None): The traceback corresponding to the
              exception, if any.
        """
        self.shutdown()

    # Internal compatibility wrappers.

    def _recording_worker(self):
        return run_recording_worker(self)

    def _realtime_worker(self):
        return run_realtime_worker(self)

    @staticmethod
    def _audio_data_worker(
            audio_queue,
            target_sample_rate,
            buffer_size,
            input_device_index,
            shutdown_event,
            interrupt_stop_event,
            use_microphone):
        return run_audio_data_worker(
            audio_queue,
            target_sample_rate,
            buffer_size,
            input_device_index,
            shutdown_event,
            interrupt_stop_event,
            use_microphone,
        )

    def _is_silero_speech(self, chunk, generation=None):
        return is_silero_speech(self, chunk, generation)

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        return is_webrtc_speech(self, chunk, all_frames_must_be_true)

    def _check_voice_activity(self, data):
        return check_voice_activity(
            self,
            data,
            thread_factory=threading.Thread,
        )

    def _selected_pre_recording_buffer_frames(self):
        return selected_pre_recording_buffer_frames(self)

    def _set_audio_from_frames(
            self,
            frames,
            backdate_stop_seconds=0.0,
            backdate_resume_seconds=0.0):
        return set_audio_from_frames(
            self,
            frames,
            backdate_stop_seconds,
            backdate_resume_seconds,
        )

    def _queue_recorded_audio(
            self,
            frames,
            backdate_stop_seconds=0.0,
            backdate_resume_seconds=0.0):
        return queue_recorded_audio(
            self,
            frames,
            backdate_stop_seconds,
            backdate_resume_seconds,
        )

    def _get_next_recorded_audio(self):
        return get_next_recorded_audio(self)

    def _is_voice_active(self):
        return is_voice_active(self)

    def _run_callback(self, cb, *args, **kwargs):
        return run_callback(self, cb, *args, **kwargs)

    def _set_state(self, new_state):
        return set_recorder_state(self, new_state)

    def _set_spinner(self, text):
        return set_spinner(self, text)

    def _preprocess_output(self, text, preview=False):
        return preprocess_output(
            text,
            preview=preview,
            ensure_sentence_starting_uppercase=(
                self.ensure_sentence_starting_uppercase
            ),
            ensure_sentence_ends_with_period=(
                self.ensure_sentence_ends_with_period
            ),
        )

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        return find_tail_match_in_text(text1, text2, length_of_match)

    # Internal facade-level helpers.

    # Keep this lifecycle boundary on the facade until worker process/thread
    # startup has stronger direct test coverage.
    def _start_thread(self, target=None, args=()):
        """
        Implement a consistent threading model across the library.

        This method is used to start any thread in this library. It uses the
        standard threading. Thread for Linux and for all others uses the pytorch
        MultiProcessing library 'Process'.
        Args:
            target (callable object): is the callable object to be invoked by
              the run() method. Defaults to None, meaning nothing is called.
            args (tuple): is a list or tuple of arguments for the target
              invocation. Defaults to ().
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

    def _read_stdout(self):
        return read_stdout_pipe(self)

    def _set_state_after_transcription(self):
        if self.is_recording:
            set_recorder_state(self, "recording")
        else:
            set_recorder_state(self, "inactive")

    def _on_realtime_transcription_stabilized(self, text):
        """
        Callback method invoked when the real-time transcription stabilizes.

        This method is called internally when the transcription text is
        considered "stable" meaning it's less likely to change significantly
        with additional audio input. It notifies any registered external
        listener about the stabilized text if recording is still ongoing.
        This is particularly useful for applications that need to display
        live transcription results to users and want to highlight parts of the
        transcription that are less likely to change.

        Args:
            text (str): The stabilized transcription text.
        """
        if self.on_realtime_transcription_stabilized:
            if self.is_recording:
                run_callback(self, self.on_realtime_transcription_stabilized, text)

    def _on_realtime_transcription_update(self, text):
        """
        Callback method invoked when there's an update in the real-time
        transcription.

        This method is called internally whenever there's a change in the
        transcription text, notifying any registered external listener about
        the update if recording is still ongoing. This provides a mechanism
        for applications to receive and possibly display live transcription
        updates, which could be partial and still subject to change.

        Args:
            text (str): The updated transcription text.
        """
        if self.on_realtime_transcription_update:
            if self.is_recording:
                run_callback(self, self.on_realtime_transcription_update, text)
