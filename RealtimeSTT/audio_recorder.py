"""

The AudioToTextRecorder class in the provided code facilitates
fast speech-to-text transcription.

The class employs a pluggable transcription engine to transcribe the recorded
audio into text using machine learning models, which can be run either on a GPU or
CPU. Voice activity detection (VAD) is built in, meaning the software can
automatically start or stop recording based on the presence or absence of
speech. It integrates optional wake word detection through Porcupine or
OpenWakeWord, allowing the software to initiate recording when a specific word
or phrase is spoken. The system provides real-time feedback and can be further
customized.

Features:
- Voice Activity Detection: Automatically starts/stops recording when speech
  is detected or when speech ends.
- Wake Word Detection: Starts recording when a specified wake word (or words)
  is detected.
- Event Callbacks: Customizable callbacks for when recording starts
  or finishes.
- Fast Transcription: Returns the transcribed text from the audio as fast
  as possible.

Author: Kolja Beigel

"""

from typing import Callable, Iterable, List, Optional, Union
import torch.multiprocessing as mp
from scipy.signal import resample
import signal as system_signal
from scipy import signal
from .core.realtime_text_stabilizer import RealtimeTextStabilizer
from .core.initialization import initialize_recorder
from .core.recording import run_recording_worker
from .core.realtime import run_realtime_worker
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
    clear_pre_recording_buffer,
    is_silero_speech,
    is_voice_active,
    is_webrtc_speech,
    reset_silero_vad_state,
    selected_pre_recording_buffer_frames,
)
import numpy as np
import traceback
import threading
import platform
import logging
import base64
import queue
import halo
import time
import copy
import os
import re
import gc

# Named logger for this module.
logger = logging.getLogger("realtimestt")
logger.propagate = False

# Set OpenMP runtime duplicate library handling to OK (Use only for development!)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
DEACTIVITY_SILENCE_CONFIRMATION_DURATION = 0.16
INT16_MAX_ABS_VALUE = 32768.0

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True


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

        initialize_recorder(
            self,
            AudioToTextRecorder,
            locals().copy(),
            normalize_wakeword_backend=_normalize_wakeword_backend,
            load_porcupine_module=_load_porcupine_module,
            load_openwakeword_modules=_load_openwakeword_modules,
        )
                   
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
        while not self.shutdown_event.is_set():
            try:
                if self.parent_stdout_pipe.poll(0.1):
                    logger.debug("Receive from stdout pipe")
                    message = self.parent_stdout_pipe.recv()
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

    def _run_callback(self, cb, *args, **kwargs):
        if self.start_callback_in_new_thread:
            # Run the callback in a new thread to avoid blocking the main thread
            threading.Thread(target=cb, args=args, kwargs=kwargs, daemon=True).start()
        else:
            # Run the callback in the main thread to avoid threading issues
            cb(*args, **kwargs)

    @staticmethod
    def _audio_data_worker(
        audio_queue,
        target_sample_rate,
        buffer_size,
        input_device_index,
        shutdown_event,
        interrupt_stop_event,
        use_microphone
    ):
        """
        Worker method that handles the audio recording process.

        This method runs in a separate process and is responsible for:
        - Setting up the audio input stream for recording at the highest possible sample rate.
        - Continuously reading audio data from the input stream, resampling if necessary,
        preprocessing the data, and placing complete chunks in a queue.
        - Handling errors during the recording process.
        - Gracefully terminating the recording process when a shutdown event is set.

        Args:
            audio_queue (queue.Queue): A queue where recorded audio data is placed.
            target_sample_rate (int): The desired sample rate for the output audio (for Silero VAD).
            buffer_size (int): The number of samples expected by the Silero VAD model.
            input_device_index (int): The index of the audio input device.
            shutdown_event (threading.Event): An event that, when set, signals this worker method to terminate.
            interrupt_stop_event (threading.Event): An event to signal keyboard interrupt.
            use_microphone (multiprocessing.Value): A shared value indicating whether to use the microphone.

        Raises:
            Exception: If there is an error while initializing the audio recording.
        """
        import pyaudio
        import numpy as np
        from scipy import signal

        if __name__ == '__main__':
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        def get_highest_sample_rate(audio_interface, device_index):
            """Get the highest supported sample rate for the specified device."""
            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                logger.debug(f"Retrieving highest sample rate for device index {device_index}: {device_info}")
                max_rate = int(device_info['defaultSampleRate'])

                if 'supportedSampleRates' in device_info:
                    supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
                    if supported_rates:
                        max_rate = max(supported_rates)

                logger.debug(f"Highest supported sample rate for device index {device_index} is {max_rate}")
                return max_rate
            except Exception as e:
                logger.warning(f"Failed to get highest sample rate: {e}")
                return 48000  # Fallback to a common high sample rate

        def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
            nonlocal input_device_index

            def validate_device(device_index):
                """Validate that the device exists and is actually available for input."""
                try:
                    device_info = audio_interface.get_device_info_by_index(device_index)
                    logger.debug(f"Validating device index {device_index} with info: {device_info}")
                    if not device_info.get('maxInputChannels', 0) > 0:
                        logger.debug("Device has no input channels, invalid for recording.")
                        return False

                    # Try to actually read from the device
                    test_stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=target_sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=device_index,
                        start=False  # Don't start the stream yet
                    )

                    test_stream.start_stream()
                    test_data = test_stream.read(chunk_size, exception_on_overflow=False)
                    test_stream.stop_stream()
                    test_stream.close()

                    if len(test_data) == 0:
                        logger.debug("Device produced no data, invalid for recording.")
                        return False

                    logger.debug(f"Device index {device_index} successfully validated.")
                    return True

                except Exception as e:
                    logger.debug(f"Device validation failed for index {device_index}: {e}")
                    return False

            """Initialize the audio stream with error handling."""
            while not shutdown_event.is_set():
                try:
                    # First, get a list of all available input devices
                    input_devices = []
                    device_count = audio_interface.get_device_count()
                    logger.debug(f"Found {device_count} total audio devices on the system.")
                    for i in range(device_count):
                        try:
                            device_info = audio_interface.get_device_info_by_index(i)
                            if device_info.get('maxInputChannels', 0) > 0:
                                input_devices.append(i)
                        except Exception as e:
                            logger.debug(f"Could not retrieve info for device index {i}: {e}")
                            continue

                    logger.debug(f"Available input devices with input channels: {input_devices}")
                    if not input_devices:
                        raise Exception("No input devices found")

                    # If input_device_index is None or invalid, try to find a working device
                    if input_device_index is None or input_device_index not in input_devices:
                        # First try the default device
                        try:
                            default_device = audio_interface.get_default_input_device_info()
                            logger.debug(f"Default device info: {default_device}")
                            if validate_device(default_device['index']):
                                input_device_index = default_device['index']
                                logger.debug(f"Default device {input_device_index} selected.")
                        except Exception:
                            # If default device fails, try other available input devices
                            logger.debug("Default device validation failed, checking other devices...")
                            for device_index in input_devices:
                                if validate_device(device_index):
                                    input_device_index = device_index
                                    logger.debug(f"Device {input_device_index} selected.")
                                    break
                            else:
                                raise Exception("No working input devices found")

                    # Validate the selected device one final time
                    if not validate_device(input_device_index):
                        raise Exception("Selected device validation failed")

                    # If we get here, we have a validated device
                    logger.debug(f"Opening stream with device index {input_device_index}, "
                                f"sample_rate={sample_rate}, chunk_size={chunk_size}")
                    stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=input_device_index,
                    )

                    logger.info(f"Microphone connected and validated (device index: {input_device_index}, "
                                f"sample rate: {sample_rate}, chunk size: {chunk_size})")
                    return stream

                except Exception as e:
                    logger.error(f"Microphone connection failed: {e}. Retrying...", exc_info=True)
                    input_device_index = None
                    time.sleep(3)  # Wait before retrying
                    continue

        def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
            """Preprocess audio chunk similar to feed_audio method."""
            if isinstance(chunk, np.ndarray):
                # Handle stereo to mono conversion if necessary
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)

                # Resample to target_sample_rate if necessary
                if original_sample_rate != target_sample_rate:
                    logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)

                chunk = chunk.astype(np.int16)
            else:
                # If chunk is bytes, convert to numpy array
                chunk = np.frombuffer(chunk, dtype=np.int16)

                # Resample if necessary
                if original_sample_rate != target_sample_rate:
                    logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)
                    chunk = chunk.astype(np.int16)

            return chunk.tobytes()

        audio_interface = None
        stream = None
        device_sample_rate = None
        chunk_size = 1024  # Increased chunk size for better performance

        def setup_audio():  
            nonlocal audio_interface, stream, device_sample_rate, input_device_index
            try:
                if audio_interface is None:
                    logger.debug("Creating PyAudio interface...")
                    audio_interface = pyaudio.PyAudio()

                if input_device_index is None:
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        input_device_index = default_device['index']
                        logger.debug(f"No device index supplied; using default device {input_device_index}")
                    except OSError as e:
                        logger.debug(f"Default device retrieval failed: {e}")
                        input_device_index = None

                # We'll try 16000 Hz first, then the highest rate we detect, then fallback if needed
                sample_rates_to_try = [16000]
                if input_device_index is not None:
                    highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
                    if highest_rate != 16000:
                        sample_rates_to_try.append(highest_rate)
                else:
                    sample_rates_to_try.append(48000)

                logger.debug(f"Sample rates to try for device {input_device_index}: {sample_rates_to_try}")

                for rate in sample_rates_to_try:
                    try:
                        device_sample_rate = rate
                        logger.debug(f"Attempting to initialize audio stream at {device_sample_rate} Hz.")
                        stream = initialize_audio_stream(audio_interface, device_sample_rate, chunk_size)
                        if stream is not None:
                            logger.debug(
                                f"Audio recording initialized successfully at {device_sample_rate} Hz, "
                                f"reading {chunk_size} frames at a time"
                            )
                            return True
                    except Exception as e:
                        logger.warning(f"Failed to initialize audio stream at {device_sample_rate} Hz: {e}")
                        continue

                # If we reach here, none of the sample rates worked
                raise Exception("Failed to initialize audio stream with all sample rates.")

            except Exception as e:
                logger.exception(f"Error initializing pyaudio audio recording: {e}")
                if audio_interface:
                    audio_interface.terminate()
                return False

        logger.debug(f"Starting audio data worker with target_sample_rate={target_sample_rate}, "
                    f"buffer_size={buffer_size}, input_device_index={input_device_index}")

        if not setup_audio():
            raise Exception("Failed to set up audio recording.")

        buffer = bytearray()
        silero_buffer_size = 2 * buffer_size  # Silero complains if too short

        time_since_last_buffer_message = 0

        try:
            while not shutdown_event.is_set():
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)

                    if use_microphone.value:
                        processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
                        buffer += processed_data

                        # Check if the buffer has reached or exceeded the silero_buffer_size
                        while len(buffer) >= silero_buffer_size:
                            # Extract silero_buffer_size amount of data from the buffer
                            to_process = buffer[:silero_buffer_size]
                            buffer = buffer[silero_buffer_size:]

                            # Feed the extracted data to the audio_queue
                            if time_since_last_buffer_message:
                                time_passed = time.time() - time_since_last_buffer_message
                                if time_passed > 1:
                                    logger.debug("_audio_data_worker writing audio data into queue.")
                                    time_since_last_buffer_message = time.time()
                            else:
                                time_since_last_buffer_message = time.time()

                            audio_queue.put(to_process)

                except OSError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logger.warning("Input overflowed. Frame dropped.")
                    else:
                        logger.error(f"OSError during recording: {e}", exc_info=True)
                        # Attempt to reinitialize the stream
                        logger.error("Attempting to reinitialize the audio stream...")

                        try:
                            if stream:
                                stream.stop_stream()
                                stream.close()
                        except Exception:
                            pass

                        time.sleep(1)
                        if not setup_audio():
                            logger.error("Failed to reinitialize audio stream. Exiting.")
                            break
                        else:
                            logger.error("Audio stream reinitialized successfully.")
                    continue

                except Exception as e:
                    logger.error(f"Unknown error during recording: {e}")
                    tb_str = traceback.format_exc()
                    logger.error(f"Traceback: {tb_str}")
                    logger.error(f"Error: {e}")
                    # Attempt to reinitialize the stream
                    logger.info("Attempting to reinitialize the audio stream...")
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                    except Exception:
                        pass

                    time.sleep(1)
                    if not setup_audio():
                        logger.error("Failed to reinitialize audio stream. Exiting.")
                        break
                    else:
                        logger.info("Audio stream reinitialized successfully.")
                    continue

        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logger.debug("Audio data worker process finished due to KeyboardInterrupt")
        finally:
            # After recording stops, feed any remaining audio data
            if buffer:
                audio_queue.put(bytes(buffer))

            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            if audio_interface:
                audio_interface.terminate()

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
            self._set_state("transcribing")
        self.was_interrupted.clear()
        if self.is_recording: # if recording, make sure to stop the recorder
            self.stop()


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

            queued_recording = self._get_next_recorded_audio()

            # If not yet started recording, wait for voice activity to initiate.
            if queued_recording is None and not self.is_recording and not self.frames:
                self._set_state("listening")
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
                queued_recording = self._get_next_recorded_audio()

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

            frames_to_read = self._set_audio_from_frames(
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
                self._set_state("inactive")

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

    def _set_audio_from_frames(
            self,
            frames,
            backdate_stop_seconds=0.0,
            backdate_resume_seconds=0.0,
    ):
        frames = frames or []

        # Calculate samples needed for backdating resume
        samples_to_keep = int(self.sample_rate * backdate_resume_seconds)

        # First convert all current frames to audio array
        full_audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
        full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

        # Calculate how many samples we need to keep for backdating resume
        if samples_to_keep > 0:
            samples_to_keep = min(samples_to_keep, len(full_audio))
            # Keep the last N samples for backdating resume
            frames_to_read_audio = full_audio[-samples_to_keep:]

            # Convert the audio back to int16 bytes for frames
            frames_to_read_int16 = (frames_to_read_audio * INT16_MAX_ABS_VALUE).astype(np.int16)
            frame_bytes = frames_to_read_int16.tobytes()

            # Split into appropriate frame sizes (assuming standard frame size)
            FRAME_SIZE = 2048  # Typical frame size
            frames_to_read = []
            for i in range(0, len(frame_bytes), FRAME_SIZE):
                frame = frame_bytes[i:i + FRAME_SIZE]
                if frame:  # Only add non-empty frames
                    frames_to_read.append(frame)
        else:
            frames_to_read = []

        # Process backdate stop seconds
        samples_to_remove = int(self.sample_rate * backdate_stop_seconds)

        if samples_to_remove > 0:
            if samples_to_remove < len(full_audio):
                self.audio = full_audio[:-samples_to_remove]
                logger.debug(f"Removed {samples_to_remove} samples "
                    f"({samples_to_remove/self.sample_rate:.3f}s) from end of audio")
            else:
                self.audio = np.array([], dtype=np.float32)
                logger.debug("Cleared audio (samples_to_remove >= audio length)")
        else:
            self.audio = full_audio
            logger.debug(f"No samples removed, final audio length: {len(self.audio)}")

        return frames_to_read

    def _queue_recorded_audio(
            self,
            frames,
            backdate_stop_seconds=0.0,
            backdate_resume_seconds=0.0,
    ):
        if not frames:
            return

        self.recorded_audio_queue.put({
            "frames": copy.deepcopy(frames),
            "backdate_stop_seconds": backdate_stop_seconds,
            "backdate_resume_seconds": backdate_resume_seconds,
        })

    def _get_next_recorded_audio(self):
        try:
            return self.recorded_audio_queue.get_nowait()
        except queue.Empty:
            return None

    def has_pending_recordings(self):
        return not self.recorded_audio_queue.empty()

    def _set_state_after_transcription(self):
        if self.is_recording:
            self._set_state("recording")
        else:
            self._set_state("inactive")

    def flush_buffered_audio(self, min_abs_level=50):
        if self.is_recording:
            self.stop()
            return True

        frames = list(self.audio_buffer)
        if not frames:
            return False

        audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
        if audio_array.size == 0:
            return False

        if np.max(np.abs(audio_array.astype(np.int32))) < min_abs_level:
            return False

        self._queue_recorded_audio(frames)
        clear_pre_recording_buffer(self)
        return True


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
                    transcription = self._preprocess_output(result.text)
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
        self._set_state("transcribing")
        if self.on_transcription_start:
            abort_value = self.on_transcription_start(audio_copy)
            if not abort_value:
                return self.perform_final_transcription(audio_copy)
            return None
        else:
            return self.perform_final_transcription(audio_copy)


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


    def format_number(self, num):
        # Convert the number to a string
        num_str = f"{num:.10f}"  # Ensure precision is sufficient
        # Split the number into integer and decimal parts
        integer_part, decimal_part = num_str.split('.')
        # Take the last two digits of the integer part and the first two digits of the decimal part
        result = f"{integer_part[-2:]}.{decimal_part[:2]}"
        return result

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
        self._set_state("recording")
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
            self._run_callback(self.on_recording_start)

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
        self._queue_recorded_audio(
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
            self._run_callback(self.on_recording_stop)

        return self

    def listen(self):
        """
        Puts recorder in immediate "listen" state.
        This is the state after a wake word detection, for example.
        The recorder now "listens" for voice activation.
        Once voice is detected we enter "recording" state.
        """
        self.listen_start = time.time()
        self._set_state("listening")
        reset_silero_vad_state(self)
        self.start_recording_on_voice_activity = True

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

    def set_microphone(self, microphone_on=True):
        """
        Set the microphone on or off.
        """
        logger.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

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

    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio
        input for voice activity and accordingly starts/stops the recording.
        """

        return run_recording_worker(self)

    def _realtime_worker(self):
        """
        Performs real-time transcription if the feature is enabled.

        This worker is intentionally defensive:
        - realtime transcription must never crash the recorder
        - empty/None buffers are skipped
        - frame buffers are snapshotted before transcription
        - model/pipe errors are logged and skipped
        """

        return run_realtime_worker(self)

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

    def clear_audio_queue(self):
        """
        Safely empties the audio queue to ensure no remaining audio 
        fragments get processed e.g. after waking up the recorder.
        """
        clear_pre_recording_buffer(self)
        try:
            while True:
                self.audio_queue.get_nowait()
        except:
            # PyTorch's mp.Queue doesn't have a specific Empty exception
            # so we catch any exception that might occur when the queue is empty
            pass

    def _is_voice_active(self):
        return is_voice_active(self)

    def _set_state(self, new_state):
        """
        Update the current state of the recorder and execute
        corresponding state-change callbacks.

        Args:
            new_state (str): The new state to set.

        """
        # Check if the state has actually changed
        if new_state == self.state:
            return

        # Store the current state for later comparison
        old_state = self.state

        # Update to the new state
        self.state = new_state

        # Log the state change
        logger.info(f"State changed from '{old_state}' to '{new_state}'")

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self._run_callback(self.on_vad_detect_stop)
        elif old_state == "wakeword":
            if self.on_wakeword_detection_end:
                self._run_callback(self.on_wakeword_detection_end)

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            if self.on_vad_detect_start:
                self._run_callback(self.on_vad_detect_start)
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start:
                self._run_callback(self.on_wakeword_detection_start)
            self._set_spinner(f"say {self.wake_words}")
            if self.spinner and self.halo:
                self.halo._interval = 500
        elif new_state == "transcribing":
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new
        spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing
        whitespace, converting all whitespace sequences to a single space
        character, and capitalizing the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'\s+', ' ', text.strip())

        if self.ensure_sentence_starting_uppercase:
            if text:
                text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation
        # if it ends with an alphanumeric character
        if not preview:
            if self.ensure_sentence_ends_with_period:
                if text and text[-1].isalnum():
                    text += '.'

        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """
        Find the position where the last 'n' characters of text1
        match with a substring in text2.

        This method takes two texts, extracts the last 'n' characters from
        text1 (where 'n' is determined by the variable 'length_of_match'), and
        searches for an occurrence of this substring in text2, starting from
        the end of text2 and moving towards the beginning.

        Parameters:
        - text1 (str): The text containing the substring that we want to find
          in text2.
        - text2 (str): The text in which we want to find the matching
          substring.
        - length_of_match(int): The length of the matching string that we are
          looking for

        Returns:
        int: The position (0-based index) in text2 where the matching
          substring starts. If no match is found or either of the texts is
          too short, returns -1.
        """

        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]

        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2
            # to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                # Position in text2 where the match starts
                return len(text2) - i

        return -1

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
                self._run_callback(self.on_realtime_transcription_stabilized, text)

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
                self._run_callback(self.on_realtime_transcription_update, text)

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
