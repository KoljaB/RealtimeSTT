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

from importlib import import_module
from typing import Callable, Iterable, List, Optional, Union
import torch.multiprocessing as mp
from scipy.signal import resample
import signal as system_signal
from ctypes import c_bool
from scipy import signal
from .safepipe import SafePipe
from .realtime_boundary_detector import RealtimeSpeechBoundaryDetector
from .realtime_text_stabilizer import (
    RealtimeTextObservation,
    RealtimeTextStabilizer,
)
from .preroll import PrerollFrameMetadata, select_preroll_frames
from .silero_vad import create_silero_vad_model
from .transcription_engines import (
    TranscriptionEngineConfig,
    create_transcription_engine,
)
import soundfile as sf
import collections
import numpy as np
import traceback
import threading
import webrtcvad
import datetime
import platform
import logging
import struct
import base64
import queue
import torch
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

PORCUPINE_WAKEWORD_BACKENDS = {"pvp", "pvporcupine", "porcupine"}
OPENWAKEWORD_BACKENDS = {
    "oww",
    "openwakeword",
    "openwakewords",
    "open_wakeword",
    "open_wakewords",
}


def _normalize_wakeword_backend(wakeword_backend, wake_words):
    backend = (wakeword_backend or "").strip().lower().replace("-", "_")
    if not backend and wake_words:
        return "pvporcupine"
    return backend


def _load_porcupine_module():
    try:
        return import_module("pvporcupine")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Porcupine wake word detection requires the optional "
            "'pvporcupine' package. Install it with "
            "'pip install \"RealtimeSTT[porcupine]\"'."
        ) from exc


def _load_openwakeword_modules():
    try:
        openwakeword_module = import_module("openwakeword")
        model_module = import_module("openwakeword.model")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenWakeWord wake word detection requires the optional "
            "'openwakeword' package. Install it with "
            "'pip install \"RealtimeSTT[openwakeword]\"'."
        ) from exc
    return openwakeword_module, model_module.Model


class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe, transcription_engine, transcription_engine_options, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter, normalize_audio):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.transcription_engine = transcription_engine
        self.transcription_engine_options = transcription_engine_options or {}
        self.model_path = model_path
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.batch_size = batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        while not self.shutdown_event.is_set():
            try:
                # Use a longer timeout to reduce polling frequency
                if self.conn.poll(0.01):  # Increased from 0.01 to 0.5 seconds
                    data = self.conn.recv()
                    self.queue.put(data)
                else:
                    # Sleep only if no data, but use a shorter sleep
                    time.sleep(TIME_SLEEP)
            except Exception as e:
                logging.error(f"Error receiving data from connection: {e}", exc_info=True)
                time.sleep(TIME_SLEEP)

    def run(self):
        if __name__ == "__main__":
             system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
             __builtins__['print'] = self.custom_print

        logging.info(
            f"Initializing {self.transcription_engine} main transcription model {self.model_path}"
        )

        try:
            engine = create_transcription_engine(
                self.transcription_engine,
                TranscriptionEngineConfig(
                    model=self.model_path,
                    download_root=self.download_root,
                    compute_type=self.compute_type,
                    gpu_device_index=self.gpu_device_index,
                    device=self.device,
                    beam_size=self.beam_size,
                    initial_prompt=self.initial_prompt,
                    suppress_tokens=self.suppress_tokens,
                    batch_size=self.batch_size,
                    vad_filter=self.faster_whisper_vad_filter,
                    normalize_audio=self.normalize_audio,
                    engine_options=self.transcription_engine_options,
                ),
            )

            # Run a warm-up transcription
            current_dir = os.path.dirname(os.path.realpath(__file__))
            warmup_audio_path = os.path.join(
                current_dir, "warmup_audio.wav"
            )
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            engine.warmup(warmup_audio_data)
        except Exception as e:
            logging.exception(
                f"Error initializing main {self.transcription_engine} transcription model: {e}"
            )
            raise

        self.ready_event.set()
        logging.debug(
            f"{self.transcription_engine} main speech to text transcription model initialized successfully"
        )

        # Start the polling thread
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language, use_prompt = self.queue.get(timeout=0.1)
                    try:
                        logging.debug(f"Transcribing audio with language {language}")
                        start_t = time.time()
                        transcription_result = engine.transcribe(
                            audio,
                            language=language,
                            use_prompt=use_prompt,
                        )
                        elapsed = time.time() - start_t
                        logging.debug(
                            f"Final text detected with main model: {transcription_result.text} in {elapsed:.4f}s"
                        )
                        self.conn.send(('success', transcription_result))
                    except Exception as e:
                        logging.error(f"General error in transcription: {e}", exc_info=True)
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}", exc_info=True)
        finally:
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish


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

        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = (
            ensure_sentence_starting_uppercase
        )
        self.ensure_sentence_ends_with_period = (
            ensure_sentence_ends_with_period
        )
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.pre_recording_buffer_trim_config = dict(
            pre_recording_buffer_trim_config or {}
        )
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_turn_detection_start = on_turn_detection_start
        self.on_turn_detection_stop = on_turn_detection_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.enable_realtime_transcription = enable_realtime_transcription
        self.use_main_model_for_realtime = use_main_model_for_realtime
        self.transcription_engine = transcription_engine
        self.transcription_engine_options = transcription_engine_options or {}
        self.realtime_transcription_engine = (
            realtime_transcription_engine or transcription_engine
        )
        self.realtime_transcription_engine_options = (
            realtime_transcription_engine_options
            if realtime_transcription_engine_options is not None
            else self.transcription_engine_options
        )
        self.main_model_type = model
        if not download_root:
            download_root = None
        self.download_root = download_root
        self.realtime_model_type = realtime_model_type
        self.realtime_transcription_model = None
        self.realtime_processing_pause = realtime_processing_pause
        self.init_realtime_after_seconds = init_realtime_after_seconds
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.on_realtime_transcription_stabilized = (
            on_realtime_transcription_stabilized
        )
        self.on_realtime_text_stabilization_update = (
            on_realtime_text_stabilization_update
        )
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = allowed_latency_limit
        self.batch_size = batch_size
        self.realtime_batch_size = realtime_batch_size

        self.level = level
        self.audio_queue = mp.Queue() if use_microphone else queue.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_start_monotonic = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.silero_vad_lock = threading.Lock()
        self._silero_vad_generation = 0
        self.speech_end_silence_start = 0
        self.speech_end_silence_candidate_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_use_onnx = silero_use_onnx
        self.silero_backend = silero_backend
        self.silero_onnx_model_path = silero_onnx_model_path
        self.silero_onnx_threads = silero_onnx_threads
        self.silero_deactivity_detection = silero_deactivity_detection
        self.webrtc_sensitivity = webrtc_sensitivity
        self.warmup_vad = warmup_vad
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.realtime_text_stabilizer = RealtimeTextStabilizer()
        self.realtime_recording_id = 0
        self.realtime_observation_sequence = 0
        self.realtime_text_stabilization_event = None
        self.realtime_stabilization_accepted_count = 0
        self.realtime_stabilization_outlier_count = 0
        self.realtime_stabilization_stable_delta_count = 0
        self.realtime_stabilization_final_mismatch_count = 0
        self.is_webrtc_speech_active = False
        self.last_webrtc_speech_time = 0
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.recorded_audio_queue = queue.Queue()
        self.continuous_listening = False
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.last_transcription_metadata = None
        self.last_preroll_selection = None
        self._pending_preroll_selection = None
        self.initial_prompt = initial_prompt
        self.initial_prompt_realtime = initial_prompt_realtime
        self.suppress_tokens = suppress_tokens
        normalized_wakeword_backend = _normalize_wakeword_backend(
            wakeword_backend,
            wake_words,
        )
        self.use_wake_words = bool(
            wake_words or normalized_wakeword_backend in OPENWAKEWORD_BACKENDS
        )
        self.detected_language = None
        self.detected_language_probability = 0
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.transcription_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.transcribe_count = 0
        self.realtime_transcription_count = 0
        self.realtime_transcription_success_count = 0
        self.realtime_transcription_empty_count = 0
        self.realtime_transcription_trigger_counts = {}
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.use_extended_logging = use_extended_logging
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.awaiting_speech_end = False
        self.start_callback_in_new_thread = start_callback_in_new_thread
        self.realtime_transcription_use_syllable_boundaries = (
            realtime_transcription_use_syllable_boundaries
        )
        self.realtime_boundary_detector_sensitivity = realtime_boundary_detector_sensitivity
        self.realtime_boundary_followup_delays = realtime_boundary_followup_delays
        self.transcription_executor = transcription_executor
        self.realtime_transcription_executor = realtime_transcription_executor
        self._uses_external_transcription_executor = transcription_executor is not None
        self._uses_external_realtime_transcription_executor = (
            realtime_transcription_executor is not None
        )
        self._external_transcription_results = queue.Queue()
        self._external_transcription_threads = []

        # ----------------------------------------------------------------------------
        # Named logger configuration
        # By default, let's set it up so it logs at 'level' to the console.
        # If you do NOT want this default configuration, remove the lines below
        # and manage your "realtimestt" logger from your application code.
        logger.setLevel(logging.DEBUG)  # We capture all, then filter via handlers

        log_format = "RealTimeSTT: %(name)s - %(levelname)s - %(message)s"
        file_log_format = "%(asctime)s.%(msecs)03d - " + log_format

        # Create and set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(logging.Formatter(log_format))

        logger.addHandler(console_handler)

        if not no_log_file:
            file_handler = logging.FileHandler('realtimesst.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(file_log_format, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(file_handler)
        # ----------------------------------------------------------------------------

        self.is_shut_down = False
        self.shutdown_event = mp.Event()
        
        try:
            # Only set the start method if it hasn't been set already
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logger.info(f"Start method has already been set. Details: {e}")

        logger.info("Starting RealTimeSTT")

        if use_extended_logging:
            logger.info("RealtimeSTT was called with these parameters:")
            for param, value in locals().items():
                logger.info(f"{param}: {value}")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()

        self.transcript_process = None
        self.stdout_thread = None
        self.parent_transcription_pipe = None
        self.parent_stdout_pipe = None
        child_transcription_pipe = None
        child_stdout_pipe = None

        if not self._uses_external_transcription_executor:
            self.parent_transcription_pipe, child_transcription_pipe = SafePipe()
            self.parent_stdout_pipe, child_stdout_pipe = SafePipe()

        # Set device for model
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        if self._uses_external_transcription_executor:
            logger.info("Using external main transcription executor")
            self.main_transcription_ready_event.set()
        else:
            self.transcript_process = self._start_thread(
                target=AudioToTextRecorder._transcription_worker,
                args=(
                    child_transcription_pipe,
                    child_stdout_pipe,
                    self.transcription_engine,
                    self.transcription_engine_options,
                    self.main_model_type,
                    self.download_root,
                    self.compute_type,
                    self.gpu_device_index,
                    self.device,
                    self.main_transcription_ready_event,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.beam_size,
                    self.initial_prompt,
                    self.suppress_tokens,
                    self.batch_size,
                    self.faster_whisper_vad_filter,
                    self.normalize_audio,
                )
            )

        # Start audio data reading process
        if self.use_microphone.value:
            logger.info("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        # Initialize the realtime transcription model
        if (
            self.enable_realtime_transcription
            and not self.use_main_model_for_realtime
            and not self._uses_external_realtime_transcription_executor
        ):
            try:
                logger.info(
                             f"Initializing {self.realtime_transcription_engine} realtime "
                             f"transcription model {self.realtime_model_type}, "
                             f"default device: {self.device}, "
                             f"compute type: {self.compute_type}, "
                             f"device index: {self.gpu_device_index}, "
                             f"download root: {self.download_root}"
                             )
                self.realtime_transcription_model = create_transcription_engine(
                    self.realtime_transcription_engine,
                    TranscriptionEngineConfig(
                        model=self.realtime_model_type,
                        download_root=self.download_root,
                        compute_type=self.compute_type,
                        gpu_device_index=self.gpu_device_index,
                        device=self.device,
                        beam_size=self.beam_size_realtime,
                        initial_prompt=self.initial_prompt_realtime,
                        suppress_tokens=self.suppress_tokens,
                        batch_size=self.realtime_batch_size,
                        vad_filter=self.faster_whisper_vad_filter,
                        normalize_audio=self.normalize_audio,
                        engine_options=self.realtime_transcription_engine_options,
                    ),
                )

                # Run a warm-up transcription
                current_dir = os.path.dirname(os.path.realpath(__file__))
                warmup_audio_path = os.path.join(
                    current_dir, "warmup_audio.wav"
                )
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                self.realtime_transcription_model.warmup(warmup_audio_data)
            except Exception as e:
                logger.exception(
                    f"Error initializing {self.realtime_transcription_engine} realtime transcription model: {e}"
                )
                raise

            logger.debug(
                f"{self.realtime_transcription_engine} realtime speech to text transcription model initialized successfully"
            )

        # Setup wake word detection
        if self.use_wake_words or normalized_wakeword_backend in PORCUPINE_WAKEWORD_BACKENDS:
            self.wakeword_backend = normalized_wakeword_backend

            self.wake_words_list = [
                word.strip() for word in wake_words.lower().split(',')
                if word.strip()
            ] if wake_words else []
            self.wake_words_sensitivity = wake_words_sensitivity
            self.wake_words_sensitivities = [
                float(wake_words_sensitivity)
                for _ in range(len(self.wake_words_list))
            ]

            if self.wakeword_backend in PORCUPINE_WAKEWORD_BACKENDS:
                if not self.wake_words_list:
                    raise ValueError(
                        "Porcupine wake word detection requires wake_words. "
                        "Pass a comma-separated Porcupine keyword list, or use "
                        "wakeword_backend='openwakeword' for OpenWakeWord models."
                    )

                try:
                    pvporcupine = _load_porcupine_module()
                    self.porcupine = pvporcupine.create(
                        keywords=self.wake_words_list,
                        sensitivities=self.wake_words_sensitivities
                    )
                    self.buffer_size = self.porcupine.frame_length
                    self.sample_rate = self.porcupine.sample_rate

                except Exception as e:
                    logger.exception(
                        "Error initializing porcupine "
                        f"wake word detection engine: {e}. "
                        f"Wakewords: {self.wake_words_list}."
                    )
                    raise

                logger.debug(
                    "Porcupine wake word detection engine initialized successfully"
                )

            elif self.wakeword_backend in OPENWAKEWORD_BACKENDS:
                    
                try:
                    openwakeword, Model = _load_openwakeword_modules()
                    openwakeword.utils.download_models()

                    if openwakeword_model_paths:
                        model_paths = openwakeword_model_paths.split(',')
                        self.owwModel = Model(
                            wakeword_models=model_paths,
                            inference_framework=openwakeword_inference_framework
                        )
                        logger.info(
                            "Successfully loaded wakeword model(s): "
                            f"{openwakeword_model_paths}"
                        )
                    else:
                        self.owwModel = Model(
                            inference_framework=openwakeword_inference_framework)
                    
                    self.oww_n_models = len(self.owwModel.models.keys())
                    if not self.oww_n_models:
                        logger.error(
                            "No wake word models loaded."
                        )

                    for model_key in self.owwModel.models.keys():
                        logger.info(
                            "Successfully loaded openwakeword model: "
                            f"{model_key}"
                        )

                except Exception as e:
                    logger.exception(
                        "Error initializing openwakeword "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logger.debug(
                    "Open wake word detection engine initialized successfully"
                )
            
            else:
                raise ValueError(
                    f"Wakeword engine {self.wakeword_backend} unknown or unsupported. "
                    "Please specify one of: pvporcupine, openwakeword."
                )


        # Setup voice activity detection model WebRTC
        try:
            logger.info("Initializing WebRTC voice with "
                         f"Sensitivity {webrtc_sensitivity}"
                         )
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        except Exception as e:
            logger.exception("Error initializing WebRTC voice "
                              f"activity detection engine: {e}"
                              )
            raise

        logger.debug("WebRTC VAD voice activity detection "
                      "engine initialized successfully"
                      )

        # Setup voice activity detection model Silero VAD
        try:
            self.silero_vad_model = create_silero_vad_model(
                backend=silero_backend,
                silero_use_onnx=silero_use_onnx,
                onnx_model_path=silero_onnx_model_path,
                onnx_threads=silero_onnx_threads,
                sample_rate=self.sample_rate,
                chunk_samples=self.buffer_size,
                logger=logger,
            )

        except Exception as e:
            logger.exception(f"Error initializing Silero VAD "
                              f"voice activity detection engine: {e}"
                              )
            raise

        logger.debug(
            "Silero VAD voice activity detection engine initialized "
            "successfully with backend %s",
            getattr(self.silero_vad_model, "backend", silero_backend),
        )

        if self.warmup_vad:
            self._warmup_voice_activity_detectors()

        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_duration)
        )
        self.audio_buffer_metadata = collections.deque(
            maxlen=self.audio_buffer.maxlen
        )
        self.last_words_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       0.3)
        )
        self.frames = []
        self.last_frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start the realtime transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
                   
        # Wait for transcription models to start
        logger.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logger.debug('Main transcription model ready')

        if self.parent_stdout_pipe is not None:
            self.stdout_thread = threading.Thread(target=self._read_stdout)
            self.stdout_thread.daemon = True
            self.stdout_thread.start()

        logger.debug('RealtimeSTT initialization completed successfully')
                   
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

    def _transcription_worker(*args, **kwargs):
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()

    def _call_transcription_executor(self, executor, audio, language, use_prompt):
        if hasattr(executor, "transcribe"):
            return executor.transcribe(
                audio,
                language=language if language else None,
                use_prompt=use_prompt,
            )
        return executor(
            audio,
            language=language if language else None,
            use_prompt=use_prompt,
        )

    def _submit_transcription_request(self, audio, language, use_prompt):
        if self._uses_external_transcription_executor:
            audio_copy = copy.deepcopy(audio)

            def _run_external_transcription():
                try:
                    result = self._call_transcription_executor(
                        self.transcription_executor,
                        audio_copy,
                        language,
                        use_prompt,
                    )
                    self._external_transcription_results.put(("success", result))
                except Exception as exc:
                    self._external_transcription_results.put(("error", str(exc)))

            self.transcribe_count += 1
            thread = threading.Thread(
                target=_run_external_transcription,
                name="RealtimeSTTExternalFinalTranscription",
                daemon=True,
            )
            self._external_transcription_threads.append(thread)
            thread.start()
            return

        self.parent_transcription_pipe.send((audio, language, use_prompt))
        self.transcribe_count += 1

    def _receive_transcription_result(self, timeout=0.1):
        if self._uses_external_transcription_executor:
            try:
                return self._external_transcription_results.get(timeout=timeout)
            except queue.Empty:
                return None

        if not self.parent_transcription_pipe.poll(timeout):
            return None
        return self.parent_transcription_pipe.recv()

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
                self._reset_silero_vad_state()
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
                self._reset_silero_vad_state()
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
        self._clear_pre_recording_buffer()
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
                    self._submit_transcription_request(
                        audio_bytes,
                        self.language,
                        use_prompt,
                    )

                while self.transcribe_count > 0:
                    logger.debug(F"Receive from parent_transcription_pipe after sendiung transcription request, transcribe_count: {self.transcribe_count}")
                    response = self._receive_transcription_result(timeout=0.1)
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


    def _process_wakeword(self, data):
        """
        Processes audio data to detect wake words.
        """
        if self.wakeword_backend in PORCUPINE_WAKEWORD_BACKENDS:
            pcm = struct.unpack_from(
                "h" * self.buffer_size,
                data
            )
            porcupine_index = self.porcupine.process(pcm)
            if self.debug_mode:
                logger.info(f"wake words porcupine_index: {porcupine_index}")
            return porcupine_index

        elif self.wakeword_backend in OPENWAKEWORD_BACKENDS:
            pcm = np.frombuffer(data, dtype=np.int16)
            prediction = self.owwModel.predict(pcm)
            max_score = -1
            max_index = -1
            wake_words_in_prediction = len(self.owwModel.prediction_buffer.keys())
            self.wake_words_sensitivities
            if wake_words_in_prediction:
                for idx, mdl in enumerate(self.owwModel.prediction_buffer.keys()):
                    scores = list(self.owwModel.prediction_buffer[mdl])
                    if scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
                        max_score = scores[-1]
                        max_index = idx
                if self.debug_mode:
                    logger.info(f"wake words oww max_index, max_score: {max_index} {max_score}")
                return max_index  
            else:
                if self.debug_mode:
                    logger.info(f"wake words oww_index: -1")
                return -1

        if self.debug_mode:        
            logger.info("wake words no match")

        return -1

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
        self._reset_silero_vad_state()
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
        self._reset_silero_vad_state()
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
        self._reset_silero_vad_state()
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
                        self._run_callback(self.on_recorded_chunk, data)

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
                                self._run_callback(self.on_wakeword_timeout)
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
                            self._set_state("wakeword")
                        else:
                            if self.listen_start:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Setting state to "listening"')
                                self._set_state("listening")
                            else:
                                if self.use_extended_logging:
                                    logger.debug('Debug: Setting state to "inactive"')
                                self._set_state("inactive")

                    if self.use_extended_logging:
                        logger.debug('Debug: Checking wake word conditions')
                    if self.use_wake_words and wake_word_activation_delay_passed:
                        try:
                            if self.use_extended_logging:
                                logger.debug('Debug: Processing wakeword')
                            wakeword_index = self._process_wakeword(data)

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
                                self._run_callback(self.on_wakeword_detected)

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

                        if self._is_voice_active():

                            if self.on_vad_start:
                               self._run_callback(self.on_vad_start)

                            if self.use_extended_logging:
                                logger.debug('Debug: Voice activity detected')
                            logger.info("voice activity detected")

                            if self.use_extended_logging:
                                logger.debug('Debug: Starting recording')
                            pre_recording_frames = self._selected_pre_recording_buffer_frames()
                            self.start()

                            self.start_recording_on_voice_activity = False

                            if self.use_extended_logging:
                                logger.debug('Debug: Adding buffered audio to frames')
                            # Add the buffered audio
                            # to the recording frames
                            self.frames.extend(pre_recording_frames)
                            self._clear_pre_recording_buffer()

                            if self.use_extended_logging:
                                logger.debug('Debug: Resetting Silero VAD model states')
                            self._reset_silero_vad_state()
                        else:
                            if self.use_extended_logging:
                                logger.debug('Debug: Checking voice activity')
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)

                    if self.use_extended_logging:
                        logger.debug('Debug: Resetting speech_end_silence_start')

                    if self.speech_end_silence_start != 0:
                        self.speech_end_silence_start = 0
                        if self.on_turn_detection_stop:
                            if self.use_extended_logging:
                                logger.debug('Debug: Calling on_turn_detection_stop')
                            self._run_callback(self.on_turn_detection_stop)

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
                            self._is_silero_speech(data) if self.silero_deactivity_detection
                            else self._is_webrtc_speech(data)
                        )
                        if is_speech:
                            self.speech_end_silence_candidate_start = 0
                        elif not self.speech_end_silence_start:
                            now = time.time()
                            if not self.speech_end_silence_candidate_start:
                                self.speech_end_silence_candidate_start = now
                            if (
                                now - self.speech_end_silence_candidate_start
                                < DEACTIVITY_SILENCE_CONFIRMATION_DURATION
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

                                    self._run_callback(self.on_turn_detection_start)

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
                                    self._submit_transcription_request(
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
                                        self._run_callback(self.on_turn_detection_stop)

                                self.allowed_to_early_transcribe = True

                        if self.use_extended_logging:
                            logger.debug('Debug: Checking if silence duration exceeds threshold')
                        # Wait for silence to stop recording after speech
                        if self.speech_end_silence_start and time.time() - \
                                self.speech_end_silence_start >= \
                                self.post_speech_silence_duration:

                            if self.on_vad_stop:
                                self._run_callback(self.on_vad_stop)

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
                                        self._run_callback(self.on_turn_detection_stop)

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
                    self._clear_pre_recording_buffer()

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
                        self._run_callback(self.on_wakeword_timeout)
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
                    self._append_to_pre_recording_buffer(data)

        except Exception as e:
            logger.debug('Debug: Caught exception in main try block')
            if not self.interrupt_stop_event.is_set():
                logger.error(f"Unhandled exeption in _recording_worker: {e}", exc_info=True)
                raise

        if self.use_extended_logging:
            logger.debug('Debug: Exiting _recording_worker method')

    def _realtime_worker(self):
        """
        Performs real-time transcription if the feature is enabled.

        This worker is intentionally defensive:
        - realtime transcription must never crash the recorder
        - empty/None buffers are skipped
        - frame buffers are snapshotted before transcription
        - model/pipe errors are logged and skipped
        """

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
                    return self._call_transcription_executor(
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
                    return self._call_transcription_executor(
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
                self._run_callback(callback, *args)
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
                self._preprocess_output(stabilized_display_text, True),
            )

            _safe_realtime_callback(
                self._on_realtime_transcription_update,
                self._preprocess_output(raw_text.strip(), True),
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


    # def _realtime_worker(self):
    #     """
    #     Performs real-time transcription if the feature is enabled.

    #     The method is responsible transcribing recorded audio frames
    #       in real-time based on the specified resolution interval.
    #     The transcribed text is stored in `self.realtime_transcription_text`
    #       and a callback
    #     function is invoked with this text if specified.
    #     """

    #     try:

    #         logger.debug('Starting realtime worker')

    #         # Return immediately if real-time transcription is not enabled
    #         if not self.enable_realtime_transcription:
    #             return

    #         # Track time of last transcription
    #         last_transcription_time = time.time()

    #         while self.is_running:

    #             if self.is_recording:

    #                 # MODIFIED SLEEP LOGIC:
    #                 # Wait until realtime_processing_pause has elapsed,
    #                 # but check often so we can respond to changes quickly.
    #                 while (
    #                     time.time() - last_transcription_time
    #                 ) < self.realtime_processing_pause:
    #                     time.sleep(0.001)
    #                     if not self.is_running or not self.is_recording:
    #                         break

    #                 if self.awaiting_speech_end:
    #                     time.sleep(0.001)
    #                     continue

    #                 # Update transcription time
    #                 last_transcription_time = time.time()

    #                 # Convert the buffer frames to a NumPy array
    #                 audio_array = np.frombuffer(
    #                     b''.join(self.frames),
    #                     dtype=np.int16
    #                     )

    #                 logger.debug(f"Current realtime buffer size: {len(audio_array)}")

    #                 # Normalize the array to a [-1, 1] range
    #                 audio_array = audio_array.astype(np.float32) / \
    #                     INT16_MAX_ABS_VALUE

    #                 if self.use_main_model_for_realtime:
    #                     with self.transcription_lock:
    #                         try:
    #                             self.parent_transcription_pipe.send((audio_array, self.language, True))
    #                             if self.parent_transcription_pipe.poll(timeout=5):  # Wait for 5 seconds
    #                                 logger.debug("Receive from realtime worker after transcription request to main model")
    #                                 status, result = self.parent_transcription_pipe.recv()
    #                                 if status == 'success':
    #                                     self.detected_realtime_language = (
    #                                         result.info.language if result.info.language_probability > 0 else None
    #                                     )
    #                                     self.detected_realtime_language_probability = result.info.language_probability
    #                                     realtime_text = result.text
    #                                     logger.debug(f"Realtime text detected with main model: {realtime_text}")
    #                                 else:
    #                                     logger.error(f"Realtime transcription error: {result}")
    #                                     continue
    #                             else:
    #                                 logger.warning("Realtime transcription timed out")
    #                                 continue
    #                         except Exception as e:
    #                             logger.error(f"Error in realtime transcription: {str(e)}", exc_info=True)
    #                             continue
    #                 else:
    #                     transcription_result = self.realtime_transcription_model.transcribe(
    #                         audio_array,
    #                         language=self.language if self.language else None,
    #                         use_prompt=True,
    #                     )
    #                     self.detected_realtime_language = (
    #                         transcription_result.info.language
    #                         if transcription_result.info.language_probability > 0
    #                         else None
    #                     )
    #                     self.detected_realtime_language_probability = (
    #                         transcription_result.info.language_probability
    #                     )
    #                     realtime_text = transcription_result.text
    #                     logger.debug(f"Realtime text detected: {realtime_text}")

    #                 # double check recording state
    #                 # because it could have changed mid-transcription
    #                 if self.is_recording and time.time() - \
    #                         self.recording_start_time > self.init_realtime_after_seconds:

    #                     self.realtime_transcription_text = realtime_text
    #                     self.realtime_transcription_text = \
    #                         self.realtime_transcription_text.strip()

    #                     self.text_storage.append(
    #                         self.realtime_transcription_text
    #                         )

    #                     # Take the last two texts in storage, if they exist
    #                     if len(self.text_storage) >= 2:
    #                         last_two_texts = self.text_storage[-2:]

    #                         # Find the longest common prefix
    #                         # between the two texts
    #                         prefix = os.path.commonprefix(
    #                             [last_two_texts[0], last_two_texts[1]]
    #                             )

    #                         # This prefix is the text that was transcripted
    #                         # two times in the same way
    #                         # Store as "safely detected text"
    #                         if len(prefix) >= \
    #                                 len(self.realtime_stabilized_safetext):

    #                             # Only store when longer than the previous
    #                             # as additional security
    #                             self.realtime_stabilized_safetext = prefix

    #                     # Find parts of the stabilized text
    #                     # in the freshly transcripted text
    #                     matching_pos = self._find_tail_match_in_text(
    #                         self.realtime_stabilized_safetext,
    #                         self.realtime_transcription_text
    #                         )

    #                     if matching_pos < 0:
    #                         # pick which text to send
    #                         text_to_send = (
    #                             self.realtime_stabilized_safetext
    #                             if self.realtime_stabilized_safetext
    #                             else self.realtime_transcription_text
    #                         )
    #                         # preprocess once
    #                         processed = self._preprocess_output(text_to_send, True)
    #                         # invoke on its own thread
    #                         self._run_callback(self._on_realtime_transcription_stabilized, processed)

    #                     else:
    #                         # We found parts of the stabilized text
    #                         # in the transcripted text
    #                         # We now take the stabilized text
    #                         # and add only the freshly transcripted part to it
    #                         output_text = self.realtime_stabilized_safetext + \
    #                             self.realtime_transcription_text[matching_pos:]

    #                         # This yields us the "left" text part as stabilized
    #                         # AND at the same time delivers fresh detected
    #                         # parts on the first run without the need for
    #                         # two transcriptions
    #                         self._run_callback(self._on_realtime_transcription_stabilized, self._preprocess_output(output_text, True))

    #                     # Invoke the callback with the transcribed text
    #                     self._run_callback(self._on_realtime_transcription_update, self._preprocess_output(self.realtime_transcription_text,True))

    #             # If not recording, sleep briefly before checking again
    #             else:
    #                 time.sleep(TIME_SLEEP)

    #     except Exception as e:
    #         logger.error(f"Unhandled exeption in _realtime_worker: {e}", exc_info=True)
    #         raise

    def _silero_vad_probability(self, audio_chunk):
        result = self.silero_vad_model(audio_chunk, SAMPLE_RATE)
        if isinstance(result, (float, int)):
            return float(result)
        if hasattr(result, "item"):
            return float(result.item())
        return float(np.asarray(result).reshape(-1)[0])

    def _reset_silero_vad_state(self):
        """
        Reset Silero's recurrent state and the recorder-side Silero flag.

        Silero VAD keeps hidden state between chunk calls. That is useful while
        evaluating one continuous stream, but it must not leak across warmup,
        listening attempts, or completed recordings.
        """
        self._silero_vad_generation = (
            getattr(self, "_silero_vad_generation", 0) + 1
        )
        reset_states = getattr(
            getattr(self, "silero_vad_model", None),
            "reset_states",
            None,
        )
        if reset_states:
            try:
                lock = getattr(self, "silero_vad_lock", None)
                if lock is None:
                    reset_states()
                else:
                    with lock:
                        reset_states()
            except Exception:
                logger.debug("Silero VAD state reset skipped", exc_info=True)
        self.is_silero_speech_active = False

    def _warmup_voice_activity_detectors(self):
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
            self.webrtc_vad_model.is_speech(silence_frame, 16000)
        except Exception:
            logger.debug("WebRTC VAD warmup skipped", exc_info=True)

        try:
            sample_count = max(1, int(getattr(self, "buffer_size", BUFFER_SIZE)))
            t = np.arange(sample_count, dtype=np.float32) / float(SAMPLE_RATE)
            # A quiet tone exercises the model path without marking the
            # recorder as actively recording or speech-active.
            tone = (0.03 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            silence = np.zeros(sample_count, dtype=np.float32)

            self._silero_vad_probability(tone)
            self._silero_vad_probability(silence)

            self._reset_silero_vad_state()
        except Exception:
            logger.debug("Silero VAD warmup skipped", exc_info=True)

        self.silero_working = False
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.last_webrtc_speech_time = 0

    def _is_silero_speech(self, chunk, generation=None):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        if generation is None:
            generation = getattr(self, "_silero_vad_generation", 0)

        self.silero_working = True
        try:
            if generation != getattr(self, "_silero_vad_generation", 0):
                return False

            if self.sample_rate != 16000:
                pcm_data = np.frombuffer(chunk, dtype=np.int16)
                data_16000 = signal.resample_poly(
                    pcm_data, 16000, self.sample_rate)
                chunk = data_16000.astype(np.int16).tobytes()

            if generation != getattr(self, "_silero_vad_generation", 0):
                return False

            audio_chunk = np.frombuffer(chunk, dtype=np.int16)
            audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
            lock = getattr(self, "silero_vad_lock", None)
            if lock is None:
                vad_prob = self._silero_vad_probability(audio_chunk)
            else:
                with lock:
                    if generation != getattr(self, "_silero_vad_generation", 0):
                        return False
                    vad_prob = self._silero_vad_probability(audio_chunk)

            if generation != getattr(self, "_silero_vad_generation", 0):
                return False

            is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
            if is_silero_speech_active:
                if not self.is_silero_speech_active and self.use_extended_logging:
                    logger.info(f"{bcolors.OKGREEN}Silero VAD detected speech{bcolors.ENDC}")
            elif self.is_silero_speech_active and self.use_extended_logging:
                logger.info(f"{bcolors.WARNING}Silero VAD detected silence{bcolors.ENDC}")
            self.is_silero_speech_active = is_silero_speech_active
            return is_silero_speech_active
        finally:
            self.silero_working = False

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with
            16000 sample rate and 16 bits per sample)
        """
        speech_str = f"{bcolors.OKGREEN}WebRTC VAD detected speech{bcolors.ENDC}"
        silence_str = f"{bcolors.WARNING}WebRTC VAD detected silence{bcolors.ENDC}"
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        # Number of audio frames per millisecond
        frame_length = int(16000 * 0.01)  # for 10ms frame
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    if self.debug_mode:
                        logger.info(f"Speech detected in frame {i + 1}"
                              f" of {num_frames}")
                    if not self.is_webrtc_speech_active and self.use_extended_logging:
                        logger.info(speech_str)
                    self.is_webrtc_speech_active = True
                    self.last_webrtc_speech_time = time.time()
                    return True
        if all_frames_must_be_true:
            if self.debug_mode and speech_frames == num_frames:
                logger.info(f"Speech detected in {speech_frames} of "
                      f"{num_frames} frames")
            elif self.debug_mode:
                logger.info(f"Speech not detected in all {num_frames} frames")
            speech_detected = speech_frames == num_frames
            if speech_detected and not self.is_webrtc_speech_active and self.use_extended_logging:
                logger.info(speech_str)
            elif not speech_detected and self.is_webrtc_speech_active and self.use_extended_logging:
                logger.info(silence_str)
            self.is_webrtc_speech_active = speech_detected
            return speech_detected
        else:
            if self.debug_mode:
                logger.info(f"Speech not detected in any of {num_frames} frames")
            if self.is_webrtc_speech_active and self.use_extended_logging:
                logger.info(silence_str)
            self.is_webrtc_speech_active = False
            return False

    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        was_webrtc_speech_active = self.is_webrtc_speech_active
        self._is_webrtc_speech(data)

        # First quick performing check for voice activity using WebRTC
        if self.is_webrtc_speech_active:

            if not self.silero_working:
                if not was_webrtc_speech_active:
                    self._reset_silero_vad_state()
                self.silero_working = True
                silero_generation = getattr(self, "_silero_vad_generation", 0)

                # Run the intensive check in a separate thread
                threading.Thread(
                    target=self._is_silero_speech,
                    args=(data, silero_generation)).start()

    def _pre_recording_buffer_trim_enabled(self):
        config = getattr(self, "pre_recording_buffer_trim_config", None) or {}
        return bool(config.get("enabled", False))

    def _append_to_pre_recording_buffer(self, data):
        self.audio_buffer.append(data)
        metadata_buffer = getattr(self, "audio_buffer_metadata", None)
        if metadata_buffer is not None:
            metadata_buffer.append(self._preroll_frame_metadata(data))

    def _clear_pre_recording_buffer(self):
        self.audio_buffer.clear()
        metadata_buffer = getattr(self, "audio_buffer_metadata", None)
        if metadata_buffer is not None:
            metadata_buffer.clear()

    def _selected_pre_recording_buffer_frames(self):
        frames = list(self.audio_buffer)
        self._pending_preroll_selection = None
        if not frames:
            return frames

        if not self._pre_recording_buffer_trim_enabled():
            return frames

        metadata = list(getattr(self, "audio_buffer_metadata", ()))
        if len(metadata) != len(frames):
            metadata = [self._metadata_for_frame_without_vad(frame) for frame in frames]

        config = getattr(self, "pre_recording_buffer_trim_config", None) or {}
        selection = select_preroll_frames(
            metadata,
            int(getattr(self, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE),
            min_silence_ms=config.get("min_silence_ms", 200.0),
            guard_ms=config.get("guard_ms", 160.0),
            max_gap_ms=config.get("max_gap_ms", 80.0),
            min_included_ms=config.get("min_included_ms", 600.0),
            energy_silence_rms=config.get("energy_silence_rms"),
            noise_floor_multiplier=config.get("noise_floor_multiplier", 2.5),
            energy_margin_rms=config.get("energy_margin_rms", 25.0),
        )
        selection.diagnostics.update(self._webrtc_replay_preroll_diagnostics(frames))
        self._pending_preroll_selection = selection
        return frames[selection.start_index:]

    def _preroll_frame_metadata(self, data):
        sample_count = max(0, len(data) // 2)
        rms = self._frame_rms(data)
        webrtc_is_speech = bool(getattr(self, "is_webrtc_speech_active", False))
        silero_is_speech = bool(getattr(self, "is_silero_speech_active", False))
        is_speech = webrtc_is_speech or silero_is_speech
        return PrerollFrameMetadata(
            sample_count=sample_count,
            is_speech=is_speech,
            rms=rms,
            webrtc_is_speech=webrtc_is_speech,
            silero_is_speech=silero_is_speech,
        )

    def _metadata_for_frame_without_vad(self, frame):
        return PrerollFrameMetadata(
            sample_count=max(0, len(frame) // 2),
            is_speech=None,
            rms=self._frame_rms(frame),
        )

    def _frame_rms(self, data):
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

    def _webrtc_replay_preroll_diagnostics(self, frames):
        speech_sample_count = 0
        analyzed_sample_count = 0
        frame_length = int(16000 * 0.01)
        sample_rate = int(getattr(self, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)

        try:
            vad_model = webrtcvad.Vad(int(getattr(self, "webrtc_sensitivity", 3)))
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

    def clear_audio_queue(self):
        """
        Safely empties the audio queue to ensure no remaining audio 
        fragments get processed e.g. after waking up the recorder.
        """
        self._clear_pre_recording_buffer()
        try:
            while True:
                self.audio_queue.get_nowait()
        except:
            # PyTorch's mp.Queue doesn't have a specific Empty exception
            # so we catch any exception that might occur when the queue is empty
            pass

    def _is_voice_active(self):
        """
        Determine if voice is active.

        Returns:
            bool: True if voice is active, False otherwise.
        """
        webrtc_speech_recent = (
            time.time() - getattr(self, "last_webrtc_speech_time", 0) <= 1.0
        )
        return (
            (self.is_webrtc_speech_active or webrtc_speech_recent)
            and self.is_silero_speech_active
        )

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
