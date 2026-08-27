"""
Internal recorder initialization helpers.
"""

from ctypes import c_bool
import collections
import logging
import os
import queue
import threading

import soundfile as sf
import torch
import torch.multiprocessing as mp
import webrtcvad

from .realtime import (
    _normalize_realtime_punctuation_split_marks,
    run_realtime_worker,
)
from .realtime_text_stabilizer import RealtimeTextStabilizer
from .realtime_merge import StickyRealtimeTranscriptionMerger
from .preview_transcription import start_preview_transcription_worker
from .recording import run_recording_worker
from ..transcription_engines import (
    TranscriptionEngineConfig,
    create_transcription_engine,
)
from .silero_vad import create_silero_vad_model
from .safepipe import SafePipe
from .wakeword import OPENWAKEWORD_BACKENDS, setup_wakeword_detection
from .audio_input_worker import run_audio_data_worker
from .runtime import read_stdout_pipe, start_recorder_worker
from .transcription import (
    SharedFinalModelExecutor,
    run_transcription_worker,
)
from .voice_activity import warmup_voice_activity_detectors


logger = logging.getLogger("realtimestt")


def initialize_recorder(
    recorder,
    recorder_cls,
    init_args,
    normalize_wakeword_backend,
    load_porcupine_module,
    load_openwakeword_modules,
):
    """
    Initializes recorder state, models, workers, and callbacks.
    """
    normalized_wakeword_backend = _assign_initial_attributes(
        recorder,
        init_args,
        normalize_wakeword_backend,
    )

    _configure_logger(recorder, init_args["no_log_file"], init_args)
    _initialize_shutdown_state(recorder)
    _configure_multiprocessing_start_method()

    logger.info("Starting RealTimeSTT")

    _log_extended_initialization(init_args["use_extended_logging"], init_args)
    _initialize_transcription_runtime(recorder, recorder_cls)
    _start_audio_reader(recorder, recorder_cls)
    _initialize_realtime_transcription_model(recorder)
    _initialize_preview_transcription_model(recorder)
    _initialize_wakeword_detection(
        recorder,
        normalized_wakeword_backend,
        init_args["wake_words"],
        init_args["wake_words_sensitivity"],
        init_args["openwakeword_model_paths"],
        init_args["openwakeword_inference_framework"],
        load_porcupine_module,
        load_openwakeword_modules,
    )
    _initialize_voice_activity_detection(
        recorder,
        init_args["webrtc_sensitivity"],
        init_args["silero_backend"],
        init_args["silero_use_onnx"],
        init_args["silero_onnx_model_path"],
        init_args["silero_onnx_threads"],
    )
    _initialize_recording_buffers(recorder)
    _start_worker_threads(recorder)
    _finish_initialization(recorder)


def _assign_initial_attributes(recorder, init_args, normalize_wakeword_backend):
    """
    Copies constructor settings onto the recorder instance.
    """

    recorder.language = init_args["language"]
    recorder.compute_type = init_args["compute_type"]
    recorder.input_device_index = init_args["input_device_index"]
    recorder.gpu_device_index = init_args["gpu_device_index"]
    recorder.device = init_args["device"]
    recorder.wake_words = init_args["wake_words"]
    recorder.wake_word_activation_delay = init_args["wake_word_activation_delay"]
    recorder.wake_word_timeout = init_args["wake_word_timeout"]
    recorder.wake_word_buffer_duration = init_args["wake_word_buffer_duration"]
    recorder.ensure_sentence_starting_uppercase = (
        init_args["ensure_sentence_starting_uppercase"]
    )
    recorder.ensure_sentence_ends_with_period = (
        init_args["ensure_sentence_ends_with_period"]
    )
    recorder.use_microphone = mp.Value(c_bool, init_args["use_microphone"])
    recorder.min_gap_between_recordings = init_args["min_gap_between_recordings"]
    recorder.min_length_of_recording = init_args["min_length_of_recording"]
    recorder.pre_recording_buffer_duration = (
        init_args["pre_recording_buffer_duration"]
    )
    recorder.pre_recording_buffer_trim_config = dict(
        init_args["pre_recording_buffer_trim_config"] or {}
    )
    recorder.post_speech_silence_duration = (
        init_args["post_speech_silence_duration"]
    )
    recorder.deactivity_silence_confirmation_duration = (
        init_args["deactivity_silence_confirmation_duration"]
    )
    recorder.on_recording_start = init_args["on_recording_start"]
    recorder.on_recording_stop = init_args["on_recording_stop"]
    recorder.on_wakeword_detected = init_args["on_wakeword_detected"]
    recorder.on_wakeword_timeout = init_args["on_wakeword_timeout"]
    recorder.on_vad_start = init_args["on_vad_start"]
    recorder.on_vad_stop = init_args["on_vad_stop"]
    recorder.on_vad_detect_start = init_args["on_vad_detect_start"]
    recorder.on_vad_detect_stop = init_args["on_vad_detect_stop"]
    recorder.on_turn_detection_start = init_args["on_turn_detection_start"]
    recorder.on_turn_detection_stop = init_args["on_turn_detection_stop"]
    recorder.on_wakeword_detection_start = (
        init_args["on_wakeword_detection_start"]
    )
    recorder.on_wakeword_detection_end = init_args["on_wakeword_detection_end"]
    recorder.on_recorded_chunk = init_args["on_recorded_chunk"]
    recorder.on_transcription_start = init_args["on_transcription_start"]
    recorder.enable_realtime_transcription = (
        init_args["enable_realtime_transcription"]
    )
    recorder.use_main_model_for_realtime = init_args["use_main_model_for_realtime"]
    recorder.transcription_engine = init_args["transcription_engine"]
    recorder.transcription_engine_options = (
        init_args["transcription_engine_options"] or {}
    )
    recorder.realtime_transcription_engine = (
        init_args["realtime_transcription_engine"]
        or init_args["transcription_engine"]
    )
    recorder.realtime_transcription_engine_options = (
        init_args["realtime_transcription_engine_options"]
        if init_args["realtime_transcription_engine_options"] is not None
        else recorder.transcription_engine_options
    )
    recorder.main_model_type = init_args["model"]
    download_root = init_args["download_root"]
    if not download_root:
        download_root = None
    init_args["download_root"] = download_root
    recorder.download_root = download_root
    recorder.realtime_model_type = init_args["realtime_model_type"]
    recorder.realtime_transcription_model = None
    recorder.realtime_processing_pause = init_args["realtime_processing_pause"]
    recorder.init_realtime_after_seconds = (
        init_args["init_realtime_after_seconds"]
    )
    recorder.on_realtime_transcription_update = (
        init_args["on_realtime_transcription_update"]
    )
    recorder.on_realtime_transcription_stabilized = (
        init_args["on_realtime_transcription_stabilized"]
    )
    recorder.on_realtime_text_stabilization_update = (
        init_args["on_realtime_text_stabilization_update"]
    )
    recorder.ultrafast_realtime_model_type = init_args[
        "ultrafast_realtime_model_type"
    ]
    recorder.ultrafast_realtime_transcription_engine = (
        init_args["ultrafast_realtime_transcription_engine"]
        or recorder.realtime_transcription_engine
    )
    recorder.ultrafast_realtime_transcription_engine_options = (
        init_args["ultrafast_realtime_transcription_engine_options"]
        if init_args["ultrafast_realtime_transcription_engine_options"]
        is not None
        else recorder.realtime_transcription_engine_options
    )
    recorder.ultrafast_realtime_transcription_model = None
    recorder._ultrafast_uses_realtime_model = False
    recorder.on_ultrafast_transcription_update = init_args[
        "on_ultrafast_transcription_update"
    ]
    recorder.on_merged_realtime_transcription_update = init_args[
        "on_merged_realtime_transcription_update"
    ]
    recorder.on_realtime_transcription_merge_update = init_args[
        "on_realtime_transcription_merge_update"
    ]
    recorder.ultrafast_realtime_max_tail_words = int(
        init_args["ultrafast_realtime_max_tail_words"]
    )

    recorder.enable_preview_transcription = bool(
        init_args["enable_preview_transcription"]
    )
    recorder.preview_model_type = (
        init_args["preview_model_type"] or init_args["model"]
    )
    recorder.preview_transcription_engine = (
        init_args["preview_transcription_engine"]
        or init_args["transcription_engine"]
    )
    recorder.preview_transcription_engine_options = (
        init_args["preview_transcription_engine_options"]
        if init_args["preview_transcription_engine_options"] is not None
        else recorder.transcription_engine_options
    )
    recorder.preview_transcription_tail_seconds = float(
        init_args["preview_transcription_tail_seconds"]
    )
    recorder.preview_transcription_min_live_words_for_fuzzy_repair = int(
        init_args["preview_transcription_min_live_words_for_fuzzy_repair"]
    )
    recorder.on_preview_transcription_finished = (
        init_args["on_preview_transcription_finished"]
    )
    recorder.preview_transcription_executor = (
        init_args["preview_transcription_executor"]
    )
    recorder._uses_external_preview_transcription_executor = (
        recorder.preview_transcription_executor is not None
    )
    recorder.preview_transcription_uses_main_model = False
    recorder._preview_uses_main_model = False
    recorder._preview_uses_shared_final_worker = False
    recorder.preview_transcription_model = None
    recorder.preview_transcription_worker = None
    recorder.realtime_punctuation_split_marks = (
        _normalize_realtime_punctuation_split_marks(
            init_args["realtime_punctuation_split_marks"]
        )
    )
    recorder.debug_mode = init_args["debug_mode"]
    recorder.handle_buffer_overflow = init_args["handle_buffer_overflow"]
    recorder.beam_size = init_args["beam_size"]
    recorder.beam_size_realtime = init_args["beam_size_realtime"]
    recorder.allowed_latency_limit = init_args["allowed_latency_limit"]
    recorder.batch_size = init_args["batch_size"]
    recorder.realtime_batch_size = init_args["realtime_batch_size"]

    recorder.level = init_args["level"]
    recorder.audio_queue = (
        mp.Queue() if init_args["use_microphone"] else queue.Queue()
    )
    recorder.buffer_size = init_args["buffer_size"]
    recorder.sample_rate = init_args["sample_rate"]
    recorder.recording_start_time = 0
    recorder.recording_start_monotonic = 0
    recorder.recording_stop_time = 0
    recorder.last_recording_start_time = 0
    recorder.last_recording_stop_time = 0
    recorder.wake_word_detect_time = 0
    recorder.silero_check_time = 0
    recorder.silero_working = False
    recorder.silero_vad_lock = threading.Lock()
    recorder._silero_vad_generation = 0
    recorder.speech_end_silence_start = 0
    recorder.speech_end_silence_candidate_start = 0
    recorder.silero_sensitivity = init_args["silero_sensitivity"]
    recorder.silero_use_onnx = init_args["silero_use_onnx"]
    recorder.silero_backend = init_args["silero_backend"]
    recorder.silero_onnx_model_path = init_args["silero_onnx_model_path"]
    recorder.silero_onnx_threads = init_args["silero_onnx_threads"]
    recorder.silero_deactivity_detection = (
        init_args["silero_deactivity_detection"]
    )
    recorder.webrtc_sensitivity = init_args["webrtc_sensitivity"]
    recorder.warmup_vad = init_args["warmup_vad"]
    recorder.listen_start = 0
    recorder.spinner = init_args["spinner"]
    recorder.halo = None
    recorder.state = "inactive"
    recorder.wakeword_detected = False
    recorder.text_storage = []
    recorder.realtime_stabilized_text = ""
    recorder.realtime_stabilized_safetext = ""
    recorder.realtime_text_stabilizer = RealtimeTextStabilizer()
    recorder.realtime_transcription_merger = StickyRealtimeTranscriptionMerger(
        max_ultrafast_tail_words=(
            recorder.ultrafast_realtime_max_tail_words
        )
    )
    recorder.realtime_transcription_merger.reset(0)
    recorder.realtime_recording_id = 0
    recorder.realtime_observation_sequence = 0
    recorder.ultrafast_realtime_observation_sequence = 0
    recorder.ultrafast_realtime_transcription_text = ""
    recorder.merged_realtime_transcription_text = ""
    recorder.last_ultrafast_transcription = ""
    recorder.last_merged_realtime_transcription = ""
    recorder.last_realtime_transcription_merge_result = None
    recorder.realtime_text_stabilization_event = None
    recorder.realtime_stabilization_accepted_count = 0
    recorder.realtime_stabilization_outlier_count = 0
    recorder.realtime_stabilization_stable_delta_count = 0
    recorder.realtime_stabilization_final_mismatch_count = 0
    recorder.is_webrtc_speech_active = False
    recorder.last_webrtc_speech_time = 0
    recorder.is_silero_speech_active = False
    recorder.recording_thread = None
    recorder.realtime_thread = None
    recorder.audio_interface = None
    recorder.audio = None
    recorder.stream = None
    recorder.start_recording_event = threading.Event()
    recorder.stop_recording_event = threading.Event()
    recorder.backdate_stop_seconds = 0.0
    recorder.backdate_resume_seconds = 0.0
    recorder.recorded_audio_queue = queue.Queue()
    recorder._force_current_recording_lowercase_start = False
    recorder._current_transcription_force_lowercase_start = False
    recorder._realtime_punctuation_split_lock = threading.RLock()
    recorder._realtime_punctuation_split_busy = False
    recorder._last_realtime_punctuation_split_attempt_text = ""
    recorder._realtime_punctuation_split_candidate = None
    recorder.continuous_listening = False
    recorder.last_transcription_bytes = None
    recorder.last_transcription_bytes_b64 = None
    recorder.last_transcription_metadata = None
    recorder.active_speech_tail_buffer = bytearray()
    recorder._current_transcription_live_text = None
    recorder._current_transcription_tail_audio = None
    recorder._preview_transcription_submitted = False
    recorder._pending_vad_live_text = None
    recorder.preview_transcription_queue = None
    recorder.preview_transcription_stop_event = None
    recorder.preview_transcription_thread = None
    recorder.last_preview_transcription = ""
    recorder.last_preview_transcription_result = None
    recorder.last_preroll_selection = None
    recorder._pending_preroll_selection = None
    recorder.initial_prompt = init_args["initial_prompt"]
    recorder.initial_prompt_realtime = init_args["initial_prompt_realtime"]
    recorder.suppress_tokens = init_args["suppress_tokens"]
    normalized_wakeword_backend = normalize_wakeword_backend(
        init_args["wakeword_backend"],
        init_args["wake_words"],
    )
    init_args["normalized_wakeword_backend"] = normalized_wakeword_backend
    recorder.use_wake_words = bool(
        init_args["wake_words"]
        or normalized_wakeword_backend in OPENWAKEWORD_BACKENDS
    )
    recorder.detected_language = None
    recorder.detected_language_probability = 0
    recorder.detected_realtime_language = None
    recorder.detected_realtime_language_probability = 0
    recorder.transcription_lock = threading.Lock()
    recorder.shutdown_lock = threading.Lock()
    recorder.transcribe_count = 0
    recorder.realtime_transcription_count = 0
    recorder.realtime_transcription_success_count = 0
    recorder.realtime_transcription_empty_count = 0
    recorder.realtime_transcription_trigger_counts = {}
    recorder.print_transcription_time = init_args["print_transcription_time"]
    recorder.early_transcription_on_silence = (
        init_args["early_transcription_on_silence"]
    )
    recorder.use_extended_logging = init_args["use_extended_logging"]
    recorder.faster_whisper_vad_filter = init_args["faster_whisper_vad_filter"]
    recorder.normalize_audio = init_args["normalize_audio"]
    recorder.final_transcription_word_timestamps = init_args[
        "final_transcription_word_timestamps"
    ]
    recorder.awaiting_speech_end = False
    recorder.start_callback_in_new_thread = (
        init_args["start_callback_in_new_thread"]
    )
    recorder.realtime_transcription_use_syllable_boundaries = (
        init_args["realtime_transcription_use_syllable_boundaries"]
    )
    recorder.realtime_boundary_detector_sensitivity = (
        init_args["realtime_boundary_detector_sensitivity"]
    )
    recorder.realtime_boundary_followup_delays = (
        init_args["realtime_boundary_followup_delays"]
    )
    recorder.transcription_executor = init_args["transcription_executor"]
    recorder.realtime_transcription_executor = (
        init_args["realtime_transcription_executor"]
    )
    recorder._uses_external_transcription_executor = (
        init_args["transcription_executor"] is not None
    )
    recorder._uses_external_realtime_transcription_executor = (
        init_args["realtime_transcription_executor"] is not None
    )
    recorder._external_transcription_results = queue.Queue()
    recorder._external_transcription_threads = []

    return normalized_wakeword_backend


def _configure_logger(recorder, no_log_file, init_args=None):
    """
    Configures recorder logging and optional file output.

    Uses the package's historical default handlers.
    """

    logger.setLevel(logging.DEBUG)  # We capture all, then filter via handlers

    log_format = "RealTimeSTT: %(name)s - %(levelname)s - %(message)s"
    if init_args is not None:
        init_args["log_format"] = log_format
    file_log_format = "%(asctime)s.%(msecs)03d - " + log_format
    if init_args is not None:
        init_args["file_log_format"] = file_log_format

    console_handler = logging.StreamHandler()
    console_handler.setLevel(recorder.level)
    console_handler.setFormatter(logging.Formatter(log_format))
    if init_args is not None:
        init_args["console_handler"] = console_handler

    logger.addHandler(console_handler)

    if not no_log_file:
        file_handler = logging.FileHandler('realtimesst.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_log_format, datefmt='%Y-%m-%d %H:%M:%S'))
        if init_args is not None:
            init_args["file_handler"] = file_handler
        logger.addHandler(file_handler)


def _initialize_shutdown_state(recorder):
    """
    Initializes shutdown flags and synchronization state.
    """

    recorder.is_shut_down = False
    recorder.shutdown_event = mp.Event()


def _configure_multiprocessing_start_method():
    """
    Configures the multiprocessing start method on Windows.
    """

    try:
        # Multiprocessing start method is process-global; leave existing settings intact.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
    except RuntimeError as e:
        logger.info(f"Start method has already been set. Details: {e}")


def _log_extended_initialization(use_extended_logging, init_args):
    """
    Logs constructor arguments when extended logging is enabled.
    """

    if use_extended_logging:
        logger.info("RealtimeSTT was called with these parameters:")
        for param, value in init_args.items():
            logger.info(f"{param}: {value}")


def _initialize_transcription_runtime(recorder, recorder_cls):
    """
    Initializes queues and the final transcription worker.
    """

    recorder.interrupt_stop_event = mp.Event()
    recorder.was_interrupted = mp.Event()
    recorder.main_transcription_ready_event = mp.Event()

    recorder.transcript_process = None
    recorder.stdout_thread = None
    recorder.parent_transcription_pipe = None
    recorder.parent_stdout_pipe = None
    recorder.shared_preview_transcription_result_queue = None
    child_transcription_pipe = None
    child_stdout_pipe = None

    if not recorder._uses_external_transcription_executor:
        recorder.parent_transcription_pipe, child_transcription_pipe = SafePipe()
        recorder.parent_stdout_pipe, child_stdout_pipe = SafePipe()
        if _should_share_preview_model(recorder):
            recorder.shared_preview_transcription_result_queue = mp.Queue()
            recorder._preview_uses_shared_final_worker = True

    recorder.device = "cuda" if recorder.device == "cuda" and torch.cuda.is_available() else "cpu"

    if recorder._uses_external_transcription_executor:
        logger.info("Using external main transcription executor")
        recorder.main_transcription_ready_event.set()
    else:
        recorder.transcript_process = start_recorder_worker(
            target=run_transcription_worker,
            args=(
                child_transcription_pipe,
                child_stdout_pipe,
                recorder.transcription_engine,
                recorder.transcription_engine_options,
                recorder.main_model_type,
                recorder.download_root,
                recorder.compute_type,
                recorder.gpu_device_index,
                recorder.device,
                recorder.main_transcription_ready_event,
                recorder.shutdown_event,
                recorder.interrupt_stop_event,
                recorder.beam_size,
                recorder.initial_prompt,
                recorder.suppress_tokens,
                recorder.batch_size,
                recorder.faster_whisper_vad_filter,
                recorder.normalize_audio,
                recorder.shared_preview_transcription_result_queue,
            )
        )


def _values_equal(left, right):
    """Compares optional engine settings without assuming scalar values."""

    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def _should_share_preview_model(recorder):
    """Returns whether Preview can use the loaded Final model safely."""

    return bool(
        getattr(recorder, "enable_preview_transcription", False)
        and not getattr(
            recorder,
            "_uses_external_preview_transcription_executor",
            False,
        )
        and recorder.preview_model_type == recorder.main_model_type
        and recorder.preview_transcription_engine
        == recorder.transcription_engine
        and _values_equal(
            recorder.preview_transcription_engine_options,
            recorder.transcription_engine_options,
        )
    )


def _should_share_ultrafast_realtime_model(recorder):
    """Returns whether both realtime lanes can share one loaded model."""

    return bool(
        getattr(recorder, "ultrafast_realtime_model_type", None)
        and getattr(recorder, "realtime_transcription_model", None) is not None
        and recorder.ultrafast_realtime_model_type
        == recorder.realtime_model_type
        and recorder.ultrafast_realtime_transcription_engine
        == recorder.realtime_transcription_engine
        and _values_equal(
            recorder.ultrafast_realtime_transcription_engine_options,
            recorder.realtime_transcription_engine_options,
        )
    )


def _require_streaming_realtime_model(model, lane_name):
    """Rejects an optional lane that cannot maintain streaming sessions."""

    if not getattr(model, "supports_streaming", False) or not hasattr(
        model,
        "create_streaming_session",
    ):
        raise ValueError(
            f"{lane_name} realtime model must support streaming sessions"
        )


def _start_audio_reader(recorder, recorder_cls):
    """
    Starts the background audio reader process.
    """

    if recorder.use_microphone.value:
        logger.info("Initializing audio recording"
                     " (creating pyAudio input stream,"
                     f" sample rate: {recorder.sample_rate}"
                     f" buffer size: {recorder.buffer_size}"
                     )
        recorder.reader_process = start_recorder_worker(
            target=run_audio_data_worker,
            args=(
                recorder.audio_queue,
                recorder.sample_rate,
                recorder.buffer_size,
                recorder.input_device_index,
                recorder.shutdown_event,
                recorder.interrupt_stop_event,
                recorder.use_microphone
            )
        )


def _initialize_realtime_transcription_model(recorder):
    """
    Initializes the realtime transcription backend.
    """

    if (
        recorder.enable_realtime_transcription
        and not recorder.use_main_model_for_realtime
        and not recorder._uses_external_realtime_transcription_executor
    ):
        try:
            logger.info(
                         f"Initializing {recorder.realtime_transcription_engine} realtime "
                         f"transcription model {recorder.realtime_model_type}, "
                         f"default device: {recorder.device}, "
                         f"compute type: {recorder.compute_type}, "
                         f"device index: {recorder.gpu_device_index}, "
                         f"download root: {recorder.download_root}"
                         )
            recorder.realtime_transcription_model = create_transcription_engine(
                recorder.realtime_transcription_engine,
                TranscriptionEngineConfig(
                    model=recorder.realtime_model_type,
                    download_root=recorder.download_root,
                    compute_type=recorder.compute_type,
                    gpu_device_index=recorder.gpu_device_index,
                    device=recorder.device,
                    beam_size=recorder.beam_size_realtime,
                    initial_prompt=recorder.initial_prompt_realtime,
                    suppress_tokens=recorder.suppress_tokens,
                    batch_size=recorder.realtime_batch_size,
                    vad_filter=recorder.faster_whisper_vad_filter,
                    normalize_audio=recorder.normalize_audio,
                    engine_options=recorder.realtime_transcription_engine_options,
                ),
            )

            # Warmup pays model startup cost before the first realtime chunk.
            current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            warmup_audio_path = os.path.join(
                current_dir, "assets", "warmup_audio.wav"
            )
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            recorder.realtime_transcription_model.warmup(warmup_audio_data)
        except Exception as e:
            logger.exception(
                f"Error initializing {recorder.realtime_transcription_engine} realtime transcription model: {e}"
            )
            raise

        logger.debug(
            f"{recorder.realtime_transcription_engine} realtime speech to text transcription model initialized successfully"
        )

    if not (
        recorder.enable_realtime_transcription
        and recorder.ultrafast_realtime_model_type
    ):
        return

    try:
        if _should_share_ultrafast_realtime_model(recorder):
            recorder.ultrafast_realtime_transcription_model = (
                recorder.realtime_transcription_model
            )
            recorder._ultrafast_uses_realtime_model = True
        else:
            logger.info(
                "Initializing %s ultrafast realtime transcription model %s",
                recorder.ultrafast_realtime_transcription_engine,
                recorder.ultrafast_realtime_model_type,
            )
            recorder.ultrafast_realtime_transcription_model = (
                create_transcription_engine(
                    recorder.ultrafast_realtime_transcription_engine,
                    TranscriptionEngineConfig(
                        model=recorder.ultrafast_realtime_model_type,
                        download_root=recorder.download_root,
                        compute_type=recorder.compute_type,
                        gpu_device_index=recorder.gpu_device_index,
                        device=recorder.device,
                        beam_size=recorder.beam_size_realtime,
                        initial_prompt=recorder.initial_prompt_realtime,
                        suppress_tokens=recorder.suppress_tokens,
                        batch_size=recorder.realtime_batch_size,
                        vad_filter=recorder.faster_whisper_vad_filter,
                        normalize_audio=recorder.normalize_audio,
                        engine_options=(
                            recorder.ultrafast_realtime_transcription_engine_options
                        ),
                    ),
                )
            )

        _require_streaming_realtime_model(
            recorder.ultrafast_realtime_transcription_model,
            "ultrafast",
        )

        if not recorder._ultrafast_uses_realtime_model:
            current_dir = os.path.dirname(
                os.path.dirname(os.path.realpath(__file__))
            )
            warmup_audio_path = os.path.join(
                current_dir,
                "assets",
                "warmup_audio.wav",
            )
            warmup_audio_data, _ = sf.read(
                warmup_audio_path,
                dtype="float32",
            )
            recorder.ultrafast_realtime_transcription_model.warmup(
                warmup_audio_data
            )
    except Exception as error:
        logger.exception(
            "Error initializing ultrafast realtime transcription model: %s",
            error,
        )
        raise

    logger.debug("Ultrafast realtime transcription model initialized")


def _initialize_preview_transcription_model(recorder):
    """
    Initializes the optional independent speculative Preview model.
    """

    if not recorder.enable_preview_transcription:
        return

    recorder._preview_uses_main_model = _should_share_preview_model(recorder)
    if recorder._preview_uses_main_model:
        recorder.preview_transcription_uses_main_model = True
        if recorder._uses_external_transcription_executor:
            recorder.preview_transcription_executor = (
                recorder.transcription_executor
            )
        else:
            recorder.preview_transcription_executor = SharedFinalModelExecutor(
                recorder
            )
        recorder._uses_external_preview_transcription_executor = True
        start_preview_transcription_worker(recorder)
        return

    if not recorder._uses_external_preview_transcription_executor:
        try:
            logger.info(
                f"Initializing {recorder.preview_transcription_engine} preview "
                f"transcription model {recorder.preview_model_type}, "
                f"device: {recorder.device}, compute type: {recorder.compute_type}, "
                f"download root: {recorder.download_root}"
            )
            recorder.preview_transcription_model = create_transcription_engine(
                recorder.preview_transcription_engine,
                TranscriptionEngineConfig(
                    model=recorder.preview_model_type,
                    download_root=recorder.download_root,
                    compute_type=recorder.compute_type,
                    gpu_device_index=recorder.gpu_device_index,
                    device=recorder.device,
                    beam_size=recorder.beam_size_realtime,
                    initial_prompt=recorder.initial_prompt,
                    suppress_tokens=recorder.suppress_tokens,
                    batch_size=0,
                    vad_filter=recorder.faster_whisper_vad_filter,
                    normalize_audio=recorder.normalize_audio,
                    engine_options=recorder.preview_transcription_engine_options,
                ),
            )

            current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            warmup_audio_path = os.path.join(
                current_dir,
                "assets",
                "warmup_audio.wav",
            )
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            recorder.preview_transcription_model.warmup(warmup_audio_data)
        except Exception as error:
            logger.exception(
                "Error initializing preview transcription model: %s",
                error,
            )
            raise

    start_preview_transcription_worker(recorder)


def _initialize_wakeword_detection(
    recorder,
    normalized_wakeword_backend,
    wake_words,
    wake_words_sensitivity,
    openwakeword_model_paths,
    openwakeword_inference_framework,
    load_porcupine_module,
    load_openwakeword_modules,
):
    """
    Initializes wake-word detection backends and callbacks.
    """

    setup_wakeword_detection(
        recorder,
        normalized_wakeword_backend,
        wake_words,
        wake_words_sensitivity,
        openwakeword_model_paths,
        openwakeword_inference_framework,
        load_porcupine_module=load_porcupine_module,
        load_openwakeword_modules=load_openwakeword_modules,
    )


def _initialize_voice_activity_detection(
    recorder,
    webrtc_sensitivity,
    silero_backend,
    silero_use_onnx,
    silero_onnx_model_path,
    silero_onnx_threads,
):
    """
    Initializes WebRTC and Silero voice activity detection.
    """

    try:
        logger.info("Initializing WebRTC voice with "
                     f"Sensitivity {webrtc_sensitivity}"
                     )
        recorder.webrtc_vad_model = webrtcvad.Vad()
        recorder.webrtc_vad_model.set_mode(webrtc_sensitivity)

    except Exception as e:
        logger.exception("Error initializing WebRTC voice "
                          f"activity detection engine: {e}"
                          )
        raise

    logger.debug("WebRTC VAD voice activity detection "
                  "engine initialized successfully"
                  )

    try:
        recorder.silero_vad_model = create_silero_vad_model(
            backend=silero_backend,
            silero_use_onnx=silero_use_onnx,
            onnx_model_path=silero_onnx_model_path,
            onnx_threads=silero_onnx_threads,
            sample_rate=recorder.sample_rate,
            chunk_samples=recorder.buffer_size,
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
        getattr(recorder.silero_vad_model, "backend", silero_backend),
    )

    if recorder.warmup_vad:
        warmup_voice_activity_detectors(recorder)


def _initialize_recording_buffers(recorder):
    """
    Initializes recording buffers and pre-roll state.
    """

    recorder.audio_buffer = collections.deque(
        maxlen=int((recorder.sample_rate // recorder.buffer_size) *
                   recorder.pre_recording_buffer_duration)
    )
    recorder.audio_buffer_metadata = collections.deque(
        maxlen=recorder.audio_buffer.maxlen
    )
    recorder.last_words_buffer = collections.deque(
        maxlen=int((recorder.sample_rate // recorder.buffer_size) *
                   0.3)
    )
    recorder.frames_lock = threading.RLock()
    recorder.frames = []
    recorder.last_frames = []

    recorder.is_recording = False
    recorder.is_running = True
    recorder.start_recording_on_voice_activity = False
    recorder.stop_recording_on_voice_deactivity = False


def _start_worker_threads(recorder):
    """
    Starts recorder worker threads.
    """

    recorder.recording_thread = threading.Thread(
        target=run_recording_worker,
        args=(recorder,),
    )
    recorder.recording_thread.daemon = True
    recorder.recording_thread.start()

    recorder.realtime_thread = threading.Thread(
        target=run_realtime_worker,
        args=(recorder,),
    )
    recorder.realtime_thread.daemon = True
    recorder.realtime_thread.start()


def _finish_initialization(recorder):
    """
    Finishes recorder initialization state and startup logging.
    """

    logger.debug('Waiting for main transcription model to start')
    recorder.main_transcription_ready_event.wait()
    logger.debug('Main transcription model ready')

    if recorder.parent_stdout_pipe is not None:
        recorder.stdout_thread = threading.Thread(
            target=read_stdout_pipe,
            args=(recorder,),
        )
        recorder.stdout_thread.daemon = True
        recorder.stdout_thread.start()

    logger.debug('RealtimeSTT initialization completed successfully')
