"""
Speech-to-Text (STT) Server with Real-Time Transcription and WebSocket Interface

This server provides real-time speech-to-text (STT) transcription using the RealtimeSTT library. It allows clients to connect via WebSocket to send audio data and receive real-time transcription updates. The server supports configurable audio recording parameters, voice activity detection (VAD), and wake word detection. It is designed to handle continuous transcription as well as post-recording processing, enabling real-time feedback with the option to improve final transcription quality after the complete sentence is recognized.

### Features:
- Real-time transcription using pre-configured or user-defined STT models.
- WebSocket-based communication for control and data handling.
- Flexible recording and transcription options, including configurable pauses for sentence detection.
- Supports Silero and WebRTC VAD for robust voice activity detection.

### Starting the Server:
You can start the server using the command-line interface (CLI) command `stt-server`, passing the desired configuration options.

```bash
stt-server [OPTIONS]
```

### Available Parameters:
    - `-m, --model`: Model path or size; default 'large-v2'.
    - `-r, --rt-model, --realtime_model_type`: Real-time model size; default 'tiny.en'.
    - `-l, --lang, --language`: Language code for transcription; default 'en'.
    - `-i, --input-device, --input_device_index`: Audio input device index; default 1.
    - `-c, --control, --control_port`: WebSocket control port; default 8011.
    - `-d, --data, --data_port`: WebSocket data port; default 8012.
    - `-w, --wake_words`: Wake word(s) to trigger listening; default "".
    - `-D, --debug`: Enable debug logging.
    - `-W, --write`: Save audio to WAV file.
    - `-s, --silence_timing`: Enable dynamic silence duration for sentence detection; default True. 
    - `-b, --batch, --batch_size`: Batch size for inference; default 16.
    - `--root, --download_root`: Specifies the root path were the Whisper models are downloaded to.
    - `--silero_sensitivity`: Silero VAD sensitivity (0-1); default 0.05.
    - `--silero_use_onnx`: Use Silero ONNX model; default False.
    - `--webrtc_sensitivity`: WebRTC VAD sensitivity (0-3); default 3.
    - `--min_length_of_recording`: Minimum recording duration in seconds; default 1.1.
    - `--min_gap_between_recordings`: Min time between recordings in seconds; default 0.
    - `--enable_realtime_transcription`: Enable real-time transcription; default True.
    - `--realtime_processing_pause`: Pause between audio chunk processing; default 0.02.
    - `--silero_deactivity_detection`: Use Silero for end-of-speech detection; default True.
    - `--early_transcription_on_silence`: Start transcription after silence in seconds; default 0.2.
    - `--beam_size`: Beam size for main model; default 5.
    - `--beam_size_realtime`: Beam size for real-time model; default 3.
    - `--init_realtime_after_seconds`: Initial waiting time for realtime transcription; default 0.2.
    - `--realtime_batch_size`: Batch size for the real-time transcription model; default 16.
    - `--initial_prompt`: Initial main transcription guidance prompt.
    - `--initial_prompt_realtime`: Initial realtime transcription guidance prompt.
    - `--end_of_sentence_detection_pause`: Silence duration for sentence end detection; default 0.45.
    - `--unknown_sentence_detection_pause`: Pause duration for incomplete sentence detection; default 0.7.
    - `--mid_sentence_detection_pause`: Pause for mid-sentence break; default 2.0.
    - `--wake_words_sensitivity`: Wake word detection sensitivity (0-1); default 0.5.
    - `--wake_word_timeout`: Wake word timeout in seconds; default 5.0.
    - `--wake_word_activation_delay`: Delay before wake word activation; default 20.
    - `--wakeword_backend`: Backend for wake word detection; default 'none'.
    - `--openwakeword_model_paths`: Paths to OpenWakeWord models.
    - `--openwakeword_inference_framework`: OpenWakeWord inference framework; default 'tensorflow'.
    - `--wake_word_buffer_duration`: Wake word buffer duration in seconds; default 1.0.
    - `--use_main_model_for_realtime`: Use main model for real-time transcription.
    - `--use_extended_logging`: Enable extensive log messages.
    - `--logchunks`: Log incoming audio chunks.
    - `--compute_type`: Type of computation to use.
    - `--input_device_index`: Index of the audio input device.
    - `--gpu_device_index`: Index of the GPU device.
    - `--device`: Device to use for computation.
    - `--handle_buffer_overflow`: Handle buffer overflow during transcription.
    - `--suppress_tokens`: Suppress tokens during transcription.
    - `--allowed_latency_limit`: Allowed latency limit for real-time transcription.
    - `--faster_whisper_vad_filter`: Enable VAD filter for Faster Whisper; default False.


### WebSocket Interface:
The server supports two WebSocket connections:
1. **Control WebSocket**: Used to send and receive commands, such as setting parameters or calling recorder methods.
2. **Data WebSocket**: Used to send audio data for transcription and receive real-time transcription updates.

The server will broadcast real-time transcription updates to all connected clients on the data WebSocket.
"""

from .install_packages import check_and_install_packages
from difflib import SequenceMatcher
from collections import deque
from datetime import datetime
import logging
import asyncio
import pyaudio
import base64
import sys


debug_logging = False
extended_logging = False
send_recorded_chunk = False
log_incoming_chunks = False
silence_timing = False
writechunks = False
wav_file = None

hard_break_even_on_background_noise = 3.0
hard_break_even_on_background_noise_min_texts = 3
hard_break_even_on_background_noise_min_similarity = 0.99
hard_break_even_on_background_noise_min_chars = 15


text_time_deque = deque()
loglevel = logging.WARNING

FORMAT = pyaudio.paInt16
CHANNELS = 1


if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


check_and_install_packages([
    {
        'module_name': 'RealtimeSTT',                 # Import module
        'attribute': 'AudioToTextRecorder',           # Specific class to check
        'install_name': 'RealtimeSTT',                # Package name for pip install
    },
    {
        'module_name': 'websockets',                  # Import module
        'install_name': 'websockets',                 # Package name for pip install
    },
    {
        'module_name': 'numpy',                       # Import module
        'install_name': 'numpy',                      # Package name for pip install
    },
    {
        'module_name': 'scipy.signal',                # Submodule of scipy
        'attribute': 'resample',                      # Specific function to check
        'install_name': 'scipy',                      # Package name for pip install
    }
])

# Define ANSI color codes for terminal output
class bcolors:
    HEADER = '\033[95m'   # Magenta
    OKBLUE = '\033[94m'   # Blue
    OKCYAN = '\033[96m'   # Cyan
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'     # Red
    ENDC = '\033[0m'      # Reset to default
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f"{bcolors.BOLD}{bcolors.OKCYAN}Starting server, please wait...{bcolors.ENDC}")

# Initialize colorama
from colorama import init, Fore, Style
init()

from RealtimeSTT import AudioToTextRecorder
from scipy.signal import resample
import numpy as np
import websockets
import threading
import logging
import wave
import json
import time

global_args = None
recorder = None
recorder_config = {}
recorder_ready = threading.Event()
recorder_thread = None
stop_recorder = False
prev_text = ""

# Define allowed methods and parameters for security
allowed_methods = [
    'set_microphone',
    'abort',
    'stop',
    'clear_audio_queue',
    'wakeup',
    'shutdown',
    'text',
]
allowed_parameters = [
    'language',
    'silero_sensitivity',
    'wake_word_activation_delay',
    'post_speech_silence_duration',
    'listen_start',
    'recording_stop_time',
    'last_transcription_bytes',
    'last_transcription_bytes_b64',
    'speech_end_silence_start',
    'is_recording',
    'use_wake_words',
]

# Queues and connections for control and data
control_connections = set()
data_connections = set()
control_queue = asyncio.Queue()
audio_queue = asyncio.Queue()

def preprocess_text(text):
    # Remove leading whitespaces
    text = text.lstrip()

    # Remove starting ellipses if present
    if text.startswith("..."):
        text = text[3:]

    if text.endswith("...'."):
        text = text[:-1]

    if text.endswith("...'"):
        text = text[:-1]

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def debug_print(message):
    if debug_logging:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        thread_name = threading.current_thread().name
        print(f"{Fore.CYAN}[DEBUG][{timestamp}][{thread_name}] {message}{Style.RESET_ALL}", file=sys.stderr)

def format_timestamp_ns(timestamp_ns: int) -> str:
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

def text_detected(text, loop):
    global prev_text

    text = preprocess_text(text)

    if silence_timing:
        def ends_with_ellipsis(text: str):
            if text.endswith("..."):
                return True
            if len(text) > 1 and text[:-1].endswith("..."):
                return True
            return False

        def sentence_end(text: str):
            sentence_end_marks = ['.', '!', '?', '。']
            if text and text[-1] in sentence_end_marks:
                return True
            return False


        if ends_with_ellipsis(text):
            recorder.post_speech_silence_duration = global_args.mid_sentence_detection_pause
        elif sentence_end(text) and sentence_end(prev_text) and not ends_with_ellipsis(prev_text):
            recorder.post_speech_silence_duration = global_args.end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause


        # Append the new text with its timestamp
        current_time = time.time()
        text_time_deque.append((current_time, text))

        # Remove texts older than hard_break_even_on_background_noise seconds
        while text_time_deque and text_time_deque[0][0] < current_time - hard_break_even_on_background_noise:
            text_time_deque.popleft()

        # Check if at least hard_break_even_on_background_noise_min_texts texts have arrived within the last hard_break_even_on_background_noise seconds
        if len(text_time_deque) >= hard_break_even_on_background_noise_min_texts:
            texts = [t[1] for t in text_time_deque]
            first_text = texts[0]
            last_text = texts[-1]

            # Compute the similarity ratio between the first and last texts
            similarity = SequenceMatcher(None, first_text, last_text).ratio()

            if similarity > hard_break_even_on_background_noise_min_similarity and len(first_text) > hard_break_even_on_background_noise_min_chars:
                recorder.stop()
                recorder.clear_audio_queue()
                prev_text = ""

    prev_text = text

    # Put the message in the audio queue to be sent to clients
    message = json.dumps({
        'type': 'realtime',
        'text': text
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

    # Get current timestamp in HH:MM:SS.nnn format
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

    if extended_logging:
        print(f"  [{timestamp}] Realtime text: {bcolors.OKCYAN}{text}{bcolors.ENDC}\n", flush=True, end="")
    else:
        print(f"\r[{timestamp}] {bcolors.OKCYAN}{text}{bcolors.ENDC}", flush=True, end='')

def on_recording_start(loop):
    message = json.dumps({
        'type': 'recording_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_recording_stop(loop):
    message = json.dumps({
        'type': 'recording_stop'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_vad_detect_start(loop):
    message = json.dumps({
        'type': 'vad_detect_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_vad_detect_stop(loop):
    message = json.dumps({
        'type': 'vad_detect_stop'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detected(loop):
    message = json.dumps({
        'type': 'wakeword_detected'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_start(loop):
    message = json.dumps({
        'type': 'wakeword_detection_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_end(loop):
    message = json.dumps({
        'type': 'wakeword_detection_end'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_transcription_start(_audio_bytes, loop):
    bytes_b64 = base64.b64encode(_audio_bytes.tobytes()).decode('utf-8')
    message = json.dumps({
        'type': 'transcription_start',
        'audio_bytes_base64': bytes_b64
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_turn_detection_start(loop):
    print("&&& stt_server on_turn_detection_start")
    message = json.dumps({
        'type': 'start_turn_detection'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_turn_detection_stop(loop):
    print("&&& stt_server on_turn_detection_stop")
    message = json.dumps({
        'type': 'stop_turn_detection'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)


# def on_realtime_transcription_update(text, loop):
#     # Send real-time transcription updates to the client
#     text = preprocess_text(text)
#     message = json.dumps({
#         'type': 'realtime_update',
#         'text': text
#     })
#     asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

# def on_recorded_chunk(chunk, loop):
#     if send_recorded_chunk:
#         bytes_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
#         message = json.dumps({
#             'type': 'recorded_chunk',
#             'bytes': bytes_b64
#         })
#         asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

# Define the server's arguments
def parse_arguments():
    global debug_logging, extended_logging, loglevel, writechunks, log_incoming_chunks, dynamic_silence_timing

    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server with various configuration options.')

    parser.add_argument('-m', '--model', type=str, default='large-v2',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='tiny',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')
    
    parser.add_argument('-l', '--lang', '--language', type=str, default='en',
                help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')

    parser.add_argument('-i', '--input-device', '--input-device-index', type=int, default=1,
                    help='Index of the audio input device to use. Use this option to specify a particular microphone or audio input device based on your system. Default is 1.')

    parser.add_argument('-c', '--control', '--control_port', type=int, default=8011,
                        help='The port number used for the control WebSocket connection. Control connections are used to send and receive commands to the server. Default is port 8011.')

    parser.add_argument('-d', '--data', '--data_port', type=int, default=8012,
                        help='The port number used for the data WebSocket connection. Data connections are used to send audio data and receive transcription updates in real time. Default is port 8012.')

    parser.add_argument('-w', '--wake_words', type=str, default="",
                        help='Specify the wake word(s) that will trigger the server to start listening. For example, setting this to "Jarvis" will make the system start transcribing when it detects the wake word "Jarvis". Default is "Jarvis".')

    parser.add_argument('-D', '--debug', action='store_true', help='Enable debug logging for detailed server operations')

    parser.add_argument('--debug_websockets', action='store_true', help='Enable debug logging for detailed server websocket operations')

    parser.add_argument('-W', '--write', metavar='FILE', help='Save received audio to a WAV file')
    
    parser.add_argument('-b', '--batch', '--batch_size', type=int, default=16, help='Batch size for inference. This parameter controls the number of audio chunks processed in parallel during transcription. Default is 16.')

    parser.add_argument('--root', '--download_root', type=str,default=None, help='Specifies the root path where the Whisper models are downloaded to. Default is None.')

    parser.add_argument('-s', '--silence_timing', action='store_true', default=True,
                    help='Enable dynamic adjustment of silence duration for sentence detection. Adjusts post-speech silence duration based on detected sentence structure and punctuation. Default is False.')

    parser.add_argument('--init_realtime_after_seconds', type=float, default=0.2,
                        help='The initial waiting time in seconds before real-time transcription starts. This delay helps prevent false positives at the beginning of a session. Default is 0.2 seconds.')  
    
    parser.add_argument('--realtime_batch_size', type=int, default=16,
                        help='Batch size for the real-time transcription model. This parameter controls the number of audio chunks processed in parallel during real-time transcription. Default is 16.')
    
    parser.add_argument('--initial_prompt_realtime', type=str, default="", help='Initial prompt that guides the real-time transcription model to produce transcriptions in a particular style or format.')

    parser.add_argument('--silero_sensitivity', type=float, default=0.05,
                        help='Sensitivity level for Silero Voice Activity Detection (VAD), with a range from 0 to 1. Lower values make the model less sensitive, useful for noisy environments. Default is 0.05.')

    parser.add_argument('--silero_use_onnx', action='store_true', default=False,
                        help='Enable ONNX version of Silero model for faster performance with lower resource usage. Default is False.')

    parser.add_argument('--webrtc_sensitivity', type=int, default=3,
                        help='Sensitivity level for WebRTC Voice Activity Detection (VAD), with a range from 0 to 3. Higher values make the model less sensitive, useful for cleaner environments. Default is 3.')

    parser.add_argument('--min_length_of_recording', type=float, default=1.1,
                        help='Minimum duration of valid recordings in seconds. This prevents very short recordings from being processed, which could be caused by noise or accidental sounds. Default is 1.1 seconds.')

    parser.add_argument('--min_gap_between_recordings', type=float, default=0,
                        help='Minimum time (in seconds) between consecutive recordings. Setting this helps avoid overlapping recordings when there’s a brief silence between them. Default is 0 seconds.')

    parser.add_argument('--enable_realtime_transcription', action='store_true', default=True,
                        help='Enable continuous real-time transcription of audio as it is received. When enabled, transcriptions are sent in near real-time. Default is True.')

    parser.add_argument('--realtime_processing_pause', type=float, default=0.02,
                        help='Time interval (in seconds) between processing audio chunks for real-time transcription. Lower values increase responsiveness but may put more load on the CPU. Default is 0.02 seconds.')

    parser.add_argument('--silero_deactivity_detection', action='store_true', default=True,
                        help='Use the Silero model for end-of-speech detection. This option can provide more robust silence detection in noisy environments, though it consumes more GPU resources. Default is True.')

    parser.add_argument('--early_transcription_on_silence', type=float, default=0.2,
                        help='Start transcription after the specified seconds of silence. This is useful when you want to trigger transcription mid-speech when there is a brief pause. Should be lower than post_speech_silence_duration. Set to 0 to disable. Default is 0.2 seconds.')

    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for the main transcription model. Larger values may improve transcription accuracy but increase the processing time. Default is 5.')

    parser.add_argument('--beam_size_realtime', type=int, default=3,
                        help='Beam size for the real-time transcription model. A smaller beam size allows for faster real-time processing but may reduce accuracy. Default is 3.')

    parser.add_argument('--initial_prompt', type=str,
                        default="Incomplete thoughts should end with '...'. Examples of complete thoughts: 'The sky is blue.' 'She walked home.' Examples of incomplete thoughts: 'When the sky...' 'Because he...'",
                        help='Initial prompt that guides the transcription model to produce transcriptions in a particular style or format. The default provides instructions for handling sentence completions and ellipsis usage.')

    parser.add_argument('--end_of_sentence_detection_pause', type=float, default=0.45,
                        help='The duration of silence (in seconds) that the model should interpret as the end of a sentence. This helps the system detect when to finalize the transcription of a sentence. Default is 0.45 seconds.')

    parser.add_argument('--unknown_sentence_detection_pause', type=float, default=0.7,
                        help='The duration of pause (in seconds) that the model should interpret as an incomplete or unknown sentence. This is useful for identifying when a sentence is trailing off or unfinished. Default is 0.7 seconds.')

    parser.add_argument('--mid_sentence_detection_pause', type=float, default=2.0,
                        help='The duration of pause (in seconds) that the model should interpret as a mid-sentence break. Longer pauses can indicate a pause in speech but not necessarily the end of a sentence. Default is 2.0 seconds.')

    parser.add_argument('--wake_words_sensitivity', type=float, default=0.5,
                        help='Sensitivity level for wake word detection, with a range from 0 (most sensitive) to 1 (least sensitive). Adjust this value based on your environment to ensure reliable wake word detection. Default is 0.5.')

    parser.add_argument('--wake_word_timeout', type=float, default=5.0,
                        help='Maximum time in seconds that the system will wait for a wake word before timing out. After this timeout, the system stops listening for wake words until reactivated. Default is 5.0 seconds.')

    parser.add_argument('--wake_word_activation_delay', type=float, default=0,
                        help='The delay in seconds before the wake word detection is activated after the system starts listening. This prevents false positives during the start of a session. Default is 0 seconds.')

    parser.add_argument('--wakeword_backend', type=str, default='none',
                        help='The backend used for wake word detection. You can specify different backends such as "default" or any custom implementations depending on your setup. Default is "pvporcupine".')

    parser.add_argument('--openwakeword_model_paths', type=str, nargs='*',
                        help='A list of file paths to OpenWakeWord models. This is useful if you are using OpenWakeWord for wake word detection and need to specify custom models.')

    parser.add_argument('--openwakeword_inference_framework', type=str, default='tensorflow',
                        help='The inference framework to use for OpenWakeWord models. Supported frameworks could include "tensorflow", "pytorch", etc. Default is "tensorflow".')

    parser.add_argument('--wake_word_buffer_duration', type=float, default=1.0,
                        help='Duration of the buffer in seconds for wake word detection. This sets how long the system will store the audio before and after detecting the wake word. Default is 1.0 seconds.')

    parser.add_argument('--use_main_model_for_realtime', action='store_true',
                        help='Enable this option if you want to use the main model for real-time transcription, instead of the smaller, faster real-time model. Using the main model may provide better accuracy but at the cost of higher processing time.')

    parser.add_argument('--use_extended_logging', action='store_true',
                        help='Writes extensive log messages for the recording worker, that processes the audio chunks.')

    parser.add_argument('--compute_type', type=str, default='default',
                        help='Type of computation to use. See https://opennmt.net/CTranslate2/quantization.html')

    parser.add_argument('--gpu_device_index', type=int, default=0,
                        help='Index of the GPU device to use. Default is None.')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for model to use. Can either be "cuda" or "cpu". Default is cuda.')
    
    parser.add_argument('--handle_buffer_overflow', action='store_true',
                        help='Handle buffer overflow during transcription. Default is False.')

    parser.add_argument('--suppress_tokens', type=int, default=[-1], nargs='*', help='Suppress tokens during transcription. Default is [-1].')

    parser.add_argument('--allowed_latency_limit', type=int, default=100,
                        help='Maximal amount of chunks that can be unprocessed in queue before discarding chunks.. Default is 100.')

    parser.add_argument('--faster_whisper_vad_filter', action='store_true',
                        help='Enable VAD filter for Faster Whisper. Default is False.')

    parser.add_argument('--logchunks', action='store_true', help='Enable logging of incoming audio chunks (periods)')

    # Parse arguments
    args = parser.parse_args()

    debug_logging = args.debug
    extended_logging = args.use_extended_logging
    writechunks = args.write
    log_incoming_chunks = args.logchunks
    dynamic_silence_timing = args.silence_timing


    ws_logger = logging.getLogger('websockets')
    if args.debug_websockets:
        # If app debug is on, let websockets be verbose too
        ws_logger.setLevel(logging.DEBUG)
        # Ensure it uses the handler configured by basicConfig
        ws_logger.propagate = False # Prevent duplicate messages if it also propagates to root
    else:
        # If app debug is off, silence websockets below WARNING
        ws_logger.setLevel(logging.WARNING)
        ws_logger.propagate = True # Allow WARNING/ERROR messages to reach root logger's handler

    # Replace escaped newlines with actual newlines in initial_prompt
    if args.initial_prompt:
        args.initial_prompt = args.initial_prompt.replace("\\n", "\n")

    if args.initial_prompt_realtime:
        args.initial_prompt_realtime = args.initial_prompt_realtime.replace("\\n", "\n")

    return args

def _recorder_thread(loop):
    global recorder, stop_recorder
    print(f"{bcolors.OKGREEN}Initializing RealtimeSTT server with parameters:{bcolors.ENDC}")
    for key, value in recorder_config.items():
        print(f"    {bcolors.OKBLUE}{key}{bcolors.ENDC}: {value}")
    recorder = AudioToTextRecorder(**recorder_config)
    print(f"{bcolors.OKGREEN}{bcolors.BOLD}RealtimeSTT initialized{bcolors.ENDC}")
    recorder_ready.set()
    
    def process_text(full_sentence):
        global prev_text
        prev_text = ""
        full_sentence = preprocess_text(full_sentence)
        message = json.dumps({
            'type': 'fullSentence',
            'text': full_sentence
        })
        # Use the passed event loop here
        asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        if extended_logging:
            print(f"  [{timestamp}] Full text: {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n", flush=True, end="")
        else:
            print(f"\r[{timestamp}] {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n")
    try:
        while not stop_recorder:
            recorder.text(process_text)
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Exiting application due to keyboard interrupt{bcolors.ENDC}")

def decode_and_resample(
        audio_data,
        original_sample_rate,
        target_sample_rate):

    # Decode 16-bit PCM data to numpy array
    if original_sample_rate == target_sample_rate:
        return audio_data

    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Calculate the number of samples after resampling
    num_original_samples = len(audio_np)
    num_target_samples = int(num_original_samples * target_sample_rate /
                                original_sample_rate)

    # Resample the audio
    resampled_audio = resample(audio_np, num_target_samples)

    return resampled_audio.astype(np.int16).tobytes()

async def control_handler(websocket):
    debug_print(f"New control connection from {websocket.remote_address}")
    print(f"{bcolors.OKGREEN}Control client connected{bcolors.ENDC}")
    global recorder
    control_connections.add(websocket)
    try:
        async for message in websocket:
            debug_print(f"Received control message: {message[:200]}...")
            if not recorder_ready.is_set():
                print(f"{bcolors.WARNING}Recorder not ready{bcolors.ENDC}")
                continue
            if isinstance(message, str):
                # Handle text message (command)
                try:
                    command_data = json.loads(message)
                    command = command_data.get("command")
                    if command == "set_parameter":
                        parameter = command_data.get("parameter")
                        value = command_data.get("value")
                        if parameter in allowed_parameters and hasattr(recorder, parameter):
                            setattr(recorder, parameter, value)
                            # Format the value for output
                            if isinstance(value, float):
                                value_formatted = f"{value:.2f}"
                            else:
                                value_formatted = value
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            if extended_logging:
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Set recorder.{parameter} to: {bcolors.OKBLUE}{value_formatted}{bcolors.ENDC}")
                            # Optionally send a response back to the client
                            await websocket.send(json.dumps({"status": "success", "message": f"Parameter {parameter} set to {value}"}))
                        else:
                            if not parameter in allowed_parameters:
                                print(f"{bcolors.WARNING}Parameter {parameter} is not allowed (set_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} is not allowed (set_parameter)"}))
                            else:
                                print(f"{bcolors.WARNING}Parameter {parameter} does not exist (set_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} does not exist (set_parameter)"}))

                    elif command == "get_parameter":
                        parameter = command_data.get("parameter")
                        request_id = command_data.get("request_id")  # Get the request_id from the command data
                        if parameter in allowed_parameters and hasattr(recorder, parameter):
                            value = getattr(recorder, parameter)
                            if isinstance(value, float):
                                value_formatted = f"{value:.2f}"
                            else:
                                value_formatted = f"{value}"

                            value_truncated = value_formatted[:39] + "…" if len(value_formatted) > 40 else value_formatted

                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            if extended_logging:
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Get recorder.{parameter}: {bcolors.OKBLUE}{value_truncated}{bcolors.ENDC}")
                            response = {"status": "success", "parameter": parameter, "value": value}
                            if request_id is not None:
                                response["request_id"] = request_id
                            await websocket.send(json.dumps(response))
                        else:
                            if not parameter in allowed_parameters:
                                print(f"{bcolors.WARNING}Parameter {parameter} is not allowed (get_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} is not allowed (get_parameter)"}))
                            else:
                                print(f"{bcolors.WARNING}Parameter {parameter} does not exist (get_parameter){bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} does not exist (get_parameter)"}))
                    elif command == "call_method":
                        method_name = command_data.get("method")
                        if method_name in allowed_methods:
                            method = getattr(recorder, method_name, None)
                            if method and callable(method):
                                args = command_data.get("args", [])
                                kwargs = command_data.get("kwargs", {})
                                method(*args, **kwargs)
                                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Called method recorder.{bcolors.OKBLUE}{method_name}{bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "success", "message": f"Method {method_name} called"}))
                            else:
                                print(f"{bcolors.WARNING}Recorder does not have method {method_name}{bcolors.ENDC}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Recorder does not have method {method_name}"}))
                        else:
                            print(f"{bcolors.WARNING}Method {method_name} is not allowed{bcolors.ENDC}")
                            await websocket.send(json.dumps({"status": "error", "message": f"Method {method_name} is not allowed"}))
                    else:
                        print(f"{bcolors.WARNING}Unknown command: {command}{bcolors.ENDC}")
                        await websocket.send(json.dumps({"status": "error", "message": f"Unknown command {command}"}))
                except json.JSONDecodeError:
                    print(f"{bcolors.WARNING}Received invalid JSON command{bcolors.ENDC}")
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON command"}))
            else:
                print(f"{bcolors.WARNING}Received unknown message type on control connection{bcolors.ENDC}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"{bcolors.WARNING}Control client disconnected: {e}{bcolors.ENDC}")
    finally:
        control_connections.remove(websocket)

async def data_handler(websocket):
    global writechunks, wav_file
    print(f"{bcolors.OKGREEN}Data client connected{bcolors.ENDC}")
    data_connections.add(websocket)
    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                if extended_logging:
                    debug_print(f"Received audio chunk (size: {len(message)} bytes)")
                elif log_incoming_chunks:
                    print(".", end='', flush=True)
                # Handle binary message (audio data)
                metadata_length = int.from_bytes(message[:4], byteorder='little')
                metadata_json = message[4:4+metadata_length].decode('utf-8')
                metadata = json.loads(metadata_json)
                sample_rate = metadata['sampleRate']

                if 'server_sent_to_stt' in metadata:
                    stt_received_ns = time.time_ns()
                    metadata["stt_received"] = stt_received_ns
                    metadata["stt_received_formatted"] = format_timestamp_ns(stt_received_ns)
                    print(f"Server received audio chunk of length {len(message)} bytes, metadata: {metadata}")

                if extended_logging:
                    debug_print(f"Processing audio chunk with sample rate {sample_rate}")
                chunk = message[4+metadata_length:]

                if writechunks:
                    if not wav_file:
                        wav_file = wave.open(writechunks, 'wb')
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
                        wav_file.setframerate(sample_rate)

                    wav_file.writeframes(chunk)

                if sample_rate != 16000:
                    resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
                    if extended_logging:
                        debug_print(f"Resampled chunk size: {len(resampled_chunk)} bytes")
                    recorder.feed_audio(resampled_chunk)
                else:
                    recorder.feed_audio(chunk)
            else:
                print(f"{bcolors.WARNING}Received non-binary message on data connection{bcolors.ENDC}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"{bcolors.WARNING}Data client disconnected: {e}{bcolors.ENDC}")
    finally:
        data_connections.remove(websocket)
        recorder.clear_audio_queue()  # Ensure audio queue is cleared if client disconnects

async def broadcast_audio_messages():
    while True:
        message = await audio_queue.get()
        for conn in list(data_connections):
            try:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                if extended_logging:
                    print(f"  [{timestamp}] Sending message: {bcolors.OKBLUE}{message}{bcolors.ENDC}\n", flush=True, end="")
                await conn.send(message)
            except websockets.exceptions.ConnectionClosed:
                data_connections.remove(conn)

# Helper function to create event loop bound closures for callbacks
def make_callback(loop, callback):
    def inner_callback(*args, **kwargs):
        callback(*args, **kwargs, loop=loop)
    return inner_callback

async def main_async():            
    global stop_recorder, recorder_config, global_args
    args = parse_arguments()
    global_args = args

    # Get the event loop here and pass it to the recorder thread
    loop = asyncio.get_event_loop()

    recorder_config = {
        'model': args.model,
        'download_root': args.root,
        'realtime_model_type': args.rt_model,
        'language': args.lang,
        'batch_size': args.batch,
        'init_realtime_after_seconds': args.init_realtime_after_seconds,
        'realtime_batch_size': args.realtime_batch_size,
        'initial_prompt_realtime': args.initial_prompt_realtime,
        'input_device_index': args.input_device,
        'silero_sensitivity': args.silero_sensitivity,
        'silero_use_onnx': args.silero_use_onnx,
        'webrtc_sensitivity': args.webrtc_sensitivity,
        'post_speech_silence_duration': args.unknown_sentence_detection_pause,
        'min_length_of_recording': args.min_length_of_recording,
        'min_gap_between_recordings': args.min_gap_between_recordings,
        'enable_realtime_transcription': args.enable_realtime_transcription,
        'realtime_processing_pause': args.realtime_processing_pause,
        'silero_deactivity_detection': args.silero_deactivity_detection,
        'early_transcription_on_silence': args.early_transcription_on_silence,
        'beam_size': args.beam_size,
        'beam_size_realtime': args.beam_size_realtime,
        'initial_prompt': args.initial_prompt,
        'wake_words': args.wake_words,
        'wake_words_sensitivity': args.wake_words_sensitivity,
        'wake_word_timeout': args.wake_word_timeout,
        'wake_word_activation_delay': args.wake_word_activation_delay,
        'wakeword_backend': args.wakeword_backend,
        'openwakeword_model_paths': args.openwakeword_model_paths,
        'openwakeword_inference_framework': args.openwakeword_inference_framework,
        'wake_word_buffer_duration': args.wake_word_buffer_duration,
        'use_main_model_for_realtime': args.use_main_model_for_realtime,
        'spinner': False,
        'use_microphone': False,

        'on_realtime_transcription_update': make_callback(loop, text_detected),
        'on_recording_start': make_callback(loop, on_recording_start),
        'on_recording_stop': make_callback(loop, on_recording_stop),
        'on_vad_detect_start': make_callback(loop, on_vad_detect_start),
        'on_vad_detect_stop': make_callback(loop, on_vad_detect_stop),
        'on_wakeword_detected': make_callback(loop, on_wakeword_detected),
        'on_wakeword_detection_start': make_callback(loop, on_wakeword_detection_start),
        'on_wakeword_detection_end': make_callback(loop, on_wakeword_detection_end),
        'on_transcription_start': make_callback(loop, on_transcription_start),
        'on_turn_detection_start': make_callback(loop, on_turn_detection_start),
        'on_turn_detection_stop': make_callback(loop, on_turn_detection_stop),

        # 'on_recorded_chunk': make_callback(loop, on_recorded_chunk),
        'no_log_file': True,  # Disable logging to file
        'use_extended_logging': args.use_extended_logging,
        'level': loglevel,
        'compute_type': args.compute_type,
        'gpu_device_index': args.gpu_device_index,
        'device': args.device,
        'handle_buffer_overflow': args.handle_buffer_overflow,
        'suppress_tokens': args.suppress_tokens,
        'allowed_latency_limit': args.allowed_latency_limit,
        'faster_whisper_vad_filter': args.faster_whisper_vad_filter,
    }

    try:
        # Attempt to start control and data servers
        control_server = await websockets.serve(control_handler, "localhost", args.control)
        data_server = await websockets.serve(data_handler, "localhost", args.data)
        print(f"{bcolors.OKGREEN}Control server started on {bcolors.OKBLUE}ws://localhost:{args.control}{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Data server started on {bcolors.OKBLUE}ws://localhost:{args.data}{bcolors.ENDC}")

        # Start the broadcast and recorder threads
        broadcast_task = asyncio.create_task(broadcast_audio_messages())

        recorder_thread = threading.Thread(target=_recorder_thread, args=(loop,))
        recorder_thread.start()
        recorder_ready.wait()

        print(f"{bcolors.OKGREEN}Server started. Press Ctrl+C to stop the server.{bcolors.ENDC}")

        # Run server tasks
        await asyncio.gather(control_server.wait_closed(), data_server.wait_closed(), broadcast_task)
    except OSError as e:
        print(f"{bcolors.FAIL}Error: Could not start server on specified ports. It’s possible another instance of the server is already running, or the ports are being used by another application.{bcolors.ENDC}")
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Server interrupted by user, shutting down...{bcolors.ENDC}")
    finally:
        # Shutdown procedures for recorder and server threads
        await shutdown_procedure()
        print(f"{bcolors.OKGREEN}Server shutdown complete.{bcolors.ENDC}")

async def shutdown_procedure():
    global stop_recorder, recorder_thread
    if recorder:
        stop_recorder = True
        recorder.abort()
        recorder.stop()
        recorder.shutdown()
        print(f"{bcolors.OKGREEN}Recorder shut down{bcolors.ENDC}")

        if recorder_thread:
            recorder_thread.join()
            print(f"{bcolors.OKGREEN}Recorder thread finished{bcolors.ENDC}")

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print(f"{bcolors.OKGREEN}All tasks cancelled, closing event loop now.{bcolors.ENDC}")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        # Capture any final KeyboardInterrupt to prevent it from showing up in logs
        print(f"{bcolors.WARNING}Server interrupted by user.{bcolors.ENDC}")
        exit(0)

if __name__ == '__main__':
    main()
