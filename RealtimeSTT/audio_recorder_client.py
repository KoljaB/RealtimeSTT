from typing import Iterable, List, Optional, Union
from urllib.parse import urlparse
import subprocess
import websocket
import threading
import platform
import logging
import pyaudio
import socket
import struct
import signal
import json
import time
import sys
import os

DEFAULT_CONTROL_URL = "ws://127.0.0.1:8011"
DEFAULT_DATA_URL = "ws://127.0.0.1:8012"

INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
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

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
BUFFER_SIZE = 512

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True

class AudioToTextRecorderClient:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    `faster_whisper` model.
    """

    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
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

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 use_main_model_for_realtime=False,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
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
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,

                 # Wake word parameters
                 wakeword_backend: str = "pvporcupine",
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
                 suppress_tokens: Optional[List[int]] = [-1],
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logging: bool = False,

                 # Server urls
                 control_url: str = DEFAULT_CONTROL_URL,
                 data_url: str = DEFAULT_DATA_URL,
                 autostart_server: bool = True,
                 ):

        # Set instance variables from constructor parameters
        self.model = model
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_transcription_start = on_transcription_start
        self.ensure_sentence_starting_uppercase = ensure_sentence_starting_uppercase
        self.ensure_sentence_ends_with_period = ensure_sentence_ends_with_period
        self.use_microphone = use_microphone
        self.spinner = spinner
        self.level = level

        # Real-time transcription parameters
        self.enable_realtime_transcription = enable_realtime_transcription
        self.use_main_model_for_realtime = use_main_model_for_realtime
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.on_realtime_transcription_stabilized = on_realtime_transcription_stabilized

        # Voice activation parameters
        self.silero_sensitivity = silero_sensitivity
        self.silero_use_onnx = silero_use_onnx
        self.silero_deactivity_detection = silero_deactivity_detection
        self.webrtc_sensitivity = webrtc_sensitivity
        self.post_speech_silence_duration = post_speech_silence_duration
        self.min_length_of_recording = min_length_of_recording
        self.min_gap_between_recordings = min_gap_between_recordings
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop

        # Wake word parameters
        self.wakeword_backend = wakeword_backend
        self.openwakeword_model_paths = openwakeword_model_paths
        self.openwakeword_inference_framework = openwakeword_inference_framework
        self.wake_words = wake_words
        self.wake_words_sensitivity = wake_words_sensitivity
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.allowed_latency_limit = allowed_latency_limit
        self.no_log_file = no_log_file
        self.use_extended_logging = use_extended_logging

        # Server URLs
        self.control_url = control_url
        self.data_url = data_url
        self.autostart_server = autostart_server

        # Instance variables
        self.muted = False
        self.recording_thread = None
        self.is_running = True
        self.connection_established = threading.Event()
        self.recording_start = threading.Event()
        self.final_text_ready = threading.Event()
        self.realtime_text = ""
        self.final_text = ""

        if self.debug_mode:
            print("Checking STT server")
        if not self.connect():
            print("Failed to connect to the server.", file=sys.stderr)
        else:
            if self.debug_mode:
                print("STT server is running and connected.")

        if self.use_microphone:
            self.start_recording()

    def text(self, on_transcription_finished=None):
        self.realtime_text = ""
        self.submitted_realtime_text = ""
        self.final_text = ""
        self.final_text_ready.clear()

        self.recording_start.set()

        try:
            total_wait_time = 0
            wait_interval = 0.02  # Wait in small intervals, e.g., 100ms
            max_wait_time = 60  # Timeout after 60 seconds

            while total_wait_time < max_wait_time:
                if self.final_text_ready.wait(timeout=wait_interval):
                    break  # Break if transcription is ready
                
                if not self.realtime_text == self.submitted_realtime_text:
                    if self.on_realtime_transcription_update:
                        self.on_realtime_transcription_update(self.realtime_text)
                    self.submitted_realtime_text = self.realtime_text

                total_wait_time += wait_interval
                
                # Check if a manual interrupt has occurred
                if total_wait_time >= max_wait_time:
                    if self.debug_mode:
                        print("Timeout while waiting for text from the server.")
                    self.recording_start.clear()
                    if on_transcription_finished:
                        threading.Thread(target=on_transcription_finished, args=("",)).start()
                    return ""

            self.recording_start.clear()

            if on_transcription_finished:
                threading.Thread(target=on_transcription_finished, args=(self.final_text,)).start()

            return self.final_text

        except KeyboardInterrupt:
            if self.debug_mode:
                print("KeyboardInterrupt in record_and_send_audio, exiting...")
            raise KeyboardInterrupt

        except Exception as e:
            print(f"Error in AudioToTextRecorderClient.text(): {e}")
            return ""

    def feed_audio(self, chunk, original_sample_rate=16000):
        metadata = {"sampleRate": original_sample_rate}
        metadata_json = json.dumps(metadata)
        metadata_length = len(metadata_json)
        message = struct.pack('<I', metadata_length) + metadata_json.encode('utf-8') + chunk

        if self.is_running:
            self.data_ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)

    def set_microphone(self, microphone_on=True):
        """
        Set the microphone on or off.
        """
        self.muted = not microphone_on
        #self.call_method("set_microphone", [microphone_on])
        # self.use_microphone.value = microphone_on

    def abort(self):
        self.call_method("abort")

    def wakeup(self):
        self.call_method("wakeup")

    def clear_audio_queue(self):
        self.call_method("clear_audio_queue")

    def stop(self):
        self.call_method("stop")

    def connect(self):
        if not self.ensure_server_running():
            print("Cannot start STT server. Exiting.")
            return False
        
        try:
            # Connect to control WebSocket
            self.control_ws = websocket.WebSocketApp(self.control_url,
                                                     on_message=self.on_control_message,
                                                     on_error=self.on_error,
                                                     on_close=self.on_close,
                                                     on_open=self.on_control_open)

            self.control_ws_thread = threading.Thread(target=self.control_ws.run_forever)
            self.control_ws_thread.daemon = False
            self.control_ws_thread.start()

            # Connect to data WebSocket
            self.data_ws = websocket.WebSocketApp(self.data_url,
                                                  on_message=self.on_data_message,
                                                  on_error=self.on_error,
                                                  on_close=self.on_close,
                                                  on_open=self.on_data_open)

            self.data_ws_thread = threading.Thread(target=self.data_ws.run_forever)
            self.data_ws_thread.daemon = False
            self.data_ws_thread.start()

            # Wait for the connections to be established
            if not self.connection_established.wait(timeout=10):
                print("Timeout while connecting to the server.")
                return False

            if self.debug_mode:
                print("WebSocket connections established successfully.")
            return True
        except Exception as e:
            print(f"Error while connecting to the server: {e}")
            return False

    def start_server(self):
        args = ['stt-server']

        # Map constructor parameters to server arguments
        if self.model:
            args += ['--model', self.model]
        if self.realtime_model_type:
            args += ['--realtime_model_type', self.realtime_model_type]
        if self.language:
            args += ['--language', self.language]
        if self.silero_sensitivity is not None:
            args += ['--silero_sensitivity', str(self.silero_sensitivity)]
        if self.webrtc_sensitivity is not None:
            args += ['--webrtc_sensitivity', str(self.webrtc_sensitivity)]
        if self.min_length_of_recording is not None:
            args += ['--min_length_of_recording', str(self.min_length_of_recording)]
        if self.min_gap_between_recordings is not None:
            args += ['--min_gap_between_recordings', str(self.min_gap_between_recordings)]
        if self.realtime_processing_pause is not None:
            args += ['--realtime_processing_pause', str(self.realtime_processing_pause)]
        if self.early_transcription_on_silence is not None:
            args += ['--early_transcription_on_silence', str(self.early_transcription_on_silence)]
        if self.beam_size is not None:
            args += ['--beam_size', str(self.beam_size)]
        if self.beam_size_realtime is not None:
            args += ['--beam_size_realtime', str(self.beam_size_realtime)]
        if self.initial_prompt:
            args += ['--initial_prompt', self.initial_prompt]
        if self.control_url:
            parsed_control_url = urlparse(self.control_url)
            if parsed_control_url.port:
                args += ['--control_port', str(parsed_control_url.port)]
        if self.data_url:
            parsed_data_url = urlparse(self.data_url)
            if parsed_data_url.port:
                args += ['--data_port', str(parsed_data_url.port)]

        # Start the subprocess with the mapped arguments
        if os.name == 'nt':  # Windows
            cmd = 'start /min cmd /c ' + subprocess.list2cmdline(args)
            subprocess.Popen(cmd, shell=True)
        else:  # Unix-like systems
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print("STT server start command issued. Please wait a moment for it to initialize.", file=sys.stderr)

    def is_server_running(self):
        parsed_url = urlparse(self.control_url)
        host = parsed_url.hostname
        port = parsed_url.port or 80
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def ensure_server_running(self):
        if not self.is_server_running():
            if self.debug_mode:
                print("STT server is not running.", file=sys.stderr)
            if self.autostart_server or self.ask_to_start_server():
                self.start_server()
                if self.debug_mode:
                    print("Waiting for STT server to start...", file=sys.stderr)
                for _ in range(20):  # Wait up to 20 seconds
                    if self.is_server_running():
                        if self.debug_mode:
                            print("STT server started successfully.", file=sys.stderr)
                        time.sleep(2)  # Give the server a moment to fully initialize
                        return True
                    time.sleep(1)
                print("Failed to start STT server.", file=sys.stderr)
                return False
            else:
                print("STT server is required. Please start it manually.", file=sys.stderr)
                return False
        return True

    def start_recording(self):
        self.recording_thread = threading.Thread(target=self.record_and_send_audio)
        self.recording_thread.daemon = False
        self.recording_thread.start()

    def setup_audio(self):
        try:
            self.audio_interface = pyaudio.PyAudio()
            self.input_device_index = None
            try:
                default_device = self.audio_interface.get_default_input_device_info()
                self.input_device_index = default_device['index']
            except OSError as e:
                print(f"No default input device found: {e}")
                return False

            self.device_sample_rate = 16000  # Try 16000 Hz first

            try:
                self.stream = self.audio_interface.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=self.device_sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=self.input_device_index,
                )
                if self.debug_mode:
                    print(f"Audio recording initialized successfully at {self.device_sample_rate} Hz")
                return True
            except Exception as e:
                print(f"Failed to initialize audio stream at {self.device_sample_rate} Hz: {e}")
                return False

        except Exception as e:
            print(f"Error initializing audio recording: {e}")
            if self.audio_interface:
                self.audio_interface.terminate()
            return False

    def record_and_send_audio(self):
        try:
            if not self.setup_audio():
                raise Exception("Failed to set up audio recording.")

            if self.debug_mode:
                print("Recording and sending audio...")

            while self.is_running:
                if self.muted:
                    time.sleep(0.01)
                    continue

                try:
                    audio_data = self.stream.read(CHUNK)

                    if self.on_recorded_chunk:
                        self.on_recorded_chunk(audio_data)

                    if self.muted:
                        continue

                    if self.recording_start.is_set():
                        metadata = {"sampleRate": self.device_sample_rate}
                        metadata_json = json.dumps(metadata)
                        metadata_length = len(metadata_json)
                        message = struct.pack('<I', metadata_length) + metadata_json.encode('utf-8') + audio_data

                        if self.is_running:
                            self.data_ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
                except KeyboardInterrupt:  # handle manual interruption (Ctrl+C)
                    if self.debug_mode:
                        print("KeyboardInterrupt in record_and_send_audio, exiting...")
                    break
                except Exception as e:
                    print(f"Error sending audio data: {e}")
                    break  # Exit the recording loop

        except Exception as e:
            print(f"Error in record_and_send_audio: {e}")
        finally:
            self.cleanup_audio()

    def cleanup_audio(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None
        except Exception as e:
            print(f"Error cleaning up audio resources: {e}")

    def on_control_message(self, ws, message):
        try:
            data = json.loads(message)
            # Handle server response with status
            if 'status' in data:
                if data['status'] == 'success':
                    if 'parameter' in data and 'value' in data:
                        if self.debug_mode:
                            print(f"Parameter {data['parameter']} = {data['value']}")
                elif data['status'] == 'error':
                    print(f"Server Error: {data.get('message', '')}")
            else:
                print(f"Unknown control message format: {data}")
        except json.JSONDecodeError:
            print(f"Received non-JSON control message: {message}")
        except Exception as e:
            print(f"Error processing control message: {e}")

    # Handle real-time transcription and full sentence updates
    def on_data_message(self, ws, message):
        try:
            data = json.loads(message)
            # Handle real-time transcription updates
            if data.get('type') == 'realtime':
                if data['text'] != self.realtime_text:
                    self.realtime_text = data['text']

            # Handle full sentences
            elif data.get('type') == 'fullSentence':
                self.final_text = data['text']
                self.final_text_ready.set()

            elif data.get('type') == 'recording_start':
                if self.on_recording_start:
                    self.on_recording_start()
            elif data.get('type') == 'recording_stop':
                if self.on_recording_stop:
                    self.on_recording_stop()
            elif data.get('type') == 'transcription_start':
                if self.on_transcription_start:
                    self.on_transcription_start()
            elif data.get('type') == 'vad_detect_start':
                if self.on_vad_detect_start:
                    self.on_vad_detect_start()
            elif data.get('type') == 'wakeword_detection_start':
                if self.on_wakeword_detection_start:
                    self.on_wakeword_detection_start()
            elif data.get('type') == 'wakeword_detection_end':
                if self.on_wakeword_detection_end:
                    self.on_wakeword_detection_end()

            else:
                print(f"Unknown data message format: {data}")

        except json.JSONDecodeError:
            print(f"Received non-JSON data message: {message}")
        except Exception as e:
            print(f"Error processing data message: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        if self.debug_mode:
            if ws == self.data_ws:
                print(f"Data WebSocket connection closed: {close_status_code} - {close_msg}")
            elif ws == self.control_ws:
                print(f"Control WebSocket connection closed: {close_status_code} - {close_msg}")
        
        self.is_running = False

    def on_control_open(self, ws):
        if self.debug_mode:
            print("Control WebSocket connection opened.")
        self.connection_established.set()

    def on_data_open(self, ws):
        if self.debug_mode:
            print("Data WebSocket connection opened.")

    def set_parameter(self, parameter, value):
        command = {
            "command": "set_parameter",
            "parameter": parameter,
            "value": value
        }
        self.control_ws.send(json.dumps(command))

    def get_parameter(self, parameter):
        command = {
            "command": "get_parameter",
            "parameter": parameter
        }
        self.control_ws.send(json.dumps(command))

    def call_method(self, method, args=None, kwargs=None):
        command = {
            "command": "call_method",
            "method": method,
            "args": args or [],
            "kwargs": kwargs or {}
        }
        self.control_ws.send(json.dumps(command))

    def shutdown(self):
        self.is_running = False
        #self.stop_event.set()
        if self.control_ws:
            self.control_ws.close()
        if self.data_ws:
            self.data_ws.close()

        # Join threads to ensure they finish before exiting
        if self.control_ws_thread:
            self.control_ws_thread.join()
        if self.data_ws_thread:
            self.data_ws_thread.join()
        if self.recording_thread:
            self.recording_thread.join()

        # Clean up audio resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()

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
