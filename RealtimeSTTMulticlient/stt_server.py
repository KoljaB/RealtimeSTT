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
    - `-r, --rt-model, --realtime_model_type`: Real-time model size; default 'base'.
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

from install_packages import check_and_install_packages
from difflib import SequenceMatcher
from collections import deque
from datetime import datetime
import logging
import asyncio
import pyaudio
import base64
import sys
import time


# Global constants that don't change
send_recorded_chunk = False  # Keep this as it's not used in the current implementation

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
from collections import deque
import signal
import atexit
import uuid

# JSON schema validation
def validate_command_json(data):
    """Validate JSON command structure to prevent injection attacks"""
    if not isinstance(data, dict):
        raise ValueError("Command must be a JSON object")
    
    # Check required fields
    if 'command' not in data:
        raise ValueError("Command field is required")
    
    command = data.get('command')
    if not isinstance(command, str):
        raise ValueError("Command must be a string")
    
    # Validate command types and their required fields
    if command == 'set_parameter':
        if 'parameter' not in data or 'value' not in data:
            raise ValueError("set_parameter requires 'parameter' and 'value' fields")
        if not isinstance(data['parameter'], str):
            raise ValueError("Parameter name must be a string")
        # Value can be string, int, float, or bool
        if not isinstance(data['value'], (str, int, float, bool)):
            raise ValueError("Parameter value must be string, number, or boolean")
    
    elif command == 'get_parameter':
        if 'parameter' not in data:
            raise ValueError("get_parameter requires 'parameter' field")
        if not isinstance(data['parameter'], str):
            raise ValueError("Parameter name must be a string")
        # request_id is optional but must be string/number if present
        if 'request_id' in data and not isinstance(data['request_id'], (str, int)):
            raise ValueError("request_id must be string or number")
    
    elif command == 'call_method':
        if 'method' not in data:
            raise ValueError("call_method requires 'method' field")
        if not isinstance(data['method'], str):
            raise ValueError("Method name must be a string")
        # args must be list if present
        if 'args' in data and not isinstance(data['args'], list):
            raise ValueError("args must be a list")
        # kwargs must be dict if present
        if 'kwargs' in data and not isinstance(data['kwargs'], dict):
            raise ValueError("kwargs must be an object")
    
    else:
        raise ValueError(f"Unknown command: {command}")
    
    return True

class ClientSession:
    """Encapsulates the state for a single client connection"""
    
    def __init__(self, session_id, control_socket, data_socket, recorder_config, global_args, loop, server_config=None):
        self.session_id = session_id
        self.control_socket = control_socket
        self.data_socket = data_socket
        self.recorder = None
        self.recorder_config = recorder_config.copy()
        self.global_args = global_args
        self.loop = loop
        self.server_config = server_config  # Reference to server configuration
        self.recorder_ready = threading.Event()
        self.recorder_thread = None
        self.last_activity = time.time()  # Track last activity for timeout
        self.stop_recorder = False
        self.cleanup_started = False  # Prevent multiple cleanup calls
        self.cleanup_completed = False  # Track cleanup state
        
        # Session state management to prevent race conditions
        self.session_state = "initializing"  # States: initializing, control_connected, ready, cleanup
        self.control_connected = False
        self.data_connected = False
        self.state_timeout = time.time() + 30.0  # 30 second timeout for session establishment
        self.prev_text = ""
        self.text_time_deque = deque()
        self.wav_file = None
        
        # Hard break constants (moved from global)
        self.hard_break_even_on_background_noise = 3.0
        self.hard_break_even_on_background_noise_min_texts = 3
        self.hard_break_even_on_background_noise_min_similarity = 0.99
        self.hard_break_even_on_background_noise_min_chars = 15
        
        self._initialize_recorder()
    
    def _initialize_recorder(self):
        """Initialize the AudioToTextRecorder for this session"""
        def recorder_thread_func():
            try:
                # Update callbacks to use session-specific methods
                self.recorder_config.update({
                    'on_realtime_transcription_update': self._make_callback(self._text_detected),
                    'on_recording_start': self._make_callback(self._on_recording_start),
                    'on_recording_stop': self._make_callback(self._on_recording_stop),
                    'on_vad_detect_start': self._make_callback(self._on_vad_detect_start),
                    'on_vad_detect_stop': self._make_callback(self._on_vad_detect_stop),
                    'on_wakeword_detected': self._make_callback(self._on_wakeword_detected),
                    'on_wakeword_detection_start': self._make_callback(self._on_wakeword_detection_start),
                    'on_wakeword_detection_end': self._make_callback(self._on_wakeword_detection_end),
                    'on_transcription_start': self._make_callback(self._on_transcription_start),
                    'on_turn_detection_start': self._make_callback(self._on_turn_detection_start),
                    'on_turn_detection_stop': self._make_callback(self._on_turn_detection_stop),
                })
                
                self.recorder = AudioToTextRecorder(**self.recorder_config)
                print(f"{bcolors.OKGREEN}Session {self.session_id}: RealtimeSTT initialized{bcolors.ENDC}")
                self.recorder_ready.set()
                
                def process_text(full_sentence):
                    self.prev_text = ""
                    full_sentence = preprocess_text(full_sentence)
                    message = json.dumps({
                        'type': 'fullSentence',
                        'text': full_sentence
                    })
                    if self.server_config and self.server_config.extended_logging:
                        print(f"{bcolors.OKBLUE}📝 Session {self.session_id}: Generated fullSentence message: {message}{bcolors.ENDC}")
                    
                    self._send_to_client_sync(message)

                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    if self.server_config and self.server_config.extended_logging:
                        print(f"  [{timestamp}] Session {self.session_id} - Full text: {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n", flush=True, end="")
                    else:
                        print(f"\r[{timestamp}] Session {self.session_id} - {bcolors.BOLD}Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{full_sentence}{bcolors.ENDC}\n")
                
                while not self.stop_recorder:
                    if self.recorder:
                        self.recorder.text(process_text)
                        
            except Exception as e:
                error_msg = f"Failed to initialize RealtimeSTT recorder for session {self.session_id}: {str(e)}"
                print(f"{bcolors.FAIL}ERROR: {error_msg}{bcolors.ENDC}")
                self.recorder = None
                self.recorder_ready.set()
        
        self.recorder_thread = threading.Thread(target=recorder_thread_func)
        self.recorder_thread.start()
    
    def _make_callback(self, callback_func):
        """Create a callback that passes the loop to the function"""
        def inner_callback(*args, **kwargs):
            callback_func(*args, **kwargs)
        return inner_callback
    
    def _send_to_client_sync(self, message):
        """Send message to client - synchronous version for thread safety"""
        # Don't send messages if cleanup has started
        if self.cleanup_started:
            return
        
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        if self.server_config and self.server_config.extended_logging:
            print(f"  [{timestamp}] Session {self.session_id}: ATTEMPTING to send message: {message[:100]}...")
        
        if not self.data_socket:
            print(f"{bcolors.WARNING}Session {self.session_id}: No data socket available{bcolors.ENDC}")
            return
            
        # Check WebSocket state properly
        try:
            is_closed = self.data_socket.closed
        except AttributeError:
            # Different websockets versions use different attributes
            is_closed = getattr(self.data_socket, 'close_code', None) is not None
        
        if self.server_config and self.server_config.extended_logging:
            print(f"  [{timestamp}] Session {self.session_id}: Data socket state - closed: {is_closed}")
        
        if is_closed:
            print(f"{bcolors.WARNING}Session {self.session_id}: Data socket is closed{bcolors.ENDC}")
            return
        
        # Queue the async send operation without waiting for it
        async def async_send():
            try:
                if self.server_config and self.server_config.extended_logging:
                    print(f"  [{timestamp}] Session {self.session_id}: About to call websocket.send()...")
                await self.data_socket.send(message)
                if self.server_config and self.server_config.extended_logging:
                    print(f"{bcolors.OKGREEN}  [{timestamp}] Session {self.session_id}: ✅ Message SENT successfully{bcolors.ENDC}")
            except websockets.exceptions.ConnectionClosed as e:
                print(f"{bcolors.WARNING}Session {self.session_id}: ❌ Connection closed during send: {e}{bcolors.ENDC}")
            except Exception as e:
                print(f"{bcolors.FAIL}Session {self.session_id}: ❌ Error sending message: {type(e).__name__}: {e}{bcolors.ENDC}")
        
        # Schedule the coroutine without waiting for result
        try:
            asyncio.run_coroutine_threadsafe(async_send(), self.loop)
            if self.server_config and self.server_config.extended_logging:
                print(f"{bcolors.OKBLUE}  [{timestamp}] Session {self.session_id}: Message scheduled for async sending{bcolors.ENDC}")
        except Exception as e:
            print(f"{bcolors.FAIL}Session {self.session_id}: Failed to schedule message: {e}{bcolors.ENDC}")
    
    def _text_detected(self, text):
        """Handle real-time transcription updates for this session"""
        text = preprocess_text(text)
        
        if self.server_config and self.server_config.silence_timing:
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
                self.recorder.post_speech_silence_duration = self.global_args.mid_sentence_detection_pause
            elif sentence_end(text) and sentence_end(self.prev_text) and not ends_with_ellipsis(self.prev_text):
                self.recorder.post_speech_silence_duration = self.global_args.end_of_sentence_detection_pause
            else:
                self.recorder.post_speech_silence_duration = self.global_args.unknown_sentence_detection_pause

            current_time = time.time()
            self.text_time_deque.append((current_time, text))

            while self.text_time_deque and self.text_time_deque[0][0] < current_time - self.hard_break_even_on_background_noise:
                self.text_time_deque.popleft()

            if len(self.text_time_deque) >= self.hard_break_even_on_background_noise_min_texts:
                texts = [t[1] for t in self.text_time_deque]
                first_text = texts[0]
                last_text = texts[-1]

                similarity = SequenceMatcher(None, first_text, last_text).ratio()

                if similarity > self.hard_break_even_on_background_noise_min_similarity and len(first_text) > self.hard_break_even_on_background_noise_min_chars:
                    self.recorder.stop()
                    self.recorder.clear_audio_queue()
                    self.prev_text = ""
                    return

        self.prev_text = text

        message = json.dumps({
            'type': 'realtime',
            'text': text
        })
        if self.server_config and self.server_config.extended_logging:
            print(f"{bcolors.OKBLUE}🔄 Session {self.session_id}: Generated realtime message: {message}{bcolors.ENDC}")
        
        self._send_to_client_sync(message)

        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        if self.server_config and self.server_config.extended_logging:
            print(f"  [{timestamp}] Session {self.session_id} - Realtime text: {bcolors.OKCYAN}{text}{bcolors.ENDC}\n", flush=True, end="")
        else:
            print(f"\r[{timestamp}] Session {self.session_id} - {bcolors.OKCYAN}{text}{bcolors.ENDC}", flush=True, end='')
    
    def _on_recording_start(self):
        message = json.dumps({'type': 'recording_start'})
        self._send_to_client_sync(message)
    
    def _on_recording_stop(self):
        message = json.dumps({'type': 'recording_stop'})
        self._send_to_client_sync(message)
    
    def _on_vad_detect_start(self):
        if self.server_config and self.server_config.extended_logging:
            print(f"{bcolors.OKBLUE}🗢️ Session {self.session_id}: VAD detect start callback triggered{bcolors.ENDC}")
        message = json.dumps({'type': 'vad_detect_start'})
        self._send_to_client_sync(message)
    
    def _on_vad_detect_stop(self):
        message = json.dumps({'type': 'vad_detect_stop'})
        self._send_to_client_sync(message)
    
    def _on_wakeword_detected(self):
        message = json.dumps({'type': 'wakeword_detected'})
        self._send_to_client_sync(message)
    
    def _on_wakeword_detection_start(self):
        message = json.dumps({'type': 'wakeword_detection_start'})
        self._send_to_client_sync(message)
    
    def _on_wakeword_detection_end(self):
        message = json.dumps({'type': 'wakeword_detection_end'})
        self._send_to_client_sync(message)
    
    def _on_transcription_start(self, audio_bytes):
        bytes_b64 = base64.b64encode(audio_bytes.tobytes()).decode('utf-8')
        message = json.dumps({
            'type': 'transcription_start',
            'audio_bytes_base64': bytes_b64
        })
        self._send_to_client_sync(message)
    
    def _on_turn_detection_start(self):
        message = json.dumps({'type': 'start_turn_detection'})
        self._send_to_client_sync(message)
    
    def _on_turn_detection_stop(self):
        message = json.dumps({'type': 'stop_turn_detection'})
        self._send_to_client_sync(message)
    
    def wait_for_recorder_ready(self):
        """Wait for the recorder to be ready"""
        self.recorder_ready.wait()
        return self.recorder is not None
    
    def feed_audio(self, audio_data):
        """Feed audio data to this session's recorder"""
        if self.should_process_audio():
            self.recorder.feed_audio(audio_data)
    
    def handle_control_command(self, command_data):
        """Handle control commands for this session"""
        if not self.recorder:
            return {"status": "error", "message": "Recorder not ready for this session"}
        
        command = command_data.get("command")
        
        if command == "set_parameter":
            parameter = command_data.get("parameter")
            value = command_data.get("value")
            if parameter in allowed_parameters and hasattr(self.recorder, parameter):
                setattr(self.recorder, parameter, value)
                return {"status": "success", "message": f"Parameter {parameter} set to {value}"}
            else:
                return {"status": "error", "message": f"Parameter {parameter} not allowed or doesn't exist"}
        
        elif command == "get_parameter":
            parameter = command_data.get("parameter")
            request_id = command_data.get("request_id")
            if parameter in allowed_parameters and hasattr(self.recorder, parameter):
                value = getattr(self.recorder, parameter)
                response = {"status": "success", "parameter": parameter, "value": value}
                if request_id is not None:
                    response["request_id"] = request_id
                return response
            else:
                return {"status": "error", "message": f"Parameter {parameter} not allowed or doesn't exist"}
        
        elif command == "call_method":
            method_name = command_data.get("method")
            if method_name in allowed_methods:
                method = getattr(self.recorder, method_name, None)
                if method and callable(method):
                    args = command_data.get("args", [])
                    kwargs = command_data.get("kwargs", {})
                    method(*args, **kwargs)
                    return {"status": "success", "message": f"Method {method_name} called"}
                else:
                    return {"status": "error", "message": f"Method {method_name} not found"}
            else:
                return {"status": "error", "message": f"Method {method_name} not allowed"}
        
        return {"status": "error", "message": f"Unknown command {command}"}
    
    def update_activity(self):
        """Update the last activity timestamp"""
        if not self.cleanup_started:  # Don't update activity during cleanup
            self.last_activity = time.time()
    
    def is_inactive(self, timeout_seconds):
        """Check if session has been inactive for longer than timeout"""
        return (time.time() - self.last_activity) > timeout_seconds
    
    def is_state_expired(self):
        """Check if session state timeout has been exceeded"""
        return (self.session_state != "ready" and 
                self.session_state != "cleanup" and 
                time.time() > self.state_timeout)
    
    def is_cleanup_completed(self):
        """Check if cleanup has been completed"""
        return self.cleanup_completed
    
    def should_process_audio(self):
        """Check if session should still process audio data"""
        return (not self.stop_recorder and 
                not self.cleanup_started and 
                self.recorder is not None and
                self.session_state == "ready")
    
    def set_control_connected(self, control_socket):
        """Mark control socket as connected and update session state"""
        self.control_socket = control_socket
        self.control_connected = True
        self.session_state = "control_connected"
        print(f"{bcolors.OKGREEN}Session {self.session_id}: Control socket connected{bcolors.ENDC}")
    
    def set_data_connected(self, data_socket):
        """Mark data socket as connected and update session state if ready"""
        self.data_socket = data_socket
        self.data_connected = True
        
        # Only transition to ready if control is also connected
        if self.control_connected and self.session_state == "control_connected":
            self.session_state = "ready"
            print(f"{bcolors.OKGREEN}Session {self.session_id}: Both sockets connected, session ready{bcolors.ENDC}")
        else:
            print(f"{bcolors.WARNING}Session {self.session_id}: Data socket connected but waiting for control socket{bcolors.ENDC}")
    
    def is_session_ready(self):
        """Check if session is fully ready for audio processing"""
        return (self.session_state == "ready" and 
                self.control_connected and 
                self.data_connected and 
                not self.cleanup_started)
    
    def get_session_state(self):
        """Get current session state for debugging"""
        return {
            "session_id": self.session_id,
            "state": self.session_state,
            "control_connected": self.control_connected,
            "data_connected": self.data_connected,
            "cleanup_started": self.cleanup_started,
            "recorder_ready": self.recorder is not None
        }
    
    def cleanup(self):
        """Clean up resources for this session with proper thread termination"""
        # Prevent multiple cleanup calls (thread-safe)
        if self.cleanup_started:
            print(f"{bcolors.WARNING}Session {self.session_id}: Cleanup already in progress or completed{bcolors.ENDC}")
            return
        
        self.cleanup_started = True
        self.session_state = "cleanup"  # Mark session as in cleanup state
        print(f"{bcolors.WARNING}Cleaning up session {self.session_id}{bcolors.ENDC}")
        
        # Step 1: Signal the thread to stop
        self.stop_recorder = True
        
        # Step 2: Stop the recorder first to interrupt any blocking operations
        if self.recorder:
            try:
                print(f"{bcolors.WARNING}Session {self.session_id}: Stopping recorder...{bcolors.ENDC}")
                self.recorder.abort()  # Immediately abort any ongoing operations
                self.recorder.stop()   # Signal the recorder to stop
                self.recorder.clear_audio_queue()  # Clear any pending audio
            except Exception as e:
                print(f"{bcolors.WARNING}Session {self.session_id}: Error stopping recorder: {e}{bcolors.ENDC}")
        
        # Step 3: Wait for thread termination with escalating timeouts
        if self.recorder_thread and self.recorder_thread.is_alive():
            print(f"{bcolors.WARNING}Session {self.session_id}: Waiting for recorder thread to terminate...{bcolors.ENDC}")
            
            # First attempt: Give it 10 seconds to terminate gracefully
            self.recorder_thread.join(timeout=10.0)
            
            if self.recorder_thread.is_alive():
                print(f"{bcolors.WARNING}Session {self.session_id}: Thread still alive after 10s, forcing shutdown...{bcolors.ENDC}")
                
                # Second attempt: Force recorder shutdown and wait another 5 seconds
                if self.recorder:
                    try:
                        self.recorder.shutdown()  # More aggressive shutdown
                    except Exception as e:
                        print(f"{bcolors.WARNING}Session {self.session_id}: Error during forced shutdown: {e}{bcolors.ENDC}")
                
                self.recorder_thread.join(timeout=5.0)
                
                # Final check: Log if thread is still alive (zombie thread)
                if self.recorder_thread.is_alive():
                    print(f"{bcolors.FAIL}Session {self.session_id}: WARNING - Thread failed to terminate! Zombie thread detected.{bcolors.ENDC}")
                    print(f"{bcolors.FAIL}Session {self.session_id}: Thread will be abandoned but may cause memory leaks.{bcolors.ENDC}")
                else:
                    print(f"{bcolors.OKGREEN}Session {self.session_id}: Thread terminated successfully after forced shutdown.{bcolors.ENDC}")
            else:
                print(f"{bcolors.OKGREEN}Session {self.session_id}: Thread terminated gracefully.{bcolors.ENDC}")
        
        # Step 4: Clean up recorder resources (only after thread is stopped/abandoned)
        if self.recorder:
            try:
                # Final cleanup of recorder resources
                self.recorder = None
                print(f"{bcolors.OKGREEN}Session {self.session_id}: Recorder resources cleaned up.{bcolors.ENDC}")
            except Exception as e:
                print(f"{bcolors.WARNING}Session {self.session_id}: Error cleaning up recorder: {e}{bcolors.ENDC}")
        
        # Step 5: Clean up file resources (safe to do after thread termination)
        if self.wav_file:
            try:
                self.wav_file.close()
                print(f"{bcolors.OKGREEN}Session {self.session_id}: WAV file closed successfully.{bcolors.ENDC}")
            except Exception as e:
                print(f"{bcolors.WARNING}Session {self.session_id}: Error closing WAV file: {e}{bcolors.ENDC}")
            finally:
                self.wav_file = None
        
        # Step 6: Clear session data structures
        self.text_time_deque.clear()
        self.prev_text = ""
        
        # Step 7: Mark cleanup as completed
        self.cleanup_completed = True
        print(f"{bcolors.OKGREEN}Session {self.session_id}: Cleanup completed successfully.{bcolors.ENDC}")
            
class WebSocketServer:
    """Main server application managing multiple client sessions"""
    
    def __init__(self, args):
        self.args = args
        self.sessions = {}  # session_id -> ClientSession
        self.session_counter = 0
        self.loop = None
        self.timeout_task = None  # Background task for checking timeouts
        
        # Move global configuration variables here
        self.debug_logging = args.debug if hasattr(args, 'debug') else False
        self.extended_logging = args.use_extended_logging if hasattr(args, 'use_extended_logging') else False
        self.writechunks = args.write if hasattr(args, 'write') else False
        self.log_incoming_chunks = args.logchunks if hasattr(args, 'logchunks') else False
        self.silence_timing = args.silence_timing if hasattr(args, 'silence_timing') else False
        self.loglevel = logging.WARNING
        
        # Timeout checker state management for robustness
        self.timeout_checker_failures = 0
        self.timeout_checker_max_failures = 5
        self.timeout_checker_backoff = 10  # seconds
        self.timeout_checker_running = False
        self.timeout_checker_last_success = time.time()
    
    def _generate_session_id(self):
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    async def _send_session_id(self, websocket, session_id):
        """Send session ID to client"""
        message = json.dumps({
            'type': 'session_init',
            'session_id': session_id
        })
        await websocket.send(message)
    
    def _get_session_from_message(self, message_data):
        """Extract session ID from message and return corresponding session"""
        session_id = message_data.get('session_id')
        if not session_id:
            return None
        return self.sessions.get(session_id)
    
    async def _check_session_timeouts(self):
        """Periodically check for and cleanup inactive sessions with robust error handling"""
        self.timeout_checker_running = True
        
        while self.timeout_checker_running:
            try:
                # Adaptive sleep - longer delays after failures
                sleep_duration = 10 + (self.timeout_checker_failures * self.timeout_checker_backoff)
                await asyncio.sleep(min(sleep_duration, 60))  # Cap at 60 seconds
                
                # Circuit breaker - stop trying if too many failures
                if self.timeout_checker_failures >= self.timeout_checker_max_failures:
                    print(f"{bcolors.FAIL}Timeout checker circuit breaker OPEN - too many failures ({self.timeout_checker_failures}){bcolors.ENDC}")
                    print(f"{bcolors.FAIL}Timeout checker suspended to prevent server crash{bcolors.ENDC}")
                    await asyncio.sleep(300)  # Wait 5 minutes before trying again
                    self.timeout_checker_failures = 0  # Reset failure count
                    print(f"{bcolors.WARNING}Timeout checker attempting to resume...{bcolors.ENDC}")
                    continue
                
                # Attempt session cleanup with comprehensive error handling
                await self._perform_session_cleanup()
                
                # Reset failure count on success
                self.timeout_checker_failures = 0
                self.timeout_checker_last_success = time.time()
                
            except asyncio.CancelledError:
                print(f"{bcolors.WARNING}Timeout checker cancelled - shutting down gracefully{bcolors.ENDC}")
                break
                
            except KeyboardInterrupt:
                print(f"{bcolors.WARNING}Timeout checker interrupted - shutting down{bcolors.ENDC}")
                break
                
            except Exception as e:
                self.timeout_checker_failures += 1
                failure_time = time.time() - self.timeout_checker_last_success
                
                print(f"{bcolors.FAIL}Timeout checker error #{self.timeout_checker_failures}: {type(e).__name__}: {e}{bcolors.ENDC}")
                print(f"{bcolors.FAIL}Last successful run: {failure_time:.1f} seconds ago{bcolors.ENDC}")
                
                # Log additional context for debugging
                try:
                    print(f"{bcolors.FAIL}Active sessions: {len(self.sessions)}, Timeout: {getattr(self.args, 'session_timeout', 'unknown')}s{bcolors.ENDC}")
                except Exception as debug_error:
                    print(f"{bcolors.FAIL}Could not log debug info: {debug_error}{bcolors.ENDC}")
                
                # Progressive backoff - longer delays after more failures
                if self.timeout_checker_failures < self.timeout_checker_max_failures:
                    backoff_delay = min(self.timeout_checker_failures * 30, 300)  # Max 5 minutes
                    print(f"{bcolors.WARNING}Timeout checker will retry in {backoff_delay} seconds{bcolors.ENDC}")
                
        self.timeout_checker_running = False
        print(f"{bcolors.OKGREEN}Timeout checker stopped cleanly{bcolors.ENDC}")
    
    async def _perform_session_cleanup(self):
        """Perform the actual session cleanup with detailed error handling"""
        try:
            timeout_seconds = self.args.session_timeout
            sessions_to_cleanup = []
            
            # Step 1: Identify sessions to cleanup (with error handling)
            try:
                # Create a copy of sessions dict to avoid modification during iteration
                sessions_snapshot = dict(self.sessions)
                
                for session_id, session in sessions_snapshot.items():
                    try:
                        # Check for both inactivity timeout and state expiration
                        if session.is_inactive(timeout_seconds):
                            print(f"{bcolors.WARNING}Session {session_id} inactive for {timeout_seconds} seconds{bcolors.ENDC}")
                            sessions_to_cleanup.append(session_id)
                        elif session.is_state_expired():
                            print(f"{bcolors.WARNING}Session {session_id} state expired (stuck in {session.session_state}){bcolors.ENDC}")
                            sessions_to_cleanup.append(session_id)
                            
                    except Exception as session_check_error:
                        print(f"{bcolors.WARNING}Error checking session {session_id}: {session_check_error}{bcolors.ENDC}")
                        # Add to cleanup list if we can't check it properly
                        sessions_to_cleanup.append(session_id)
                        
            except Exception as identification_error:
                print(f"{bcolors.FAIL}Error identifying sessions for cleanup: {identification_error}{bcolors.ENDC}")
                raise  # Re-raise to trigger circuit breaker
            
            # Step 2: Clean up identified sessions
            cleanup_errors = []
            for session_id in sessions_to_cleanup:
                try:
                    await self._cleanup_single_session(session_id, timeout_seconds)
                except Exception as cleanup_error:
                    cleanup_errors.append(f"Session {session_id}: {cleanup_error}")
            
            # Report cleanup errors but don't fail the entire operation
            if cleanup_errors:
                print(f"{bcolors.WARNING}Some session cleanups failed:{bcolors.ENDC}")
                for error in cleanup_errors:
                    print(f"{bcolors.WARNING}  - {error}{bcolors.ENDC}")
                    
        except Exception as overall_error:
            print(f"{bcolors.FAIL}Critical error in session cleanup: {overall_error}{bcolors.ENDC}")
            raise  # Re-raise to increment failure count
    
    async def _cleanup_single_session(self, session_id, timeout_seconds):
        """Clean up a single session with comprehensive error handling"""
        if session_id not in self.sessions:
            return  # Session already cleaned up
            
        session = self.sessions[session_id]
        print(f"{bcolors.WARNING}Session {session_id} inactive for {timeout_seconds}s, cleaning up{bcolors.ENDC}")
        
        # Step 1: Close WebSocket connections with specific error handling
        websocket_errors = []
        
        if session.control_socket and hasattr(session.control_socket, 'close'):
            try:
                await session.control_socket.close()
            except Exception as control_error:
                websocket_errors.append(f"Control socket: {control_error}")
        
        if session.data_socket and hasattr(session.data_socket, 'close'):
            try:
                await session.data_socket.close()
            except Exception as data_error:
                websocket_errors.append(f"Data socket: {data_error}")
        
        # Step 2: Clean up session resources
        try:
            session.cleanup()
        except Exception as resource_cleanup_error:
            print(f"{bcolors.WARNING}Session {session_id} resource cleanup error: {resource_cleanup_error}{bcolors.ENDC}")
            # Continue with session removal even if cleanup fails
        
        # Step 3: Remove session from active sessions
        try:
            del self.sessions[session_id]
            max_users = self.args.max_concurrent_users
            print(f"{bcolors.OKGREEN}Session {session_id} cleaned up successfully ({len(self.sessions)}/{max_users} users){bcolors.ENDC}")
        except Exception as removal_error:
            print(f"{bcolors.WARNING}Error removing session {session_id} from active sessions: {removal_error}{bcolors.ENDC}")
        
        # Report WebSocket errors (non-critical)
        if websocket_errors:
            print(f"{bcolors.WARNING}WebSocket cleanup issues for session {session_id}:{bcolors.ENDC}")
            for error in websocket_errors:
                print(f"{bcolors.WARNING}  - {error}{bcolors.ENDC}")
    
    def get_timeout_checker_status(self):
        """Get current status of the timeout checker for monitoring"""
        current_time = time.time()
        time_since_success = current_time - self.timeout_checker_last_success
        
        if not self.timeout_checker_running:
            status = "stopped"
        elif self.timeout_checker_failures >= self.timeout_checker_max_failures:
            status = "circuit_breaker_open"
        elif self.timeout_checker_failures > 0:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "running": self.timeout_checker_running,
            "failure_count": self.timeout_checker_failures,
            "max_failures": self.timeout_checker_max_failures,
            "time_since_last_success": time_since_success,
            "last_success_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timeout_checker_last_success)),
            "circuit_breaker_open": self.timeout_checker_failures >= self.timeout_checker_max_failures,
            "active_sessions": len(self.sessions)
        }
    
    def force_timeout_checker_reset(self):
        """Emergency reset of timeout checker state (for debugging/recovery)"""
        old_failures = self.timeout_checker_failures
        self.timeout_checker_failures = 0
        self.timeout_checker_last_success = time.time()
        print(f"{bcolors.OKGREEN}Timeout checker state reset - failures cleared from {old_failures} to 0{bcolors.ENDC}")
        return True
    
    async def stop_timeout_checker(self):
        """Gracefully stop the timeout checker"""
        if self.timeout_checker_running:
            print(f"{bcolors.WARNING}Stopping timeout checker gracefully...{bcolors.ENDC}")
            self.timeout_checker_running = False
            
            # Cancel the timeout task if it exists
            if self.timeout_task and not self.timeout_task.done():
                self.timeout_task.cancel()
                try:
                    await self.timeout_task
                except asyncio.CancelledError:
                    pass
            
            print(f"{bcolors.OKGREEN}Timeout checker stopped{bcolors.ENDC}")
    
    async def control_handler(self, websocket):
        """Handle control WebSocket connections"""
        # Rate limiting: Check for too many connections from same IP
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        current_time = time.time()
        
        # Clean old connection attempts (older than 10 seconds)
        if not hasattr(self, 'connection_attempts'):
            self.connection_attempts = {}
            
        # Clean old attempts
        self.connection_attempts = {
            ip: timestamps for ip, timestamps in self.connection_attempts.items() 
            if any(t > current_time - 10 for t in timestamps)
        }
        
        # Check rate limit (max 5 connections per IP per 10 seconds)
        if client_ip in self.connection_attempts:
            recent_attempts = [t for t in self.connection_attempts[client_ip] if t > current_time - 10]
            if len(recent_attempts) >= 5:
                print(f"{bcolors.WARNING}Rate limit exceeded for IP {client_ip} - rejecting connection{bcolors.ENDC}")
                await websocket.close()
                return
            self.connection_attempts[client_ip].append(current_time)
        else:
            self.connection_attempts[client_ip] = [current_time]
        
        # Check if we've reached the maximum concurrent users
        max_users = self.args.max_concurrent_users
        if len(self.sessions) >= max_users:
            await websocket.send(json.dumps({
                "status": "error", 
                "message": f"Server has reached maximum capacity ({max_users} concurrent users). Please try again later."
            }))
            await websocket.close()
            print(f"{bcolors.WARNING}Connection rejected - Maximum capacity ({max_users} users) reached from {client_ip}{bcolors.ENDC}")
            return
            
        session_id = self._generate_session_id()
        debug_print(f"New control connection from {websocket.remote_address} - Session: {session_id}", self.debug_logging)
        print(f"{bcolors.OKGREEN}Control client connected - Session: {session_id} ({len(self.sessions)+1}/{max_users}){bcolors.ENDC}")
        
        # Create new session with only control socket initially
        session = ClientSession(session_id, None, None, self._get_recorder_config(), self.args, self.loop, self)
        self.sessions[session_id] = session
        
        # Set control socket connection (this updates session state)
        session.set_control_connected(websocket)
        
        # Send session ID to client
        await self._send_session_id(websocket, session_id)
        
        try:
            async for message in websocket:
                debug_print(f"Received control message for session {session_id}: {message[:200]}...", self.debug_logging)
                
                if isinstance(message, str):
                    try:
                        command_data = json.loads(message)
                        validate_command_json(command_data)
                        
                        # Get session from message or use the connection's session
                        session = self._get_session_from_message(command_data)
                        if not session:
                            # If no session_id in message, use the connection's session
                            session = self.sessions.get(session_id)
                        
                        if not session:
                            await websocket.send(json.dumps({"status": "error", "message": "Session not found"}))
                            continue
                        
                        # Update activity for any control command
                        session.update_activity()
                        
                        if not session.wait_for_recorder_ready():
                            await websocket.send(json.dumps({"status": "error", "message": "Recorder not ready"}))
                            continue
                        
                        response = session.handle_control_command(command_data)
                        await websocket.send(json.dumps(response))
                        
                    except json.JSONDecodeError:
                        print(f"{bcolors.WARNING}Session {session_id}: Received invalid JSON command{bcolors.ENDC}")
                        await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON format"}))
                    except ValueError as e:
                        print(f"{bcolors.WARNING}Session {session_id}: Invalid command structure: {e}{bcolors.ENDC}")
                        await websocket.send(json.dumps({"status": "error", "message": f"Invalid command: {str(e)}"}))
                else:
                    print(f"{bcolors.WARNING}Session {session_id}: Received unknown message type on control connection{bcolors.ENDC}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"{bcolors.WARNING}Control client disconnected - Session: {session_id} - {e}{bcolors.ENDC}")
        finally:
            # Don't immediately clean up session when control connection closes
            # The data connection might still be active and needs to send messages
            # Session cleanup will happen when data connection closes
            if session_id in self.sessions:
                print(f"{bcolors.WARNING}Control connection closed for session {session_id}, but keeping session active for data connection{bcolors.ENDC}")
                # Just clear the control socket reference, don't cleanup the whole session
                self.sessions[session_id].control_socket = None
    
    async def data_handler(self, websocket):
        """Handle data WebSocket connections"""
        session_id = None
        print(f"{bcolors.OKGREEN}Data client connected{bcolors.ENDC}")
        
        # Wait for the first message to contain session_id
        try:
            first_message = await websocket.recv()
            if isinstance(first_message, str):
                try:
                    init_data = json.loads(first_message)
                    session_id = init_data.get('session_id')
                    if not session_id:
                        await websocket.send(json.dumps({"status": "error", "message": "Session ID required"}))
                        return
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON in first message"}))
                    return
            else:
                await websocket.send(json.dumps({"status": "error", "message": "First message must contain session_id"}))
                return
        except:
            return
        
        # Validate that session exists and was properly created by control handler
        if session_id not in self.sessions:
            error_msg = f"Session {session_id} not found. Control connection must be established first."
            print(f"{bcolors.FAIL}Data connection rejected: {error_msg}{bcolors.ENDC}")
            await websocket.send(json.dumps({
                "status": "error", 
                "message": error_msg
            }))
            return
        
        session = self.sessions[session_id]
        
        # Check if session is in a valid state to accept data connection
        if session.session_state == "cleanup":
            error_msg = f"Session {session_id} is being cleaned up. Cannot accept data connection."
            print(f"{bcolors.FAIL}Data connection rejected: {error_msg}{bcolors.ENDC}")
            await websocket.send(json.dumps({
                "status": "error", 
                "message": error_msg
            }))
            return
        
        # Set data socket connection (this updates session state to ready if control is connected)
        session.set_data_connected(websocket)
        print(f"{bcolors.OKGREEN}Data client connected to session: {session_id}{bcolors.ENDC}")
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if self.extended_logging:
                        debug_print(f"Session {session_id}: Received audio chunk (size: {len(message)} bytes)", self.debug_logging)
                    elif self.log_incoming_chunks:
                        print(".", end='', flush=True)
                    
                    try:
                        metadata_length = int.from_bytes(message[:4], byteorder='little')
                        
                        if metadata_length < 0 or metadata_length > len(message) - 4:
                            raise ValueError("Invalid metadata length")
                        
                        metadata_json = message[4:4+metadata_length].decode('utf-8')
                        metadata = json.loads(metadata_json)
                        
                        validate_audio_metadata(metadata)
                        sample_rate = metadata['sampleRate']
                    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
                        if self.extended_logging:
                            debug_print(f"Session {session_id}: Invalid audio metadata: {e}", self.debug_logging)
                        continue

                    if 'server_sent_to_stt' in metadata:
                        stt_received_ns = time.time_ns()
                        metadata["stt_received"] = stt_received_ns
                        metadata["stt_received_formatted"] = format_timestamp_ns(stt_received_ns)
                        print(f"Session {session_id}: Server received audio chunk of length {len(message)} bytes, metadata: {metadata}")

                    if self.extended_logging:
                        debug_print(f"Session {session_id}: Processing audio chunk with sample rate {sample_rate}", self.debug_logging)
                    
                    chunk = message[4+metadata_length:]

                    if self.writechunks:
                        try:
                            if not session.wav_file:
                                session.wav_file = wave.open(f"{self.writechunks}_{session_id}.wav", 'wb')
                                session.wav_file.setnchannels(CHANNELS)
                                session.wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
                                session.wav_file.setframerate(sample_rate)
                                if self.extended_logging:
                                    debug_print(f"Session {session_id}: Created WAV file: {self.writechunks}_{session_id}.wav", self.debug_logging)

                            session.wav_file.writeframes(chunk)
                        except Exception as e:
                            print(f"{bcolors.WARNING}Session {session_id}: Error writing to WAV file: {e}{bcolors.ENDC}")
                            if session.wav_file:
                                try:
                                    session.wav_file.close()
                                except:
                                    pass
                                session.wav_file = None

                    # Check if session is ready for audio processing
                    if not session.is_session_ready():
                        if self.extended_logging:
                            debug_print(f"Session {session_id}: Skipping audio chunk - session not ready (state: {session.session_state})", self.debug_logging)
                        continue
                    
                    if not session.wait_for_recorder_ready():
                        if self.extended_logging:
                            debug_print(f"Session {session_id}: Skipping audio chunk - recorder not initialized", self.debug_logging)
                        continue
                    
                    # Update activity when receiving audio data
                    session.update_activity()
                    
                    if sample_rate != 16000:
                        resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
                        if self.extended_logging:
                            debug_print(f"Session {session_id}: Resampled chunk size: {len(resampled_chunk)} bytes", self.debug_logging)
                        session.feed_audio(resampled_chunk)
                    else:
                        session.feed_audio(chunk)
                else:
                    print(f"{bcolors.WARNING}Session {session_id}: Received non-binary message on data connection{bcolors.ENDC}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"{bcolors.WARNING}Data client disconnected - Session: {session_id} - {e}{bcolors.ENDC}")
        finally:
            if session and session.recorder:
                session.recorder.clear_audio_queue()
            
            # Clean up session when data connection closes
            if session_id and session_id in self.sessions:
                self.sessions[session_id].cleanup()
                del self.sessions[session_id]
                max_users = self.args.max_concurrent_users
                print(f"{bcolors.WARNING}Data connection closed, session {session_id} cleaned up ({len(self.sessions)}/{max_users} users){bcolors.ENDC}")
    
    def _get_recorder_config(self):
        """Get recorder configuration based on args"""
        return {
            'model': self.args.model,
            'download_root': self.args.root,
            'realtime_model_type': self.args.rt_model,
            'language': self.args.lang,
            'batch_size': self.args.batch,
            'init_realtime_after_seconds': self.args.init_realtime_after_seconds,
            'realtime_batch_size': self.args.realtime_batch_size,
            'initial_prompt_realtime': self.args.initial_prompt_realtime,
            'input_device_index': self.args.input_device,
            'silero_sensitivity': self.args.silero_sensitivity,
            'silero_use_onnx': self.args.silero_use_onnx,
            'webrtc_sensitivity': self.args.webrtc_sensitivity,
            'post_speech_silence_duration': self.args.unknown_sentence_detection_pause,
            'min_length_of_recording': self.args.min_length_of_recording,
            'min_gap_between_recordings': self.args.min_gap_between_recordings,
            'enable_realtime_transcription': self.args.enable_realtime_transcription,
            'realtime_processing_pause': self.args.realtime_processing_pause,
            'silero_deactivity_detection': self.args.silero_deactivity_detection,
            'early_transcription_on_silence': self.args.early_transcription_on_silence,
            'beam_size': self.args.beam_size,
            'beam_size_realtime': self.args.beam_size_realtime,
            'initial_prompt': self.args.initial_prompt,
            'wake_words': self.args.wake_words,
            'wake_words_sensitivity': self.args.wake_words_sensitivity,
            'wake_word_timeout': self.args.wake_word_timeout,
            'wake_word_activation_delay': self.args.wake_word_activation_delay,
            'wakeword_backend': self.args.wakeword_backend,
            'openwakeword_model_paths': self.args.openwakeword_model_paths,
            'openwakeword_inference_framework': self.args.openwakeword_inference_framework,
            'wake_word_buffer_duration': self.args.wake_word_buffer_duration,
            'use_main_model_for_realtime': self.args.use_main_model_for_realtime,
            'spinner': False,
            'use_microphone': False,
            'no_log_file': True,
            'use_extended_logging': self.args.use_extended_logging,
            'level': self.loglevel,
            'compute_type': self.args.compute_type,
            'gpu_device_index': self.args.gpu_device_index,
            'device': self.args.device,
            'handle_buffer_overflow': self.args.handle_buffer_overflow,
            'suppress_tokens': self.args.suppress_tokens,
            'allowed_latency_limit': self.args.allowed_latency_limit,
            'faster_whisper_vad_filter': self.args.faster_whisper_vad_filter,
        }
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.loop = asyncio.get_event_loop()
        
        try:
            control_server = await websockets.serve(self.control_handler, '0.0.0.0', self.args.control)
            data_server = await websockets.serve(self.data_handler, '0.0.0.0', self.args.data)
            print(f"{bcolors.OKGREEN}Control server started on {bcolors.OKBLUE}ws://0.0.0.0:{self.args.control}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}Data server started on {bcolors.OKBLUE}ws://0.0.0.0:{self.args.data}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}Session timeout: {self.args.session_timeout}s{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}Server started. Press Ctrl+C to stop the server.{bcolors.ENDC}")
            
            # Start timeout checker
            self.timeout_task = asyncio.create_task(self._check_session_timeouts())
            
            await asyncio.gather(control_server.wait_closed(), data_server.wait_closed(), self.timeout_task)
            
        except OSError as e:
            print(f"{bcolors.FAIL}Error: Could not start server on specified ports. It's possible another instance of the server is already running, or the ports are being used by another application.{bcolors.ENDC}")
        except KeyboardInterrupt:
            print(f"{bcolors.WARNING}Server interrupted by user, shutting down...{bcolors.ENDC}")
        finally:
            await self.shutdown()
            print(f"{bcolors.OKGREEN}Server shutdown complete.{bcolors.ENDC}")
    
    async def shutdown(self):
        """Shutdown all sessions and cleanup resources"""
        print(f"{bcolors.WARNING}Shutting down {len(self.sessions)} active sessions...{bcolors.ENDC}")
        
        # Gracefully stop timeout checker first
        await self.stop_timeout_checker()
        
        for session_id, session in list(self.sessions.items()):
            session.cleanup()
        
        self.sessions.clear()
        
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"{bcolors.OKGREEN}All tasks cancelled, closing event loop now.{bcolors.ENDC}")

def validate_audio_metadata(metadata):
    """Validate audio metadata structure to prevent injection attacks"""
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    
    # Check required fields
    if 'sampleRate' not in metadata:
        raise ValueError("sampleRate field is required in metadata")
    
    sample_rate = metadata.get('sampleRate')
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sampleRate must be a positive integer")
    
    # Validate sample rate range (reasonable audio sample rates)
    if sample_rate < 8000 or sample_rate > 192000:
        raise ValueError("sampleRate must be between 8000 and 192000 Hz")
    
    # Validate optional fields if present
    if 'server_sent_to_stt' in metadata:
        if not isinstance(metadata['server_sent_to_stt'], (int, float)):
            raise ValueError("server_sent_to_stt must be a number")
    
    # Prevent excessive metadata size
    if len(str(metadata)) > 1000:
        raise ValueError("Metadata too large")
    
    return True

# WAV file cleanup is now handled per-session in ClientSession.cleanup()

# Register cleanup function for signal handlers
def signal_handler(signum, frame):
    exit(0)

# Handle common termination signals
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGINT'):
    signal.signal(signal.SIGINT, signal_handler)

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

def debug_print(message, debug_logging=False):
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

# Callback functions moved to ClientSession class


# Old callback functions removed - now handled in ClientSession

# Define the server's arguments
def parse_arguments():

    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server with various configuration options.')

    parser.add_argument('-m', '--model', type=str, default='small.en',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='base.en',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is base.')
    
    parser.add_argument('-l', '--lang', '--language', type=str, default='en',
                help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')

    parser.add_argument('-i', '--input-device', '--input-device-index', type=int, default=1,
                    help='Index of the audio input device to use. Use this option to specify a particular microphone or audio input device based on your system. Default is 1.')

    parser.add_argument('-c', '--control', '--control_port', type=int, default=8011,
                        help='The port number used for the control WebSocket connection. Control connections are used to send and receive commands to the server. Default is port 8011.')

    parser.add_argument('-d', '--data', '--data_port', type=int, default=8012,
                        help='The port number used for the data WebSocket connection. Data connections are used to send audio data and receive transcription updates in real time. Default is port 8012.')

    parser.add_argument('--host', '--bind-address', type=str, default='localhost',
                        help='Host address to bind the server to. Use "0.0.0.0" to accept connections from any device on the network, or "localhost" to only accept local connections. Default is localhost.')

    parser.add_argument('-w', '--wake_words', type=str, default="",
                        help='Specify the wake word(s) that will trigger the server to start listening. For example, setting this to "Jarvis" will make the system start transcribing when it detects the wake word "Jarvis". Default is "Jarvis".')

    parser.add_argument('-D', '--debug', action='store_true', help='Enable debug logging for detailed server operations')

    parser.add_argument('--debug_websockets', action='store_true', help='Enable debug logging for detailed server websocket operations')

    parser.add_argument('-W', '--write', metavar='FILE', help='Save received audio to a WAV file')
    
    parser.add_argument('-b', '--batch', '--batch_size', type=int, default=16, help='Batch size for inference. This parameter controls the number of audio chunks processed in parallel during transcription. Default is 16.')

    parser.add_argument('--root', '--download_root', type=str, default=None, help='Specifies the root path where the Whisper models are downloaded to. Default is None.')

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
    
    parser.add_argument('--max_concurrent_users', type=int, default=3,
                        help='Maximum number of concurrent users allowed on the server. Default is 3.')
    
    parser.add_argument('--session_timeout', type=float, default=60.0,
                        help='Session inactivity timeout in seconds. Sessions will be automatically cleaned up after this period of inactivity. Default is 60.0 seconds.')

    parser.add_argument('--suppress_tokens', type=int, default=[-1], nargs='*', help='Suppress tokens during transcription. Default is [-1].')

    parser.add_argument('--allowed_latency_limit', type=int, default=100,
                        help='Maximal amount of chunks that can be unprocessed in queue before discarding chunks.. Default is 100.')

    parser.add_argument('--faster_whisper_vad_filter', action='store_true',
                        help='Enable VAD filter for Faster Whisper. Default is False.')

    parser.add_argument('--logchunks', action='store_true', help='Enable logging of incoming audio chunks (periods)')

    # Parse arguments
    args = parser.parse_args()

    # Global variables are now moved to WebSocketServer class


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

# Old _recorder_thread function removed - now handled in ClientSession

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

# Old handler functions removed - now handled in WebSocketServer

async def main_async():
    args = parse_arguments()
    print(f"{bcolors.OKGREEN}Initializing RealtimeSTT server with parameters:{bcolors.ENDC}")
    
    server = WebSocketServer(args)
    await server.start_server()

# Shutdown procedure now handled in WebSocketServer.shutdown()

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        # Capture any final KeyboardInterrupt to prevent it from showing up in logs
        print(f"{bcolors.WARNING}Server interrupted by user.{bcolors.ENDC}")
        exit(0)

if __name__ == '__main__':
    main()