"""
Speech-to-Text (STT) Client CLI for WebSocket Server Interaction

This command-line interface (CLI) allows interaction with the Speech-to-Text (STT) WebSocket server. It connects to the server via control and data WebSocket URLs to facilitate real-time speech transcription, control the server, and manage various parameters related to the STT process.

The client can be used to start recording audio, set or retrieve STT parameters, and interact with the server using commands. Additionally, the CLI can disable real-time updates or run in debug mode for detailed output.

### Features:
- Connects to STT WebSocket server for real-time transcription and control.
- Supports setting and retrieving parameters via the command line.
- Allows calling server methods (e.g., start/stop recording).
- Option to disable real-time updates during transcription.
- Debug mode available for verbose logging.

### Starting the Client:
You can start the client using the command `stt` and optionally pass configuration options or commands for interacting with the server.

```bash
stt [OPTIONS]
```

### Available Parameters:
- `--control-url` (default: "ws://localhost:8011"): The WebSocket URL for server control commands.
  
- `--data-url` (default: "ws://localhost:8012"): The WebSocket URL for sending audio data and receiving transcription updates.

- `--debug`: Enable debug mode, which prints detailed logs to stderr.

- `--nort` or `--norealtime`: Disable real-time output of transcription results.

- `--set-param PARAM VALUE`: Set a recorder parameter (e.g., silero_sensitivity, beam_size, etc.). This option can be used multiple times to set different parameters.

- `--get-param PARAM`: Retrieve the value of a specific recorder parameter. This option can be used multiple times to retrieve different parameters.

- `--call-method METHOD [ARGS]`: Call a method on the recorder with optional arguments. This option can be used multiple times for different methods.

### Example Usage:
1. **Start the client with default settings:**
   ```bash
   stt
   ```

2. **Set a recorder parameter (e.g., set Silero sensitivity to 0.1):**
   ```bash
   stt --set-param silero_sensitivity 0.1
   ```

3. **Retrieve the value of a recorder parameter (e.g., get the current Silero sensitivity):**
   ```bash
   stt --get-param silero_sensitivity
   ```

4. **Call a method on the recorder (e.g., start the microphone input):**
   ```bash
   stt --call-method set_microphone
   ```

5. **Run in debug mode:**
   ```bash
   stt --debug
   ```

### WebSocket Interface:
- **Control WebSocket**: Used for sending control commands like setting parameters or invoking methods.
- **Data WebSocket**: Used for sending audio data for real-time transcription and receiving transcription results.

The client can be used to send audio data to the server for transcription and to control the behavior of the server remotely.
"""

import os
import sys
import pyaudio
import numpy as np
from scipy import signal
import logging
import websocket
import argparse
import json
import threading
import time
import struct
import socket
import shutil
from urllib.parse import urlparse
import queue 
from queue import Queue

os.environ['ALSA_LOG_LEVEL'] = 'none'

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEFAULT_CONTROL_URL = "ws://localhost:8011"
DEFAULT_DATA_URL = "ws://localhost:8012"

# Initialize colorama
from colorama import init, Fore, Style
init()

# Stop websocket from spamming the log
websocket.enableTrace(False)

class STTWebSocketClient:
    def __init__(self, control_url, data_url, debug=False, file_output=None, norealtime=False):
        self.control_url = control_url
        self.data_url = data_url
        self.control_ws = None
        self.data_ws = None
        self.is_running = False
        self.debug = debug
        self.file_output = file_output
        self.last_text = ""
        self.console_width = shutil.get_terminal_size().columns
        self.recording_indicator = "🔴"
        self.norealtime = norealtime
        self.connection_established = threading.Event()
        self.message_queue = Queue()
        self.commands = Queue()
        self.stop_event = threading.Event()

        # Audio attributes
        self.audio_interface = None
        self.stream = None
        self.device_sample_rate = None
        self.input_device_index = None

        # Threads
        self.control_ws_thread = None
        self.data_ws_thread = None
        self.recording_thread = None

    def debug_print(self, message):
        if self.debug:
            print(message, file=sys.stderr)

    def connect(self):
        if not self.ensure_server_running():
            self.debug_print("Cannot start STT server. Exiting.")
            return False

        try:
            # Connect to control WebSocket
            self.control_ws = websocket.WebSocketApp(self.control_url,
                                                     on_message=self.on_control_message,
                                                     on_error=self.on_error,
                                                     on_close=self.on_close,
                                                     on_open=self.on_control_open)

            self.control_ws_thread = threading.Thread(target=self.control_ws.run_forever)
            self.control_ws_thread.daemon = False  # Set to False to ensure proper shutdown
            self.control_ws_thread.start()

            # Connect to data WebSocket
            self.data_ws = websocket.WebSocketApp(self.data_url,
                                                  on_message=self.on_data_message,
                                                  on_error=self.on_error,
                                                  on_close=self.on_close,
                                                  on_open=self.on_data_open)

            self.data_ws_thread = threading.Thread(target=self.data_ws.run_forever)
            self.data_ws_thread.daemon = False  # Set to False to ensure proper shutdown
            self.data_ws_thread.start()

            # Wait for the connections to be established
            if not self.connection_established.wait(timeout=10):
                self.debug_print("Timeout while connecting to the server.")
                return False

            self.debug_print("WebSocket connections established successfully.")
            return True
        except Exception as e:
            self.debug_print(f"Error while connecting to the server: {e}")
            return False

    def on_control_open(self, ws):
        self.debug_print("Control WebSocket connection opened.")
        self.connection_established.set()
        self.start_command_processor()

    def on_data_open(self, ws):
        self.debug_print("Data WebSocket connection opened.")
        self.is_running = True
        self.start_recording()

    def on_error(self, ws, error):
        self.debug_print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.debug_print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_running = False
        self.stop_event.set()

    def is_server_running(self):
        parsed_url = urlparse(self.control_url)
        host = parsed_url.hostname
        port = parsed_url.port or 80
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def ask_to_start_server(self):
        response = input("Would you like to start the STT server now? (y/n): ").strip().lower()
        return response == 'y' or response == 'yes'

    def start_server(self):
        if os.name == 'nt':  # Windows
            subprocess.Popen('start /min cmd /c stt-server', shell=True)
        else:  # Unix-like systems
            subprocess.Popen(['stt-server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print("STT server start command issued. Please wait a moment for it to initialize.", file=sys.stderr)

    def ensure_server_running(self):
        if not self.is_server_running():
            print("STT server is not running.", file=sys.stderr)
            if self.ask_to_start_server():
                self.start_server()
                print("Waiting for STT server to start...", file=sys.stderr)
                for _ in range(20):  # Wait up to 20 seconds
                    if self.is_server_running():
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

    # Handle control messages like set_parameter, get_parameter, etc.
    def on_control_message(self, ws, message):
        try:
            data = json.loads(message)
            # Handle server response with status
            if 'status' in data:
                if data['status'] == 'success':
                    # print(f"Server Response: {data.get('message', '')}")
                    if 'parameter' in data and 'value' in data:
                        print(f"Parameter {data['parameter']} = {data['value']}")
                elif data['status'] == 'error':
                    print(f"Server Error: {data.get('message', '')}")
            else:
                self.debug_print(f"Unknown control message format: {data}")
        except json.JSONDecodeError:
            self.debug_print(f"Received non-JSON control message: {message}")
        except Exception as e:
            self.debug_print(f"Error processing control message: {e}")

    # Handle real-time transcription and full sentence updates
    def on_data_message(self, ws, message):
        try:
            data = json.loads(message)
            # Handle real-time transcription updates
            if data.get('type') == 'realtime':
                if data['text'] != self.last_text:
                    self.last_text = data['text']
                    if not self.norealtime:
                        self.update_progress_bar(self.last_text)

            # Handle full sentences
            elif data.get('type') == 'fullSentence':
                if self.file_output:
                    sys.stderr.write('\r\033[K')
                    sys.stderr.write(data['text'])
                    sys.stderr.write('\n')
                    sys.stderr.flush()
                    print(data['text'], file=self.file_output)
                    self.file_output.flush()  # Ensure it's written immediately
                else:
                    self.finish_progress_bar()
                    print(f"{data['text']}")
                self.stop()

            else:
                self.debug_print(f"Unknown data message format: {data}")

        except json.JSONDecodeError:
            self.debug_print(f"Received non-JSON data message: {message}")
        except Exception as e:
            self.debug_print(f"Error processing data message: {e}")

    def show_initial_indicator(self):
        if self.norealtime:
            return
        initial_text = f"{self.recording_indicator}\b\b"
        sys.stderr.write(initial_text)
        sys.stderr.flush()

    def update_progress_bar(self, text):
        try:
            available_width = self.console_width - 5  # Adjust for progress bar decorations
            sys.stderr.write('\r\033[K')  # Clear the current line
            words = text.split()
            last_chars = ""
            for word in reversed(words):
                if len(last_chars) + len(word) + 1 > available_width:
                    break
                last_chars = word + " " + last_chars
            last_chars = last_chars.strip()
            colored_text = f"{Fore.YELLOW}{last_chars}{Style.RESET_ALL}{self.recording_indicator}\b\b"
            sys.stderr.write(colored_text)
            sys.stderr.flush()
        except Exception as e:
            self.debug_print(f"Error updating progress bar: {e}")

    def finish_progress_bar(self):
        try:
            sys.stderr.write('\r\033[K')
            sys.stderr.flush()
        except Exception as e:
            self.debug_print(f"Error finishing progress bar: {e}")

    def stop(self):
        self.finish_progress_bar()
        self.is_running = False
        self.stop_event.set()
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

    def start_recording(self):
        self.recording_thread = threading.Thread(target=self.record_and_send_audio)
        self.recording_thread.daemon = False  # Set to False to ensure proper shutdown
        self.recording_thread.start()

    def record_and_send_audio(self):
        try:
            if not self.setup_audio():
                raise Exception("Failed to set up audio recording.")

            self.debug_print("Recording and sending audio...")
            self.show_initial_indicator()

            while self.is_running:
                try:
                    audio_data = self.stream.read(CHUNK)
                    metadata = {"sampleRate": self.device_sample_rate}
                    metadata_json = json.dumps(metadata)
                    metadata_length = len(metadata_json)
                    message = struct.pack('<I', metadata_length) + metadata_json.encode('utf-8') + audio_data
                    self.data_ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    self.debug_print(f"Error sending audio data: {e}")
                    break  # Exit the recording loop

        except Exception as e:
            self.debug_print(f"Error in record_and_send_audio: {e}")
        finally:
            self.cleanup_audio()

    def setup_audio(self):
        try:
            self.audio_interface = pyaudio.PyAudio()
            self.input_device_index = None
            try:
                default_device = self.audio_interface.get_default_input_device_info()
                self.input_device_index = default_device['index']
            except OSError as e:
                self.debug_print(f"No default input device found: {e}")
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
                self.debug_print(f"Audio recording initialized successfully at {self.device_sample_rate} Hz")
                return True
            except Exception as e:
                self.debug_print(f"Failed to initialize audio stream at {self.device_sample_rate} Hz: {e}")
                return False

        except Exception as e:
            self.debug_print(f"Error initializing audio recording: {e}")
            if self.audio_interface:
                self.audio_interface.terminate()
            return False

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
            self.debug_print(f"Error cleaning up audio resources: {e}")

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

    def start_command_processor(self):
        self.command_thread = threading.Thread(target=self.command_processor)
        self.command_thread.daemon = False  # Ensure it is not a daemon thread
        self.command_thread.start()

    def command_processor(self):
        # print(f"Starting command processor")
        self.debug_print(f"Starting command processor")
        #while self.is_running and not self.stop_event.is_set():
        while not self.stop_event.is_set():
            try:
                command = self.commands.get(timeout=0.1)
                if command['type'] == 'set_parameter':
                    self.set_parameter(command['parameter'], command['value'])
                elif command['type'] == 'get_parameter':
                    self.get_parameter(command['parameter'])
                elif command['type'] == 'call_method':
                    self.call_method(command['method'], command.get('args'), command.get('kwargs'))
            except queue.Empty:  # Use queue.Empty instead of Queue.Empty
                continue  # Queue was empty, just loop again
            except Exception as e:
                self.debug_print(f"Error in command processor: {e}")
            # finally:
        #print(f"Leaving command processor")
        self.debug_print(f"Leaving command processor")


    def add_command(self, command):
        self.commands.put(command)

def main():
    parser = argparse.ArgumentParser(description="STT Client")
    parser.add_argument("--control-url", default=DEFAULT_CONTROL_URL, help="STT Control WebSocket URL")
    parser.add_argument("--data-url", default=DEFAULT_DATA_URL, help="STT Data WebSocket URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-nort", "--norealtime", action="store_true", help="Disable real-time output")
    parser.add_argument("--set-param", nargs=2, metavar=('PARAM', 'VALUE'), action='append',
                        help="Set a recorder parameter. Can be used multiple times.")
    parser.add_argument("--call-method", nargs='+', metavar='METHOD', action='append',
                        help="Call a recorder method with optional arguments.")
    parser.add_argument("--get-param", nargs=1, metavar='PARAM', action='append',
                        help="Get the value of a recorder parameter. Can be used multiple times.")
    args = parser.parse_args()

    # Check if output is being redirected
    if not os.isatty(sys.stdout.fileno()):
        file_output = sys.stdout
    else:
        file_output = None

    client = STTWebSocketClient(args.control_url, args.data_url, args.debug, file_output, args.norealtime)

    def signal_handler(sig, frame):
        client.stop()
        sys.exit(0)

    import signal
    signal.signal(signal.SIGINT, signal_handler)

    try:
        if client.connect():
            # Process command-line parameters
            if args.set_param:
                for param, value in args.set_param:
                    try:
                        # Attempt to parse the value to the appropriate type
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string if not a number

                    client.add_command({
                        'type': 'set_parameter',
                        'parameter': param,
                        'value': value
                    })

            if args.get_param:
                for param_list in args.get_param:
                    param = param_list[0]
                    client.add_command({
                        'type': 'get_parameter',
                        'parameter': param
                    })

            if args.call_method:
                for method_call in args.call_method:
                    method = method_call[0]
                    args_list = method_call[1:] if len(method_call) > 1 else []
                    client.add_command({
                        'type': 'call_method',
                        'method': method,
                        'args': args_list
                    })

            # If command-line parameters were used (like --get-param), wait for them to be processed
            if args.set_param or args.get_param or args.call_method:
                while not client.commands.empty():
                    time.sleep(0.1)

            # Start recording directly if no command-line params were provided
            while client.is_running:
                time.sleep(0.1)

        else:
            print("Failed to connect to the server.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.stop()

if __name__ == "__main__":
    main()
