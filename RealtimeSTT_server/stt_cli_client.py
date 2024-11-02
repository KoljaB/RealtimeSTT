"""
This is a command-line client for the Speech-to-Text (STT) server.
It records audio from the default input device and sends it to the server for speech recognition.
It can also process commands to set parameters, get parameter values, or call methods on the server.

Usage:
    stt [--control-url CONTROL_URL] [--data-url DATA_URL] [--debug] [--norealtime] [--set-param PARAM VALUE] [--call-method METHOD [ARGS ...]] [--get-param PARAM]

Options:
    --control-url CONTROL_URL       STT Control WebSocket URL
    --data-url DATA_URL             STT Data WebSocket URL
    --debug                         Enable debug mode
    --norealtime                    Disable real-time output
    --set-param PARAM VALUE         Set a recorder parameter. Can be used multiple times.
    --call-method METHOD [ARGS ...] Call a recorder method with optional arguments.
    --get-param PARAM               Get the value of a recorder parameter. Can be used multiple times.
"""

from urllib.parse import urlparse
from scipy import signal
from queue import Queue
import numpy as np
import subprocess
import threading
import websocket
import argparse
import pyaudio
import struct
import socket
import shutil
import queue 
import json
import time
import wave
import sys
import os

os.environ['ALSA_LOG_LEVEL'] = 'none'

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEFAULT_CONTROL_URL = "ws://127.0.0.1:8011"
DEFAULT_DATA_URL = "ws://127.0.0.1:8012"

# Initialize colorama
from colorama import init, Fore, Style
init()

# Stop websocket from spamming the log
websocket.enableTrace(False)

class STTWebSocketClient:
    def __init__(self, control_url, data_url, debug=False, file_output=None, norealtime=False, writechunks=None):
        self.control_url = control_url
        self.data_url = data_url
        self.control_ws = None
        self.data_ws_app = None
        self.data_ws_connected = None  # WebSocket object that will be used for sending
        self.is_running = True
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
        self.chunks_sent = 0
        self.last_chunk_time = time.time()
        self.writechunks = writechunks  # Add this to store the file name for writing audio chunks

        self.debug_print("Initializing STT WebSocket Client")
        self.debug_print(f"Control URL: {control_url}")
        self.debug_print(f"Data URL: {data_url}")
        self.debug_print(f"File Output: {file_output}")
        self.debug_print(f"No Realtime: {norealtime}")
        self.debug_print(f"Write Chunks: {writechunks}")

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
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            thread_name = threading.current_thread().name
            print(f"{Fore.CYAN}[DEBUG][{timestamp}][{thread_name}] {message}{Style.RESET_ALL}", file=sys.stderr)

    def connect(self):
        if not self.ensure_server_running():
            self.debug_print("Cannot start STT server. Exiting.")
            return False

        try:
            self.debug_print("Attempting to establish WebSocket connections...")

            # Connect to control WebSocket
            self.debug_print(f"Connecting to control WebSocket at {self.control_url}")
            self.control_ws = websocket.WebSocketApp(self.control_url,
                                                     on_message=self.on_control_message,
                                                     on_error=self.on_error,
                                                     on_close=self.on_close,
                                                     on_open=self.on_control_open)

            self.control_ws_thread = threading.Thread(target=self.control_ws.run_forever)
            self.control_ws_thread.daemon = False
            self.debug_print("Starting control WebSocket thread")
            self.control_ws_thread.start()

            # Connect to data WebSocket
            self.debug_print(f"Connecting to data WebSocket at {self.data_url}")
            self.data_ws_app = websocket.WebSocketApp(self.data_url,
                                                      on_message=self.on_data_message,
                                                      on_error=self.on_error,
                                                      on_close=self.on_close,
                                                      on_open=self.on_data_open)

            self.data_ws_thread = threading.Thread(target=self.data_ws_app.run_forever)
            self.data_ws_thread.daemon = False
            self.debug_print("Starting data WebSocket thread")
            self.data_ws_thread.start()

            self.debug_print("Waiting for connections to be established...")
            if not self.connection_established.wait(timeout=10):
                self.debug_print("Timeout while connecting to the server.")
                return False

            self.debug_print("WebSocket connections established successfully.")
            return True
        except Exception as e:
            self.debug_print(f"Error while connecting to the server: {str(e)}")
            return False


    def on_control_open(self, ws):
        self.debug_print("Control WebSocket connection opened successfully")
        self.connection_established.set()
        self.start_command_processor()

    def on_data_open(self, ws):
        self.debug_print("Data WebSocket connection opened successfully")
        self.data_ws_connected = ws
        self.start_recording()

    def on_error(self, ws, error):
        self.debug_print(f"WebSocket error occurred: {str(error)}")
        self.debug_print(f"WebSocket object: {ws}")
        self.debug_print(f"Error type: {type(error)}")

    def on_close(self, ws, close_status_code, close_msg):
        self.debug_print(f"WebSocket connection closed")
        self.debug_print(f"Close status code: {close_status_code}")
        self.debug_print(f"Close message: {close_msg}")
        self.debug_print(f"WebSocket object: {ws}")
        self.is_running = False
        self.stop_event.set()

    def is_server_running(self):
        parsed_url = urlparse(self.control_url)
        host = parsed_url.hostname
        port = parsed_url.port or 80
        self.debug_print(f"Checking if server is running at {host}:{port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex((host, port)) == 0
            self.debug_print(f"Server status check result: {'running' if result else 'not running'}")
            return result

    def ask_to_start_server(self):
        response = input("Would you like to start the STT server now? (y/n): ").strip().lower()
        return response == 'y' or response == 'yes'

    def start_server(self):
        if os.name == 'nt':  # Windows
            subprocess.Popen('start /min cmd /c stt-server', shell=True)
        else:  # Unix-like systems
            terminal_emulators = [
                'gnome-terminal',
                'x-terminal-emulator',
                'konsole',
                'xfce4-terminal',
                'lxterminal',
                'xterm',
                'mate-terminal',
                'terminator',
                'tilix',
                'alacritty',
                'urxvt',
                'eterm',
                'rxvt',
                'kitty',
                'hyper'
            ]

            terminal = None
            for term in terminal_emulators:
                if shutil.which(term):
                    terminal = term
                    break

            if terminal:
                terminal_exec_options = {
                    'x-terminal-emulator': ['--'],
                    'gnome-terminal': ['--'],
                    'mate-terminal': ['--'],
                    'terminator': ['--'],
                    'tilix': ['--'],
                    'konsole': ['-e'],
                    'xfce4-terminal': ['-e'],
                    'lxterminal': ['-e'],
                    'alacritty': ['-e'],
                    'xterm': ['-e'],
                    'rxvt': ['-e'],
                    'urxvt': ['-e'],
                    'eterm': ['-e'],
                    'kitty': [],
                    'hyper': ['--command']
                }

                exec_option = terminal_exec_options.get(terminal, None)
                if exec_option is not None:
                    subprocess.Popen([terminal] + exec_option + ['stt-server'], start_new_session=True)
                    print(f"STT server started in a new terminal window using {terminal}.", file=sys.stderr)
                else:
                    print(f"Unsupported terminal emulator '{terminal}'. Please start the STT server manually.", file=sys.stderr)
            else:
                print("No supported terminal emulator found. Please start the STT server manually.", file=sys.stderr)

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

    def on_control_message(self, ws, message):
        try:
            self.debug_print(f"Received control message: {message}")
            data = json.loads(message)
            if 'status' in data:
                self.debug_print(f"Message status: {data['status']}")
                if data['status'] == 'success':
                    if 'parameter' in data and 'value' in data:
                        self.debug_print(f"Parameter update: {data['parameter']} = {data['value']}")
                        print(f"Parameter {data['parameter']} = {data['value']}")
                elif data['status'] == 'error':
                    self.debug_print(f"Server error received: {data.get('message', '')}")
                    print(f"Server Error: {data.get('message', '')}")
            else:
                self.debug_print(f"Unknown control message format: {data}")
        except json.JSONDecodeError:
            self.debug_print(f"Failed to decode JSON control message: {message}")
        except Exception as e:
            self.debug_print(f"Error processing control message: {str(e)}")

    def on_data_message(self, ws, message):
        try:
            self.debug_print(f"Received data message: {message}")
            data = json.loads(message)
            message_type = data.get('type')
            self.debug_print(f"Message type: {message_type}")

            if message_type == 'realtime':
                if data['text'] != self.last_text:
                    self.debug_print(f"New realtime text received: {data['text']}")
                    self.last_text = data['text']
                    if not self.norealtime:
                        self.update_progress_bar(self.last_text)
            elif message_type == 'fullSentence':
                self.debug_print(f"Full sentence received: {data['text']}")
                if self.file_output:
                    self.debug_print("Writing to file output")
                    sys.stderr.write('\r\033[K')
                    sys.stderr.write(data['text'])
                    sys.stderr.write('\n')
                    sys.stderr.flush()
                    print(data['text'], file=self.file_output)
                    self.file_output.flush()
                else:
                    self.finish_progress_bar()
                    print(f"{data['text']}")
                self.stop()
            elif message_type in {
                'vad_detect_start',
                'vad_detect_stop',
                'recording_start',
                'recording_stop',
                'wakeword_detected',
                'wakeword_detection_start',
                'wakeword_detection_end',
                'transcription_start'}:
                pass  # Known message types, no action needed
            else:
                self.debug_print(f"Other message type received: {message_type}")

        except json.JSONDecodeError:
            self.debug_print(f"Failed to decode JSON data message: {message}")
        except Exception as e:
            self.debug_print(f"Error processing data message: {str(e)}")

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
        self.debug_print("Stopping client and cleaning up resources.")
        if self.control_ws:
            self.control_ws.close()
        if self.data_ws_connected:
            self.data_ws_connected.close()

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
                self.debug_print("Failed to set up audio recording")
                raise Exception("Failed to set up audio recording.")

            # Initialize WAV file writer if writechunks is provided
            if self.writechunks:
                self.wav_file = wave.open(self.writechunks, 'wb')
                self.wav_file.setnchannels(CHANNELS)
                self.wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
                self.wav_file.setframerate(self.device_sample_rate)  # Use self.device_sample_rate

            self.debug_print("Starting audio recording and transmission")
            self.show_initial_indicator()

            while self.is_running:
                try:
                    audio_data = self.stream.read(CHUNK)
                    self.chunks_sent += 1
                    current_time = time.time()
                    elapsed = current_time - self.last_chunk_time

                    # Write to WAV file if enabled
                    if self.writechunks:
                        self.wav_file.writeframes(audio_data)

                    if self.chunks_sent % 100 == 0:  # Log every 100 chunks
                        self.debug_print(f"Sent {self.chunks_sent} chunks. Last chunk took {elapsed:.3f}s")

                    metadata = {"sampleRate": self.device_sample_rate}
                    metadata_json = json.dumps(metadata)
                    metadata_length = len(metadata_json)
                    message = struct.pack('<I', metadata_length) + metadata_json.encode('utf-8') + audio_data

                    self.debug_print(f"Sending audio chunk {self.chunks_sent}: {len(audio_data)} bytes, metadata: {metadata_json}")
                    self.data_ws_connected.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
                    self.last_chunk_time = current_time

                except Exception as e:
                    self.debug_print(f"Error sending audio data: {str(e)}")
                    break

        except Exception as e:
            self.debug_print(f"Error in record_and_send_audio: {str(e)}")
        finally:
            self.cleanup_audio()

    def setup_audio(self):
        try:
            self.debug_print("Initializing PyAudio interface")
            self.audio_interface = pyaudio.PyAudio()
            self.input_device_index = None

            try:
                default_device = self.audio_interface.get_default_input_device_info()
                self.input_device_index = default_device['index']
                self.debug_print(f"Default input device found: {default_device}")
            except OSError as e:
                self.debug_print(f"No default input device found: {str(e)}")
                return False

            self.device_sample_rate = 16000
            self.debug_print(f"Attempting to open audio stream with sample rate {self.device_sample_rate} Hz")

            try:
                self.stream = self.audio_interface.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=self.device_sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=self.input_device_index,
                )
                self.debug_print(f"Audio stream initialized successfully")
                self.debug_print(f"Audio parameters: rate={self.device_sample_rate}, channels={CHANNELS}, format={FORMAT}, chunk={CHUNK}")
                return True
            except Exception as e:
                self.debug_print(f"Failed to initialize audio stream: {str(e)}")
                return False

        except Exception as e:
            self.debug_print(f"Error in setup_audio: {str(e)}")
            if self.audio_interface:
                self.audio_interface.terminate()
            return False

    def cleanup_audio(self):
        self.debug_print("Cleaning up audio resources")
        try:
            if self.stream:
                self.debug_print("Stopping and closing audio stream")
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.audio_interface:
                self.debug_print("Terminating PyAudio interface")
                self.audio_interface.terminate()
                self.audio_interface = None
            if self.writechunks and self.wav_file:
                self.debug_print("Closing WAV file")
                self.wav_file.close()
        except Exception as e:
            self.debug_print(f"Error during audio cleanup: {str(e)}")

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
        self.debug_print("Starting command processor thread")
        while not self.stop_event.is_set():
            try:
                command = self.commands.get(timeout=0.1)
                self.debug_print(f"Processing command: {command}")
                if command['type'] == 'set_parameter':
                    self.debug_print(f"Setting parameter: {command['parameter']} = {command['value']}")
                    self.set_parameter(command['parameter'], command['value'])
                elif command['type'] == 'get_parameter':
                    self.debug_print(f"Getting parameter: {command['parameter']}")
                    self.get_parameter(command['parameter'])
                elif command['type'] == 'call_method':
                    self.debug_print(f"Calling method: {command['method']} with args: {command.get('args')} and kwargs: {command.get('kwargs')}")
                    self.call_method(command['method'], command.get('args'), command.get('kwargs'))
            except queue.Empty:
                continue
            except Exception as e:
                self.debug_print(f"Error in command processor: {str(e)}")

        self.debug_print("Command processor thread stopping")

    def add_command(self, command):
        self.commands.put(command)

def main():
    parser = argparse.ArgumentParser(description="STT Client")
    parser.add_argument("--control-url", default=DEFAULT_CONTROL_URL, help="STT Control WebSocket URL")
    parser.add_argument("--data-url", default=DEFAULT_DATA_URL, help="STT Data WebSocket URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-nort", "--norealtime", action="store_true", help="Disable real-time output")
    parser.add_argument("--writechunks", metavar="FILE", help="Save recorded audio chunks to a WAV file")
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

    client = STTWebSocketClient(args.control_url, args.data_url, args.debug, file_output, args.norealtime, args.writechunks)

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

