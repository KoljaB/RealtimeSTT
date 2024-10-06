import os
import sys
import pyaudio
import numpy as np
from scipy import signal
import logging
os.environ['ALSA_LOG_LEVEL'] = 'none'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Default fallback rate
input_device_index = None
audio_interface = None
stream = None
device_sample_rate = None
chunk_size = CHUNK

def get_highest_sample_rate(audio_interface, device_index):
    """Get the highest supported sample rate for the specified device."""
    try:
        device_info = audio_interface.get_device_info_by_index(device_index)
        max_rate = int(device_info['defaultSampleRate'])

        if 'supportedSampleRates' in device_info:
            supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
            if supported_rates:
                max_rate = max(supported_rates)

        return max_rate
    except Exception as e:
        logging.warning(f"Failed to get highest sample rate: {e}")
        return 48000  # Fallback to a common high sample rate

def initialize_audio_stream(audio_interface, device_index, sample_rate, chunk_size):
    """Initialize the audio stream with error handling."""
    try:
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            input_device_index=device_index,
        )
        return stream
    except Exception as e:
        logging.error(f"Error initializing audio stream: {e}")
        raise

def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
    """Preprocess audio chunk similar to feed_audio method."""
    if isinstance(chunk, np.ndarray):
        if chunk.ndim == 2:  # Stereo to mono conversion
            chunk = np.mean(chunk, axis=1)

        # Resample if needed
        if original_sample_rate != target_sample_rate:
            num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
            chunk = signal.resample(chunk, num_samples)

        chunk = chunk.astype(np.int16)
    else:
        chunk = np.frombuffer(chunk, dtype=np.int16)

        if original_sample_rate != target_sample_rate:
            num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
            chunk = signal.resample(chunk, num_samples)
            chunk = chunk.astype(np.int16)

    return chunk.tobytes()

def setup_audio():
    global audio_interface, stream, device_sample_rate, input_device_index
    try:
        audio_interface = pyaudio.PyAudio()
        if input_device_index is None:
            try:
                default_device = audio_interface.get_default_input_device_info()
                input_device_index = default_device['index']
            except OSError as e:
                input_device_index = None

        sample_rates_to_try = [16000]  # Try 16000 Hz first
        if input_device_index is not None:
            highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
            if highest_rate != 16000:
                sample_rates_to_try.append(highest_rate)
        else:
            sample_rates_to_try.append(48000)  # Fallback sample rate

        for rate in sample_rates_to_try:
            try:
                device_sample_rate = rate
                stream = initialize_audio_stream(audio_interface, input_device_index, device_sample_rate, chunk_size)
                if stream is not None:
                    logging.debug(f"Audio recording initialized successfully at {device_sample_rate} Hz, reading {chunk_size} frames at a time")
                    return True
            except Exception as e:
                logging.warning(f"Failed to initialize audio stream at {device_sample_rate} Hz: {e}")
                continue

        raise Exception("Failed to initialize audio stream with all sample rates.")
    except Exception as e:
        logging.exception(f"Error initializing audio recording: {e}")
        if audio_interface:
            audio_interface.terminate()
        return False

from .install_packages import check_and_install_packages

check_and_install_packages([
    {
        'module_name': 'websocket',                    # Import module
        'install_name': 'websocket-client',            # Package name for pip install (websocket-client is the correct package for websocket)
    },
    {
        'module_name': 'pyaudio',                      # Import module
        'install_name': 'pyaudio',                     # Package name for pip install
    },
    {
        'module_name': 'colorama',                     # Import module
        'attribute': 'init',                           # Attribute to check (init method from colorama)
        'install_name': 'colorama',                    # Package name for pip install
        'version': '',                                 # Optional version constraint
    },
])

import websocket
import pyaudio
from colorama import init, Fore, Style

import argparse
import json
import threading
import time
import struct
import socket
import subprocess
import shutil
from urllib.parse import urlparse
from queue import Queue

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEFAULT_SERVER_URL = "ws://localhost:8011"

class STTWebSocketClient:
    def __init__(self, server_url, debug=False, file_output=None, norealtime=False):
        self.server_url = server_url
        self.ws = None
        self.is_running = False
        self.debug = debug
        self.file_output = file_output
        self.last_text = ""
        self.pbar = None
        self.console_width = shutil.get_terminal_size().columns
        self.recording_indicator = "🔴"
        self.norealtime = norealtime
        self.connection_established = threading.Event()
        self.message_queue = Queue()

    def debug_print(self, message):
        if self.debug:
            print(message, file=sys.stderr)

    def connect(self):
        if not self.ensure_server_running():
            self.debug_print("Cannot start STT server. Exiting.")
            return False

        websocket.enableTrace(self.debug)
        try:
            
            self.ws = websocket.WebSocketApp(self.server_url,
                                             on_message=self.on_message,
                                             on_error=self.on_error,
                                             on_close=self.on_close,
                                             on_open=self.on_open)
            
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()

            # Wait for the connection to be established
            if not self.connection_established.wait(timeout=10):
                self.debug_print("Timeout while connecting to the server.")
                return False
            
            self.debug_print("WebSocket connection established successfully.")
            return True
        except Exception as e:
            self.debug_print(f"Error while connecting to the server: {e}")
            return False

    def on_open(self, ws):
        self.debug_print("WebSocket connection opened.")
        self.is_running = True
        self.connection_established.set()
        self.start_recording()

    def on_error(self, ws, error):
        self.debug_print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.debug_print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_running = False

    def is_server_running(self):
        parsed_url = urlparse(self.server_url)
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

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data['type'] == 'realtime':
                if data['text'] != self.last_text:
                    self.last_text = data['text']
                    if not self.norealtime:
                        self.update_progress_bar(self.last_text) 
            elif data['type'] == 'fullSentence':
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
                
        except json.JSONDecodeError:
            self.debug_print(f"\nReceived non-JSON message: {message}")

    def show_initial_indicator(self):
        if self.norealtime:
            return

        initial_text = f"{self.recording_indicator}\b\b"
        sys.stderr.write(initial_text)
        sys.stderr.flush()

    def update_progress_bar(self, text):
        # Reserve some space for the progress bar decorations
        available_width = self.console_width - 5
        
        # Clear the current line
        sys.stderr.write('\r\033[K')  # Move to the beginning of the line and clear it

        # Get the last 'available_width' characters, but don't cut words
        words = text.split()
        last_chars = ""
        for word in reversed(words):
            if len(last_chars) + len(word) + 1 > available_width:
                break
            last_chars = word + " " + last_chars

        last_chars = last_chars.strip()

        # Color the text yellow and add recording indicator
        colored_text = f"{Fore.YELLOW}{last_chars}{Style.RESET_ALL}{self.recording_indicator}\b\b"

        sys.stderr.write(colored_text)
        sys.stderr.flush()

    def finish_progress_bar(self):
        # Clear the current line
        sys.stderr.write('\r\033[K')
        sys.stderr.flush()

    def stop(self):
        self.finish_progress_bar()
        self.is_running = False
        if self.ws:
            self.ws.close()
        #if hasattr(self, 'ws_thread'):
        #    self.ws_thread.join(timeout=2)

    def start_recording(self):
        threading.Thread(target=self.record_and_send_audio).start()

    def record_and_send_audio(self):
        if not setup_audio():
            raise Exception("Failed to set up audio recording.")

        self.debug_print("Recording and sending audio...")
        self.show_initial_indicator()

        while self.is_running:
            try:
                audio_data = stream.read(CHUNK)

                # Prepare metadata
                metadata = {
                    "sampleRate": device_sample_rate
                }
                metadata_json = json.dumps(metadata)
                metadata_length = len(metadata_json)

                # Construct the message
                message = struct.pack('<I', metadata_length) + metadata_json.encode('utf-8') + audio_data

                self.ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                self.debug_print(f"Error sending audio data: {e}")
                break

        self.debug_print("Stopped recording.")
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()


def main():
    parser = argparse.ArgumentParser(description="STT Client")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="STT WebSocket server URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-nort", "--norealtime", action="store_true", help="Disable real-time output")    
    args = parser.parse_args()

    # Check if output is being redirected
    if not os.isatty(sys.stdout.fileno()):
        file_output = sys.stdout
    else:
        file_output = None
    
    client = STTWebSocketClient(args.server, args.debug, file_output, args.norealtime)
  
    def signal_handler(sig, frame):
        # print("\nInterrupted by user, shutting down...")
        client.stop()
        sys.exit(0)

    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if client.connect():
            # print("Connection established. Recording... (Press Ctrl+C to stop)", file=sys.stderr)
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
