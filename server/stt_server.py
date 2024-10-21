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
- `--model` (str, default: 'medium.en'): Path to the STT model or model size. Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model like `deepdml/faster-whisper-large-v3-turbo-ct2`.
  
- `--realtime_model_type` (str, default: 'tiny.en'): Model size for real-time transcription. Same options as `--model`.

- `--language` (str, default: 'en'): Language code for the STT model. Leave empty for auto-detection.

- `--input_device_index` (int, default: 1): Index of the audio input device to use.

- `--silero_sensitivity` (float, default: 0.05): Sensitivity for Silero Voice Activity Detection (VAD). Lower values are less sensitive.

- `--webrtc_sensitivity` (int, default: 3): Sensitivity for WebRTC VAD. Higher values are less sensitive.

- `--min_length_of_recording` (float, default: 1.1): Minimum duration (in seconds) for a valid recording. Prevents short recordings.

- `--min_gap_between_recordings` (float, default: 0): Minimum time (in seconds) between consecutive recordings.

- `--enable_realtime_transcription` (flag, default: True): Enable real-time transcription of audio.

- `--realtime_processing_pause` (float, default: 0.02): Time interval (in seconds) between processing audio chunks for real-time transcription. Lower values increase responsiveness.

- `--silero_deactivity_detection` (flag, default: True): Use Silero model for end-of-speech detection.

- `--early_transcription_on_silence` (float, default: 0.2): Start transcription after specified seconds of silence.

- `--beam_size` (int, default: 5): Beam size for the main transcription model.

- `--beam_size_realtime` (int, default: 3): Beam size for the real-time transcription model.

- `--initial_prompt` (str, default: 'Add periods only for complete sentences...'): Initial prompt for the transcription model to guide its output format and style.

- `--end_of_sentence_detection_pause` (float, default: 0.45): Duration of pause (in seconds) to consider as the end of a sentence.

- `--unknown_sentence_detection_pause` (float, default: 0.7): Duration of pause (in seconds) to consider as an unknown or incomplete sentence.

- `--mid_sentence_detection_pause` (float, default: 2.0): Duration of pause (in seconds) to consider as a mid-sentence break.

- `--control_port` (int, default: 8011): Port for the control WebSocket connection.

- `--data_port` (int, default: 8012): Port for the data WebSocket connection.

### WebSocket Interface:
The server supports two WebSocket connections:
1. **Control WebSocket**: Used to send and receive commands, such as setting parameters or calling recorder methods.
2. **Data WebSocket**: Used to send audio data for transcription and receive real-time transcription updates.

The server will broadcast real-time transcription updates to all connected clients on the data WebSocket.
"""


import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from .install_packages import check_and_install_packages

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

print("Starting server, please wait...")

import threading
import json
import websockets
from RealtimeSTT import AudioToTextRecorder
import numpy as np
from scipy.signal import resample

global_args = None
recorder = None
recorder_config = {}
recorder_ready = threading.Event()
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
    'silero_sensitivity',
    'wake_word_activation_delay',
    'post_speech_silence_duration',
    'listen_start',
    'recording_stop_time',
    'last_transcription_bytes',
    'last_transcription_bytes_b64',
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

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def text_detected(text, loop):
    global prev_text

    text = preprocess_text(text)

    sentence_end_marks = ['.', '!', '?', '。'] 
    if text.endswith("..."):
        recorder.post_speech_silence_duration = global_args.mid_sentence_detection_pause
    elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
        recorder.post_speech_silence_duration = global_args.end_of_sentence_detection_pause
    else:
        recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause

    prev_text = text

    # Put the message in the audio queue to be sent to clients
    message = json.dumps({
        'type': 'realtime',
        'text': text
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)
    print(f"\r{text}", flush=True, end='')

def on_recording_start(loop):
    # Send a message to the client indicating recording has started
    message = json.dumps({
        'type': 'recording_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_recording_stop(loop):
    # Send a message to the client indicating recording has stopped
    message = json.dumps({
        'type': 'recording_stop'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_vad_detect_start(loop):
    message = json.dumps({
        'type': 'vad_detect_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_start(loop):
    # Send a message to the client when wake word detection starts
    message = json.dumps({
        'type': 'wakeword_detection_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_end(loop):
    # Send a message to the client when wake word detection ends
    message = json.dumps({
        'type': 'wakeword_detection_end'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_transcription_start(loop):
    # Send a message to the client when transcription starts
    message = json.dumps({
        'type': 'transcription_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_realtime_transcription_update(text, loop):
    # Send real-time transcription updates to the client
    text = preprocess_text(text)
    message = json.dumps({
        'type': 'realtime_update',
        'text': text
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_recorded_chunk(chunk):
    # Process each recorded audio chunk (optional implementation)
    pass

# Define the server's arguments
def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server with various configuration options.')
    
    parser.add_argument('--model', type=str, default='medium.en',
                        help='Path to the STT model or model size. Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2 or any hugginface CTranslate2 stt model like deepdml/faster-whisper-large-v3-turbo-ct2. Default: medium.en')
    
    parser.add_argument('--realtime_model_type', type=str, default='tiny.en',
                        help='Model size for real-time transcription. Same options as --model. Used only if real-time transcription is enabled. Default: tiny.en')
    
    parser.add_argument('--language', type=str, default='en',
                        help='Language code for the STT model. Leave empty for auto-detection. Default: en')
    
    parser.add_argument('--input_device_index', type=int, default=1,
                        help='Index of the audio input device to use. Default: 1')
    
    parser.add_argument('--silero_sensitivity', type=float, default=0.05,
                        help='Sensitivity for Silero Voice Activity Detection (0 to 1). Lower values are less sensitive. Default: 0.05')
    
    parser.add_argument('--webrtc_sensitivity', type=int, default=3,
                        help='Sensitivity for WebRTC Voice Activity Detection (0 to 3). Higher values are less sensitive. Default: 3')
    
    parser.add_argument('--min_length_of_recording', type=float, default=1.1,
                        help='Minimum duration (in seconds) for a valid recording. Prevents excessively short recordings. Default: 1.1')
    
    parser.add_argument('--min_gap_between_recordings', type=float, default=0,
                        help='Minimum time (in seconds) between consecutive recordings. Prevents rapid successive recordings. Default: 0')
    
    parser.add_argument('--enable_realtime_transcription', action='store_true', default=True,
                        help='Enable continuous real-time transcription of audio. Default: True')
    
    parser.add_argument('--realtime_processing_pause', type=float, default=0.02,
                        help='Time interval (in seconds) between processing audio chunks for real-time transcription. Lower values increase responsiveness but may increase CPU load. Default: 0.02')
    
    parser.add_argument('--silero_deactivity_detection', action='store_true', default=True,
                        help='Use Silero model for end-of-speech detection. More robust against background noise but uses more GPU resources. Default: True')
    
    parser.add_argument('--early_transcription_on_silence', type=float, default=0.2,
                        help='Start transcription after specified seconds of silence. Should be lower than post_speech_silence_duration. Set to 0 to disable. Default: 0.2')
    
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for the main transcription model. Larger values may improve accuracy but increase processing time. Default: 5')
    
    parser.add_argument('--beam_size_realtime', type=int, default=3,
                        help='Beam size for the real-time transcription model. Smaller than main beam_size for faster processing. Default: 3')
    
    parser.add_argument('--initial_prompt', type=str, 
                        default='Add periods only for complete sentences. Use ellipsis (...) for unfinished thoughts or unclear endings. Examples: \n- Complete: "I went to the store."\n- Incomplete: "I think it was..."',
                        help='Initial prompt for the transcription model to guide its output format and style. Default provides instructions for sentence completion and ellipsis usage.')
    
    parser.add_argument('--end_of_sentence_detection_pause', type=float, default=0.45,
                        help='Duration of pause (in seconds) to consider as end of a sentence. Default: 0.45')
    
    parser.add_argument('--unknown_sentence_detection_pause', type=float, default=0.7,
                        help='Duration of pause (in seconds) to consider as an unknown or incomplete sentence. Default: 0.7')
    
    parser.add_argument('--mid_sentence_detection_pause', type=float, default=2.0,
                        help='Duration of pause (in seconds) to consider as a mid-sentence break. Default: 2.0')
    
    parser.add_argument('--control_port', type=int, default=8011,
                        help='Port for the control WebSocket connection. Default: 8011')
    
    parser.add_argument('--data_port', type=int, default=8012,
                        help='Port for the data WebSocket connection. Default: 8012')
    
    return parser.parse_args()

def _recorder_thread(loop):
    global recorder, prev_text, stop_recorder
    print(f"Initializing RealtimeSTT server with parameters {recorder_config}")
    recorder = AudioToTextRecorder(**recorder_config)
    print("RealtimeSTT initialized")
    recorder_ready.set()
    
    def process_text(full_sentence):
        full_sentence = preprocess_text(full_sentence)
        message = json.dumps({
            'type': 'fullSentence',
            'text': full_sentence
        })
        # Use the passed event loop here
        asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)
        print(f"\rSentence: {full_sentence}")

    try:
        while not stop_recorder:
            recorder.text(process_text)
    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")

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

async def control_handler(websocket, path):
    print("Control client connected")
    global recorder
    control_connections.add(websocket)
    try:
        async for message in websocket:
            if not recorder_ready.is_set():
                print("Recorder not ready")
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
                            print(f"Set recorder.{parameter} to {value}")
                            # Optionally send a response back to the client
                            await websocket.send(json.dumps({"status": "success", "message": f"Parameter {parameter} set to {value}"}))
                        else:
                            if not parameter in allowed_parameters:
                                print(f"Parameter {parameter} is not allowed (set_parameter)")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} is not allowed (set_parameter)"}))
                            else:
                                print(f"Parameter {parameter} does not exist (set_parameter)")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} does not exist (set_parameter)"}))
                    elif command == "get_parameter":
                        parameter = command_data.get("parameter")
                        if parameter in allowed_parameters and hasattr(recorder, parameter):
                            value = getattr(recorder, parameter)
                            await websocket.send(json.dumps({"status": "success", "parameter": parameter, "value": value}))
                        else:
                            if not parameter in allowed_parameters:
                                print(f"Parameter {parameter} is not allowed (get_parameter)")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} is not allowed (get_parameter)"}))
                            else:
                                print(f"Parameter {parameter} does not exist (get_parameter)")
                                await websocket.send(json.dumps({"status": "error", "message": f"Parameter {parameter} does not exist (get_parameter)"}))
                    elif command == "call_method":
                        method_name = command_data.get("method")
                        if method_name in allowed_methods:
                            method = getattr(recorder, method_name, None)
                            if method and callable(method):
                                args = command_data.get("args", [])
                                kwargs = command_data.get("kwargs", {})
                                method(*args, **kwargs)
                                print(f"Called method recorder.{method_name}")
                                await websocket.send(json.dumps({"status": "success", "message": f"Method {method_name} called"}))
                            else:
                                print(f"Recorder does not have method {method_name}")
                                await websocket.send(json.dumps({"status": "error", "message": f"Recorder does not have method {method_name}"}))
                        else:
                            print(f"Method {method_name} is not allowed")
                            await websocket.send(json.dumps({"status": "error", "message": f"Method {method_name} is not allowed"}))
                    else:
                        print(f"Unknown command: {command}")
                        await websocket.send(json.dumps({"status": "error", "message": f"Unknown command {command}"}))
                except json.JSONDecodeError:
                    print("Received invalid JSON command")
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON command"}))
            else:
                print("Received unknown message type on control connection")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Control client disconnected: {e}")
    finally:
        control_connections.remove(websocket)

async def data_handler(websocket, path):
    print("Data client connected")
    data_connections.add(websocket)
    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                # Handle binary message (audio data)
                metadata_length = int.from_bytes(message[:4], byteorder='little')
                metadata_json = message[4:4+metadata_length].decode('utf-8')
                metadata = json.loads(metadata_json)
                sample_rate = metadata['sampleRate']
                chunk = message[4+metadata_length:]
                resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
                recorder.feed_audio(resampled_chunk)
            else:
                print("Received non-binary message on data connection")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Data client disconnected: {e}")
    finally:
        data_connections.remove(websocket)
        recorder.clear_audio_queue()  # Ensure audio queue is cleared if client disconnects

async def broadcast_audio_messages():
    while True:
        message = await audio_queue.get()
        for conn in list(data_connections):
            try:
                # print(f"Sending message: {message}")
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
        'realtime_model_type': args.realtime_model_type,
        'language': args.language,
        'input_device_index': args.input_device_index,
        'silero_sensitivity': args.silero_sensitivity,
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

        'spinner': False,
        'use_microphone': False,

        'on_realtime_transcription_update': make_callback(loop, text_detected),
        'on_recording_start': make_callback(loop, on_recording_start),
        'on_recording_stop': make_callback(loop, on_recording_stop),
        'on_vad_detect_start': make_callback(loop, on_vad_detect_start),
        'on_wakeword_detection_start': make_callback(loop, on_wakeword_detection_start),
        'on_wakeword_detection_end': make_callback(loop, on_wakeword_detection_end),
        'on_transcription_start': make_callback(loop, on_transcription_start),
        'no_log_file': True,
    }

    control_server = await websockets.serve(control_handler, "localhost", args.control_port)
    data_server = await websockets.serve(data_handler, "localhost", args.data_port)
    print(f"Control server started on ws://localhost:{args.control_port}")
    print(f"Data server started on ws://localhost:{args.data_port}")

    # Task to broadcast audio messages
    broadcast_task = asyncio.create_task(broadcast_audio_messages())

    recorder_thread = threading.Thread(target=_recorder_thread, args=(loop,))
    recorder_thread.start()
    recorder_ready.wait()

    print("Server started. Press Ctrl+C to stop the server.")

    try:
        await asyncio.gather(control_server.wait_closed(), data_server.wait_closed(), broadcast_task)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Shut down the recorder
        if recorder:
            stop_recorder = True
            recorder.abort()
            recorder.stop()
            recorder.shutdown()
            print("Recorder shut down")

            recorder_thread.join()
            print("Recorder thread finished")
        
        # Cancel all active tasks in the event loop
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # Run pending tasks and handle cancellation
        await asyncio.gather(*tasks, return_exceptions=True)

        print("All tasks cancelled, closing event loop now.")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        # Capture any final KeyboardInterrupt to prevent it from showing up in logs
        print("Server interrupted by user.")
        exit(0)

if __name__ == '__main__':
    main()
