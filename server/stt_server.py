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

import asyncio
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
client_websocket = None
stop_recorder = False
prev_text = ""


async def send_to_client(message):
    global client_websocket
    if client_websocket and client_websocket.open:
        try:
            await client_websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            print("Client websocket is closed, resetting client_websocket")
            client_websocket = None
    else:
        print("No client connected or connection is closed.")
        client_websocket = None  # Ensure it resets

def preprocess_text(text):
    # Remove leading whitespaces
    text = text.lstrip()

    #  Remove starting ellipses if present
    if text.startswith("..."):
        text = text[3:]

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def text_detected(text):
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

    try:
        asyncio.new_event_loop().run_until_complete(
            send_to_client(
                json.dumps({
                    'type': 'realtime',
                    'text': text
                })
            )
        )
    except Exception as e:
        print(f"Error in text_detected while sending to client: {e}")
    print(f"\r{text}", flush=True, end='')

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
    
    parser.add_argument('--webrtc_sensitivity', type=float, default=3,
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
    
    return parser.parse_args()

def _recorder_thread():
    global recorder, prev_text, stop_recorder
    # print("Initializing RealtimeSTT...")
    print(f"Initializing RealtimeSTT server with parameters {recorder_config}")
    recorder = AudioToTextRecorder(**recorder_config)
    print("RealtimeSTT initialized")
    recorder_ready.set()
    
    def process_text(full_sentence):
        full_sentence = preprocess_text(full_sentence)
        prev_text = ""
        try:
            asyncio.new_event_loop().run_until_complete(
                send_to_client(
                    json.dumps({
                        'type': 'fullSentence',
                        'text': full_sentence
                    })
                )
            )
        except Exception as e:
            print(f"Error in _recorder_thread while sending to client: {e}")
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

async def echo(websocket, path):
    print("Client connected")
    global client_websocket
    client_websocket = websocket
    recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause
    try:
        async for message in websocket:
            if not recorder_ready.is_set():
                print("Recorder not ready")
                continue

            metadata_length = int.from_bytes(message[:4], byteorder='little')
            metadata_json = message[4:4+metadata_length].decode('utf-8')
            metadata = json.loads(metadata_json)
            sample_rate = metadata['sampleRate']
            chunk = message[4+metadata_length:]
            resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
            recorder.feed_audio(resampled_chunk)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")
    finally:
        print("Resetting client_websocket after disconnect")
        client_websocket = None  # Reset websocket reference

async def main_async():            
    global stop_recorder, recorder_config, global_args
    args = parse_arguments()
    global_args = args

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
        'on_realtime_transcription_update': text_detected,
        'no_log_file': True,
    }

    start_server = await websockets.serve(echo, "localhost", 8011)

    recorder_thread = threading.Thread(target=_recorder_thread)
    recorder_thread.start()
    recorder_ready.wait()

    print("Server started. Press Ctrl+C to stop the server.")
    
    try:
        await start_server.wait_closed()  # This will keep the server running
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
