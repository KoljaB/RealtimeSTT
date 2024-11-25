end_of_sentence_detection_pause = 0.45
unknown_sentence_detection_pause = 0.7
mid_sentence_detection_pause = 2.0

from install_packages import check_and_install_packages

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

recorder = None
recorder_ready = threading.Event()
client_websocket = None
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
        recorder.post_speech_silence_duration = mid_sentence_detection_pause
    elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
        recorder.post_speech_silence_duration = end_of_sentence_detection_pause
    else:
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

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


# Recorder configuration
recorder_config = {
    'spinner': False,
    'use_microphone': False,
    'model': 'medium.en', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
    'input_device_index': 1,
    'realtime_model_type': 'tiny.en', # or small.en or distil-small.en or ...
    'language': 'en',
    'silero_sensitivity': 0.05,
    'webrtc_sensitivity': 3,
    'post_speech_silence_duration': unknown_sentence_detection_pause,
    'min_length_of_recording': 1.1,        
    'min_gap_between_recordings': 0,                
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0.02,
    'on_realtime_transcription_update': text_detected,
    #'on_realtime_transcription_stabilized': text_detected,
    'silero_deactivity_detection': True,
    'early_transcription_on_silence': 0.2,
    'beam_size': 5,
    'beam_size_realtime': 3,
    'no_log_file': True,
    'initial_prompt': 'Add periods only for complete sentences. Use ellipsis (...) for unfinished thoughts or unclear endings. Examples: \n- Complete: "I went to the store."\n- Incomplete: "I think it was..."'
    #  'initial_prompt': "Only add a period at the end of a sentence if you are 100 percent certain that the speaker has finished their statement. If you're unsure or the sentence seems incomplete, leave the sentence open or use ellipses to reflect continuation. For example: 'I went to the...' or 'I think it was...'"
    # 'initial_prompt': "Use ellipses for incomplete sentences like: I went to the..."        
}

def _recorder_thread():
    global recorder, prev_text
    print("Initializing RealtimeSTT...")
    recorder = AudioToTextRecorder(**recorder_config)
    print("RealtimeSTT initialized")
    recorder_ready.set()
    
    def process_text(full_sentence):
        print(f"\rSentence1: {full_sentence}")
        full_sentence = preprocess_text(full_sentence)
        print(f"\rSentence2: {full_sentence}")
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
        print(f"\rSentence3: {full_sentence}")

    while True:
        recorder.text(process_text)

def decode_and_resample(
        audio_data,
        original_sample_rate,
        target_sample_rate):

    # Decode 16-bit PCM data to numpy array
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
    recorder.post_speech_silence_duration = unknown_sentence_detection_pause
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

def main():            
    start_server = websockets.serve(echo, "localhost", 8011)

    recorder_thread = threading.Thread(target=_recorder_thread)
    recorder_thread.start()
    recorder_ready.wait()

    print("Server started. Press Ctrl+C to stop the server.")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    main()
