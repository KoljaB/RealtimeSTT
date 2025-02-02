if __name__ == '__main__':
    print("Starting server, please wait...")
    from RealtimeSTT import AudioToTextRecorder
    import asyncio
    import websockets
    import threading
    import numpy as np
    from scipy.signal import resample
    import json
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger('websockets').setLevel(logging.WARNING)

    recorder = None
    recorder_ready = threading.Event()
    client_websocket = None

    async def send_to_client(message):
        global client_websocket
        if client_websocket:
            try:
                await client_websocket.send(message)  # Add a newline or delimiter
            except websockets.exceptions.ConnectionClosed:
                client_websocket = None
                print("Client disconnected")

    def text_detected(text):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                send_to_client(
                    json.dumps({
                        'type': 'realtime',
                        'text': text
                    })
                )
            )
        finally:
            loop.close()
        print(f"\r{text}", flush=True, end='')

    recorder_config = {
        'spinner': False,
        'use_microphone': False,
        'model': 'large-v2',
        'language': 'en',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.7,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_stabilized': text_detected,
    }

    def run_recorder():
        global recorder
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print("Initializing RealtimeSTT...")
        recorder = AudioToTextRecorder(**recorder_config)
        print("RealtimeSTT initialized")
        recorder_ready.set()
        
        try:
            while True:
                try:
                    full_sentence = recorder.text()
                    if full_sentence:
                        loop.run_until_complete(
                            send_to_client(
                                json.dumps({
                                    'type': 'fullSentence',
                                    'text': full_sentence
                                })
                            )
                        )
                        print(f"\rSentence: {full_sentence}")
                except Exception as e:
                    print(f"Error in recorder thread: {e}")
                    continue
        finally:
            loop.close()

    def decode_and_resample(
            audio_data,
            original_sample_rate,
            target_sample_rate):
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            num_original_samples = len(audio_np)
            num_target_samples = int(num_original_samples * target_sample_rate / 
                                   original_sample_rate)
            resampled_audio = resample(audio_np, num_target_samples)
            return resampled_audio.astype(np.int16).tobytes()
        except Exception as e:
            print(f"Error in resampling: {e}")
            return audio_data

    async def echo(websocket):
        global client_websocket  #global
        print("Client connected")
        client_websocket = websocket
        
        try:
            async for message in websocket:
                if not recorder_ready.is_set():
                    print("Recorder not ready")
                    continue

                try:
                    metadata_length = int.from_bytes(message[:4], byteorder='little')
                    metadata_json = message[4:4+metadata_length].decode('utf-8')
                    metadata = json.loads(metadata_json)
                    sample_rate = metadata['sampleRate']
                    chunk = message[4+metadata_length:]
                    resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
                    recorder.feed_audio(resampled_chunk)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            if client_websocket == websocket:
                client_websocket = None

    async def main():
        recorder_thread = threading.Thread(target=run_recorder)
        recorder_thread.daemon = True
        recorder_thread.start()
        recorder_ready.wait()
        
        print("Server started. Press Ctrl+C to stop the server.")
        async with websockets.serve(echo, "localhost", 8001):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                print("\nShutting down server...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        if recorder:
            del recorder
