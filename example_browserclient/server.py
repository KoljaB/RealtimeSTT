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
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger('websockets').setLevel(logging.WARNING)

    is_running = True
    recorder = None
    recorder_ready = threading.Event()
    client_websocket = None
    main_loop = None  # This will hold our primary event loop

    async def send_to_client(message):
        global client_websocket
        if client_websocket:
            try:
                await client_websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                client_websocket = None
                print("Client disconnected")

    # Called from the recorder thread on stabilized realtime text.
    def text_detected(text):
        global main_loop
        if main_loop is not None:
            # Schedule the sending on the main event loop
            asyncio.run_coroutine_threadsafe(
                send_to_client(json.dumps({
                    'type': 'realtime',
                    'text': text
                })), main_loop)
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
        global recorder, main_loop, is_running
        print("Initializing RealtimeSTT...")
        recorder = AudioToTextRecorder(**recorder_config)
        print("RealtimeSTT initialized")
        recorder_ready.set()

        # Loop indefinitely checking for full sentence output.
        while is_running:
            try:
                full_sentence = recorder.text()
                if full_sentence:
                    if main_loop is not None:
                        asyncio.run_coroutine_threadsafe(
                            send_to_client(json.dumps({
                                'type': 'fullSentence',
                                'text': full_sentence
                            })), main_loop)
                    print(f"\rSentence: {full_sentence}")
            except Exception as e:
                print(f"Error in recorder thread: {e}")
                continue

    def decode_and_resample(audio_data, original_sample_rate, target_sample_rate):
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            num_original_samples = len(audio_np)
            num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
            resampled_audio = resample(audio_np, num_target_samples)
            return resampled_audio.astype(np.int16).tobytes()
        except Exception as e:
            print(f"Error in resampling: {e}")
            return audio_data

    async def echo(websocket):
        global client_websocket
        print("Client connected")
        client_websocket = websocket

        try:
            async for message in websocket:
                if not recorder_ready.is_set():
                    print("Recorder not ready")
                    continue

                try:
                    # Read the metadata length (first 4 bytes)
                    metadata_length = int.from_bytes(message[:4], byteorder='little')
                    # Get the metadata JSON string
                    metadata_json = message[4:4+metadata_length].decode('utf-8')
                    metadata = json.loads(metadata_json)
                    sample_rate = metadata['sampleRate']
                    # Get the audio chunk following the metadata
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
        global main_loop
        main_loop = asyncio.get_running_loop()

        recorder_thread = threading.Thread(target=run_recorder)
        recorder_thread.daemon = True
        recorder_thread.start()
        recorder_ready.wait()

        print("Server started. Press Ctrl+C to stop the server.")
        async with websockets.serve(echo, "localhost", 8001):
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                print("\nShutting down server...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        is_running = False
        recorder.stop()
        recorder.shutdown()
    finally:
        if recorder:
            del recorder
