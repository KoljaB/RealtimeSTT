if __name__ == "__main__":
    import threading
    import pyaudio
    from RealtimeSTT import AudioToTextRecorder

    # Audio stream configuration constants
    CHUNK = 1024                  # Number of audio samples per buffer
    FORMAT = pyaudio.paInt16      # Sample format (16-bit integer)
    CHANNELS = 1                  # Mono audio
    RATE = 16000                  # Sampling rate in Hz (expected by the recorder)

    # Initialize the audio-to-text recorder without using the microphone directly
    # Since we are feeding audio data manually, set use_microphone to False
    recorder = AudioToTextRecorder(
        use_microphone=False,     # Disable built-in microphone usage
        spinner=False             # Disable spinner animation in the console
    )

    # Event to signal when to stop the threads
    stop_event = threading.Event()

    def feed_audio_thread():
        """Thread function to read audio data and feed it to the recorder."""
        p = pyaudio.PyAudio()

        # Open an input audio stream with the specified configuration
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        try:
            print("Speak now")
            while not stop_event.is_set():
                # Read audio data from the stream (in the expected format)
                data = stream.read(CHUNK)
                # Feed the audio data to the recorder
                recorder.feed_audio(data)
        except Exception as e:
            print(f"feed_audio_thread encountered an error: {e}")
        finally:
            # Clean up the audio stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio stream closed.")

    def recorder_transcription_thread():
        """Thread function to handle transcription and process the text."""
        def process_text(full_sentence):
            """Callback function to process the transcribed text."""
            print("Transcribed text:", full_sentence)
            # Check for the stop command in the transcribed text
            if "stop recording" in full_sentence.lower():
                print("Stop command detected. Stopping threads...")
                stop_event.set()
                recorder.abort()
        try:
            while not stop_event.is_set():
                # Get transcribed text and process it using the callback
                recorder.text(process_text)
        except Exception as e:
            print(f"transcription_thread encountered an error: {e}")
        finally:
            print("Transcription thread exiting.")

    # Create and start the audio feeding thread
    audio_thread = threading.Thread(target=feed_audio_thread)
    audio_thread.daemon = False    # Ensure the thread doesn't exit prematurely
    audio_thread.start()

    # Create and start the transcription thread
    transcription_thread = threading.Thread(target=recorder_transcription_thread)
    transcription_thread.daemon = False    # Ensure the thread doesn't exit prematurely
    transcription_thread.start()

    # Wait for both threads to finish
    audio_thread.join()
    transcription_thread.join()

    print("Recording and transcription have stopped.")
    recorder.shutdown()
