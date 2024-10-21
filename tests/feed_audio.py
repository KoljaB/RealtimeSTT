if __name__ == "__main__":
    import threading
    import pyaudio
    from RealtimeSTT import AudioToTextRecorder

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    recorder = AudioToTextRecorder(
        use_microphone=False,
        spinner=False
    )

    stop_event = threading.Event()

    def feed_audio_thread():
        p = pyaudio.PyAudio()

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
                data = stream.read(CHUNK)
                recorder.feed_audio(data)
        except Exception as e:
            print(f"feed_audio_thread encountered an error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio stream closed.")

    def recorder_transcription_thread():
        def process_text(full_sentence):
            print("Transcribed text:", full_sentence)
            if "stop recording" in full_sentence.lower():
                print("Stop command detected. Stopping threads...")
                stop_event.set()
                recorder.abort()
        try:
            while not stop_event.is_set():
                recorder.text(process_text)
        except Exception as e:
            print(f"transcription_thread encountered an error: {e}")
        finally:
            print("Transcription thread exiting.")

    audio_thread = threading.Thread(target=feed_audio_thread)
    audio_thread.daemon = False
    audio_thread.start()

    transcription_thread = threading.Thread(target=recorder_transcription_thread)
    transcription_thread.daemon = False
    transcription_thread.start()

    audio_thread.join()
    transcription_thread.join()

    print("Recording and transcription have stopped.")
    recorder.shutdown()