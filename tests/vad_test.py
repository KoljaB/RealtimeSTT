import asyncio
import time

from RealtimeSTT import AudioToTextRecorder


# Voice Activity Detection (VAD) start handler
def on_vad_detect_start():
    print(f"VAD Start detected at {time.time():.2f}")


# Voice Activity Detection (VAD) stop handler
def on_vad_detect_stop():
    print(f"VAD Stop detected at {time.time():.2f}")


# Transcription completion handler
def on_transcription_finished(text):
    print(f"Transcribed text: {text}")


async def run_recording(recorder):
    # Start recording and process audio in a loop
    print("Starting recording...")
    while True:
        # Use text() to process audio and get transcription
        recorder.text(on_transcription_finished=on_transcription_finished)
        await asyncio.sleep(0.1)  # Prevent tight loop


async def main():
    # Initialize AudioToTextRecorder with VAD event handlers
    recorder = AudioToTextRecorder(
        # model="deepdml/faster-whisper-large-v3-turbo-ct2",
        spinner=False,
        on_vad_detect_start=on_vad_detect_start,
        on_vad_detect_stop=on_vad_detect_stop,
    )

    # Start recording task in a separate thread
    recording_task = asyncio.create_task(run_recording(recorder))

    # Run for 20 seconds to observe VAD events
    await asyncio.sleep(20)

    # Stop recording and shutdown
    print("Stopping recording...")
    recorder.stop()
    recorder.shutdown()

    # Cancel and wait for the recording task to complete
    recording_task.cancel()
    try:
        await recording_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())