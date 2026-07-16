# Quick Start

This page starts with the smallest working examples and then adds the common
recording patterns most applications need.

Use an `if __name__ == "__main__":` guard in runnable scripts, especially on
Windows, because RealtimeSTT uses multiprocessing for model work.

## One Utterance From The Microphone

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    with AudioToTextRecorder() as recorder:
        print("Speak now")
        print(recorder.text())
```

`text()` waits until voice activity starts and stops, then returns the final
transcription.

## Continuous Automatic Recording

Use a callback when you want to keep listening in a loop:

```python
from RealtimeSTT import AudioToTextRecorder


def print_text(text):
    print(text)


if __name__ == "__main__":
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(print_text)
```

## Manual Start And Stop

Use `start()` and `stop()` when the application decides when recording begins
and ends:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder()
    recorder.start()
    input("Press Enter to stop recording...")
    recorder.stop()
    print(recorder.text())
    recorder.shutdown()
```

## Realtime Text Updates

Realtime updates are interim text for the current recording. Final text still
comes from `text()`:

```python
from RealtimeSTT import AudioToTextRecorder


def update(text):
    print("live:", text)


if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        enable_realtime_transcription=True,
        on_realtime_transcription_update=update,
        realtime_model_type="tiny.en",
        model="small.en",
    )

    while True:
        print("final:", recorder.text())
```

Use a smaller realtime model than the final model when you want faster interim
text.

## Wake Word Activation

Porcupine wake words can be enabled with a comma-separated `wake_words` list:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder(wake_words="jarvis")
    print('Say "Jarvis" and then speak.')
    print(recorder.text())
    recorder.shutdown()
```

OpenWakeWord uses `wakeword_backend="oww"` and model file paths. See
[wake-words.md](wake-words.md) for setup details.

## External Audio

Set `use_microphone=False` and feed PCM audio into the recorder:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder(use_microphone=False)

    with open("audio_chunk.pcm", "rb") as audio_file:
        recorder.feed_audio(audio_file.read(), original_sample_rate=16000)

    print(recorder.text())
    recorder.shutdown()
```

For file streams, websocket clients, and process pipelines, see
[external-audio.md](external-audio.md).

## CPU-Friendly Engine Example

The recommended `faster_whisper` path is installed with
`RealtimeSTT[faster-whisper]`. For CPU-focused local testing with whisper.cpp:

```bash
python -m pip install "RealtimeSTT[whisper-cpp]"
```

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        transcription_engine="whisper_cpp",
        model="tiny.en",
        device="cpu",
        beam_size=1,
    )
    print(recorder.text())
    recorder.shutdown()
```

See [transcription-engines.md](transcription-engines.md) before choosing an
engine for production.
