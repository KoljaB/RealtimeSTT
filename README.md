# RealtimeSTT

RealtimeSTT is a Python speech-to-text library for applications that need
voice activity detection, fast transcription, optional realtime text updates,
wake words, and direct access to audio streams. It is designed for assistants,
dictation tools, browser streaming servers, and prototypes that need to turn
speech into text with only a few lines of code.

The recommended default path uses `faster_whisper`. Other engines are available
through install extras when their optional dependencies and models are present.

## Demo

https://github.com/user-attachments/assets/797e6552-27cd-41b1-a7f3-e5cbc72094f5

[CLI demo code (reproduces the video above)](tests/realtimestt_test.py)

## Install

```bash
pip install "RealtimeSTT[faster-whisper]"
```

On Linux, install PortAudio headers before installing the package:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
```

On macOS:

```bash
brew install portaudio
```

For CUDA, platform notes, and optional engine stacks, see
[docs/installation.md](docs/installation.md).

## Microphone Example

This waits for speech, stops after the detected utterance, and prints the final
transcript:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    with AudioToTextRecorder() as recorder:
        print("Speak now")
        print(recorder.text())
```

Use the `if __name__ == "__main__":` guard when running scripts, especially on
Windows, because RealtimeSTT uses multiprocessing for model work.

## Automatic Recording Loop

For continuous dictation, pass a callback to `text()` so transcription work can
complete asynchronously while your loop keeps listening:

```python
from RealtimeSTT import AudioToTextRecorder


def process_text(text):
    print(text)


if __name__ == "__main__":
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)
```

## External Audio

Set `use_microphone=False` when audio comes from a file, stream, websocket, or
another process. Feed 16-bit mono PCM chunks at 16 kHz, or pass the original
sample rate so RealtimeSTT can resample:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    recorder = AudioToTextRecorder(use_microphone=False)

    with open("audio_chunk.pcm", "rb") as audio_file:
        recorder.feed_audio(audio_file.read(), original_sample_rate=16000)

    print(recorder.text())
    recorder.shutdown()
```

More examples are in [docs/quick-start.md](docs/quick-start.md) and
[docs/external-audio.md](docs/external-audio.md).

## Configuration Reference

Every `AudioToTextRecorder` constructor parameter is documented in
[docs/configuration.md](docs/configuration.md), including model/engine
selection, realtime transcription, VAD timing, wake words, callbacks, external
audio, logging, and executor injection.

## Features

- Voice activity detection with WebRTC VAD and Silero VAD.
- Final and realtime transcription with selectable engines.
- Optional wake word activation through Porcupine or OpenWakeWord.
- Direct microphone input or application-fed audio chunks.
- Event callbacks for recording, VAD, realtime text, transcription, and wake
  word state.
- A FastAPI browser streaming server example with multi-user session isolation,
  shared inference resources, metrics, and health endpoints.

## Documentation

- [Quick start](docs/quick-start.md): shortest demos and common recording
  patterns.
- [Installation](docs/installation.md): platform setup, CUDA notes, and optional
  dependencies.
- [Configuration](docs/configuration.md): complete `AudioToTextRecorder`
  parameter reference.
- [Transcription engines](docs/transcription-engines.md): engine selection and
  setup links.
- [Wake words](docs/wake-words.md): Porcupine and OpenWakeWord setup.
- [External audio](docs/external-audio.md): feeding audio without a microphone.
- [Testing](docs/testing.md): maintained unit and opt-in golden test workflow.
- [Test scripts](docs/test-scripts.md): demos, manual tests, regressions, and
  legacy experiments under `tests/`.
- [FastAPI server](docs/fastapi-server.md): browser server configuration,
  protocol, metrics, and deployment notes.
- [Troubleshooting](docs/troubleshooting.md): common install, audio, CUDA,
  model, dependency, and runtime errors.

Engine-specific references:

- [faster-whisper](docs/engines/faster-whisper.md)
- [whisper.cpp](docs/engines/whisper-cpp.md)
- [OpenAI Whisper](docs/engines/openai-whisper.md)
- [Moonshine](docs/engines/moonshine.md)
- [sherpa-onnx](docs/engines/sherpa-onnx.md)
- [Kroko-ONNX](docs/engines/kroko-onnx.md)
- [Parakeet NeMo](docs/engines/parakeet-nemo.md)
- [Meta Omnilingual ASR](docs/engines/omnilingual-asr.md)
- [Transformers engines](docs/engines/hf-transformers.md)
- [Cohere Transcribe](docs/engines/cohere.md)

## Server Example

The browser server lives in `example_fastapi_server`:

```bash
python -m pip install -r example_fastapi_server/requirements.txt
python example_fastapi_server/server.py --host 0.0.0.0 --port 8010
```

Open `http://localhost:8010`. See [docs/fastapi-server.md](docs/fastapi-server.md)
for engine recipes, websocket protocol details, health checks, and metrics.

## Contributing

Focused tests and small changes are easiest to review. The project keeps fast
unit tests separate from opt-in real-model tests; see [docs/testing.md](docs/testing.md).

## License

MIT

## Author

Kolja Beigel
