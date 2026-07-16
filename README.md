# RealtimeSTT

RealtimeSTT is a Python speech-to-text library for applications that need
voice activity detection, fast transcription, optional realtime text updates,
wake words, and direct access to audio streams. It is designed for assistants,
dictation tools, browser streaming servers, and prototypes that need to turn
speech into text with only a few lines of code.

The recommended default path uses `faster_whisper`. Other engines are available
through install extras when their optional dependencies and models are present.

### Support RealtimeSTT

If RealtimeSTT saved you time, one GitHub star is a simple way to help make it more stable.

Stars improve visibility and visibility brings more users, more real-world testing, more bug reports, more fixes, and better releases for everyone.

## Demo

https://github.com/user-attachments/assets/797e6552-27cd-41b1-a7f3-e5cbc72094f5

[CLI demo code (reproduces the video above)](tests/realtimestt_test.py)

## Featured Integration: Kroko/Banafo ASR

RealtimeSTT includes native support for `kroko_onnx`, the local streaming ASR
engine from the Kroko/Banafo team.

This integration has been on my wishlist for a long time. Kroko is a strong fit
for RealtimeSTT's goals: fast, accurate local speech recognition.

Start with the public Community models for local testing, or see Kroko/Banafo's
commercial model options if you need production licensing and higher-end models.

```bash
pip install "RealtimeSTT[kroko-builder,silero-onnx-cpu]"
stt-install-kroko --build
```

The `silero-onnx-cpu` extra gives `AudioToTextRecorder` a local VAD backend for
recorder-based smoke tests and live microphone use.

See the [Kroko-ONNX engine guide](docs/engines/kroko-onnx.md),
[Kroko ASR docs](https://docs.kroko.ai/on-premise/), and
[kroko-onnx on GitHub](https://github.com/kroko-ai/kroko-onnx).

## Install

Use Python 3.11 or newer for the current pinned dependency set.

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
- [Engine licenses](docs/licenses.md): license notes for optional engine
  runtimes and model families.

Engine-specific references:

- [faster-whisper](docs/engines/faster-whisper.md)
- [whisper.cpp](docs/engines/whisper-cpp.md)
- [OpenAI Whisper](docs/engines/openai-whisper.md)
- [Moonshine](docs/engines/moonshine.md)
- [sherpa-onnx](docs/engines/sherpa-onnx.md)
- [Kroko-ONNX](docs/engines/kroko-onnx.md)
- [Parakeet NeMo](docs/engines/parakeet-nemo.md)
- [Meta Omnilingual ASR](docs/engines/omnilingual-asr.md)
- [Granite/Qwen Transformers engines](docs/engines/hf-transformers.md)
- [Cohere Transcribe](docs/engines/cohere.md)
- [FunASR](docs/engines/funasr.md)

## Server Example

The browser FastAPI reference server lives in `example_fastapi_server` and is
intended for source checkouts. It is not installed by the PyPI wheel; keeping it
source-only keeps the wheel lean and avoids adding web-server dependencies for
users who only need the recorder/API library.

```bash
python -m pip install -r example_fastapi_server/requirements.txt
python example_fastapi_server/server.py --host 0.0.0.0 --port 8010
```

For pip-only installs, use the Python recorder/API examples instead. If you
want the FastAPI reference server, clone the repository or install from Git.

Open `http://localhost:8010`. See [docs/fastapi-server.md](docs/fastapi-server.md)
for engine recipes, websocket protocol details, health checks, and metrics.

## Contributing

Focused tests and small changes are easiest to review. The project keeps fast
unit tests separate from opt-in real-model tests; see [docs/testing.md](docs/testing.md).

## License

MIT

## Author

Kolja Beigel
