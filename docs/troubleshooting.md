# Troubleshooting

This page lists common RealtimeSTT setup and runtime issues. Engine-specific
notes live under [engines/](engines/) and the full install guide lives in
[installation.md](installation.md).

## Install Errors

### PyAudio Or PortAudio Fails To Build

Install PortAudio system packages first.

Linux:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
python -m pip install "RealtimeSTT[faster-whisper]"
```

macOS:

```bash
brew install portaudio
python -m pip install "RealtimeSTT[faster-whisper]"
```

Windows users should prefer wheels from PyPI where available and install from a
normal terminal with the intended Python environment active.

### Optional Engine Import Fails

RealtimeSTT imports optional engines lazily. If an engine import fails, install
that backend's package:

| Engine | Package |
| --- | --- |
| `whisper_cpp` | `pywhispercpp` |
| `openai_whisper` | `openai-whisper` |
| `sherpa_onnx_*` | `sherpa-onnx` |
| `parakeet` | `nemo_toolkit[asr]`, `soundfile` |
| `granite_speech`, `moonshine`, `cohere_transcribe` | `transformers`, `torch` |
| `qwen3_asr` | `qwen-asr` |
| `omnilingual_asr` | `RealtimeSTT[omnilingual]` on Linux/WSL2, plus a compatible PyTorch stack |
| `kroko_onnx` | `RealtimeSTT[kroko-builder]`, then `stt-install-kroko --build` |

## Audio Device Problems

### No Microphone Input

- Confirm the OS granted microphone permission to the terminal or Python app.
- Check the default input device in system settings.
- Pass `input_device_index` if the wrong device is selected.
- Run a small PyAudio device-list script or a recorder demo from
  [quick-start.md](quick-start.md).

### Input Overflow Warnings

Overflow warnings mean audio arrives faster than it is being consumed. Try:

- A smaller/faster model.
- `device="cuda"` when CUDA is available.
- Higher `realtime_processing_pause`.
- Lower realtime beam size.
- Larger queue/capacity settings in the FastAPI server.

### External Audio Sounds Garbled

Check that chunks passed to `feed_audio()` are:

- 16-bit signed PCM bytes.
- Mono.
- Labeled with the correct `original_sample_rate`.
- In the correct byte order.

See [external-audio.md](external-audio.md).

## CUDA And PyTorch

### cuDNN Or CUDA Library Cannot Be Loaded

Install a PyTorch build that matches your CUDA runtime and driver:

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Use the PyTorch install selector for your exact CUDA version. If GPU setup is
not required, use `device="cpu"` and a small model.

### GPU Memory Is Exhausted

- Use a smaller model.
- Use a smaller realtime model than final model.
- Set `use_main_model_for_realtime=True` to avoid loading two models, accepting
  some contention.
- Reduce batch sizes.
- Use CPU INT8 engines such as `sherpa_onnx_moonshine` for CPU deployments.

## Model Download And Cache Problems

### Hugging Face Downloads Fail

- Check network access.
- Set `download_root` to a writable directory.
- Authenticate if the model is gated.
- Pre-download models in the same environment that runs the app.

### sherpa-onnx Model Files Are Missing

sherpa-onnx engines do not download model bundles automatically. Download and
extract the required archive, then pass the extracted directory:

```python
AudioToTextRecorder(
    transcription_engine="sherpa_onnx_moonshine",
    model="test-model-cache/sherpa-onnx/sherpa-onnx-moonshine-tiny-en-int8",
    device="cpu",
    language="en",
)
```

The error message names the missing file. Confirm you are pointing at the
extracted directory, not the `.tar.bz2` file.

## Runtime Behavior

### `text()` Never Returns

The recorder waits for VAD to see speech end. Try:

- Feed trailing silence when using external audio.
- Lower `post_speech_silence_duration` for faster turn finalization.
- Check microphone input and VAD sensitivity.
- Confirm speech duration exceeds `min_length_of_recording`.

### Final Text Is Slow

- Use a smaller final model.
- Use CUDA.
- Reduce `beam_size`.
- Enable `early_transcription_on_silence` carefully.
- For realtime UX, keep final accuracy high but use a smaller realtime model.

### Realtime Text Lags

- Use `realtime_model_type="tiny.en"` or another small realtime model.
- Set `beam_size_realtime=1`.
- Increase `realtime_processing_pause`.
- Enable syllable-boundary scheduling and tune follow-up delays.
- Use a separate realtime engine/model if the final model is heavy.

### Wake Word Does Not Trigger

- Confirm `wake_words` or `wakeword_backend` is set.
- For Porcupine, use one of the supported built-in keyword names.
- For OpenWakeWord, pass valid model files and the matching framework.
- Tune `wake_words_sensitivity`.
- Test with a quiet room and close microphone first.

### Wake Word Appears In The Transcript

Increase `wake_word_buffer_duration` so more wake word audio is excluded from
the following recording.

## FastAPI Server

### Browser Cannot Connect

- Confirm the server is running and listening on the expected host/port.
- Open `http://localhost:8010` from the same machine first.
- Check browser microphone permission.
- Check `/health` for startup errors.

### New Sessions Are Rejected

The server reached `--max-sessions`. Increase the limit only if the selected
engine and hardware can handle the load.

### Realtime Events Drop Under Load

The server intentionally coalesces stale realtime work so final transcription
is preserved. Tune:

- `--max-realtime-queue-age-ms`
- `--realtime-processing-pause`
- `--max-global-inference-queue-depth`
- model size and engine choice

## Windows Notes

Some multiprocessing and model-loading tests may need to run from a normal
terminal rather than a restricted sandbox. Parakeet/NeMo and Qwen vLLM are
Linux-oriented; use WSL2 for those real-model paths on a Windows workstation.
