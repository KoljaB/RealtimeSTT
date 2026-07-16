# Parakeet NeMo

`parakeet` and `nvidia_parakeet` use NVIDIA NeMo ASR to load Parakeet models.
This path is intended for Linux or WSL2 environments, especially when CUDA is
available.

For CPU INT8 inference without NeMo, use [sherpa-onnx.md](sherpa-onnx.md).

## Install

```bash
python -m pip install -U "nemo_toolkit[asr]" soundfile
```

NeMo ASR is primarily supported on Linux. On Windows, use WSL2 for real-model
testing.

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="parakeet",
    model="nvidia/parakeet-tdt-0.6b-v3",
    device="cuda",
    language="en",
)
```

## CUDA Requirements

Use a CUDA-enabled PyTorch stack that matches your NVIDIA driver. Keep the model
on GPU with `device="cuda"` when possible:

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

NeMo, PyTorch, CUDA, and Python version compatibility is stricter than the
default faster-whisper path. If you hit package conflicts, start from a fresh
Linux virtual environment.

## Model Cache Behavior

The NeMo backend calls `ASRModel.from_pretrained(model_name=...)`. NeMo handles
the model download/cache for known model ids such as:

- `nvidia/parakeet-tdt-0.6b-v3`

Backend loader options may be passed through
`transcription_engine_options["model"]`:

```python
recorder = AudioToTextRecorder(
    transcription_engine="parakeet",
    model="nvidia/parakeet-tdt-0.6b-v3",
    transcription_engine_options={
        "model": {},
        "transcribe": {"batch_size": 1},
    },
)
```

## Expected Resource Usage

Parakeet v3 is a large ASR model compared with the default tiny Whisper demo.
Expect higher memory use, longer startup time, and better results from a GPU
environment. For multi-user server deployments, prefer a shared-model server
lane rather than one model per browser session.

## Options

| Option | Meaning |
| --- | --- |
| `transcription_engine_options["model"]` | Passed to `ASRModel.from_pretrained`. |
| `transcription_engine_options["transcribe"]` | Merged into `model.transcribe(...)`. |
| `transcription_engine_options["sample_rate"]` | Sample rate used when writing in-memory audio to a temporary WAV. |
| `transcription_engine_options["timestamps"]` | Enables/disables timestamp request when supported by NeMo. |
| `batch_size` | Passed to transcribe when greater than `0`. |

## FastAPI Recipe

```bash
python example_fastapi_server/server.py \
  --engine parakeet \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --realtime-engine faster_whisper \
  --realtime-model tiny.en \
  --device cuda \
  --language en
```

Use `--use-main-model-for-realtime` only when you want a single shared model
lane and can accept realtime/final contention.

## Troubleshooting

- If `nemo.collections.asr` cannot be imported, install
  `"nemo_toolkit[asr]"` in the active environment.
- If `soundfile` errors occur, install `soundfile` and system `libsndfile`.
- If Windows package resolution fails, move real Parakeet testing to WSL2 or
  Linux.
- If startup is slow, remember that NeMo is downloading and loading a large
  model.
