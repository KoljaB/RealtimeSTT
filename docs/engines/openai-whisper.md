# OpenAI Whisper

`openai_whisper` uses OpenAI's local `openai-whisper` package. It is useful for
applications that specifically want compatibility with the original Whisper
Python API instead of faster-whisper.

## Install

```bash
python -m pip install openai-whisper
```

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="openai_whisper",
    model="tiny.en",
    device="cpu",
    compute_type="float32",
)
```

## Model Behavior

OpenAI Whisper downloads known model names automatically through its package
cache. `download_root` is passed to `whisper.load_model` when set:

```python
recorder = AudioToTextRecorder(
    transcription_engine="openai_whisper",
    model="small.en",
    download_root="models/openai-whisper",
)
```

Local model paths supported by `openai-whisper` can also be passed as `model`.

## CPU/GPU Behavior

Use `device="cuda"` for GPU inference and `device="cpu"` for CPU inference.
The adapter maps `compute_type` to OpenAI Whisper's `fp16` flag:

- `float16`, `fp16`, `half` -> `fp16=True`
- `float32`, `fp32`, `int8` -> `fp16=False`

CPU runs should generally use `tiny` or `base` model sizes.

## Options

| Option bucket | Meaning |
| --- | --- |
| `transcription_engine_options["model"]` | Passed to `whisper.load_model`. |
| `transcription_engine_options["load_model"]` | Also merged into `whisper.load_model`. |
| `transcription_engine_options["transcribe"]` | Merged into `model.transcribe(...)`. |
| `beam_size` | Passed to transcription when greater than `1`. |
| `initial_prompt` | String prompts are supported. Token iterable prompts are not supported by this adapter. |
| `suppress_tokens` | Passed to transcription when set. |

## Tradeoffs

`openai_whisper` is familiar and direct, but faster-whisper is usually the
better default for production latency and CTranslate2 quantization. Use this
backend when API compatibility matters more than peak throughput.

## Troubleshooting

- Install `ffmpeg` if the underlying package or your workflow needs it for file
  handling.
- If CPU inference is too slow, switch to a smaller model or use
  `faster_whisper` with `compute_type="int8"`.
- If prompt errors mention unsupported prompt types, pass a string
  `initial_prompt`.
