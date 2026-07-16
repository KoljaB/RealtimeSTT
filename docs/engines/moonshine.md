# Moonshine

`moonshine` and `moonshine_streaming` use the Moonshine streaming model family
through Transformers by default. The adapter currently treats Moonshine as an
English-only engine.

Moonshine can also be routed through sherpa-onnx by setting
`transcription_engine_options={"backend": "sherpa_onnx"}`; see
[sherpa-onnx.md](sherpa-onnx.md) for that path.

## Install

```bash
python -m pip install transformers torch
```

Install a CUDA-enabled PyTorch wheel first if you plan to run on GPU.

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="moonshine",
    model="UsefulSensors/moonshine-streaming-medium",
    language="en",
    device="cuda",
)
```

For CPU, start with a smaller model if available in your environment and use
`device="cpu"`.

## Model Cache Behavior

The Transformers backend downloads model and processor files automatically from
Hugging Face. `download_root` is passed as `cache_dir` for model and processor
loading unless an explicit cache directory is provided in options:

```python
recorder = AudioToTextRecorder(
    transcription_engine="moonshine",
    model="UsefulSensors/moonshine-streaming-medium",
    download_root="models/hf",
    language="en",
)
```

## CPU Usage

Moonshine is designed for streaming use, but Python/Transformers overhead can
still be significant. For CPU-focused server deployments, the sherpa-onnx
Moonshine INT8 path is usually more predictable because it uses extracted ONNX
model files and ONNX Runtime.

## Common Options

| Option bucket | Meaning |
| --- | --- |
| `transcription_engine_options["model"]` | Passed to `MoonshineStreamingForConditionalGeneration.from_pretrained`. |
| `transcription_engine_options["processor"]` | Passed to `AutoProcessor.from_pretrained`. |
| `transcription_engine_options["generate"]` | Passed to `model.generate(...)`. |
| `transcription_engine_options["sample_rate"]` | Input sample rate for the processor. Defaults to processor rate or 16000. |
| `compute_type` | Mapped to a torch dtype where supported. |

## FastAPI Recipe

```bash
python example_fastapi_server/server.py \
  --engine moonshine \
  --model UsefulSensors/moonshine-streaming-medium \
  --realtime-engine moonshine \
  --realtime-model UsefulSensors/moonshine-streaming-medium \
  --language en \
  --device cuda
```

For CPU servers, compare with the sherpa-onnx Moonshine recipes in
[fastapi-server.md](../fastapi-server.md).

## Troubleshooting

- Moonshine currently raises an error for non-English languages in this adapter.
- If model downloads fail, confirm Hugging Face access and set `download_root`
  to a writable cache directory.
- If memory use is too high, use the sherpa-onnx INT8 Moonshine engine or a
  smaller model.
