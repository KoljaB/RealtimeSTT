# faster-whisper

`faster_whisper` is the default RealtimeSTT transcription engine. It uses
CTranslate2 through the `faster-whisper` package and supports the familiar
Whisper model names plus local CTranslate2 model directories.

## Install

Install the `faster-whisper` extra:

```bash
pip install "RealtimeSTT[faster-whisper]"
```

If you are working from a source checkout:

```bash
python -m pip install -e ".[faster-whisper]"
```

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="faster_whisper",
    model="small.en",
    device="cuda",
    compute_type="default",
)
```

For CPU:

```python
recorder = AudioToTextRecorder(
    model="tiny.en",
    device="cpu",
    compute_type="int8",
)
```

## Model Behavior

Known model names such as `tiny`, `tiny.en`, `base`, `small`, `medium`,
`large-v1`, and `large-v2` are downloaded automatically by faster-whisper.
Use `download_root` to control the cache/download directory:

```python
recorder = AudioToTextRecorder(
    model="small.en",
    download_root="models/faster-whisper",
)
```

You can also pass a path to a local CTranslate2-converted model directory as
`model`.

## GPU Notes

Use `device="cuda"` for GPU inference. `gpu_device_index` can be an integer or
a list of GPU ids for compatible multi-GPU loading.

`compute_type` controls CTranslate2 precision and quantization. Common values
include:

- `default`
- `float16`
- `float32`
- `int8`
- `int8_float16`

CPU runs are usually more practical with small models and `compute_type="int8"`.

## Common Options

| RealtimeSTT parameter | faster-whisper mapping |
| --- | --- |
| `model` | `WhisperModel(model_size_or_path=...)` |
| `download_root` | `WhisperModel(download_root=...)` |
| `device` | `WhisperModel(device=...)` |
| `compute_type` | `WhisperModel(compute_type=...)` |
| `gpu_device_index` | `WhisperModel(device_index=...)` |
| `beam_size` | `model.transcribe(beam_size=...)` |
| `batch_size` | Enables `BatchedInferencePipeline` when greater than `0`. |
| `language` | Passed as the transcription language when set. |
| `initial_prompt` | Passed as `initial_prompt`. |
| `suppress_tokens` | Passed as `suppress_tokens`. |
| `faster_whisper_vad_filter` | Passed as `vad_filter`. |
| `normalize_audio` | Normalizes audio before transcription when enabled. |

## Realtime Suggestions

Use a smaller realtime model than the final model:

```python
recorder = AudioToTextRecorder(
    model="small.en",
    enable_realtime_transcription=True,
    realtime_model_type="tiny.en",
    realtime_processing_pause=0.15,
)
```

For a single shared model, set `use_main_model_for_realtime=True`. This saves
memory but can reduce responsiveness when final and realtime work contend for
the same model.

## Troubleshooting

- If CUDA libraries fail to load, reinstall PyTorch/torchaudio for the CUDA
  version on the machine.
- If model downloads fail, set `download_root` to a writable directory and test
  network access to the Hugging Face Hub.
- If realtime text lags, use a smaller realtime model, lower `beam_size_realtime`,
  increase `realtime_processing_pause`, or switch realtime to a CPU-friendly
  engine.
