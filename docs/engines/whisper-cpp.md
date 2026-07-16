# whisper.cpp

`whisper_cpp` uses the optional `pywhispercpp` package. It is useful when you
want local CPU transcription with whisper.cpp model files and a smaller Python
dependency surface than PyTorch-based engines.

## Install

```bash
python -m pip install "RealtimeSTT[whisper-cpp]"
```

Then select the engine:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="whisper_cpp",
    model="tiny.en",
    device="cpu",
)
```

## Model Behavior

`model` can be a name or path accepted by `pywhispercpp`. For known model names,
`pywhispercpp` may download the matching ggml model automatically. Use
`download_root` to keep model files in a predictable directory:

```python
recorder = AudioToTextRecorder(
    transcription_engine="whisper_cpp",
    model="small.en-q5_1",
    download_root="models/whispercpp",
    device="cpu",
)
```

If you download model files manually, pass the model path or put them where
`pywhispercpp` expects them through its `models_dir` option.

## CPU Tuning

For realtime CPU updates, use greedy decoding and streaming-friendly
`pywhispercpp` transcription options:

```python
recorder = AudioToTextRecorder(
    transcription_engine="whisper_cpp",
    model="tiny.en",
    device="cpu",
    beam_size=5,
    transcription_engine_options={
        "model": {
            "n_threads": 8,
            "redirect_whispercpp_logs_to": None,
        },
    },
    enable_realtime_transcription=True,
    realtime_transcription_engine="whisper_cpp",
    realtime_model_type="tiny.en",
    beam_size_realtime=1,
    realtime_processing_pause=0.15,
    realtime_transcription_engine_options={
        "model": {
            "n_threads": 8,
            "redirect_whispercpp_logs_to": None,
        },
        "transcribe": {
            "single_segment": True,
            "no_context": True,
            "print_timestamps": False,
        },
    },
)
```

Good starting profiles:

- Fast: `tiny.en` or `base.en-q5_1`, realtime beam size `1`.
- Balanced: `small.en-q5_1`, final beam size `3`.
- More accurate CPU: `small.en`, final beam size `5`.

`medium.en` and larger models can be too slow for interactive CPU use.

## Options

| Option bucket | Meaning |
| --- | --- |
| `transcription_engine_options["model"]` | Passed to `pywhispercpp.model.Model`. |
| `transcription_engine_options["transcribe"]` | Merged into `Model.transcribe(...)`. |
| `download_root` | Passed as `models_dir`. |
| `beam_size` | Uses whisper.cpp beam search when greater than `1`; otherwise greedy. |
| `initial_prompt` | String prompts become `initial_prompt`; token iterables become prompt token fields. |

Current adapter limitations:

- `compute_type`, `batch_size`, `faster_whisper_vad_filter`, and
  `suppress_tokens` do not map to equivalent whisper.cpp behavior.
- Language probability is not reported like faster-whisper; explicit languages
  are returned with probability `1.0`.
- Native whisper.cpp output may still appear depending on package behavior and
  options.

## Troubleshooting

- If import fails, install `pywhispercpp` in the active environment.
- If a model cannot be found, set `download_root` or pass an absolute model
  path.
- If realtime updates fall behind, reduce the model size, increase thread count
  up to the CPU's useful limit, keep `beam_size_realtime=1`, and increase
  `realtime_processing_pause`.
