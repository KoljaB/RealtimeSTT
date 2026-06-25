# FunASR

FunASR provides an experimental multilingual ASR path for models such as
SenseVoice.

## Install

Install the RealtimeSTT extra:

```bash
pip install "realtimestt[funasr]"
```

For an existing environment where RealtimeSTT is already installed:

```bash
pip install funasr
```

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="funasr",
    model="iic/SenseVoiceSmall",
    device="cuda",
    transcription_engine_options={
        "language": "auto",
        "use_itn": True,
    },
)
```

The adapter currently returns backend text mostly as FunASR provides it. Some
SenseVoice models include control tags such as language, emotion, speech type,
or inverse text-normalization markers in the transcript text; treat that output
format as backend-specific while the adapter is experimental.

For CPU:

```python
recorder = AudioToTextRecorder(
    transcription_engine="funasr",
    model="iic/SenseVoiceSmall",
    device="cpu",
)
```

If `model` is omitted, `AudioToTextRecorder` still supplies its legacy Whisper
default `"tiny"`. The FunASR adapter maps that default to
`iic/SenseVoiceSmall` unless `transcription_engine_options={"hub": "openai"}`
or `use_default_model=False` is set.

## Model Behavior

FunASR accepts ModelScope ids, Hugging Face ids when `hub="hf"`, known aliases,
and local model directories. Known model names such as `iic/SenseVoiceSmall`,
`FunAudioLLM/Fun-ASR-Nano-2512`, and `paraformer-zh` are downloaded by FunASR
when the files are not already present.

`download_root` sets default cache roots for FunASR's hub clients by filling
`MODELSCOPE_CACHE` and `HF_HOME` when those environment variables are not
already set. If you need exact placement, pass a local model directory as
`model`.

## Common Options

| RealtimeSTT parameter | FunASR mapping |
| --- | --- |
| `model` | `AutoModel(model=...)` |
| `device="cuda"` plus `gpu_device_index=0` | `device="cuda:0"` |
| `device="cpu"` | `device="cpu"` |
| `beam_size` | `AutoModel(beam_size=...)` |
| `batch_size` | `AutoModel(batch_size=...)` when greater than zero |
| `download_root` | Default `MODELSCOPE_CACHE` and `HF_HOME` |
| string `initial_prompt` | `generate(hotword=...)` |

Use `transcription_engine_options["model"]` or
`transcription_engine_options["auto_model"]` for advanced `AutoModel(...)`
arguments:

```python
recorder = AudioToTextRecorder(
    transcription_engine="funasr",
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    device="cuda",
    transcription_engine_options={
        "hub": "hf",
        "auto_model": {
            "trust_remote_code": True,
            "remote_code": "./model.py",
        },
    },
)
```

Use `transcription_engine_options["generate"]` or
`transcription_engine_options["transcribe"]` for `AutoModel.generate(...)`
arguments. Top-level shortcuts are also supported for common generation keys:

```python
recorder = AudioToTextRecorder(
    transcription_engine="funasr",
    model="iic/SenseVoiceSmall",
    device="cuda",
    transcription_engine_options={
        "language": "auto",
        "batch_size_s": 60,
        "use_itn": True,
        "sentence_timestamp": True,
    },
)
```

## VAD And Submodels

To let FunASR segment long audio internally, pass a FunASR VAD model:

```python
recorder = AudioToTextRecorder(
    transcription_engine="funasr",
    model="iic/SenseVoiceSmall",
    device="cuda",
    transcription_engine_options={
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
)
```

Set `vad_filter=False` to prevent the adapter from passing `vad_model` even if
it is present. Punctuation and speaker submodels may be passed with
`punc_model`, `punc_kwargs`, `spk_model`, and `spk_kwargs`.

## Real-Model Smoke Test

Use the standalone recorder test for manual validation:

```bash
python tests/realtimestt_funasr_test.py --init-only --device cuda
python tests/realtimestt_funasr_test.py --file-smoke --device cuda
python tests/realtimestt_funasr_test.py --device cuda
```

Add `--vad-model fsmn-vad` to test FunASR's own long-audio VAD path, or
`--device cpu` for CPU tests. The microphone loop uses final FunASR
transcription by default; add `--realtime` if you also want realtime preview
calls through FunASR.

The unit suite keeps real model loading opt-in:

```bash
set REALTIMESTT_RUN_FUNASR=1
python -m unittest tests.unit.test_funasr_engine
```

Optional environment variables:

```bash
set REALTIMESTT_FUNASR_MODEL=iic/SenseVoiceSmall
set REALTIMESTT_FUNASR_DEVICE=cuda
set REALTIMESTT_FUNASR_MODEL_DIR=models/funasr
```

## Troubleshooting

- If install fails on Windows, install a CUDA-compatible PyTorch build first,
  then install `funasr`.
- If startup checks are noisy, the adapter already passes
  `disable_update=True` and `disable_pbar=True` by default. Override them in
  `transcription_engine_options` if needed.
- If downloads fail, verify ModelScope or Hugging Face access and set
  `download_root`, `MODELSCOPE_CACHE`, or `HF_HOME` to a writable directory.
- FunASR input arrays are expected to be mono float audio at 16 kHz, matching
  RealtimeSTT's final transcription buffer.
