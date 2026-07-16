# Transformers-Backed Engines

RealtimeSTT includes thin adapters for model-family engines that use
Transformers or engine-specific Hugging Face tooling:

- `granite_speech`
- `qwen3_asr`
- `moonshine`

Cohere Transcribe has its own focused page because it requires an explicit
language and may require gated access: [cohere.md](cohere.md).

## Install

For Granite Speech and Moonshine:

```bash
python -m pip install transformers torch
```

For Qwen3-ASR:

```bash
python -m pip install -U qwen-asr
```

For the optional Qwen vLLM backend:

```bash
python -m pip install -U "qwen-asr[vllm]"
```

vLLM is Linux-oriented. On Windows, use WSL2 for real vLLM testing.

## Model Behavior

These engines download model files through Hugging Face or the engine package.
Set `download_root` to a writable cache directory when you want model files in
a predictable location:

```python
recorder = AudioToTextRecorder(
    transcription_engine="granite_speech",
    model="ibm-granite/granite-speech-4.1-2b",
    download_root="models/hf",
)
```

Some models require accepting license or gated access terms before download.

## Granite Speech

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="granite_speech",
    model="ibm-granite/granite-speech-4.1-2b",
    device="cuda",
    transcription_engine_options={
        "generate": {
            "max_new_tokens": 200,
            "do_sample": False,
        },
    },
)
```

Useful options:

- `engine_options["processor"]`: processor load options.
- `engine_options["model"]`: model load options.
- `engine_options["generate"]`: generation options.
- `engine_options["prompt"]`: prompt text used for transcription.
- `engine_options["include_language_in_prompt"]`: appends language to prompt
  when `language` is set.

## Qwen3-ASR

```python
recorder = AudioToTextRecorder(
    transcription_engine="qwen3_asr",
    model="Qwen/Qwen3-ASR-1.7B",
    language="en",
    device="cuda",
)
```

Useful options:

- `engine_options["backend"]`: `"transformers"` by default, or `"vllm"`.
- `engine_options["model"]`: model loader options.
- `engine_options["transcribe"]`: transcription options.
- `engine_options["language"]`: language when not passed through `language`.
- `engine_options["return_time_stamps"]`: request timestamps where supported.
- `engine_options["sample_rate"]`: sample rate for in-memory audio.

Two-letter language codes are mapped to names for common languages such as
English, German, Spanish, French, Japanese, Korean, Portuguese, Russian, and
Chinese.

## Moonshine

Moonshine details live in [moonshine.md](moonshine.md). It is listed here
because the default Moonshine adapter uses Transformers.

## Resource Usage

Transformers-backed engines often have larger model downloads, higher memory
requirements, and stricter CUDA/PyTorch compatibility than the default
faster-whisper path. Use them when model quality or family-specific behavior is
worth that operational cost.

For server deployments, prefer the FastAPI server's shared inference lanes
instead of creating one model per session.

## Troubleshooting

- If `transformers` lacks a required class, upgrade Transformers.
- If model downloads fail, check Hugging Face access and `download_root`.
- If CUDA memory is exhausted, reduce model size, use CPU where practical, or
  run one shared model lane in the server.
- If Qwen vLLM fails on Windows, move the run to Linux or WSL2.
