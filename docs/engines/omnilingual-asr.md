# Meta Omnilingual ASR

`omnilingual_asr` uses Meta's Omnilingual ASR package through
`ASRInferencePipeline`. This runtime is intended for Linux or WSL2. Native
Windows installs currently fail because `fairseq2n` has no Windows wheel.

## Install

Use a Linux or WSL2 environment with Python 3.10 through 3.12 and a
CUDA-enabled PyTorch stack, then install the optional backend:

```bash
python -m pip install "RealtimeSTT[omnilingual-asr]"
```

When working from a source checkout:

```bash
python -m pip install -e ".[omnilingual-asr]"
```

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="omnilingual_asr",
    model="omniASR_CTC_300M_v2",
    device="cuda",
    compute_type="float16",
)
```

If `model` is left at RealtimeSTT's default Whisper value, the Omnilingual
adapter uses `omniASR_CTC_300M_v2`.

## Models

Supported model card names include:

- `omniASR_CTC_300M_v2`
- `omniASR_CTC_1B_v2`
- `omniASR_LLM_1B_v2`
- `omniASR_CTC_3B`
- `omniASR_CTC_7B`
- `omniASR_LLM_3B`
- `omniASR_LLM_7B`
- `omniASR_LLM_7B_ZS`

The adapter does not reject larger `_v2` model card names if the installed
package exposes them.

## Language

CTC models ignore language, so the adapter does not pass `lang` for CTC model
cards. LLM models can use Omnilingual language IDs such as `eng_Latn`:

```python
recorder = AudioToTextRecorder(
    transcription_engine="omnilingual_asr",
    model="omniASR_LLM_1B_v2",
    language="eng_Latn",
)
```

Common two-letter codes such as `en`, `de`, `es`, and `fr` are mapped to their
Omnilingual language IDs before calling the LLM pipeline.

## Options

| Option | Meaning |
| --- | --- |
| `model` | Omnilingual model card. |
| `transcription_engine_options["model_card"]` | Overrides `model`. |
| `device` | Passed to `ASRInferencePipeline`; `cuda` becomes `cuda:<gpu_device_index>`. |
| `compute_type` | Mapped to torch dtype. Defaults to FP16 on CUDA, FP32 on CPU. |
| `transcription_engine_options["dtype"]` | Explicit torch dtype string such as `float16`, `bfloat16`, or `float32`. |
| `transcription_engine_options["sample_rate"]` | Sample rate for in-memory audio. Defaults to 16000. |
| `transcription_engine_options["batch_size"]` | Passed to `pipeline.transcribe(...)`. |
| `transcription_engine_options["pipeline"]` | Extra keyword arguments for `ASRInferencePipeline`. |
| `transcription_engine_options["transcribe"]` | Extra keyword arguments for `pipeline.transcribe(...)`. |
| `transcription_engine_options["max_audio_seconds"]` | In-memory audio duration guard. Defaults to 39.9 seconds. |

## Notes

- The Omnilingual package caches model assets through fairseq2 under the Linux
  user's default cache directory.
- In-memory audio is passed as
  `{"waveform": waveform_np, "sample_rate": sample_rate}`; raw float NumPy
  arrays are not passed directly to the package.
- The tested non-streaming Omnilingual pipeline requires audio shorter than 40
  seconds.
