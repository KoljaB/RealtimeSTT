# Meta Omnilingual ASR

`omnilingual_asr` uses Meta's Omnilingual ASR package through
`ASRInferencePipeline`. The adapter is lazy-loaded, so normal RealtimeSTT
imports and installs do not require the Omnilingual runtime until this engine
is selected.

This backend is intended for Linux or WSL2. Native Windows installs currently
fail because `fairseq2n` has no Windows wheel.

## Engine Names

- `omnilingual_asr`
- `omnilingual`
- `meta_omnilingual_asr`
- `omni_asr`

Hyphenated CLI forms such as `omnilingual-asr`,
`meta-omnilingual-asr`, and `omni-asr` are accepted by the generic engine-name
normalization.

## Install

Use a Linux or WSL2 environment with Python 3.10 through 3.12. Install a
CUDA-enabled PyTorch stack first when you want GPU inference, then install the
RealtimeSTT extra. The clean alias is `omnilingual`:

```bash
python -m pip install "RealtimeSTT[omnilingual]"
```

`omnilingual-asr` and `meta-omnilingual-asr` are equivalent aliases if you
prefer the explicit package-family name.

When working from a source checkout:

```bash
python -m pip install -e ".[omnilingual]"
```

The extra is guarded with a non-Windows platform marker. On Windows, create or
reuse a WSL2 environment and run the Omnilingual process there.

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

If `model` is left at RealtimeSTT's default Whisper value, the adapter selects
`omniASR_CTC_300M_v2`.

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

The adapter also accepts larger `_v2` names if the installed Omnilingual
package exposes them, for example `omniASR_CTC_3B_v2` or
`omniASR_LLM_7B_v2`.

`omniASR_CTC_300M_v2` is the recommended default for realtime-oriented local
testing. Larger CTC and LLM models may need substantially more VRAM and longer
startup time.

## Language

CTC model cards ignore language, so the adapter removes `lang` before calling
the CTC pipeline.

LLM model cards can use Omnilingual language IDs such as `eng_Latn`:

```python
recorder = AudioToTextRecorder(
    transcription_engine="omnilingual_asr",
    model="omniASR_LLM_1B_v2",
    language="eng_Latn",
    device="cuda",
    compute_type="float16",
)
```

Common short codes such as `en`, `de`, `es`, and `fr` are mapped to
Omnilingual language IDs before calling the LLM pipeline. Additional aliases
can be supplied through `transcription_engine_options["language_aliases"]`.

## Model Cache Behavior

The Omnilingual package downloads and caches model assets through its
underlying fairseq2/Hugging Face tooling in the Linux user's normal cache
locations. RealtimeSTT does not move or delete those files.

The adapter passes in-memory audio as a predecoded object:

```python
{"waveform": waveform_np, "sample_rate": sample_rate}
```

This avoids the Omnilingual package treating a raw NumPy float array as encoded
audio bytes.

## Realtime Suggestions

For realtime previews, use the smaller CTC model first:

```python
recorder = AudioToTextRecorder(
    transcription_engine="omnilingual_asr",
    model="omniASR_CTC_300M_v2",
    enable_realtime_transcription=True,
    realtime_transcription_engine="omnilingual_asr",
    realtime_model_type="omniASR_CTC_300M_v2",
    use_main_model_for_realtime=True,
    device="cuda",
    compute_type="float16",
)
```

`use_main_model_for_realtime=True` keeps one shared model in memory. This is
usually the safest first test for VRAM, but final and realtime requests can
contend for that same model. If you use separate final and realtime models,
validate VRAM headroom before increasing model size.

## Options

| Option | Meaning |
| --- | --- |
| `model` | Omnilingual model card. |
| `transcription_engine_options["model_card"]` | Overrides `model`. |
| `device` | Passed to `ASRInferencePipeline`; `cuda` becomes `cuda:<gpu_device_index>`. |
| `compute_type` | Mapped to torch dtype. Defaults to FP16 on CUDA and FP32 on CPU. |
| `transcription_engine_options["dtype"]` / `["torch_dtype"]` | Explicit torch dtype string such as `float16`, `bfloat16`, or `float32`. |
| `transcription_engine_options["sample_rate"]` | Sample rate for in-memory audio. Defaults to 16000. |
| `batch_size` | Used when greater than `0`; otherwise the adapter defaults to batch size `1`. |
| `transcription_engine_options["batch_size"]` | Overrides the RealtimeSTT `batch_size` value. |
| `transcription_engine_options["pipeline"]` | Extra keyword arguments for `ASRInferencePipeline`. |
| `transcription_engine_options["transcribe"]` | Extra keyword arguments for `pipeline.transcribe(...)`. |
| `transcription_engine_options["max_audio_seconds"]` | In-memory audio duration guard. Defaults to 39.9 seconds. Set to `false` to disable. |
| `transcription_engine_options["language"]` / `["lang"]` | Language used when no `language` parameter is set. |
| `transcription_engine_options["language_aliases"]` | Extra short-code to Omnilingual language ID mappings. |

## FastAPI Recipe

Run the FastAPI server from WSL2 when using Omnilingual:

```bash
PYTHONPATH=. python example_fastapi_server/server.py \
  --host 0.0.0.0 \
  --port 8010 \
  --engine omnilingual_asr \
  --model omniASR_CTC_300M_v2 \
  --realtime-engine omnilingual_asr \
  --realtime-model omniASR_CTC_300M_v2 \
  --use-main-model-for-realtime \
  --device cuda \
  --compute-type float16 \
  --language eng_Latn \
  --engine-options '{"batch_size":1,"sample_rate":16000}'
```

Open `http://localhost:8010/` from a Windows browser. WSL2 forwards localhost
for the default setup; if your WSL networking is customized, use the WSL host
IP instead. After the 300M model works, try `omniASR_CTC_1B_v2` if your GPU
has enough VRAM.

## Tests

Fast contract tests use mocked Omnilingual runtime objects and do not require
the optional dependency:

```powershell
python -m unittest -v tests.unit.test_omnilingual_asr_engine
```

Real model smoke tests should run inside WSL2/Linux with the Omnilingual
runtime installed. Start with `omniASR_CTC_300M_v2`; larger models are mostly
model-card plumbing unless you have enough GPU memory to validate them.

## Troubleshooting

- Missing dependency errors mean `omnilingual_asr`, PyTorch, `fairseq2`, or
  `fairseq2n` is not importable in the active Linux/WSL environment.
- Native Windows installs fail because `fairseq2n` currently has no Windows
  wheel. Run the Omnilingual runtime in WSL2 or Linux.
- If CUDA memory is exhausted, use `omniASR_CTC_300M_v2`, enable one shared
  model with `use_main_model_for_realtime=True`, or reduce concurrent server
  sessions.
- If LLM output quality is poor or empty, set an Omnilingual language code such
  as `eng_Latn`. CTC models ignore language.
- The tested non-streaming Omnilingual pipeline requires audio shorter than
  40 seconds; keep realtime/final utterances below that limit.
