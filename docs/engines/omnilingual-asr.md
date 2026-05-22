# Meta Omnilingual ASR

`omnilingual_asr` uses Meta's Omnilingual ASR package through
`ASRInferencePipeline`. The adapter is lazy-loaded, so normal RealtimeSTT
imports and installs do not require the Omnilingual runtime until this engine
is selected.

Current install target: Linux or WSL2 with Python 3.11.x. Native Windows cannot
run the Omnilingual runtime because `fairseq2n` has no Windows wheel, and
Python 3.12.x currently cannot resolve `omnilingual-asr>=0.2.0` from PyPI
because the upstream package metadata excludes normal 3.12 patch releases.

## Engine Names

- `omnilingual_asr`
- `omnilingual`
- `meta_omnilingual_asr`
- `omni_asr`

Hyphenated CLI forms such as `omnilingual-asr`,
`meta-omnilingual-asr`, and `omni-asr` are accepted by the generic engine-name
normalization.

## Install

Use a Linux or WSL2 environment with Python 3.11.x. The clean alias is
`omnilingual`:

```bash
python -m pip install "RealtimeSTT[omnilingual]"
```

`omnilingual-asr` and `meta-omnilingual-asr` are equivalent aliases if you
prefer the explicit package-family name.

The Omnilingual extra requires `omnilingual-asr>=0.2.0` and constrains
matching `torch==2.8.0` / `torchaudio==2.8.0` builds on Linux/WSL2. If you
install a CUDA-enabled PyTorch stack separately, keep `torch` and `torchaudio`
on matching releases. A mismatched pair can pass `python -m pip check` but
fail while importing `omnilingual_asr`, for example with a missing
`libcudart.so` shared-library error.

Do not use native Windows or Python 3.12.x for the Omnilingual runtime. Native
Windows installs intentionally skip the runtime dependency, and
`omnilingual-asr==0.2.0` currently declares `Requires-Python: <=3.12,>=3.10`,
which makes normal Python 3.12 patch releases fail dependency resolution.

When working from a source checkout:

```bash
python -m pip install -e ".[omnilingual]"
```

The extra is guarded with a non-Windows platform marker. On Windows, create or
reuse a Python 3.11.x WSL2 environment and run the Omnilingual process there.

## Basic Use

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="omnilingual_asr",
    model="omniASR_CTC_1B_v2",
    device="cuda",
    compute_type="float16",
)
```

If `model` is left at RealtimeSTT's default Whisper value, the adapter selects
`omniASR_CTC_1B_v2`. That v2 model card must be present in the installed
`omnilingual-asr` package.

## Models

Start with `omniASR_CTC_1B_v2` when the installed `omnilingual-asr` package
ships the v2 cards. The RealtimeSTT extra requires `omnilingual-asr>=0.2.0`
because `omnilingual-asr==0.1.0` only knew older non-v2 cards.
`omniASR_CTC_300M_v2` is a smaller card, but it is not the recommended
RealtimeSTT default because it did not pass validation for this integration.

Do not silently fall back to older non-v2 cards just because a v2 card is
unknown. Unknown v2 cards mean the installed `omnilingual-asr` dependency is
older than these docs expect.

Common current v2 model card names include:

- `omniASR_CTC_300M_v2`
- `omniASR_CTC_1B_v2`
- `omniASR_CTC_3B_v2`
- `omniASR_CTC_7B_v2`
- `omniASR_LLM_300M_v2`
- `omniASR_LLM_1B_v2`
- `omniASR_LLM_3B_v2`
- `omniASR_LLM_7B_v2`
- `omniASR_LLM_7B_ZS_v2`
- `omniASR_LLM_Unlimited_300M_v2`
- `omniASR_LLM_Unlimited_1B_v2`
- `omniASR_LLM_Unlimited_3B_v2`
- `omniASR_LLM_Unlimited_7B_v2`

Older non-v2 cards such as `omniASR_CTC_300M`, `omniASR_CTC_1B`,
`omniASR_CTC_3B`, and `omniASR_CTC_7B` may exist in older Omnilingual
packages, but they are not a release-quality fallback unless they pass your
language fixture.

`omniASR_CTC_1B_v2` is the recommended default for realtime-oriented local
testing. It needs more VRAM and startup time than the 300M card, but it is the
validated default for this integration.

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

## File Smoke Test

A pip-installed environment can test the engine without a source checkout by
constructing the engine directly and passing a short WAV file:

```python
from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig
from RealtimeSTT.transcription_engines.factory import create_transcription_engine

audio = "path/to/short-english.wav"
config = TranscriptionEngineConfig(
    model="omniASR_CTC_1B_v2",
    device="cuda",
    compute_type="float16",
    engine_options={"batch_size": 1, "sample_rate": 16000},
)

engine = create_transcription_engine("omnilingual_asr", config)
result = engine.transcribe(audio, language="en")
print(result.text)
```

Use a short file, ideally under 40 seconds. If the expected English text is not
recognizable with `omniASR_CTC_1B_v2`, treat the install as a runtime/model
blocker before documenting or shipping a release recommendation.

To keep model downloads isolated during testing, set cache-related environment
variables before running the smoke:

```bash
HOME="$PWD/.home-omnilingual-smoke" \
XDG_CACHE_HOME="$PWD/.home-omnilingual-smoke/.cache" \
python smoke_omnilingual.py
```

Omnilingual model assets are large. The 1B-family smoke can download several
GiB of model/cache files.

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

For realtime previews, use the validated CTC model with one shared model lane:

```python
recorder = AudioToTextRecorder(
    transcription_engine="omnilingual_asr",
    model="omniASR_CTC_1B_v2",
    enable_realtime_transcription=True,
    realtime_transcription_engine="omnilingual_asr",
    realtime_model_type="omniASR_CTC_1B_v2",
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

Run the FastAPI server from a source checkout in WSL2/Linux when using
Omnilingual:

```bash
PYTHONPATH=. python example_fastapi_server/server.py \
  --host 0.0.0.0 \
  --port 8010 \
  --engine omnilingual_asr \
  --model omniASR_CTC_1B_v2 \
  --realtime-engine omnilingual_asr \
  --realtime-model omniASR_CTC_1B_v2 \
  --use-main-model-for-realtime \
  --device cuda \
  --compute-type float16 \
  --realtime-processing-pause 0.05 \
  --engine-options '{"batch_size":1,"sample_rate":16000}'
```

Open `http://localhost:8010/` from a Windows browser. WSL2 forwards localhost
for the default setup; if your WSL networking is customized, use the WSL host
IP instead.

This recipe targets `example_fastapi_server/server.py`, not the installed
`stt-server` console script. Check `stt-server --help` separately for the
installed CLI's supported options.

## Tests

Fast contract tests use mocked Omnilingual runtime objects and do not require
the optional dependency, but they are source-checkout tests:

```powershell
python -m unittest -v tests.unit.test_omnilingual_asr_engine
```

That command is not expected to work from a clean pip install unless the source
tree and tests are present.

Real model smoke tests should run inside WSL2/Linux with the Omnilingual
runtime installed. Start with `omniASR_CTC_1B_v2`.

## Troubleshooting

- Missing dependency errors mean `omnilingual_asr`, PyTorch, `fairseq2`, or
  `fairseq2n` is not importable in the active Linux/WSL environment.
- Native Windows installs skip the Omnilingual runtime because `fairseq2n`
  currently has no Windows wheel. Run the Omnilingual runtime in WSL2 or Linux.
- Python 3.12.x currently fails Omnilingual dependency resolution because
  upstream `omnilingual-asr==0.2.0` declares `Requires-Python: <=3.12,>=3.10`.
  Use Python 3.11.x until upstream metadata changes.
- CUDA shared-library import errors such as missing `libcudart.so` often mean
  `torch` and `torchaudio` were resolved from incompatible builds. Install
  matching versions in the same environment.
- `ModelNotKnownError` for a `_v2` model card means the installed
  `omnilingual-asr` package does not ship that card. Upgrade or pin the
  Omnilingual package to a release that includes the documented card.
- If CUDA memory is exhausted, enable one shared model with
  `use_main_model_for_realtime=True`, reduce concurrent server sessions, or
  evaluate a smaller validated model before changing the default.
- If LLM output quality is poor or empty, set an Omnilingual language code such
  as `eng_Latn`. CTC models ignore language.
- The tested non-streaming Omnilingual pipeline requires audio shorter than
  40 seconds; keep realtime/final utterances below that limit.
