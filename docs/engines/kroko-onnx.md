# Kroko-ONNX

`kroko_onnx` uses the optional
[kroko-ai/kroko-onnx](https://github.com/kroko-ai/kroko-onnx) runtime with
Kroko/Banafo streaming `.data` models. The adapter is lazy-loaded, so normal
RealtimeSTT installs and tests do not require Kroko-ONNX.

## Engine Names

- `kroko_onnx`
- `kroko`
- `banafo_kroko`

Hyphenated CLI forms such as `kroko-onnx` and `banafo-kroko` are accepted by
the generic engine-name normalization.

## Install

Kroko-ONNX is not installed by default. RealtimeSTT exposes a small builder
helper:

```bash
python -m pip install "RealtimeSTT[kroko-builder]"
stt-install-kroko --build
```

The helper uses Kroko's `cross-platform-builds` branch. On Windows it builds a
CPython 3.12 `win_amd64` wheel with Docker Desktop, then installs that wheel.
On Linux it patches the checkout and installs from source. Add `--skip-install`
to build without installing into the active Python environment.

Windows requirements:

- Python 3.12 x64
- Git
- Docker Desktop running with the WSL2 backend

Linux requirements:

- Git
- CMake
- A working C/C++ build toolchain

Use `--variant pro` when you need licensed Pro models:

```bash
stt-install-kroko --build --variant pro
```

The `free` variant is for public Community models. The `pro` variant is for
licensed Pro/private models and may need network access for Kroko's license
check.

## Models

Public Community models are available from
[Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR). RealtimeSTT can
download known public Community `.data` files automatically when
`auto_download_model` is enabled. Bare Kroko filenames are cached under
`~/.cache/realtimestt/kroko-onnx` unless `download_root` points somewhere else.

You can also pre-download models into a project-local ignored cache:

```powershell
New-Item -ItemType Directory -Path test-model-cache\kroko-onnx -Force
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Banafo/Kroko-ASR', filename='Kroko-EN-Community-64-L-Streaming-001.data', local_dir='test-model-cache/kroko-onnx')"
```

Pro/private models are not assumed to be public. Pass an existing `.data` path,
`model_download_url`, or explicit Hugging Face repo/token options. Pass Pro keys
through environment/config or CLI options only; do not commit keys.

## Python Usage

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="kroko_onnx",
    model="Kroko-EN-Community-64-L-Streaming-001.data",
    device="cpu",
    language="en",
    transcription_engine_options={
        "provider": "cpu",
        "num_threads": 2,
    },
)
```

For realtime previews, Kroko uses a persistent native stream and receives only
new audio frames. Final transcription still uses one full-utterance call.

```python
recorder = AudioToTextRecorder(
    transcription_engine="kroko_onnx",
    model="Kroko-EN-Community-128-L-Streaming-001.data",
    enable_realtime_transcription=True,
    realtime_transcription_engine="kroko_onnx",
    realtime_model_type="Kroko-EN-Pro-16-L-Streaming-001.data",
    realtime_transcription_engine_options={
        "provider": "cpu",
        "num_threads": 4,
        "key": "...",
        "suppress_native_output": True,
    },
)
```

`Pro-16-L` is the fastest measured partial-cadence option when Pro access is
available. Kroko model names encode native streaming cadence as
`number * 20 ms`, so `16` is about `320 ms`, `32` is about `640 ms`, and `64`
is about `1280 ms`. Feeding smaller chunks does not force Kroko to emit
partials faster; it only avoids extra scheduling and buffer latency.

## Options

| Option | Meaning |
| --- | --- |
| `model_path` | Explicit `.data` model file. Overrides `model`. |
| `model_dir` | Directory containing a single `.data` file, or the default English Community filename. |
| `model_filename` | File name to use inside `model_dir`. |
| `auto_download_model` / `download_model` | Download missing public Community model files. Defaults to `True`. |
| `model_download_url` | Direct download URL for a missing `.data` file. Useful for Pro/private models. |
| `model_repo_id`, `model_revision`, `hf_token` | Optional Hugging Face download settings. |
| `key` | License key for Pro models. |
| `referralcode` | Optional Kroko referral code. |
| `provider` | `cpu`, `cuda`, or `coreml`. Defaults from `device`. |
| `num_threads` | Runtime thread count. Defaults to `1`. |
| `sample_rate` | Kroko recognizer sample rate. Defaults to `16000`. |
| `feature_dim` | Feature dimension. Defaults to `80`. |
| `decoding_method` | `greedy_search` or `modified_beam_search`. |
| `max_active_paths` | Beam paths for modified beam search. |
| `hotwords_file`, `hotwords_score` | Optional hotword biasing inputs. |
| `blank_penalty` | Blank-symbol penalty during decoding. |
| `enable_endpoint_detection` | Enables Kroko endpoint detection. |
| `rule1_min_trailing_silence`, `rule2_min_trailing_silence`, `rule3_min_utterance_length` | Endpoint rule values. |
| `tail_padding_seconds` / `finalization_padding_seconds` | Final silence padding before one-shot decoding. Defaults to `auto`, inferred from model cadence plus a small margin. |
| `suppress_native_output` | Redirect Kroko native stdout/stderr during recognizer calls and set `KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT=1`. Aliases: `suppress_output`, `quiet`, `silent`. |
| `recognizer` | Extra dictionary merged into `OnlineRecognizer.from_transducer(...)`. |

`suppress_native_output` is a Python-side mitigation plus an environment flag.
Reliable suppression of asynchronous Pro license refresh messages such as
`Remaining seconds updated: ...` requires a Kroko wheel built with RealtimeSTT's
native patch. Older/unpatched Kroko wheels may still print background license
messages.

## FastAPI Example

```powershell
$model = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
python example_fastapi_server\server.py `
  --host 0.0.0.0 `
  --port 8010 `
  --engine kroko_onnx `
  --model $model `
  --realtime-engine kroko_onnx `
  --realtime-model $model `
  --device cpu `
  --language en `
  --engine-options '{"provider":"cpu","num_threads":2}' `
  --realtime-engine-options '{"provider":"cpu","num_threads":1}'
```

## Tests

Fast contract tests use fake Kroko runtime objects and do not require the
optional dependency:

```powershell
python -m unittest -v tests.unit.test_kroko_onnx_engine
```

Opt-in real Community smoke test:

```powershell
$env:REALTIMESTT_RUN_KROKO_ONNX = "1"
$env:REALTIMESTT_KROKO_ONNX_MODEL = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
$env:REALTIMESTT_KROKO_ONNX_PROVIDER = "cpu"
$env:REALTIMESTT_KROKO_ONNX_NUM_THREADS = "1"
python -m unittest -v tests.unit.test_kroko_onnx_engine.KrokoOnnxGoldenTranscriptionTests
```

`REALTIMESTT_KROKO_ONNX_KEY`, `KROKO_ONNX_KEY`, or `KROKO_KEY` may be used for
licensed Pro-only checks. Keep those values out of committed files, shell logs,
and generated reports.

## Troubleshooting

- Missing dependency errors mean `kroko_onnx` is not importable in the active
  environment. Install Kroko-ONNX in that same environment.
- Missing model errors name the exact `.data` file path RealtimeSTT tried.
- Free Kroko wheels cannot load Pro `.data` models; the native error can look
  like a payload parsing or block-size mismatch.
- CUDA runs require both CUDA-capable hardware and a Kroko-ONNX build with CUDA
  provider support.
- On Windows, prefer the helper's `cross-platform-builds` wheel workflow over a
  direct native source build.
