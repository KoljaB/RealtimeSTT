# Kroko-ONNX

`kroko_onnx` uses the optional
[kroko-ai/kroko-onnx](https://github.com/kroko-ai/kroko-onnx) runtime with
Kroko/Banafo streaming `.data` models. It is useful when you want low-latency
streaming previews from Kroko models while keeping the default RealtimeSTT
install free of Kroko-specific native dependencies.

The adapter is lazy-loaded. A normal RealtimeSTT install can import and use
other engines without `kroko_onnx` installed.

## Engine Names

- `kroko_onnx`
- `kroko`
- `banafo_kroko`

Hyphenated CLI forms such as `kroko-onnx` and `banafo-kroko` are accepted by
the generic engine-name normalization.

## Install

Kroko-ONNX is not installed by default. Install RealtimeSTT's builder helper,
then build Kroko-ONNX for the active Python environment:

```bash
python -m pip install "RealtimeSTT[kroko-builder]"
stt-install-kroko --build
```

The helper uses Kroko's `cross-platform-builds` branch. On Windows it builds a
CPython 3.12 `win_amd64` wheel with Docker Desktop, then installs that wheel.
On Linux it patches the checkout and installs from source. Use `--skip-install`
when you only want to produce the package artifact.

Windows requirements:

- Python 3.12 x64
- Git
- Docker Desktop running with the WSL2 backend

Linux requirements:

- Git
- CMake
- A working C/C++ build toolchain

For licensed Pro models, build the Pro variant:

```bash
stt-install-kroko --build --variant pro
```

The `free` variant is for public Community models. The `pro` variant is for
licensed Pro/private models and may need network access for Kroko's license
check.

## Model Behavior

Known public Community model files are available from
[Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR). RealtimeSTT can
download known Community `.data` files automatically when
`auto_download_model` is enabled, which is the default.

Bare Kroko filenames are cached under `~/.cache/realtimestt/kroko-onnx` unless
`download_root` points somewhere else:

```python
recorder = AudioToTextRecorder(
    transcription_engine="kroko_onnx",
    model="Kroko-EN-Community-64-L-Streaming-001.data",
    download_root="models/kroko-onnx",
)
```

You can also pre-download models into an ignored project-local cache:

```powershell
New-Item -ItemType Directory -Path test-model-cache\kroko-onnx -Force
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Banafo/Kroko-ASR', filename='Kroko-EN-Community-64-L-Streaming-001.data', local_dir='test-model-cache/kroko-onnx')"
```

Pro/private models are not assumed to be public. Pass an existing `.data` path,
`model_download_url`, or explicit Hugging Face repo/token options. Keep license
keys and tokens in environment variables, trusted local config, or CLI/runtime
options. Do not commit them.

## Basic Use

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

For an existing model file:

```python
recorder = AudioToTextRecorder(
    transcription_engine="kroko_onnx",
    model="models/kroko-onnx/Kroko-EN-Community-64-L-Streaming-001.data",
    device="cpu",
    language="en",
)
```

## Realtime Suggestions

Kroko supports a persistent streaming session for realtime previews. RealtimeSTT
feeds only new audio frames to the realtime engine, while final transcription
still uses a full-utterance pass.

```python
recorder = AudioToTextRecorder(
    transcription_engine="kroko_onnx",
    model="Kroko-EN-Community-128-L-Streaming-001.data",
    enable_realtime_transcription=True,
    realtime_transcription_engine="kroko_onnx",
    realtime_model_type="Kroko-EN-Community-64-L-Streaming-001.data",
    realtime_processing_pause=0.05,
    realtime_transcription_engine_options={
        "provider": "cpu",
        "num_threads": 1,
        "suppress_native_output": True,
    },
)
```

For Pro access, a small realtime model can improve visible partial cadence:

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

Kroko model names encode the native streaming cadence as `number * 20 ms`.
For example, `16` is about `320 ms`, `32` is about `640 ms`, and `64` is about
`1280 ms`. Feeding smaller chunks does not force the native model to emit
partials faster; it only avoids extra scheduling and buffering delay.

## Common Options

| Option | Meaning |
| --- | --- |
| `model_path` | Explicit `.data` model file. Overrides `model`. |
| `model_dir` | Directory containing a single `.data` file, or the default Community filename. |
| `model_filename` | File name to use inside `model_dir`. |
| `auto_download_model`, `download_model` | Download missing public Community models. Defaults to `True`. |
| `model_download_url` | Direct URL for a missing `.data` file. Useful for private model hosting. |
| `model_repo_id`, `model_revision`, `hf_token` | Hugging Face download settings. |
| `provider` | `cpu`, `cuda`, or `coreml`. Defaults from `device`. |
| `num_threads` | Kroko runtime thread count. Defaults to `1`. |
| `sample_rate` | Input sample rate for Kroko. Defaults to `16000`. |
| `key` | License key for Pro models. |
| `referralcode` | Optional Kroko referral code. |
| `decoding_method` | Kroko decoding method, commonly `greedy_search` or `modified_beam_search`. |
| `max_active_paths` | Beam paths for modified beam search. |
| `hotwords_file`, `hotwords_score` | Optional hotword biasing inputs. |
| `blank_penalty` | Blank-symbol penalty during decoding. |
| `tail_padding_seconds`, `finalization_padding_seconds` | Final silence padding before one-shot decoding. Defaults to `auto`. |
| `suppress_native_output` | Redirects native stdout/stderr during recognizer calls and sets `KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT=1`. |
| `recognizer` | Extra dictionary merged into `OnlineRecognizer.from_transducer(...)`. |

`suppress_native_output` also accepts the aliases `suppress_output`, `quiet`,
and `silent`. Reliable suppression of asynchronous Pro license refresh messages
requires a Kroko wheel built with RealtimeSTT's native quiet-output patch.

## FastAPI Recipe

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
optional native dependency:

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

- If import fails, install Kroko-ONNX in the same environment that runs
  RealtimeSTT.
- Missing model errors name the exact `.data` path RealtimeSTT tried.
- Free Kroko wheels cannot load Pro `.data` models; the native error can look
  like a payload parsing or block-size mismatch.
- CUDA runs require CUDA-capable hardware and a Kroko-ONNX build with CUDA
  provider support.
- On Windows, prefer `stt-install-kroko --build` over a direct native source
  build.
