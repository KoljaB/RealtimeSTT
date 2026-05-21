# Testing

RealtimeSTT has two kinds of tests:

- Fast unit and contract tests that run without downloading speech models.
- Opt-in golden transcription tests that run real models against small audio fixtures.

Audio fixtures live in `tests/unit/audio` and are based on public-domain LJ Speech samples.

Manual demos, regression harnesses, and legacy experiments directly under
`tests/` are documented separately in [test-scripts.md](test-scripts.md).

## Fast Unit Tests

From the repository root:

```powershell
python -m unittest -v tests.unit.test_audio_fixtures tests.unit.test_whisper_cpp_engine tests.unit.test_openai_whisper_engine tests.unit.test_additional_transcription_engines tests.unit.test_cohere_transcribe_engine tests.unit.test_granite_speech_engine tests.unit.test_moonshine_engine tests.unit.test_sherpa_onnx_engine tests.unit.test_kroko_onnx_engine tests.unit.test_omnilingual_asr_engine tests.unit.test_realtime_streaming_transcription tests.unit.test_fastapi_server_protocol tests.unit.test_fastapi_server_multi_user
```

Use the Python executable from your active virtual environment.

Without golden-test environment variables, the slow model tests are skipped intentionally. A result with skipped tests means the fast tests passed and the opt-in model tests did not run.

## Golden Tests

Golden tests download or load real speech models and compare a fixture transcription against expected text. They are disabled by default because they are slower, may require network access on the first run, and can need extra platform permissions.

Run the faster-whisper golden test:

```powershell
$env:REALTIMESTT_RUN_GOLDEN_TRANSCRIPTION = "1"
$env:REALTIMESTT_TEST_MODEL = "tiny"
$env:REALTIMESTT_TEST_DEVICE = "cpu"
$env:REALTIMESTT_TEST_COMPUTE_TYPE = "int8"

python -m unittest -v tests.unit.test_audio_fixtures.GoldenTranscriptionTests
```

Run the whisper.cpp golden test:

```powershell
python -m pip install "RealtimeSTT[whisper-cpp]"

$env:REALTIMESTT_RUN_WHISPER_CPP = "1"
$env:REALTIMESTT_WHISPER_CPP_MODEL = "tiny.en"
$env:REALTIMESTT_WHISPER_CPP_MODEL_DIR = Join-Path (Get-Location) "test-model-cache\pywhispercpp"

python -m unittest -v tests.unit.test_whisper_cpp_engine.WhisperCppGoldenTranscriptionTests
```

Run the OpenAI Whisper golden test:

```powershell
python -m pip install openai-whisper

$env:REALTIMESTT_RUN_OPENAI_WHISPER = "1"
$env:REALTIMESTT_OPENAI_WHISPER_MODEL = "tiny.en"
$env:REALTIMESTT_OPENAI_WHISPER_DEVICE = "cpu"
$env:REALTIMESTT_OPENAI_WHISPER_COMPUTE_TYPE = "float32"
$env:REALTIMESTT_OPENAI_WHISPER_MODEL_DIR = Join-Path (Get-Location) "test-model-cache\openai-whisper"

python -m unittest -v tests.unit.test_openai_whisper_engine.OpenAIWhisperGoldenTranscriptionTests
```

Run opt-in smoke tests for the newer model-family engines:

```powershell
# Pick only the engine you want to validate.
$env:REALTIMESTT_RUN_PARAKEET = "1"
$env:REALTIMESTT_RUN_COHERE_TRANSCRIBE = "1"
$env:REALTIMESTT_RUN_GRANITE_SPEECH = "1"
$env:REALTIMESTT_RUN_QWEN3_ASR = "1"
$env:REALTIMESTT_RUN_MOONSHINE = "1"
$env:REALTIMESTT_HF_MODEL_DIR = Join-Path (Get-Location) "test-model-cache\hf"

python -m unittest -v tests.unit.test_additional_transcription_engines.AdditionalEngineGoldenTranscriptionTests
```

These smoke tests need `numpy` plus each backend's optional dependencies and model access. Cohere currently requires accepting gated Hugging Face access before weights can be downloaded.

Run the sherpa-onnx CPU INT8 contract tests:

```powershell
python -m pip install sherpa-onnx

python -m unittest -v tests.unit.test_sherpa_onnx_engine
```

The fast sherpa-onnx tests mock the runtime and do not download models. For a real RTF comparison, download and extract the sherpa-onnx model bundles under `test-model-cache\sherpa-onnx`, then run either opt-in smoke test:

```powershell
$env:REALTIMESTT_RUN_SHERPA_ONNX_PARAKEET = "1"
$env:REALTIMESTT_SHERPA_ONNX_PARAKEET_MODEL = Join-Path (Get-Location) "test-model-cache\sherpa-onnx\sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
$env:REALTIMESTT_SHERPA_ONNX_NUM_THREADS = "2"

python -m unittest -v tests.unit.test_sherpa_onnx_engine.SherpaOnnxGoldenTranscriptionTests.test_transcribes_fixture_with_real_sherpa_parakeet_backend
```

```powershell
$env:REALTIMESTT_RUN_SHERPA_ONNX_MOONSHINE = "1"
$env:REALTIMESTT_SHERPA_ONNX_MOONSHINE_MODEL = Join-Path (Get-Location) "test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-tiny-en-int8"
$env:REALTIMESTT_SHERPA_ONNX_NUM_THREADS = "1"

python -m unittest -v tests.unit.test_sherpa_onnx_engine.SherpaOnnxGoldenTranscriptionTests.test_transcribes_fixture_with_real_sherpa_moonshine_backend
```

Run both golden paths together:

```powershell
$env:REALTIMESTT_RUN_GOLDEN_TRANSCRIPTION = "1"
$env:REALTIMESTT_TEST_MODEL = "tiny"
$env:REALTIMESTT_TEST_DEVICE = "cpu"
$env:REALTIMESTT_TEST_COMPUTE_TYPE = "int8"
$env:REALTIMESTT_RUN_WHISPER_CPP = "1"
$env:REALTIMESTT_WHISPER_CPP_MODEL = "tiny.en"
$env:REALTIMESTT_WHISPER_CPP_MODEL_DIR = Join-Path (Get-Location) "test-model-cache\pywhispercpp"

python -m unittest -v tests.unit.test_audio_fixtures tests.unit.test_whisper_cpp_engine
```

The `test-model-cache/` directory is ignored by Git and can safely hold downloaded local test models.

Run the Kroko-ONNX contract tests:

```powershell
python -m unittest -v tests.unit.test_kroko_onnx_engine
python -m unittest -v tests.unit.test_realtime_streaming_transcription
```

The fast Kroko tests use fake runtime objects and do not install or import
Kroko-ONNX. For a real-model Community smoke test, install Kroko-ONNX in the
active environment and opt in:

```powershell
$env:REALTIMESTT_RUN_KROKO_ONNX = "1"
$env:REALTIMESTT_KROKO_ONNX_MODEL = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
$env:REALTIMESTT_KROKO_ONNX_PROVIDER = "cpu"
$env:REALTIMESTT_KROKO_ONNX_NUM_THREADS = "1"

python -m unittest -v tests.unit.test_kroko_onnx_engine.KrokoOnnxGoldenTranscriptionTests
```

`REALTIMESTT_KROKO_ONNX_KEY`, `KROKO_ONNX_KEY`, or `KROKO_KEY` can be set for
licensed Pro models. Do not store keys in command history, docs, generated
reports, or committed files.

Run the Omnilingual ASR contract tests:

```powershell
python -m unittest -v tests.unit.test_omnilingual_asr_engine
```

These tests use fake Omnilingual runtime objects and do not install or import
Meta's Omnilingual ASR package. Real model smoke tests should run from Linux or
WSL2; see [engines/omnilingual-asr.md](engines/omnilingual-asr.md).

## FastAPI Multi-User Load Test

The FastAPI browser server has fast fake-scheduler tests for session isolation,
fair scheduling, realtime coalescing, stale realtime discard, admission limits,
and clear/reset behavior:

```powershell
python -m unittest -v tests.unit.test_fastapi_server_protocol tests.unit.test_fastapi_server_multi_user
```

The opt-in real-engine load/performance test streams
`tests\unit\audio\asr-reference.wav` through multiple sessions in parallel,
compares final text with `asr-reference.expected_sentences.json`, checks
per-session latency skew, and prints a timing report. The report includes first
realtime latency, first final latency, final latency after audio upload ended,
first recording/VAD-start timing, realtime/final event cadence, WER, scheduler
p50/p95 latency, coalescing, and drop counters:

```powershell
$env:REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF = "1"
$env:REALTIMESTT_FASTAPI_ASR_CLIENTS = "2"
$env:REALTIMESTT_FASTAPI_ASR_ENGINE = "faster_whisper"
$env:REALTIMESTT_FASTAPI_ASR_MODEL = "small.en"

python -m unittest -v tests.unit.test_fastapi_server_multi_user_asr_integration
```

`REALTIMESTT_RUN_FASTAPI_MULTI_USER_ASR=1` runs the same test; the `PERF`
name is just the clearer switch when you are measuring latency.

To save the same report as JSON for comparing runs:

```powershell
$env:REALTIMESTT_FASTAPI_ASR_METRICS_JSON = "test-results\fastapi-multi-user-perf.json"
```

The test accepts backend and scheduler tuning variables so it can mirror a
manual server command. For example, sherpa-onnx Moonshine on CPU:

```powershell
$env:REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF = "1"
$env:REALTIMESTT_FASTAPI_ASR_CLIENTS = "2"
$env:REALTIMESTT_FASTAPI_ASR_ENGINE = "sherpa_onnx_moonshine"
$env:REALTIMESTT_FASTAPI_ASR_MODEL = "sherpa-onnx-moonshine-base-en-int8"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE = "sherpa_onnx_moonshine"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_MODEL = "sherpa-onnx-moonshine-tiny-en-int8"
$env:REALTIMESTT_FASTAPI_ASR_DEVICE = "cpu"
$env:REALTIMESTT_FASTAPI_ASR_DOWNLOAD_ROOT = "test-model-cache\sherpa-onnx"
$env:REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS = '{"num_threads":4,"provider":"cpu"}'
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE_OPTIONS = '{"num_threads":2,"provider":"cpu"}'
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_PROCESSING_PAUSE = "0.8"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_USE_SYLLABLE_BOUNDARIES = "1"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_BOUNDARY_DETECTOR_SENSITIVITY = "0.6"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_BOUNDARY_FOLLOWUP_DELAYS = "0.1,0.2,0.4"

python -m unittest -v tests.unit.test_fastapi_server_multi_user_asr_integration.FastAPIMultiUserRealEngineASRTests
```

The engine option variables also accept cmd-friendly `key=value` lists such as
`num_threads=4,provider=cpu`, which avoids Windows quote escaping issues.

For the common Windows cmd.exe sherpa-onnx Moonshine run, the same defaults are
available as a helper script:

```cmd
example_fastapi_server\run_multi_user_perf.cmd
```

Override just the values you want before running it:

```cmd
set REALTIMESTT_FASTAPI_ASR_CLIENTS=8
set REALTIMESTT_FASTAPI_ASR_METRICS_JSON=test-results\fastapi-8-user-perf.json
example_fastapi_server\run_multi_user_perf.cmd
```

Use the `REALTIMESTT_FASTAPI_ASR_*` environment variables documented in
`example_fastapi_server\README.md` to select CPU sherpa-onnx Moonshine,
whisper.cpp, Parakeet, or another installed backend.

Kroko-ONNX uses the same generic FastAPI harness path:

```powershell
$env:REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF = "1"
$env:REALTIMESTT_FASTAPI_ASR_CLIENTS = "2"
$env:REALTIMESTT_FASTAPI_ASR_ENGINE = "kroko_onnx"
$env:REALTIMESTT_FASTAPI_ASR_MODEL = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE = "kroko_onnx"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_MODEL = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
$env:REALTIMESTT_FASTAPI_ASR_DEVICE = "cpu"
$env:REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS = "provider=cpu,num_threads=2"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE_OPTIONS = "provider=cpu,num_threads=1"
$env:REALTIMESTT_FASTAPI_ASR_METRICS_JSON = "test-results\kroko-onnx-fastapi-cpu-2clients.json"

python -m unittest -v tests.unit.test_fastapi_server_multi_user_asr_integration.FastAPIMultiUserRealEngineASRTests
```

For CUDA, switch `REALTIMESTT_FASTAPI_ASR_DEVICE` and both provider options to
`cuda` after confirming the installed Kroko-ONNX build has CUDA provider
support.

## Windows Notes

Some recorder tests use multiprocessing pipes. On Windows, those may require running outside a restricted sandbox. If a golden test fails with `PermissionError: [WinError 5] Zugriff verweigert` while creating multiprocessing queues or pipes, rerun it in a normal terminal with the same environment variables.

The Parakeet/NeMo and Qwen vLLM paths are Linux-oriented. For real-model validation on a Windows workstation, use WSL2 with a CUDA-enabled Linux environment, install the optional backend dependencies there, mount or clone the repository inside the WSL filesystem, and run the same `python -m unittest` commands from that environment. Keep the default Windows unit run focused on mocked contract tests so CI and local checks do not need GPU drivers, gated model access, or multi-gigabyte downloads.

## Adding Engine Tests

For new transcription engines, add fast contract tests first:

- Factory selection and lazy import behavior.
- Missing optional dependency error messages.
- Parameter mapping from `TranscriptionEngineConfig` to the backend binding.
- Audio validation and normalization behavior.
- Conversion from backend segments into `TranscriptionResult`.

Only add a real-model golden test after the fast contract tests are stable, and keep it opt-in with an environment variable.
