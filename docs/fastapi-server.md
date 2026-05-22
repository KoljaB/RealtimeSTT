# FastAPI Browser Server

`example_fastapi_server` is the browser streaming reference app for
RealtimeSTT. It serves a local browser UI and exposes a WebSocket endpoint that
streams microphone audio into per-session recorder state machines.

## Install

```bash
python -m venv .venv-fastapi
source .venv-fastapi/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r example_fastapi_server/requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv-fastapi
.\.venv-fastapi\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r example_fastapi_server\requirements.txt
```

Install the optional engine stack you plan to run. See
[transcription-engines.md](transcription-engines.md).

## Run

```bash
python example_fastapi_server/server.py --host 0.0.0.0 --port 8010
```

Open:

```text
http://localhost:8010
```

## Server Overview

The server accepts multiple browser sessions. Each WebSocket receives a
`sessionId`; audio buffers, VAD state, transcript segment ids, clear/reset
commands, realtime text, final text, warnings, and errors are scoped to that
session.

Heavy ASR engines are shared through final and realtime inference lanes instead
of loading one model per browser. Each accepted session owns lightweight
recorder/VAD state and feeds work into the shared scheduler.

The server exposes:

- `GET /`: browser UI.
- `GET /health`: readiness, active sessions/speakers, startup errors, and
  scheduler state.
- `GET /api/config`: public settings, limits, and supported engines.
- `GET /api/metrics`: counters, queue depth, latency, coalescing, drops, and
  worker utilization.
- `WS /ws/transcribe`: browser audio stream and command channel.

## Configuration

Core engine flags:

| Flag | Meaning |
| --- | --- |
| `--engine`, `--transcription-engine` | Final transcription engine. |
| `--model` | Final model name or path. |
| `--realtime-engine`, `--realtime-transcription-engine` | Realtime engine. Defaults to final engine when omitted. |
| `--realtime-model` | Realtime model name or path. |
| `--engine-options` | JSON object passed to final engine. |
| `--realtime-engine-options` | JSON object passed to realtime engine. |
| `--download-root` | Model cache or lookup root. |
| `--device` | `cuda` or `cpu`. |
| `--compute-type` | Engine precision/quantization hint. |
| `--language` | Language code. |
| `--use-main-model-for-realtime` | Use one shared model lane for final and realtime work. |

VAD and transcription timing flags:

| Flag | Meaning |
| --- | --- |
| `--min-length-of-recording` | Minimum recording length in seconds. |
| `--min-gap-between-recordings` | Minimum gap between recordings. |
| `--post-speech-silence-duration` | Silence required before finalizing an utterance. |
| `--silero-sensitivity` | Silero VAD sensitivity. |
| `--webrtc-sensitivity` | WebRTC VAD aggressiveness. |
| `--early-transcription-on-silence` | Starts speculative final transcription during silence. |
| `--pre-recording-buffer-duration` | Per-session pre-roll duration. |
| `--realtime-processing-pause` | Fixed realtime update cadence. |
| `--realtime-use-syllable-boundaries` | Enables acoustic boundary scheduling. |
| `--realtime-boundary-detector-sensitivity` | Boundary detector sensitivity. |
| `--realtime-boundary-followup-delays` | Comma-separated follow-up realtime delays. |

Wake word flags:

| Flag | Meaning |
| --- | --- |
| `--wakeword-backend` | Wake word backend passed to `AudioToTextRecorder`, for example `pvporcupine` or `openwakeword`. |
| `--wake-words` | Comma-separated wake words or model names for the selected backend. |
| `--wake-words-sensitivity` | Wake word detection sensitivity. |
| `--wake-word-activation-delay` | Delay before wake word mode becomes active. |
| `--wake-word-timeout` | Time to wait for speech after wake detection before returning to wake wait mode. |
| `--wake-word-buffer-duration` | Wake-word audio removed from the beginning of the recorded segment. |
| `--wake-word-followup-window` | Optional post-recording grace period that keeps the session in Voice mode so follow-up speech can start without repeating the wake word. |
| `--openwakeword-model-paths` | Comma-separated OpenWakeWord model paths. |
| `--openwakeword-inference-framework` | OpenWakeWord inference framework, default `onnx`. |

Capacity and scheduling flags:

| Flag | Meaning |
| --- | --- |
| `--max-sessions` | Maximum accepted browser sessions. |
| `--max-active-speakers` | Maximum concurrent active speakers. |
| `--audio-queue-size` | Per-session input queue size. |
| `--max-audio-packet-bytes` | Maximum binary packet size. |
| `--max-audio-queue-seconds-per-session` | Force-finalizes long continuous recordings. |
| `--max-realtime-queue-age-ms` | Drops stale realtime jobs. |
| `--max-final-queue-depth-per-session` | Limits per-session final backlog. |
| `--max-global-inference-queue-depth` | Global scheduler queue limit. |
| `--realtime-degradation-threshold-ms` | Threshold for degraded realtime scheduling. |
| `--realtime-min-audio-seconds` | Minimum audio duration for realtime jobs. |
| `--realtime-max-audio-seconds` | Maximum audio duration for realtime jobs. |
| `--vad-energy-threshold` | Audio energy gate used by the server. |
| `--no-model-warmup` | Disables model warmup. |

Named tuning profiles are available through `--profile`; explicit flags
override profile defaults.

Runtime settings:

`GET /api/config` includes a `runtimeSettings` contract that separates
`activeSessionSafe`, `newSessionOnly`, and `startupOnly` settings. Runtime
changes are explicit:

```bash
curl -X PATCH http://localhost:8010/api/config \
  -H 'Content-Type: application/json' \
  -d '{"settings":{"max_sessions":8,"wake_words":"jarvis"}}'
```

Active-session-safe capacity settings affect the running service. New-session
settings are copied into future browser sessions; existing sessions keep their
recorder configuration. Startup-only settings, including ASR engines and model
paths, are rejected because shared inference workers are already initialized.

## Engine Recipes

Default faster-whisper:

```bash
python example_fastapi_server/server.py \
  --host 0.0.0.0 \
  --port 8010 \
  --engine faster_whisper \
  --model small.en \
  --realtime-model tiny.en \
  --device cuda \
  --language en
```

whisper.cpp CPU:

```bash
python -m pip install "RealtimeSTT[whisper-cpp]"
python example_fastapi_server/server.py \
  --host 0.0.0.0 \
  --port 8010 \
  --engine whisper_cpp \
  --model tiny.en \
  --realtime-engine whisper_cpp \
  --realtime-model tiny.en \
  --device cpu \
  --beam-size 5 \
  --beam-size-realtime 1 \
  --download-root test-model-cache/pywhispercpp \
  --engine-options '{"model":{"n_threads":8,"redirect_whispercpp_logs_to":null}}' \
  --realtime-engine-options '{"model":{"n_threads":8,"redirect_whispercpp_logs_to":null},"transcribe":{"single_segment":true,"no_context":true,"print_timestamps":false}}'
```

sherpa-onnx Moonshine CPU:

```bash
python -m pip install sherpa-onnx
python example_fastapi_server/server.py \
  --engine sherpa_onnx_moonshine \
  --model sherpa-onnx-moonshine-tiny-en-int8 \
  --realtime-engine sherpa_onnx_moonshine \
  --realtime-model sherpa-onnx-moonshine-tiny-en-int8 \
  --device cpu \
  --language en \
  --download-root test-model-cache/sherpa-onnx \
  --engine-options '{"num_threads":2,"provider":"cpu"}' \
  --realtime-engine-options '{"num_threads":2,"provider":"cpu"}' \
  --realtime-processing-pause 0.8 \
  --realtime-use-syllable-boundaries
```

Kroko-ONNX CPU with the same model for final and realtime:

```powershell
$model = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
python example_fastapi_server\server.py `
  --engine kroko_onnx `
  --model $model `
  --realtime-engine kroko_onnx `
  --realtime-model $model `
  --device cpu `
  --language en `
  --engine-options '{"provider":"cpu","num_threads":2}' `
  --realtime-engine-options '{"provider":"cpu","num_threads":1}'
```

Kroko-ONNX final transcription with a lighter realtime engine:

```powershell
$model = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
python example_fastapi_server\server.py `
  --engine kroko_onnx `
  --model $model `
  --realtime-engine whisper_cpp `
  --realtime-model tiny.en `
  --device cpu `
  --language en `
  --engine-options '{"provider":"cpu","num_threads":2}'
```

Parakeet final transcription with a small realtime model:

```bash
python example_fastapi_server/server.py \
  --engine parakeet \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --realtime-engine faster_whisper \
  --realtime-model tiny.en \
  --device cuda \
  --language en
```

Meta Omnilingual ASR from Linux or WSL2, using one CTC model lane for both
realtime and final transcription:

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

Open `http://localhost:8010` from a Windows browser when WSL2 localhost
forwarding is active.

This recipe targets `example_fastapi_server/server.py` from a source checkout,
not the installed `stt-server` console script. Check `stt-server --help`
separately for the installed CLI's supported options.

Wake word mode with Porcupine:

```bash
python example_fastapi_server/server.py \
  --engine faster_whisper \
  --model small.en \
  --realtime-model tiny.en \
  --wakeword-backend pvporcupine \
  --wake-words jarvis \
  --wake-words-sensitivity 0.7 \
  --wake-word-timeout 5 \
  --wake-word-followup-window 5
```

## WebSocket Protocol

The browser sends binary audio packets to `/ws/transcribe`:

- 4 bytes little-endian unsigned metadata length
- UTF-8 JSON metadata
- 16-bit little-endian mono PCM audio bytes

Metadata example:

```json
{
  "sampleRate": 48000,
  "channels": 1,
  "format": "pcm_s16le",
  "frames": 1920
}
```

Text commands are JSON objects:

```json
{"type": "start"}
```

Supported commands:

- `start`
- `stop`
- `clear`
- `ping`
- `metrics`

Server event types include:

- `hello`: assigns `clientId` and `sessionId`.
- `ready`: model lanes are initialized.
- `timeline`: timing events for wake word state, recording start/end,
  realtime updates, final transcription start, and final transcript delivery.
- `realtime`: interim text for a session-local `segmentId`.
- `final`: final text for the same session-local `segmentId`.
- `status`: session/server state.
- `warning`: recoverable issue.
- `error`: command, packet, admission, or runtime error.
- `clear`: session transcript reset.
- `pong`: ping response.
- `metrics`: per-session metrics response.

Transcript-bearing events include `sessionId` and are routed only to that
session. `realtime` and `final` events may include a `segment` object with
recording start/end timestamps, duration, pre-recording buffer range, and wake
word timing when available.

## Metrics And Health

Use `/health` for readiness checks and basic load:

```bash
curl http://localhost:8010/health
```

Use `/api/metrics` for operational detail:

```bash
curl http://localhost:8010/api/metrics
```

Metrics include active session counts, scheduler health, queue depths,
coalesced realtime jobs, dropped stale jobs, p50/p95 queue delay and inference
latency, and worker busy ratios.

## Browser UI Behavior

The UI connects to `/ws/transcribe`, sends browser microphone audio packets, and
keeps session-local realtime and final transcript blocks related by
`segmentId`. Each transcript block shows recording start, recording end,
duration, pre-roll, and wake timing when the server has that data. The left
timeline lists wake wait/detect/timeout events, recording start/end, realtime
updates, and final transcript delivery. Clear/reset affects only the issuing
session.

Admission limits are explicit. When `--max-sessions` is reached, new websocket
clients receive an admission error and close code `1013`. When active speaker
capacity is reached, accepted sessions receive warnings while existing final
work is preserved where possible.

## Tests

Fast fake-scheduler tests:

```bash
python -m unittest -v \
  tests.unit.test_fastapi_server_protocol \
  tests.unit.test_fastapi_server_multi_user
```

Opt-in real-engine load/quality/performance test:

```bash
REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF=1 \
python -m unittest -v tests.unit.test_fastapi_server_multi_user_asr_integration
```

Windows `cmd.exe` helper for a sherpa-onnx Moonshine performance run:

```cmd
example_fastapi_server\run_multi_user_perf.cmd
```

More test details are in [testing.md](testing.md).

## Deployment Notes

- Use Linux or WSL2 for CUDA-heavy engines such as Parakeet, Omnilingual ASR,
  Qwen vLLM, and larger Transformers models.
- Install Kroko-ONNX with `RealtimeSTT[kroko-builder]` and
  `stt-install-kroko --build` before selecting `kroko_onnx`. On Windows, use
  Python 3.12 x64 and start Docker Desktop first.
- Keep model caches on persistent storage so restarts do not redownload models.
- Put the server behind a reverse proxy when exposing it beyond localhost.
- Size `--max-sessions`, `--max-active-speakers`, queue depths, and model lanes
  for the selected engine and hardware.
- Use `/health` for readiness and `/api/metrics` for load/latency monitoring.
