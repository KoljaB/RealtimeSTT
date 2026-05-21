# RealtimeSTT FastAPI Browser Server

This example serves a polished browser UI and a FastAPI WebSocket endpoint for
microphone streaming into RealtimeSTT.

## Install

Use a Linux environment for CUDA-heavy engines such as Parakeet, Qwen vLLM, and
larger Transformers models.

```bash
python -m venv .venv-fastapi
source .venv-fastapi/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r example_fastapi_server/requirements.txt
```

Install optional engine stacks as needed. For Parakeet:

```bash
python -m pip install "nemo_toolkit[asr]" soundfile librosa
```

For CPU whisper.cpp:

```bash
python -m pip install pywhispercpp
```

For CPU sherpa-onnx Moonshine:

```bash
python -m pip install sherpa-onnx
```

For Kroko-ONNX:

```bash
git clone https://github.com/kroko-ai/kroko-onnx.git
cd kroko-onnx
python -m pip install .
```

Kroko-ONNX is currently Linux/Docker-oriented upstream. If native Windows builds
fail, use WSL2/Linux and run this server from that environment.

For Meta Omnilingual ASR, use Linux or WSL2:

```bash
python -m pip install -e ".[omnilingual-asr]"
```

## Run

Default faster-whisper setup:

```bash
python example_fastapi_server/server.py --host 0.0.0.0 --port 8010
```

Open:

```text
http://localhost:8010
```

## Multi-User Mode

The server now accepts multiple independent browser sessions. Each websocket is
assigned a `sessionId`; audio buffers, VAD state, transcript segment ids,
clear/reset commands, realtime text, final text, status, warnings, and errors
are scoped to that session.

Each accepted websocket owns a lightweight `AudioToTextRecorder` stream state
machine, so the browser server keeps the recorder's existing WebRTC VAD, Silero
VAD, wakeword hooks, early-final transcription, and syllable-boundary realtime
scheduling. The heavy ASR engines are injected as shared executors instead of
being loaded by every recorder. With model warmup enabled, each session also
primes its VAD path during recorder setup so the first speech chunk does not pay
Silero/WebRTC lazy runtime setup costs.

VAD is intentionally session-scoped in this server. WebRTC/Silero detection is
stateful at the stream level, so a single shared VAD object would be a correctness
risk unless the implementation proves that model weights, recurrent state, and
thread access are separated per session. A future optimization may share immutable
VAD weights, but it must keep per-session VAD state and reset semantics.

Model resources are shared instead of creating one recorder/model copy per
browser:

- default balanced mode: one shared final model lane plus one shared realtime
  model lane
- low-memory mode: `--use-main-model-for-realtime` uses one shared model lane
  for both realtime and final work

Inference is scheduled through per-session fair queues. Final jobs are
preserved up to the configured per-session limit, while stale realtime jobs are
coalesced so a noisy client cannot fill the global queue with obsolete interim
work.

Useful capacity flags:

```bash
python example_fastapi_server/server.py \
  --max-sessions 4 \
  --max-active-speakers 4 \
  --max-global-inference-queue-depth 64 \
  --max-final-queue-depth-per-session 8 \
  --max-realtime-queue-age-ms 1500 \
  --max-audio-queue-seconds-per-session 30
```

Runtime-safe configuration is exposed through `GET /api/config` and
`PATCH /api/config`. The config response lists which settings are safe for
active sessions, which apply only to future sessions, and which require a server
restart because shared inference workers are already loaded. For example:

```bash
curl -X PATCH http://localhost:8010/api/config \
  -H 'Content-Type: application/json' \
  -d '{"settings":{"max_sessions":8,"wake_words":"jarvis"}}'
```

Wake word mode is passed through to each browser session's recorder:

```bash
python example_fastapi_server/server.py \
  --wakeword-backend pvporcupine \
  --wake-words jarvis \
  --wake-words-sensitivity 0.7 \
  --wake-word-timeout 5 \
  --wake-word-followup-window 5
```

Wake wait, detection, follow-up voice windows, timeout, recording start,
recording end, realtime, and final transcript transitions are surfaced as
`timeline` websocket events and in the browser UI. Transcript blocks include
segment timing, duration, pre-roll, and wake detection metadata when available.

Health and load are exposed at:

- `/health`: readiness, active sessions/speakers, startup errors, worker state
- `/api/metrics`: per-session counters, queue depth, coalescing/drop counters,
  p50/p95 queue delay and inference latency, worker busy ratio

Capacity is explicit. New sessions above `--max-sessions` are rejected, and new
speakers above `--max-active-speakers` receive a warning while already accepted
sessions continue to preserve final transcription work where possible.
Session slots are reserved before per-session recorder/VAD construction so a
burst of concurrent connects cannot instantiate more recorders than the session
limit. Audio packets are accepted only after a `start` command. Recorder input
queues use `--audio-queue-size`, long continuous recordings are force-finalized
at `--max-audio-queue-seconds-per-session`, and completed recording backlog is
trimmed to `--max-final-queue-depth-per-session`.

Parakeet final transcription with a small realtime Whisper model:

```bash
export HF_HOME="$HOME/.cache/huggingface"
python example_fastapi_server/server.py \
  --host 0.0.0.0 \
  --port 8010 \
  --engine parakeet \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --realtime-engine faster_whisper \
  --realtime-model tiny.en \
  --device cuda \
  --language en
```

Parakeet for both realtime and final transcription:

```bash
python example_fastapi_server/server.py \
  --engine parakeet \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --use-main-model-for-realtime \
  --profile parakeet-low-latency \
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
  --language eng_Latn \
  --engine-options '{"batch_size":1,"sample_rate":16000}'
```

Open `http://localhost:8010` from a Windows browser when WSL2 localhost
forwarding is active. If VRAM is tight, start with `omniASR_CTC_300M_v2`.

## CPU Engine Recipes

These commands are intended for Windows `cmd.exe` from the repository root,
for example `D:\Projekte\STT\RealtimeSTT\RealtimeSTT`. Replace the Python path
with the active virtual environment for your checkout.

### whisper.cpp CPU

This uses `tiny.en` for both realtime and final transcription. Realtime uses
greedy decoding and single-segment/no-context settings for faster interim text.

```cmd
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe example_fastapi_server\server.py --host 0.0.0.0 --port 8010 --engine whisper_cpp --model tiny.en --realtime-engine whisper_cpp --realtime-model tiny.en --device cpu --beam-size 5 --beam-size-realtime 1 --download-root test-model-cache\pywhispercpp --engine-options "{\"model\":{\"n_threads\":8,\"redirect_whispercpp_logs_to\":null}}" --realtime-engine-options "{\"model\":{\"n_threads\":8,\"redirect_whispercpp_logs_to\":null},\"transcribe\":{\"single_segment\":true,\"no_context\":true,\"print_timestamps\":false}}"
```

### sherpa-onnx Moonshine CPU

Download and extract the Tiny and Base Moonshine sherpa-onnx models once:

```cmd
mkdir test-model-cache\sherpa-onnx
curl.exe -L -o test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-tiny-en-int8.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -c "import tarfile; tarfile.open(r'test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-tiny-en-int8.tar.bz2', 'r:bz2').extractall(r'test-model-cache\sherpa-onnx')"
curl.exe -L -o test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-base-en-int8.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -c "import tarfile; tarfile.open(r'test-model-cache\sherpa-onnx\sherpa-onnx-moonshine-base-en-int8.tar.bz2', 'r:bz2').extractall(r'test-model-cache\sherpa-onnx')"
```

Run Base for final transcription and Tiny for realtime transcription:

```cmd
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe example_fastapi_server\server.py --host 0.0.0.0 --port 8010 --engine sherpa_onnx_moonshine --model sherpa-onnx-moonshine-base-en-int8 --realtime-engine sherpa_onnx_moonshine --realtime-model sherpa-onnx-moonshine-tiny-en-int8 --device cpu --language en --download-root test-model-cache\sherpa-onnx --engine-options "{\"num_threads\":4,\"provider\":\"cpu\"}" --realtime-engine-options "{\"num_threads\":2,\"provider\":\"cpu\"}" --realtime-processing-pause 0.8 --realtime-use-syllable-boundaries --realtime-boundary-detector-sensitivity 0.6 --realtime-boundary-followup-delays 0.1,0.2,0.4
```

For lower memory usage, use Tiny for both final and realtime transcription:

```cmd
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe example_fastapi_server\server.py --host 0.0.0.0 --port 8010 --engine sherpa_onnx_moonshine --model sherpa-onnx-moonshine-tiny-en-int8 --realtime-engine sherpa_onnx_moonshine --realtime-model sherpa-onnx-moonshine-tiny-en-int8 --device cpu --language en --download-root test-model-cache\sherpa-onnx --engine-options "{\"num_threads\":2,\"provider\":\"cpu\"}" --realtime-engine-options "{\"num_threads\":2,\"provider\":\"cpu\"}" --realtime-processing-pause 0.8 --realtime-use-syllable-boundaries --realtime-boundary-detector-sensitivity 0.6 --realtime-boundary-followup-delays 0.1,0.2,0.4
```

### Kroko-ONNX CPU

Download the community English model once:

```cmd
mkdir test-model-cache\kroko-onnx
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Banafo/Kroko-ASR', filename='Kroko-EN-Community-64-L-Streaming-001.data', local_dir='test-model-cache/kroko-onnx')"
```

Run Kroko for both final and realtime:

```cmd
python example_fastapi_server\server.py --host 0.0.0.0 --port 8010 --engine kroko_onnx --model test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data --realtime-engine kroko_onnx --realtime-model test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data --device cpu --language en --engine-options "{\"provider\":\"cpu\",\"num_threads\":2}" --realtime-engine-options "{\"provider\":\"cpu\",\"num_threads\":1}"
```

Or use Kroko for final text with a lighter realtime engine:

```cmd
python example_fastapi_server\server.py --host 0.0.0.0 --port 8010 --engine kroko_onnx --model test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data --realtime-engine whisper_cpp --realtime-model tiny.en --device cpu --language en --engine-options "{\"provider\":\"cpu\",\"num_threads\":2}"
```

## Tuning Profiles

Whisper-family engines can use `--beam-size` and `--beam-size-realtime` as a
direct speed/quality tradeoff. Parakeet TDT uses a different NeMo decoder stack,
so the server exposes Parakeet profiles that tune latency through batching,
realtime cadence, and VAD/segmentation timing instead:

- `parakeet-low-latency`: frequent interim updates, smaller batches, shorter
  silence window
- `parakeet-balanced`: calmer default for browser dictation
- `parakeet-accurate-final`: slower turn finalization with more stable final
  chunks

Explicit flags still override profile values:

```bash
python example_fastapi_server/server.py \
  --engine parakeet \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --use-main-model-for-realtime \
  --profile parakeet-balanced \
  --realtime-processing-pause 0.08 \
  --post-speech-silence-duration 0.6
```

## Protocol

The browser sends binary WebSocket audio packets to `/ws/transcribe`:

- 4 bytes little-endian unsigned metadata length
- UTF-8 JSON metadata
- 16-bit little-endian mono PCM audio bytes

Metadata fields:

```json
{
  "sampleRate": 48000,
  "channels": 1,
  "format": "pcm_s16le",
  "frames": 1920
}
```

Server events:

- `hello`: assigns `clientId` and `sessionId`
- `ready`: model lanes are initialized; includes public settings and limits
- `timeline`: segment timing and wake word state transitions
- `realtime`: interim text for a session-local `segmentId`
- `final`: final text for the same session-local `segmentId`; the UI replaces
  the interim block
- `status`, `warning`, `error`, `clear`, `pong`, `metrics`

Transcript-bearing events include `sessionId` and are routed only to that
session. `clear` resets only the issuing session and discards pending stale
results from earlier session generations.

## Engine Names

The server passes engine names straight into RealtimeSTT. Hyphenated names are
accepted and normalized, so these work:

- `faster_whisper`
- `whisper_cpp`
- `openai_whisper`
- `parakeet` / `nvidia-parakeet`
- `sherpa-onnx-parakeet`
- `kroko-onnx` / `kroko` / `banafo-kroko`
- `omnilingual-asr` / `omnilingual` / `meta-omnilingual-asr` / `omni-asr`
- `cohere-transcribe`
- `granite-speech`
- `qwen3-asr`
- `moonshine-streaming`
- `sherpa-onnx-moonshine`

Backend-specific dictionaries can be passed as JSON:

```bash
python example_fastapi_server/server.py \
  --engine cohere-transcribe \
  --model CohereLabs/cohere-transcribe-03-2026 \
  --engine-options '{"language":"en"}'
```

## Tests

Fast unit coverage uses fake schedulers and does not load ASR models:

```bash
python -m unittest -v \
  tests.unit.test_fastapi_server_protocol \
  tests.unit.test_fastapi_server_multi_user
```

The opt-in real-engine load/quality/performance test streams
`tests/unit/audio/asr-reference.wav` through multiple sessions in parallel,
compares the final transcript against
`tests/unit/audio/asr-reference.expected_sentences.json`, and prints a timing
report for each run:

```bash
REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF=1 \
python -m unittest -v tests.unit.test_fastapi_server_multi_user_asr_integration
```

`REALTIMESTT_RUN_FASTAPI_MULTI_USER_ASR=1` runs the same test; use the `PERF`
name when the main goal is measuring latency.

The report includes first realtime latency, first final latency, final latency
after audio upload ended, first recording/VAD-start timing, realtime/final
event cadence, WER, scheduler p50/p95 latency, coalescing counters, and
rejected/drop counters. Set
`REALTIMESTT_FASTAPI_ASR_METRICS_JSON=/path/to/report.json` to also write the
report as JSON.

Useful overrides:

```bash
REALTIMESTT_FASTAPI_ASR_CLIENTS=4
REALTIMESTT_FASTAPI_ASR_ENGINE=sherpa_onnx_moonshine
REALTIMESTT_FASTAPI_ASR_MODEL=sherpa-onnx-moonshine-base-en-int8
REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE=sherpa_onnx_moonshine
REALTIMESTT_FASTAPI_ASR_REALTIME_MODEL=sherpa-onnx-moonshine-tiny-en-int8
REALTIMESTT_FASTAPI_ASR_DOWNLOAD_ROOT=test-model-cache/sherpa-onnx
REALTIMESTT_FASTAPI_ASR_DEVICE=cpu
REALTIMESTT_FASTAPI_ASR_MAX_WER=0.30
REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS='{"num_threads":4,"provider":"cpu"}'
REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE_OPTIONS='{"num_threads":2,"provider":"cpu"}'
REALTIMESTT_FASTAPI_ASR_REALTIME_PROCESSING_PAUSE=0.8
REALTIMESTT_FASTAPI_ASR_REALTIME_USE_SYLLABLE_BOUNDARIES=1
REALTIMESTT_FASTAPI_ASR_REALTIME_BOUNDARY_FOLLOWUP_DELAYS=0.1,0.2,0.4
```

For Kroko-ONNX performance runs, select Kroko through the same generic
variables:

```bash
REALTIMESTT_FASTAPI_ASR_ENGINE=kroko_onnx
REALTIMESTT_FASTAPI_ASR_MODEL=test-model-cache/kroko-onnx/Kroko-EN-Community-64-L-Streaming-001.data
REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE=kroko_onnx
REALTIMESTT_FASTAPI_ASR_REALTIME_MODEL=test-model-cache/kroko-onnx/Kroko-EN-Community-64-L-Streaming-001.data
REALTIMESTT_FASTAPI_ASR_DEVICE=cpu
REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS=provider=cpu,num_threads=2
REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE_OPTIONS=provider=cpu,num_threads=1
```

The engine option variables also accept `key=value` lists, for example
`REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS=num_threads=4,provider=cpu`, which is
easier to use from Windows cmd.exe.

On Windows cmd.exe, the common 4-client sherpa-onnx Moonshine configuration is
also available as:

```cmd
example_fastapi_server\run_multi_user_perf.cmd
```

You can override only the values you care about before calling it:

```cmd
set REALTIMESTT_FASTAPI_ASR_CLIENTS=8
set REALTIMESTT_FASTAPI_ASR_METRICS_JSON=test-results\fastapi-8-user-perf.json
example_fastapi_server\run_multi_user_perf.cmd
```
