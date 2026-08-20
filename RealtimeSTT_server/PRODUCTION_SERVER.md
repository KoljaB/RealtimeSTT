# Versioned production server

`RealtimeSTT_server.production_server` is the supported HTTP/WebSocket server
for remote clients. It is separate from `stt-server`, which remains the legacy
two-WebSocket recorder-client entry point.

Run it through the new console entry point wired by packaging:

```text
stt-server-production --host 127.0.0.1 --port 8010
```

The default bind is loopback. A direct non-loopback bind is rejected unless a
bearer token and both Uvicorn TLS files are configured. Keep the token out of
shell history and process listings by using the environment:

```powershell
$env:REALTIMESTT_SERVER_BEARER_TOKEN = "replace-me"
stt-server-production `
  --host 0.0.0.0 `
  --ssl-certfile C:\certs\server-chain.pem `
  --ssl-keyfile C:\certs\server-key.pem
```

Bearer tokens are accepted only through
`REALTIMESTT_SERVER_BEARER_TOKEN`; literal token command-line arguments are
rejected so secrets cannot appear in shell history or process listings. The
`--ssl-certfile` and `--ssl-keyfile` flags are passed directly to Uvicorn and
must be supplied together. The token is never included in capabilities or
public settings.

For a reverse-proxy deployment, leave this server on its loopback default and
terminate TLS at the proxy. The proxy should forward HTTPS/WSS requests to
`http://127.0.0.1:8010` (and preserve the `Authorization` header); the server
still enforces the bearer token when one is configured. This keeps the ASR
process off the network while the proxy owns certificates and public TLS.

For a CPU INT8 sherpa-onnx deployment with Parakeet as the final model and
Nemotron as the live/realtime model, install both pinned bundles into persistent
storage with resumable downloads, archive/file verification, and atomic
extraction:

```powershell
python -m pip install "RealtimeSTT[server,sherpa-onnx]"
stt-install-sherpa-models --root models/sherpa-onnx --model all
```

The `server` extra includes the local Silero ONNX VAD model/runtime required by
legacy recorder-backed server paths. The versioned production WebSocket path
does not instantiate a recorder or use VAD to decide turn finals. Server
startup and first connection do not depend on an interactive Torch Hub
download.

Then run:

```powershell
$env:REALTIMESTT_SERVER_BEARER_TOKEN = "replace-me"
stt-server-production `
  --host 127.0.0.1 `
  --port 8010 `
  --engine sherpa_onnx_parakeet `
  --model sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 `
  --realtime-engine sherpa_onnx_nemotron `
  --realtime-model sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 `
  --download-root models/sherpa-onnx `
  --engine-options '{"provider":"cpu","num_threads":4,"verify_model_files":true}' `
  --realtime-engine-options '{"provider":"cpu","num_threads":4,"verify_model_files":true}' `
  --device cpu `
  --language auto
```

This exact two-model production profile is supported on Linux x86-64. Native
Windows remains a development target: release testing found intermittent empty
Parakeet finals on voiced cumulative-recovery audio. For Windows deployment,
configure a different authoritative final engine such as `faster_whisper`, or
run the pinned Nemotron/Parakeet pair on Linux. Do not treat an empty Windows
Parakeet response as a successful speech transcription.

## HTTP API

Versioned operational endpoints are available at both `/api/v1/...` and the
short `/v1/...` aliases:

* `GET /api/v1/live` is a process liveness check.
* `GET /api/v1/ready` returns `503` until shared model workers are ready and
  healthy.
* `GET /api/v1/capabilities` reports final/live providers and models, active
  languages, PCM format/sample rates, limits, and operations.

`GET /health` remains available for HTTP ASR probes and includes the familiar
`engine`, `model`, `device`, `provider`, `compute_type`, `ready`, and warmup
fields.

`POST /transcribe-pcm16` accepts raw little-endian mono PCM16. It preserves the
existing client query names (`sample_rate`, `encoding`, `language`,
`beam_size`, `best_of`, `temperature`, `word_timestamps`, `vad_filter`,
`condition_on_previous_text`, and `without_timestamps`) and returns `text`,
language, timing, engine/model/provider, and `segments` fields.

## WebSocket API

Connect to `/api/v1/ws/transcribe`, `/api/v1/ws`, or
`/v1/audio/transcriptions/stream`. Events carry `apiVersion: "v1"`,
`protocolVersion: "realtimestt.remote.v1"`, the `sessionId`, and a strictly
increasing per-session `eventSequence`. Each session has an independent bounded
outbound queue. When a client is slow, stale partial hypotheses for the same
turn are coalesced without creating sequence gaps; final and completion events
are preserved in a two-event terminal reserve. A client that exhausts that
reserve is closed with WebSocket backpressure code 1013 instead of growing an
unbounded queue.

Commands are JSON objects:

```json
{"type":"start","turnId":"turn-1","language":"en"}
{"type":"finalize"}
{"type":"reset"}
{"type":"cancel"}
```

Binary audio uses the existing length-prefixed packet format. Production
packets must be mono `pcm_s16le` and include a contiguous `audioSequence` in
metadata, beginning at zero. WebSocket audio is canonical 16 kHz; clients must
keep one stateful resampler for the logical turn and must not restart a
resampler for each packet:

```json
{
  "sampleRate": 16000,
  "channels": 1,
  "format": "pcm_s16le",
  "frames": 640,
  "audioSequence": 0
}
```

The server feeds Nemotron only newly accepted frames through one live stream
per logical turn. Partials are display-only, changed-hypothesis deduplicated,
and rate-limited. The accepted PCM bytes are also appended exactly once to the
turn's authoritative buffer. Finalize seals and drains live input, submits the
complete canonical buffer once through the same Parakeet final lane used by
HTTP, emits exactly one `final`, then exactly one `completion`.

Empty or silent turns emit `final` with empty text and `status: "no_speech"`,
then `completion`. Validation, queue pressure, duration limits,
authentication, and model failures use structured `error` objects with stable
machine-readable codes. Cancel, reset, disconnect, and reconnect fence old
turn and transport generations. Shutdown resolves queued jobs, stops worker
threads, and releases loaded engines deterministically.

## Capacity and memory

The two-model CPU profile keeps both Nemotron and Parakeet loaded. Size hosts
for the measured resident set plus concurrent audio buffers and native runtime
headroom; do not size from model archive sizes alone. Each active turn retains
its canonical PCM until the authoritative final completes, subject to
`--max-turn-audio-seconds`. Input, inference, and outbound queues are bounded;
backpressure is a protocol outcome, not permission to buffer without limit.
Run the streaming benchmark with production packet cadence, long-run
repetitions, and parallel concurrency on the target host before raising any
session or queue limits.
