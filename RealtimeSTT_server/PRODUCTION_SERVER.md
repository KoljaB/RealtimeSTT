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

### Optional dual realtime and immediate Preview

The production protocol keeps the slower, more accurate live model as its only
text anchor. Configure the optional lower-latency lane with
`--ultrafast-realtime-model`, `--ultrafast-realtime-engine`, and
`--ultrafast-realtime-engine-options`. `--ultrafast-realtime-max-tail-words`
defaults to five. Model names are deployment-specific; a complete dual-lane
fragment is:

```powershell
  --realtime-engine <accurate-live-engine> `
  --realtime-model <accurate-1120ms-model> `
  --ultrafast-realtime-engine <ultrafast-live-engine> `
  --ultrafast-realtime-model <ultrafast-80ms-model> `
  --ultrafast-realtime-max-tail-words 5
```

The two streaming sessions receive identical accepted PCM packets. The
ultrafast text remains private until it can be safely aligned to the accurate
text. A missing or conflicting anchor never publishes raw ultrafast words.
Omitting `--ultrafast-realtime-model` preserves the legacy one-lane path.

For a Preview-only deployment, install the `transcribe-cpp` extra and its
matching native provider, place the validated GGUF locally, and use the main
model lane as the Preview model:

```powershell
stt-server-production `
  --host 127.0.0.1 `
  --port 8651 `
  --preview-only `
  --allow-late-final-transcription `
  --late-final-max-audio-seconds 30 `
  --engine transcribe_cpp `
  --model C:\models\parakeet-tdt-0.6b-v3-Q4_K_M.gguf `
  --engine-options '{"backend":"cuda","model_sha256":"b68557be1e3c40207fd7c4bd9d63f1d3316b963f15325bfb0cc16a8bb0ffd181","session":{"n_threads":7,"kv_type":"auto","n_ctx":0},"transcribe":{"timestamps":"none"}}' `
  --device cuda `
  --beam-size 1 `
  --batch-size 1 `
  --realtime-engine <accurate-live-engine> `
  --realtime-model <accurate-1120ms-model> `
  --ultrafast-realtime-engine <ultrafast-live-engine> `
  --ultrafast-realtime-model <ultrafast-80ms-model>
```

`transcribe_cpp` requires an existing local GGUF and does not download weights.
At Preview request time, the complete buffered turn is transcribed immediately.
After a correlated Resume, the complete retained logical turn is transcribed
again; the candidate boundary fences request ownership but does not truncate
the model input. Accurate live text is diagnostic context only; it neither
shortens the Preview input nor replaces the Preview model result. If the first
Preview transcription is empty, the
server appends 500 ms of zero PCM and retries exactly once. A still-empty retry
publishes `status = "empty"`; a model failure publishes `status = "error"`.
Neither path substitutes `liveText` as Preview text. Preview never waits for
either live worker and never uses `mergedText` or `ultrafastSuffix` as its
transcript source.
`--preview-only` disables complete-turn Final ASR. The optional
`--allow-late-final-transcription` flag opens only authenticated HTTP requests
explicitly marked `operation=late_full_turn_correction`; ordinary Final remains
disabled, and `--late-final-max-audio-seconds` applies an independent server
cap with a hard upper bound of 30 seconds. Without `--preview-only`, Final still
receives the complete canonical turn buffer.
Each Preview event reports its captured audio revision, packet count, and frame
count. Preview-only finalization reuses an earlier Preview only when no newer
audio was accepted. If speech resumed after that snapshot, the server runs a
fresh Preview, suppresses the obsolete result, and still completes
with `finalCount = 0`.
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
`detected_language`, `language_probability`, timing, engine/model/provider, and
`segments` fields. Auto-detection fields are `null` when the provider supplies
no detection metadata. In Preview-only mode, the opt-in late lane additionally
requires
`operation=late_full_turn_correction`.

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
{"type":"preview","previewRequestId":"pause-1"}
{"type":"resume","turnId":"turn-1","resumeId":"resume-1","requestId":"resume-1","candidateId":"candidate-1","audioSequence":12,"sampleOffset":7680,"byteOffset":15360}
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

The server feeds only newly accepted frames through the accurate live stream
and, when configured, an independent ultrafast stream. Partial events retain
legacy `text` and `partialText` as the accurate text and add `accurateText`,
`mergedText`, `ultrafastSuffix`, and `mergeStatus`. The accepted PCM bytes are
also appended exactly once to the turn buffer. Finalize seals and drains both
live queues, waits for both live workers against one shared completion
deadline, submits the complete canonical buffer once through the final lane,
emits exactly one `final`, then exactly one
`completion`. In Preview-only mode the terminal pair contains no `final`.
`resume` retains the complete turn buffer and records its exact exclusive
sample/byte endpoint as the start of a new candidate. The optional
`audioSequence`, `sampleOffset`, and `byteOffset` fields are checked against
the server's next packet and accepted PCM length. `resumeId` is the canonical
correlation field; when both fields are supplied, `resumeId` and transitional
`requestId` must match until the fleet has upgraded. The `resume_ack` and any
resume error return both names. Legacy clients may omit all resume boundary
fields and receive an auto-correlated `resume_ack`. A resumed Preview decodes
the complete retained logical-turn PCM and emits that model result as `text`,
`cumulativeText`, and `candidateCumulativeText`. The legacy candidate-only
fields remain present as empty strings because no suffix is inferred from Live
ASR. Absolute `inputSampleRange`/`inputByteRange` fields describe the full model
input, while `candidateSampleRange`/`candidateByteRange` retain the Resume
boundary. It returns both `resumeId` and the retained `resumeRequestId` alias,
plus the matching `resumeEpoch`; publication is fenced by all three values.
`inputScope: "candidate"` remains the correlation guarantee, and
`candidateInputScope: "full_turn"` together with
`previewInputCoverage: "full_turn"` makes the model-input semantics explicit.

ACK-capable clients should select the strict Resume path only when the
capabilities document advertises `resume.liveProvenance`. Each `resume_ack`
then includes an immutable `resumeEpoch`; each accurate `partial` and raw
`ultrafast` event carries that epoch, `candidateId`, `resumeId`/
`resumeRequestId`, `candidateStartSample`, and the exclusive
`audioEndSampleExclusive` that produced the hypothesis. This lets a client
discard a delayed cumulative live revision whose PCM ended before the newest
candidate boundary. That endpoint is tracked per live lane from PCM actually
fed to its stream, rather than inferred from the concurrently accepted turn
buffer. The fields are additive: requestId-only intermediate and
legacy bare-`resume` clients remain supported. A Resume also cancels queued
older Preview snapshots before they are admitted to the Preview ASR lane; work
already executing is publication-fenced by the same snapshot epoch.

The `resume_ack` is also an ordered wire barrier: a post-Resume candidate
partial follows it with a later `eventSequence`, even when an older partial was
already waiting on a slow transport.

Empty or silent turns emit `final` with empty text and `status: "no_speech"`,
then `completion`. Validation, queue pressure, duration limits,
authentication, and model failures use structured `error` objects with stable
machine-readable codes. Cancel, reset, disconnect, and reconnect fence old
turn and transport generations. Shutdown resolves queued jobs, stops worker
threads, and releases loaded engines deterministically.

## Capacity and memory

The configured profile keeps its Preview/final model plus one or two live models loaded. Size hosts
for the measured resident set plus concurrent audio buffers and native runtime
headroom; do not size from model archive sizes alone. Each active turn retains
its canonical PCM until the authoritative final completes, subject to
`--max-turn-audio-seconds`. Input, inference, and outbound queues are bounded;
backpressure is a protocol outcome, not permission to buffer without limit.
Run the streaming benchmark with production packet cadence, long-run
repetitions, and parallel concurrency on the target host before raising any
session or queue limits.
