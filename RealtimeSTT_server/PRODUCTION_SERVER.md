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

`--bearer-token`/`--auth-token` may override the environment for local
development. The `--ssl-certfile` and `--ssl-keyfile` flags are passed directly
to Uvicorn and must be supplied together. The token is never included in
capabilities or public settings.

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
recorder-backed WebSocket sessions. Server startup and first connection do not
depend on an interactive Torch Hub download.

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
are preserved.

Commands are JSON objects:

```json
{"type":"start","turnId":"turn-1","language":"en"}
{"type":"finalize"}
{"type":"reset"}
{"type":"cancel"}
```

Binary audio uses the existing length-prefixed packet format. Production
packets must be mono `pcm_s16le` and include a contiguous `audioSequence` in
metadata, beginning at zero:

```json
{
  "sampleRate": 16000,
  "channels": 1,
  "format": "pcm_s16le",
  "frames": 640,
  "audioSequence": 0
}
```

The server emits `partial`, `final`, and one `completion` event per finalized
turn. Validation, queue pressure, duration limits, authentication, and model
failures use structured `error` objects with stable machine-readable codes.
Disconnecting a client cancels its scheduler work, closes its recorder, and
releases the session slot. Shutdown waits for worker threads and releases
loaded engines deterministically.
