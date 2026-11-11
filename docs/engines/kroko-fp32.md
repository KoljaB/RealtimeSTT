# Kroko optimized FP32 CPU

`kroko_fp32` is the English-only adapter for the optimized standalone Kroko
CPU executable. It is separate from `kroko_onnx`: that existing engine uses the
optional `kroko-onnx` Python package and `.data` streaming models, while this
adapter uses `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, and `tokens.txt`.

## Engine names

- `kroko_fp32`
- `kroko_native`

Hyphenated CLI names are normalized, so `kroko-fp32` also works.

## Runtime contract

The executable must support `--stdio-server` protocol version 1. RealtimeSTT
starts one process per configured engine worker and keeps its model and ONNX
sessions resident. Audio is sent as 16 kHz mono little-endian float32 PCM, and
the native process returns one JSON result per request.

The adapter is offline-only (`supports_streaming = False`). It is suitable for
the main Preview/Final lane while a separate streaming engine continues to own
the realtime lane.

The model is fixed to English. Omitted language and English regional variants
such as `en-US` are normalized to `en`; `auto` and non-English languages are
rejected instead of silently producing an unreliable transcript.

## Required files

Pass the model directory as `model`. It must contain:

- `encoder.onnx`
- `decoder.onnx`
- `joiner.onnx`
- `tokens.txt`

By default the adapter looks for `../bin/kroko-cli` relative to the model
directory and then on `PATH`. Set `executable` explicitly for other layouts.

## Engine options

| Option | Default | Meaning |
| --- | ---: | --- |
| `executable` | auto | Path to the optimized `kroko-cli` binary. |
| `encoder_threads` | `4` | Native encoder thread count, 1 through 64. |
| `feature_workers` | `1` | CPU feature-extraction workers, 1 through 64. |
| `startup_timeout` | `15.0` | Seconds allowed for model loading and handshake. |
| `request_timeout` | `60.0` | Seconds allowed for one transcription. |
| `max_audio_seconds` | `120.0` | Adapter-side duration ceiling, at most 300 seconds. |
| `strict_length` | `false` | Reject audio that needs model-alignment tail padding. |

## Production server example

Use Kroko only for the main model and retain the existing streaming lane:

```sh
stt-server-production \
  --engine kroko_fp32 \
  --model /opt/kroko/models \
  --device cpu \
  --language en \
  --supported-languages en \
  --engine-options '{"executable":"/opt/kroko/bin/kroko-cli","encoder_threads":7,"feature_workers":1}' \
  --realtime-engine sherpa_onnx_nemotron \
  --realtime-model /opt/models/nemotron
```

The server's worker CPU affinity is inherited by the native child process on
Linux. Match `encoder_threads` to the CPUs assigned to the main worker and
benchmark on the target topology.

An active cancellation terminates the native process so in-flight ONNX work
stops promptly. The next request automatically starts a fresh resident process
and pays model loading plus warmup again.

## Validation

Run the fast adapter tests without a native binary:

```powershell
python -m unittest -v tests.unit.test_kroko_fp32_engine
```

For a real smoke test, start the production server with the command above and
transcribe a known 16 kHz mono fixture repeatedly. Report both native
`metadata.inference_ms` and end-to-end request latency; only the former excludes
server scheduling and transport.

## Licensing

This adapter contains no Kroko model or executable. Before redistributing or
using a supplied bundle in production, verify and retain the license and
attribution terms for that exact source, runtime, and model artifact.
