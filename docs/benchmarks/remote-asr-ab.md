# RealtimeSTT 1.0.3 production-server benchmark

Date: 2026-08-20

This report compares the 1.0.3 production server with the existing CPU ASR
service on the same Linux x86-64 host. The existing service remained running
and unchanged. Both final-transcription paths used the same pinned Parakeet
INT8 model, four CPU inference threads, raw mono 16 kHz PCM16 input, and the
same 36-clip corpus.

## Candidate configuration

The candidate ran on an isolated loopback port from a verified source archive:

| Item | Value |
|---|---|
| Source archive SHA-256 | `b593dbb5c07125724a2feee36d8cafbec2a4530bca62bbdbcfcfa98dc60ef41e` |
| Runtime | CPython 3.12, `sherpa-onnx==1.13.4`, CPU |
| Live model | `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11` |
| Final model | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` |
| API | `POST /transcribe-pcm16`; `WS /v1/audio/transcriptions/stream` |
| Language mode | `auto` |
| Limits | 4 sessions, 4 active speakers, 120 seconds per turn |

The versioned liveness, readiness, and capabilities endpoints reported both
workers ready, the expected model paths/providers/languages, PCM16 audio
format, and no startup errors before measurement.

The first WebSocket attempt exposed a packaging problem: the isolated
environment had initially been installed with `--no-deps`, and a normal
`[server,sherpa-onnx]` install did not include a local Silero VAD model. The
production `server` extra now includes `silero-vad[onnx-cpu]>=6.2.1`; after
installing that declared dependency, the unchanged streaming benchmark
passed. This failure is not included in the passing metrics below.

## Final-transcription A/B

`tools/benchmarks/benchmark_asr_ab.py` ran every target separately so the two
servers did not compete for CPU. Each result below covers 36 clips, 90.390
seconds of audio, three repetitions, 108 successful requests, and zero failed
requests.

### One client

| Metric | RealtimeSTT candidate | Existing reference |
|---|---:|---:|
| WER | 0.0436 | 0.0436 |
| CER | 0.0242 | 0.0242 |
| Exact match | 0.8889 | 0.8889 |
| Throughput, requests/s | 6.823 | 6.968 |
| Client latency median, s | 0.1319 | 0.1311 |
| Client latency p95, s | 0.3179 | 0.3488 |
| RTF median | 0.0600 | 0.0580 |
| RTF p95 | 0.0722 | 0.0711 |

The candidate has identical quality. Its median latency differs by 0.8 ms and
its throughput is 2.1% lower, while its p95 latency is 8.9% lower. This is
single-client performance parity rather than a material regression or win.

### Four concurrent clients

| Metric | RealtimeSTT candidate | Existing reference |
|---|---:|---:|
| WER / CER / exact | 0.0436 / 0.0242 / 0.8889 | 0.0436 / 0.0242 / 0.8889 |
| Throughput, requests/s | 9.903 | 9.214 |
| Client latency median, s | 0.3702 | 0.3620 |
| Client latency p95, s | 0.5419 | 0.8107 |
| Queue latency p95, s | 0.4375 | 0.6103 |
| RTF p95 | 0.0547 | 0.0791 |

The candidate preserved quality, processed 7.5% more requests per second, and
reduced p95 client latency by 33.2%. Its median was 8.2 ms slower. For the
production workload, the stated goal is met: single-client behavior is at
parity and concurrent tail latency/throughput are better than the reference.

Report integrity hashes:

- sequential JSON: `8dd1d8f0856e898d83814523da7a6ec33d8c50decf0c8909a8e55a1991d322d9`
- concurrency-4 JSON: `bfbb47e597e62b6b48d658a8a7d6f78d66c571fa8bbd15b5ef38141b99e33a6e`

After the model files and Python wheels were cached locally, a controlled
process restart reached the versioned ready endpoint in 1.709 seconds. The
first 1.3-second final-transcription fixture then completed in 0.0672 seconds
(RTF 0.0504) with exact text. A repeated request completed in 0.0911 seconds;
the two single samples show no first-request lazy-load penalty but are not a
latency distribution.

## Streaming acceptance

The exact-source Linux server also received seven paced English turns through
the production WebSocket alias with 100 ms packets. All seven final texts were
exact, 61 partial updates were emitted, event/audio sequence failures were
zero, no final or completion event was missing, and median completion time
after `finalize` was 0.1039 seconds. The report JSON SHA-256 is
`67965a36fca40f1b423c00955bf51209451511ded8dfa954b142250d10514a58`.

The separate external seven-language fixture covers English, German, French,
Spanish, Italian, Portuguese, and Russian. On the same candidate code and
models it passed 7/7 exact final texts at both 100 ms and 37 ms packet sizes,
with zero event/audio sequence failures. The final texts were invariant across
the two packet sizes. Median first non-empty partial latency was 1.4529 seconds
at 100 ms and 0.9963 seconds at 37 ms. These generated fixtures are external
test data and are not included in the source distribution.

## Memory and platform result

The existing final-only reference used 1,096,164 KiB RSS. The RealtimeSTT
candidate used 1,949,204 KiB after both sherpa models loaded and 2,652,468 KiB
after the recorder/VAD runtime had been exercised. Neither service used GPU
memory. Allow at least 4 GiB of free RAM for the two-model CPU server itself,
in addition to operating-system and client workload headroom.

Linux x86-64 is the production target for the pinned two-model profile. Native
Windows tests load and transcribe the deterministic engine fixtures, but the
Parakeet final engine can return empty output for some voiced cumulative-turn
audio on Windows. Use the Linux server or another authoritative final engine
there until that upstream/runtime combination is proven reliable.

## Reproduction

Run the final-only comparison against two endpoints implementing the raw PCM
contract:

```bash
python tools/benchmarks/benchmark_asr_ab.py \
  --manifest /path/to/microphone_corpus/manifest.json \
  --target candidate=http://127.0.0.1:8767 \
  --target reference=http://127.0.0.1:8766 \
  --repetitions 3 --concurrency 1 \
  --output /persistent/results/asr-ab.json
```

Repeat with `--concurrency 4` for the load result. Measure partials separately:

```bash
python tools/benchmarks/benchmark_asr_streaming.py \
  --manifest /path/to/manifest.json \
  --url ws://127.0.0.1:8767/v1/audio/transcriptions/stream \
  --chunk-ms 100 --pace 1 \
  --output /persistent/results/asr-streaming.json
```

Do not compare a live partial event with the reference server's final response
as if they were the same latency metric.
