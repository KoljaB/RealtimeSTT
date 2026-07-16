# Multi-User FastAPI Implementation Guide

This guide outlines the design work needed before turning
`example_fastapi_server` into a multi-user realtime transcription server.

The short version: do not simply remove the current single-client guard. The
server must separate per-user stream state from shared inference resources, then
schedule shared model access fairly.

## Current Shape

The current FastAPI example is intentionally single-session:

- `RealtimeSTTService` owns one `AudioToTextRecorder`.
- It owns one global `audio_queue`.
- It owns one global `SegmentState`.
- It tracks one `active_client_id`.
- `ConnectionManager.broadcast()` sends transcript events to every websocket.
- `/ws/transcribe` rejects audio from any second active client.

That design is correct for a single browser stream, but unsafe for multiple
users. If the single-client guard were removed, audio from different users
would enter the same recorder, transcript segment state would be shared, and
transcription events could leak between users.

## Main Design Principle

Split the system into two layers:

1. Per-session state
2. Shared inference resources

Per-session state should include:

- session id and optional authenticated user id
- websocket connection
- transcript segment ids
- VAD/recording state, including separate WebRTC/Silero runtime state per stream
- audio ring buffer or bounded audio queue
- realtime scheduling state
- final-transcription state
- per-session quotas, drop counters, and latency metrics

Shared inference resources should include:

- one shared final transcription model, or a small worker pool if the backend
  can safely support it
- optionally one shared realtime model
- warmup and lifecycle management
- a fair scheduler
- global admission control and overload handling

This is the difference between "many users" and "many model copies".

VAD is not the same sharing problem as ASR inference. WebRTC and Silero VAD keep
stream-local activity state, and Silero exposes reset semantics that are easy to
mix between users if one global instance is shared. The safe baseline is one VAD
state machine per accepted session. Sharing immutable VAD weights can be a later
optimization only after the API is wrapped so every session has independent
state, reset, and locking behavior.

## Model Sharing

Loading one recorder per user is the simplest mental model, but it is the wrong
first target for CUDA-heavy engines because each recorder initializes its own
model resources. That can duplicate VRAM and startup cost.

A better target is:

- one shared main/final model worker
- optionally one shared realtime model worker
- many lightweight user sessions submitting inference jobs

There are two useful operating profiles:

### Low-Memory Profile

Use one shared model for both realtime and final transcription.

Pros:

- lowest VRAM usage
- simplest resource accounting

Cons:

- final transcription can delay realtime updates
- realtime updates can delay final transcription
- fairness depends entirely on the scheduler

This should be treated as a constrained mode, not as the best latency mode.

### Balanced Profile

Use one shared main/final model and one shared smaller realtime model.

Pros:

- realtime updates are less affected by final transcriptions
- better perceived latency under load
- still avoids per-user model copies

Cons:

- uses more memory than the low-memory profile
- requires scheduling two model lanes

This is probably the better default when VRAM allows it.

## Inference Request Abstraction

Introduce an explicit request object instead of letting sessions call model
methods directly.

Suggested fields:

- `request_id`
- `session_id`
- `kind`: `realtime` or `final`
- `audio`
- `language`
- `use_prompt`
- `segment_id`
- `sequence`
- `created_at`
- `deadline_at`
- `priority`

The response should include:

- `request_id`
- `session_id`
- `kind`
- `segment_id`
- `sequence`
- `text`
- `error`
- timing fields: queue delay, inference duration, total latency

The `sequence` field is important for realtime transcription. If an older
realtime result returns after a newer one, the session should discard the stale
result.

## Fair Scheduling

A single global FIFO queue is not enough. One noisy or slow client could fill it
and make everyone else wait.

Use per-session queues plus a fair scheduler. Good first policies:

- round-robin across active sessions
- at most one outstanding realtime request per session
- coalesce realtime work so only the latest audio snapshot matters
- never let one session enqueue unlimited final jobs
- reserve some capacity for final transcription so final results do not starve
- give realtime jobs deadlines, then drop or supersede stale realtime jobs

Realtime jobs should be "latest wins". If the server is overloaded, old
realtime snapshots are usually not worth processing. Final jobs are different:
they represent user-visible transcript commits and should be preserved unless
the session is cancelled.

## Load Handling

Define explicit server capacity. "Multi-user" should not mean unlimited users.

Start with settings like:

- `max_sessions`
- `max_active_speakers`
- `max_audio_queue_seconds_per_session`
- `audio_queue_size`
- `max_realtime_queue_age_ms`
- `max_final_queue_depth_per_session`
- `max_global_inference_queue_depth`
- `realtime_degradation_threshold_ms`

When overloaded, degrade in this order:

1. Reserve session slots before constructing per-session recorder/VAD objects.
2. Reject audio packets until the session has sent `start`.
3. Bound recorder input queues with `audio_queue_size`.
4. Force-finalize continuous recordings at `max_audio_queue_seconds_per_session`.
5. Trim completed-recording backlog to `max_final_queue_depth_per_session`.
6. Coalesce pending realtime requests per session.
7. Increase realtime cadence for affected sessions.
8. Drop stale realtime jobs.
9. Notify clients that realtime quality/latency is degraded.
10. Reject new sessions or new realtime starts.
11. Keep final transcription reliable for already accepted audio whenever
   possible.

Audio backpressure also matters. Each session should have a bounded audio
buffer. The browser can watch `WebSocket.bufferedAmount`, and the server should
send warnings when it starts dropping or coalescing work.

## Session Isolation

Every server event must be routed to the owning session unless it is explicitly
a global server event.

Required protocol changes:

- assign a `sessionId` on connect
- include `sessionId` in server events
- make `clear` affect only the current session
- make transcript `segmentId` per session
- stop using global transcript broadcast for user transcript messages
- expose server-wide metrics separately from per-session status

This prevents one user from seeing another user's realtime or final transcript.

## Recorder Refactor Options

The current `AudioToTextRecorder` is stateful and owns both stream state and
model workers. Multi-user support needs a clear boundary between those
concerns.

Practical options:

### Option A: Per-User Recorder Instances

This is fastest to prototype but duplicates model resources. It is only
acceptable as a temporary CPU/small-model experiment or with a very low
`max_sessions`.

### Option B: Shared Executor Injection

Refactor `AudioToTextRecorder` so transcription calls go through injected
executors:

- `final_transcription_executor`
- `realtime_transcription_executor`

Each session can own recorder-like stream state, but all inference goes through
shared executors.

This is the best long-term direction if the library itself should become
multi-user friendly.

### Option C: External Model Service

Keep session logic in the FastAPI process, but move model inference into one or
more dedicated worker processes. Sessions submit jobs over queues or pipes.

This can isolate CUDA/model lifecycle and works well with the scheduler design,
but it requires a clean request/response protocol.

## Backend Concurrency

Do not assume all transcription engines are thread-safe or parallel-friendly.

Default to one worker per loaded model. Add backend-specific concurrency only
behind capability flags after measurement. Some backends may support concurrent
transcription, batching, or multi-GPU placement; others should be serialized.

The scheduler should own this decision. Session code should never call a shared
model directly from arbitrary async websocket tasks.

## Metrics To Add First

Add metrics before tuning. Useful counters and histograms:

- active sessions
- active speakers
- per-session audio queue depth
- dropped audio chunks
- coalesced realtime requests
- stale realtime results discarded
- realtime queue delay
- realtime inference duration
- realtime end-to-end latency
- final queue delay
- final inference duration
- model busy ratio
- rejected sessions
- overload state

These metrics will tell us whether the bottleneck is websocket ingestion, VAD,
audio buffering, model inference, or result delivery.

## Testing Strategy

Start with fake engines. Real models are too slow and nondeterministic for the
first correctness tests.

The existing `tests/unit/test_fastapi_server_protocol.py` is a fast unit-test
file for packet encoding/decoding, settings parsing, segment ids, and queue
backpressure. Keep that layer fast. Multi-user audio streaming should be added
as a separate integration-style test layer so it can use websocket sessions,
parallel clients, fake schedulers, and optional real engines without slowing
every normal unit run.

Use the existing annotated reference audio as the main real-audio fixture:

- `tests/unit/audio/asr-reference.wav`
- `tests/unit/audio/asr-reference.expected_sentences.json`
- optional snippets in
  `tests/unit/audio/asr-reference.expected_sentences.snippets/`

There are already local patterns worth reusing:

- `tests/unit/test_audio_fixtures.py` shows how tests read 16-bit PCM WAV files
  with `wave` and `numpy`.
- `tests/unit/test_slow_final_transcription_audio_gap.py` uses
  `asr-reference.wav` for deterministic recorder timing behavior.
- `tests/final_transcription_gap_regression.py` loads
  `asr-reference.expected_sentences.json`, normalizes text, and computes word
  error rate.
- `tests/realtime_transcription_count_comparison.py` is a manual integration
  benchmark for feeding WAV chunks into realtime transcription.

Recommended tests:

- two sessions receive only their own transcripts
- `clear` resets only one session
- one chatty session cannot starve another session
- realtime jobs are coalesced under load
- stale realtime results are discarded
- final jobs are not dropped during realtime overload
- session disconnect cancels pending realtime work
- admission control rejects sessions above capacity
- health/status endpoints expose global load without leaking transcript text

For the reference-WAV parallel test, create multiple simulated websocket
clients that all stream `asr-reference.wav` to `/ws/transcribe` in chunks that
match the browser protocol:

- read the WAV as mono 16-bit PCM
- chunk it at the same cadence the browser sends, for example 32 ms
- wrap each chunk with `encode_audio_packet()`
- start 2, 4, and eventually 8 clients concurrently
- collect `realtime` and `final` events per session
- compare each session's final combined transcript with
  `combined_normalized` from `asr-reference.expected_sentences.json`
- also compare utterance-level output against `utterances[*].normalized` when
  segmentation is expected to be stable enough

The real-engine version should be opt-in, for example behind an environment
variable such as `REALTIMESTT_RUN_FASTAPI_MULTI_USER_ASR=1`, because it will
load models and may require CUDA or downloaded engine assets. The default CI
version should use a fake inference scheduler that returns deterministic text
and controlled delays. That fake test should prove isolation, fairness,
coalescing, and stale-result handling without depending on ASR quality.

The real-engine test should measure both correctness and load behavior:

- final combined word error rate per session
- missing or duplicated expected utterances
- p50/p95 realtime event latency per session
- p50/p95 final event latency per session
- maximum latency skew between sessions
- dropped audio chunks and coalesced realtime jobs per session
- global model busy ratio

The important fairness assertion is not that every session has identical
latency. It is that no accepted session is consistently starved while another
session receives much lower latency under the same input stream.

After that, keep a load harness with prerecorded PCM streams and real engines.
Measure p50/p95 realtime latency and final latency for 1, 2, 4, and 8 simulated
clients on the target hardware.

## Suggested Implementation Roadmap

1. Define target semantics: max sessions, overload behavior, and whether the
   first profile is low-memory or balanced.
2. Replace global broadcast with per-session routing in
   `example_fastapi_server`.
3. Introduce `SessionStore` and `RealtimeSession` objects.
4. Add a fake `InferenceScheduler` and make one existing session use it.
5. Add per-session transcript state and per-session audio buffering.
6. Enable multiple websocket sessions with the fake scheduler.
7. Implement shared model workers for real engines.
8. Add fairness, realtime coalescing, and stale-result handling.
9. Add metrics and overload notifications.
10. Add the `asr-reference.wav` parallel websocket test with a fake scheduler.
11. Add the opt-in real-engine reference-WAV load test and tune default capacity
    and realtime cadence from those measurements.

## Things To Avoid

- Do not only remove `active_client_id`; that mixes user audio.
- Do not keep broadcasting transcript events to all websockets.
- Do not create one CUDA recorder per user as the production design.
- Do not let every audio chunk become an inference job.
- Do not use one global FIFO queue for all users.
- Do not promise equal latency without measuring real-time factor on the target
  model and hardware.

## Recommended First Target

For the first real implementation, aim for:

- multiple isolated browser sessions
- one shared final model
- one shared realtime model when memory allows
- a low-memory mode that uses the final model for realtime too
- per-session audio buffers
- fair per-session scheduling
- realtime coalescing
- explicit max session limits
- fake-engine tests before real-model load tests

That gives the project a solid multi-user foundation without pretending that
one GPU can provide unlimited realtime capacity.
