# Architecture — RealtimeSTT

RealtimeSTT is a recorder-centered, low-latency speech-to-text pipeline. Audio is normalized into short 16 kHz mono PCM chunks, turn detection decides which chunks form an utterance, and pluggable ASR engines produce realtime and final text through a small project-level adapter contract. The core recorder owns stream state, VAD, wake-word handling, pre-roll, realtime updates, finalization, callbacks, and shutdown. Server layers sit on top of that recorder instead of replacing it.

The design goal is simple: keep the public recorder API stable while moving heavy model runtimes, optional dependencies, protocol handling, and multi-user scheduling behind focused boundaries. Importing `RealtimeSTT` should not import every ASR runtime or wake-word stack. New code should follow the smaller, testable patterns in `transcription_engines/`, `realtime_boundary_detector.py`, `realtime_text_stabilizer.py`, and `example_fastapi_server/protocol.py`, not the accumulated size of `audio_recorder.py`.

## Repository Map

| Area | Main files | Responsibility |
| --- | --- | --- |
| Public API | `RealtimeSTT/__init__.py` | Lazy exports for `AudioToTextRecorder`, `AudioToTextRecorderClient`, `AudioInput`, and realtime boundary types. |
| Recorder state machine | `RealtimeSTT/audio_recorder.py` | `AudioToTextRecorder`, `TranscriptionWorker`, audio queue consumption, VAD, wake words, pre-roll, recording lifecycle, realtime updates, final transcription, callbacks, and shutdown. |
| Audio input | `RealtimeSTT/audio_input.py` | PyAudio device selection, microphone stream setup, raw chunk reads, client-side capture, and resampling helpers. |
| Worker communication | `RealtimeSTT/safepipe.py` | Thread-safe wrapper around the parent end of `multiprocessing.Pipe`, so multiple recorder threads can safely send, receive, and poll. |
| VAD and turn helpers | `RealtimeSTT/silero_vad.py`, `RealtimeSTT/preroll.py`, `RealtimeSTT/realtime_boundary_detector.py` | Silero backend selection, conservative pre-recording buffer trimming, and low-cost acoustic boundary detection. |
| Realtime text stabilization | `RealtimeSTT/realtime_text_stabilizer.py` | Pure, timestamp-driven stabilization of partial ASR text into stable deltas, unstable preview text, and diagnostics. |
| ASR engines | `RealtimeSTT/transcription_engines/base.py`, `factory.py`, and adapter modules | Project transcription contracts, engine config/result objects, lazy engine loading, backend-specific adapters, and optional streaming sessions. |
| Legacy websocket path | `RealtimeSTT_server/stt_server.py`, `RealtimeSTT/audio_recorder_client.py` | Single-user dual-port websocket server and a thin client that mirrors the local recorder API. |
| Browser reference server | `example_fastapi_server/server.py`, `protocol.py`, `static/index.html` | Source-only FastAPI browser streaming reference application with session admission, shared inference scheduling, packet validation, metrics, limits, and websocket publishing. |
| Tests and docs | `tests/unit/*`, `docs/*` | Contract tests for engines, VAD behavior, realtime behavior, server sessions, and user-facing architecture/coding guidance. |

## Dependency Direction

Dependencies flow inward toward the core library and downward toward focused adapters:

```text
example_fastapi_server  ──►  RealtimeSTT library  ──►  transcription_engines
RealtimeSTT_server      ──►  RealtimeSTT library  ──►  safepipe / VAD / realtime helpers
```

Engine adapters must not import the recorder or either server. Server protocol helpers must not import server runtime state. Third-party library details stay inside backend classes or feature adapters. Project-facing classes return project result objects such as `TranscriptionResult` and `SpeechBoundaryEvent`, not raw backend objects.

`audio_recorder.py` is the compatibility anchor, but not the template for future module shape. It currently owns many concerns because it is the public API center: microphone input, buffering, WebRTC VAD, Silero VAD, wake words, realtime transcription, final transcription, callback dispatch, and process management. New subsystems should use small interfaces, factories or registries, backend adapters, and focused unit tests.

## Public API Surface

`AudioToTextRecorder` is the main entry point for local use. Its public methods are part of the compatibility surface and should not change without a migration plan.

| Method | Role |
| --- | --- |
| `text(callback=None)` | Block until one utterance is transcribed, or dispatch through the callback path for asynchronous use. |
| `start()` / `stop()` | Manual recording control. |
| `listen()` | Enter the listening state, including post-wake-word flows. |
| `feed_audio(chunk)` | Push external audio into the same recorder pipeline used by microphone input. This is the boundary used by websocket and file-style sources. |
| `abort()` / `shutdown()` | Interrupt current work or tear down workers, queues, callbacks, and model resources. |

`RealtimeSTT/__init__.py` uses lazy exports. Accessing `AudioToTextRecorder`, the websocket client, audio input, or boundary detector types imports only the requested object. This keeps optional model stacks behind the code paths that need them.

## Core Recorder Flow

The recorder is built around one stream of normalized PCM frames and one turn state machine.

```text
microphone worker or feed_audio()
  -> audio_queue
  -> _recording_worker
  -> wake-word gate, VAD gate, and pre-roll buffer
  -> active recording frames
  -> realtime transcription worker for interim text
  -> final transcription worker or injected executor
  -> callbacks and/or text() return value
```

With `use_microphone=True`, a microphone reader worker opens PyAudio, reads from the selected input device at the device/native capture rate, resamples to the recorder's internal rate, and pushes chunks into `audio_queue`.

With `use_microphone=False`, callers use `feed_audio()`. External audio is resampled when needed and buffered into exact `buffer_size` frames before entering the same queue. This keeps websocket, browser, file, and custom audio sources on the same recorder path as local microphone input.

The recording worker is the turn state machine. While idle, it maintains a bounded pre-recording buffer so speech onset is not clipped. It can optionally wait for Porcupine or OpenWakeWord before entering the listening path. It uses a fast WebRTC check plus Silero confirmation before calling `start()`. While recording, it appends frames, tracks silence, can submit speculative final transcription during silence, and calls `stop()` only after configured minimum-duration and post-speech-silence rules are met.

Final transcription converts the recorded int16 frame list into float32 audio. The audio is then sent to either the internal transcription worker or an injected `transcription_executor`. Both routes return through the same project result path so synchronous `text()` and callback-based usage see the same normalized result shape.

## Concurrency Model

One `AudioToTextRecorder` coordinates several workers. The exact thread/process implementation is platform-dependent, but the responsibilities remain the same.

```text
Main recorder instance

  _audio_data_worker
      microphone / PyAudio / resampling
      -> audio_queue

  _recording_worker
      dequeue PCM chunks
      -> wake-word checks
      -> WebRTC + Silero VAD
      -> state transitions
      -> frame accumulation
      -> finalization decisions

  _realtime_worker
      snapshots or streams current frames while recording
      -> boundary scheduling or timer scheduling
      -> realtime engine
      -> RealtimeTextStabilizer

  TranscriptionWorker or external executor
      owns final ASR model path
      -> warmup
      -> engine.transcribe()
      -> TranscriptionResult
```

On Linux, `_start_thread()` uses `threading.Thread`. On other platforms it uses the PyTorch multiprocessing `Process` path. The internal final transcription path communicates over `SafePipe`; `SafePipe` serializes parent-side pipe operations through a dedicated worker thread so concurrent `send()`, `recv()`, and `poll()` calls do not corrupt pipe state.

When an external transcription executor is injected, the recorder does not create the internal final model worker. Instead, it deep-copies the audio, calls the provided executor in a daemon thread, and places either `("success", result)` or `("error", message)` into the external result queue. This is the hook used by the FastAPI server to share ASR engines across sessions while keeping recorder-local VAD and state transitions.

## Recorder State Machine

State transitions are centralized through `_set_state()`. That method is the single place that updates the current state, logs the transition, updates spinner text, and fires state-entry or state-exit callbacks.

```text
inactive ──► wakeword ──► listening ──► recording ──► transcribing ──► inactive
                │                           ▲
                └───────────────────────────┘
                  wake word detected, then voice active
```

The active states are:

| State | Meaning |
| --- | --- |
| `inactive` | No active listening or recording work is exposed to the user. |
| `wakeword` | The recorder is waiting for a configured wake word. |
| `listening` | Wake-word gating is satisfied or skipped; VAD is waiting for speech onset. |
| `recording` | Speech has started and frames are being accumulated. |
| `transcribing` | Final ASR is running for the completed utterance. |

The callback names and dispatch behavior are part of the public contract. State-change callbacks such as VAD start/stop and wake-word start/end should stay wired through `_set_state()`. General callback dispatch should stay behind `_run_callback()`, including the option to run callbacks in a new thread via `start_callback_in_new_thread`.

## Audio Currency, Pre-Roll, VAD, and Wake Words

The recorder's internal audio currency is 16 kHz mono PCM. Resampling belongs at boundaries: microphone capture, external `feed_audio()` input, and client/server packet handling. WebRTC and Silero assumptions should remain explicit, because both are tied to stream timing and sample format.

Voice onset uses a dual VAD system. WebRTC VAD is the fast frame-level detector and is used for onset timing. Silero VAD is the higher-accuracy confirmation path and can also participate in end-of-speech behavior through Silero deactivity detection. The combined speech-active decision requires recent or current WebRTC speech and active Silero speech. This reduces false starts while keeping onset responsive.

Wake-word detection is optional and sits before the listening/recording path. Porcupine and OpenWakeWord are feature backends, not core dependencies. They should remain lazily imported and should preserve existing recorder callbacks such as wake-word detection start, detection end, detected, timeout, and related lifecycle events.

Pre-roll is deliberately conservative. While the recorder is idle or listening, it keeps enough recent audio to avoid cutting off the beginning of an utterance. When VAD commits to speech, the selected pre-roll frames are included in the recording buffer before active frames continue to accumulate.

Recurrent VAD state must reset at recording boundaries. One utterance must not contaminate the next through leftover detector state, recent-speech flags, or streaming session state.

## Final Transcription Path

The final path is optimized for correctness and isolation from recorder state. The internal worker creates the selected engine through `create_transcription_engine()`, warms it with `warmup_audio.wav`, then receives `(audio, language, use_prompt)` requests through `SafePipe`.

For each request, the worker calls:

```text
engine.transcribe(audio, language=language, use_prompt=use_prompt)
```

The worker returns a `TranscriptionResult` on success or an error message on failure. The recorder then uses the same result handling for blocking `text()` calls and callback-based flows. This keeps model execution separate from the state machine, but avoids creating multiple result shapes.

Speculative or early final transcription may be submitted during silence while recording is still being evaluated. Final stop/finalization still depends on configured minimum recording duration, silence duration, and gap rules. Early work should never bypass the recorder's lifecycle rules.

## Realtime Transcription Path

Realtime transcription is optional and intentionally non-fatal. `_realtime_worker` runs only when `enable_realtime_transcription=True`. Exceptions in the realtime path are logged and skipped so they do not stop recording or final transcription.

There are three realtime modes:

| Mode | Behavior |
| --- | --- |
| Timer mode | Snapshots the current frame buffer every `realtime_processing_pause` seconds. |
| Boundary mode | Uses `RealtimeSpeechBoundaryDetector` to schedule heavier ASR work near acoustic valleys, with timer fallback and follow-up delays. |
| Streaming mode | Used only when the realtime engine declares `supports_streaming=True`; streaming engines receive only new frames through `StreamingTranscriptionSession`. Kroko-ONNX uses this path. |

`RealtimeSpeechBoundaryDetector` is intentionally lightweight. It watches PCM energy and emits `SpeechBoundaryEvent` objects at likely voiced energy valleys. It does not know about Whisper, recorder threads, servers, or GPU work. Its job is scheduling, not linguistic certainty.

Every realtime ASR result is wrapped as a `RealtimeTextObservation` and passed into `RealtimeTextStabilizer`. The stabilizer is pure and side-effect-free. It has no dependency on audio devices, ASR engines, callbacks, FastAPI, threads, or wall-clock reads. Callers provide ordered observations with deterministic timestamps. The stabilizer returns stable deltas, unstable preview text, evidence diagnostics, and finalization state.

This split matters because realtime ASR output often rewrites recent text. The recorder and server should publish usable partial text without pretending unstable text is final. The stabilizer gives the UI stable text when enough evidence exists and keeps the rest as preview.

## Transcription Engine Layer

The engine subsystem uses a factory plus adapter pattern. The recorder depends on the project-level contract, not a specific ASR backend.

```text
create_transcription_engine(name, config)
        │
        ▼
normalized engine name
lowercase, trim, replace "-" with "_"
        │
        ▼
lazy import from ENGINE_CLASS_PATHS
        │
        ▼
BaseTranscriptionEngine adapter
        │
        ▼
TranscriptionResult
```

The core contract lives in `RealtimeSTT/transcription_engines/base.py`:

| Object | Role |
| --- | --- |
| `TranscriptionEngineConfig` | Carries model, download root, compute type, GPU device index, device, beam size, initial prompt, suppress tokens, batch size, VAD filter flag, normalization flag, and engine-specific options. |
| `BaseTranscriptionEngine` | Abstract base for adapters. Implements shared warmup, prompt, and audio-normalization helpers; requires `transcribe()`. |
| `TranscriptionResult` | Project result object containing final text and `TranscriptionInfo`. |
| `TranscriptionInfo` | Carries detected language metadata such as language and probability. |
| `StreamingTranscriptionSession` | Optional extension for engines that can consume incremental audio. |

Current registered adapters include faster-whisper, whisper.cpp, OpenAI Whisper, OpenAI API, Moonshine, Moonshine streaming, Sherpa ONNX Parakeet, Sherpa ONNX Moonshine, Parakeet, Cohere Transcribe, Granite Speech, Qwen3 ASR, and Kroko ONNX. Engine aliases are registered in `ENGINE_CLASS_PATHS`.

Unknown engine names raise `UnsupportedTranscriptionEngineError` with a list of available names. Adapter modules should import optional packages lazily inside their loader paths and raise project transcription errors with clear install hints when dependencies are missing.

The OpenAI API name appears in the engine registry, but the safer architecture position is to document it as a placeholder when request handling is not wired. That avoids implying a working capability merely because the adapter name exists.

To add an engine:

1. Create a focused adapter under `RealtimeSTT/transcription_engines/`.
2. Implement `BaseTranscriptionEngine`.
3. Normalize backend output into `TranscriptionResult` and `TranscriptionInfo`.
4. Add aliases to `ENGINE_CLASS_PATHS`.
5. Keep optional imports lazy.
6. Add fast tests for factory aliases, option mapping, dependency-error messages, result conversion, and unsupported-engine behavior.

Tests should inject fake backends or patch imports. Engine contract tests should not require real model downloads unless the test is explicitly a model/device integration check.

## Server Layers

RealtimeSTT has two websocket/server layers above the recorder. They serve different compatibility and deployment needs.

### Legacy single-user websocket server

`RealtimeSTT_server/stt_server.py` is a thin asyncio/websocket wrapper around one `AudioToTextRecorder` instance. It is launched through the `stt-server` console script. The matching `AudioToTextRecorderClient` mirrors the local recorder API while streaming audio over websockets and receiving transcription events back.

This path uses separate control and data websocket ports, with defaults of `8011` and `8012`. It remains useful for compatibility and for clients already built around the original dual-port protocol. New browser and multi-user work should be centered on `example_fastapi_server`.

### FastAPI browser reference server

`example_fastapi_server` is the maintained browser streaming reference server. It is source-only and not part of the wheel. It is more capable than the legacy wrapper because it handles multiple browser sessions, shared inference resources, admission limits, metrics, runtime settings, packet validation, and websocket publishing.

The server keeps VAD and recorder state session-scoped, while sharing ASR engines across sessions.

```text
Browser websocket /ws/transcribe
        │
        ▼
RecorderBackedRealtimeSession    per websocket session
        │
        ▼
AudioToTextRecorder(use_microphone=False)
        │
        ├─ owns VAD, wake-word handling, pre-roll, realtime stabilization, finalization
        │
        └─ injected transcription_executor
                │
                ▼
        InferenceScheduler
                │
        ┌───────┴────────┐
        ▼                ▼
FairInferenceQueue   FairInferenceQueue
("main")             ("realtime")
        │                │
        ▼                ▼
SharedEngineWorker   SharedEngineWorker
final ASR lane       realtime ASR lane
```

`RealtimeSTTService` owns shared inference resources and session admission. Each websocket gets a `RecorderBackedRealtimeSession`, which creates an `AudioToTextRecorder(use_microphone=False)` and injects scheduler-backed transcription executors. The recorder still owns stream correctness: WebRTC/Silero VAD, wake-word hooks, pre-roll, early-final behavior, realtime scheduling, realtime stabilization, and finalization. The server owns cross-session concerns: admission, limits, metrics, queueing, packet validation, runtime settings, websocket publishing, and session cleanup.

`SessionStore` enforces `max_sessions` and `max_active_speakers`. Session slots are reserved before recorder construction so bursts cannot over-admit sessions while recorders are still being created.

`InferenceScheduler` shares ASR models across sessions. It has a main/final lane and a realtime lane unless `use_main_model_for_realtime` collapses both onto the main lane. Each lane has a `FairInferenceQueue` and a `SharedEngineWorker`.

`FairInferenceQueue` rotates across sessions, preserves final jobs up to a per-session depth limit, coalesces stale realtime jobs, drops expired realtime jobs, and enforces global queue limits. Final jobs are protected from being replaced by newer realtime updates. Realtime work is allowed to be coalesced because old partial results are less valuable once newer audio has arrived.

`SharedEngineWorker` loads one engine for its lane, warms it, records latency and queue metrics, and returns `InferenceResult` objects to the waiting recorder executor or session. This keeps model memory shared while preserving per-session recorder state.

### FastAPI audio packet format

The browser server accepts one binary audio packet format:

```text
uint32 little-endian metadata byte length
UTF-8 JSON metadata object, including sampleRate
PCM s16le audio bytes
```

`example_fastapi_server/protocol.py` owns packet encoding, decoding, metadata-size limits, JSON validation, and helper validation such as positive integer metadata fields. Protocol helpers should remain independent from the server runtime.

## Ownership Split

| Concern | Owner |
| --- | --- |
| Audio device capture | `AudioInput` and `_audio_data_worker` for local microphone use. |
| External audio ingestion | `feed_audio()` with `use_microphone=False`. |
| Stream-level VAD state | Per-recorder state, including WebRTC/Silero timing and resets. |
| Wake-word lifecycle | Recorder, with lazy backend dependencies and stable callbacks. |
| Pre-roll and active frames | Recorder. |
| Realtime ASR scheduling | Recorder-local timer/boundary/streaming logic; server may provide shared executors. |
| Realtime text stabilization | `RealtimeTextStabilizer`, independent of recorder and server runtime. |
| Final ASR execution | Internal `TranscriptionWorker` or injected `transcription_executor`. |
| ASR model sharing | FastAPI `InferenceScheduler` and `SharedEngineWorker`. |
| Multi-session admission | FastAPI `SessionStore` and service layer. |
| Websocket protocol validation | Server protocol helpers. |
| Public callback behavior | Recorder through `_set_state()` and `_run_callback()`. |

## Extension Points

| Subsystem | Extension rule |
| --- | --- |
| Transcription engine | Add a focused adapter module, register aliases in `ENGINE_CLASS_PATHS`, return `TranscriptionResult`, keep optional imports lazy, and test factory/result/error behavior. |
| Streaming realtime engine | Implement the optional streaming-session extension and reset streaming state at utterance boundaries. |
| Wake-word backend | Follow the transcription-engine pattern: backend adapter per dependency, lazy imports, helpful install/model-file errors, normalized detection results, and stable recorder callback names. |
| VAD backend | Define a small VAD adapter boundary before adding more backend-specific branches to `_recording_worker`. Expose explicit units, timestamps or sample metadata, confidence/score when available, and state reset behavior. |
| Audio source | Prefer `feed_audio()` with `use_microphone=False` for websocket, browser, file, or custom capture sources. |
| Server protocol | Keep binary framing and validation in protocol helpers; do not import runtime session or scheduler state from protocol code. |

Future VAD work should turn the current WebRTC-plus-Silero confirmation behavior into one composable strategy, not add more hard-coded control flow inside the recorder loop. Wake-word work should follow the same adapter discipline as ASR engines.

## Testing Strategy

Use focused unit tests first. The core behaviors to protect are contracts and boundaries, not just end-to-end transcripts.

| Area | Test focus |
| --- | --- |
| Engine factory | Alias normalization, unsupported names, lazy import behavior, dependency-error messages, and option mapping. |
| Engine adapters | Result normalization into `TranscriptionResult`, prompt handling, audio normalization, and fake-backend behavior. |
| Recorder state | `_set_state()` transitions, callback firing, manual start/stop/listen behavior, abort, and shutdown. |
| VAD and pre-roll | WebRTC/Silero combination behavior, recent-speech timing, silence handling, buffer trimming, and state reset between utterances. |
| Realtime boundary detection | Acoustic boundary events, timer fallback behavior, and independence from ASR/server code. |
| Realtime stabilization | Deterministic timestamp handling, stable deltas, unstable preview text, diagnostics, and finalization behavior. |
| Server sessions | Session admission, active-speaker limits, fair queue rotation, realtime coalescing, final queue depth, stale realtime drops, and scheduler shutdown. |
| Protocol | Packet framing, metadata limits, JSON validation, required metadata fields, and binary audio extraction. |

Golden tests and real-model tests should be opt-in. Use them when reviewing actual model quality, device behavior, latency, or integration with a downloaded runtime. They should not be required for ordinary adapter, scheduler, or helper changes.

## Design Invariants

- Keep `AudioToTextRecorder` constructor parameters, public methods, callback names, text formatting, error messages, and exported names backward compatible.
- Keep state transitions and lifecycle callbacks centralized through `_set_state()`.
- Keep callback dispatch centralized through `_run_callback()`.
- Preserve logging and diagnostic callbacks; integrations may depend on lifecycle visibility, not only final text.
- Keep optional engine, VAD, and wake-word dependencies lazy. Importing `RealtimeSTT` should not import every model runtime.
- Treat 16 kHz mono PCM as the recorder's internal audio currency. Resample at boundaries and keep WebRTC/Silero assumptions explicit.
- Reset recurrent VAD and streaming ASR state at recording boundaries.
- Keep pure helpers pure. `preroll.py`, `realtime_boundary_detector.py`, `realtime_text_stabilizer.py`, and protocol helpers should remain testable without devices, model downloads, threads, or servers.
- Return project result objects from project-facing code. Do not leak raw third-party objects through recorder, engine, boundary, or server APIs.
- Keep server-only scheduling, metrics, admission, runtime settings, and websocket publishing out of `AudioToTextRecorder`.
- Prefer small base interfaces, factories/registries, and backend adapters for new subsystems.
- Validate behavior with focused unit tests before relying on real-model golden tests.
