# Architecture

RealtimeSTT is a recorder-centered speech pipeline. Audio is normalized into
short PCM chunks, turn detection decides which chunks form an utterance, and
pluggable ASR engines produce realtime and final text through one small adapter
contract. Heavy optional dependencies stay behind lazy imports so the core
package can install without every engine stack.

## Package Map

| Area | Main files | Responsibility |
| --- | --- | --- |
| Public API | `RealtimeSTT/__init__.py` | Lazy exports for `AudioToTextRecorder`, the websocket client, audio input, and realtime boundary types. |
| Recorder state machine | `RealtimeSTT/audio_recorder.py` | Audio queue consumption, VAD, wake words, pre-roll, recording lifecycle, realtime updates, final transcription, callbacks, and shutdown. |
| Audio input | `RealtimeSTT/audio_input.py` | PyAudio device selection, microphone stream setup, chunk reads, and resampling helpers used by client-side capture. |
| VAD and turn helpers | `RealtimeSTT/silero_vad.py`, `RealtimeSTT/preroll.py`, `RealtimeSTT/realtime_boundary_detector.py` | Silero backend selection, conservative pre-recording buffer trimming, and low-cost acoustic boundary scheduling. |
| Text stabilization | `RealtimeSTT/realtime_text_stabilizer.py` | Pure, timestamp-driven stabilization of partial ASR text into stable deltas and display text. |
| ASR engines | `RealtimeSTT/transcription_engines/*` | Lazy engine adapters implementing a shared transcription result contract. |
| Client/server wrappers | `RealtimeSTT/audio_recorder_client.py`, `RealtimeSTT_server/*`, `example_fastapi_server/*` | Legacy websocket client/server and the source-only FastAPI browser streaming reference server. |
| Tests and docs | `tests/unit/*`, `docs/*` | Contract tests for engines, VAD, realtime behavior, server sessions, and user-facing reference docs. |

## Core Recorder Flow

```text
microphone worker or feed_audio()
  -> audio_queue
  -> _recording_worker
  -> wake word and VAD gates
  -> pre-roll + active recording frames
  -> _realtime_worker for interim text
  -> transcription worker or injected executor for final text
  -> callbacks / text() return value
```

`AudioToTextRecorder` owns the live state. With `use_microphone=True`, it starts
a microphone reader worker that pushes 16-bit mono PCM chunks into
`audio_queue`. With `use_microphone=False`, callers feed external chunks through
`feed_audio()`, which resamples NumPy input when needed and buffers exact
`buffer_size` frames.

The recording worker is the turn state machine. While idle it keeps a bounded
pre-recording buffer, optionally waits for Porcupine or OpenWakeWord, and uses a
fast WebRTC check plus Silero confirmation before calling `start()`. While
recording it appends frames, tracks silence, can submit speculative final
transcription during silence, and calls `stop()` only after the configured
minimum duration and post-speech silence requirements are met.

Final transcription converts the recorded int16 frame list to float32 audio and
sends it to either the internal transcription worker or an injected
`transcription_executor`. The internal worker creates the selected ASR engine,
warms it with `warmup_audio.wav`, then receives `(audio, language, use_prompt)`
requests through `SafePipe`. This keeps model work separated from the recorder
state machine and preserves the same result path for synchronous `text()` and
asynchronous callbacks.

## Realtime Path

Realtime transcription is optional and intentionally non-fatal. `_realtime_worker`
only runs when `enable_realtime_transcription=True`; exceptions are logged and
skipped so they do not stop recording.

There are three realtime modes:

- Timer mode snapshots the current frame buffer every
  `realtime_processing_pause` seconds.
- Boundary mode uses `RealtimeSpeechBoundaryDetector` to trigger heavier ASR
  work near acoustic valleys, with timer fallback and follow-up delays.
- Streaming mode is used only when the realtime engine declares
  `supports_streaming=True`. Kroko-ONNX uses this path and receives only new
  frames through a `StreamingTranscriptionSession`.

Each realtime result is wrapped as a `RealtimeTextObservation` and passed into
`RealtimeTextStabilizer`. The stabilizer has no audio, thread, or ASR
dependency; it accepts ordered observations and returns stable deltas, unstable
preview text, evidence diagnostics, and finalization state.

## Engine Layer

The transcription contract lives in `RealtimeSTT/transcription_engines/base.py`:

- `TranscriptionEngineConfig` carries model, device, beam, prompt, VAD filter,
  normalization, and engine-specific options.
- `BaseTranscriptionEngine.transcribe()` returns `TranscriptionResult`.
- `TranscriptionInfo` carries detected language metadata.
- `StreamingTranscriptionSession` is an optional extension for incremental
  engines.

`factory.py` normalizes engine names by lowercasing and replacing `-` with `_`,
then lazy-loads the selected adapter. Engine modules import optional packages
inside their own loader paths and raise `TranscriptionEngineError` with install
hints when a dependency is missing. `openai_api` is a placeholder and raises
immediately because request handling is not wired.

To add an engine, implement `BaseTranscriptionEngine`, map its aliases in
`factory.py`, keep optional imports lazy, normalize model output into
`TranscriptionResult`, and add fast contract tests for dependency errors,
option mapping, result conversion, and factory selection.

## Server Architecture

`example_fastapi_server` is the maintained browser streaming reference server.
It is source-only and not part of the wheel. Its websocket endpoint accepts a
single binary packet format:

```text
uint32 little-endian metadata byte length
UTF-8 JSON metadata, including sampleRate
PCM s16le audio bytes
```

`RealtimeSTTService` owns shared inference resources and session admission.
Each websocket gets a `RecorderBackedRealtimeSession`, which creates an
`AudioToTextRecorder(use_microphone=False)` and injects scheduler-backed
transcription executors. The recorder still owns VAD, wake-word handling,
pre-roll, realtime stabilization, and finalization; the server owns admission,
metrics, limits, packet validation, runtime settings, and websocket publishing.

`InferenceScheduler` shares ASR models across sessions. It has final and
realtime lanes unless `use_main_model_for_realtime` collapses both onto the main
lane. `FairInferenceQueue` rotates across sessions, coalesces stale realtime
jobs, caps final backlog, and enforces global queue limits. `SharedEngineWorker`
loads one engine per lane, warms it, records latency metrics, and sends
`InferenceResult` objects back to the waiting recorder executor or session.

`RealtimeSTT_server` and `AudioToTextRecorderClient` are the older dual-port
websocket path. They remain useful for compatibility, but new browser work is
centered on `example_fastapi_server`.

## Design Invariants

- Keep `AudioToTextRecorder` constructor parameters, callbacks, text formatting,
  public methods, error messages, and exported names backward compatible.
- Preserve logging and diagnostic callbacks; many tests and integrations depend
  on lifecycle visibility, not just returned text.
- Keep optional engine and wake-word dependencies lazy. Importing `RealtimeSTT`
  should not import every model runtime.
- Treat 16 kHz mono PCM as the recorder's internal audio currency. Resample at
  boundaries and keep WebRTC/Silero assumptions explicit.
- Reset recurrent VAD and streaming ASR state at recording boundaries so one
  utterance cannot contaminate the next.
- Keep pure helpers pure. `preroll.py`, `realtime_boundary_detector.py`, and
  `realtime_text_stabilizer.py` are deliberately testable without devices,
  model downloads, threads, or servers.
- Validate behavior with focused unit tests first, then opt into golden tests
  only when real model, device, or latency behavior is under review.
