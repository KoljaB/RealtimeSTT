# API Compatibility

This document records the public and compatibility-sensitive API surfaces that
refactors should preserve. It is a guardrail for code movement, not a replacement
for the user-facing constructor reference in `docs/configuration.md`.

## Compatibility Promise

RealtimeSTT's primary public API is the local recorder:

```python
from RealtimeSTT import AudioToTextRecorder
```

Existing users also import directly from implementation modules in tests and
integrations, especially:

```python
from RealtimeSTT.audio_recorder import AudioToTextRecorder
from RealtimeSTT import AudioToTextRecorderClient
```

Refactors must keep these import paths working. If code moves, keep a wrapper,
facade, or deprecated re-export at the old path until an explicit removal
milestone.

## Primary Recorder Surface

`AudioToTextRecorder` is the main compatibility boundary. The constructor,
callbacks, text formatting, lifecycle methods, audio feeding behavior, and
observable state are all used outside the package.

| Surface | Compatibility rule |
| --- | --- |
| Constructor parameter names and defaults | Preserve unless a breaking change is explicitly approved. Adding parameters is safer than renaming or repurposing existing ones. |
| Constructor parameter order | Treat as compatibility-sensitive. Some tests inspect signature order around legacy and new Silero options. |
| Constructor side effects | Preserve worker startup, model warmup, microphone setup, logging setup, and optional dependency laziness. |
| Public methods | Preserve method names, arguments, return values, blocking behavior, callback behavior, and exception/error text. |
| Public attributes observed by callers | Preserve at least `is_recording`, `state`, detected language fields, and final/realtime text metadata fields unless intentionally migrated with compatibility accessors. |
| Text formatting defaults | Preserve stripping/collapsing whitespace, sentence capitalization, and final punctuation behavior controlled by constructor flags. |

The complete constructor parameter list lives in
`docs/configuration.md`. Keep that page aligned with `AudioToTextRecorder.__init__`
when parameters are added.

## Constructor Compatibility

All constructor parameters are public, including model/engine settings, audio
input settings, VAD and wake-word tuning, realtime transcription settings,
callbacks, logging flags, executor injection, and Silero backend options.

High-risk constructor changes:

- Removing or renaming a parameter.
- Changing a default value.
- Changing a parameter's type interpretation.
- Moving compatibility parameters in a way that breaks signature-order tests.
- Importing optional engine, VAD, or wake-word dependencies at package import
  time instead of lazy runtime paths.
- Changing callback argument shapes.
- Changing whether `use_microphone=True` starts local capture automatically.
- Changing whether `use_microphone=False` expects audio through `feed_audio()`.

Known compatibility-sensitive constructor points:

| Parameter group | Current expectation |
| --- | --- |
| `silero_use_onnx`, `silero_backend`, `silero_onnx_model_path`, `silero_onnx_threads` | Legacy and newer Silero options coexist. `silero_use_onnx` currently defaults to `None`; `silero_backend` defaults to `"auto"`. |
| `transcription_executor`, `realtime_transcription_executor` | Server/tests can inject shared inference executors instead of using the default process/model path. |
| `realtime_transcription_engine_options` | `None` reuses main engine options; explicit dictionaries apply to realtime only. |
| `start_callback_in_new_thread` | Controls whether callbacks run in a new thread. Callback ordering and arguments still need to remain stable. |

## Main Method Contracts

### `text(on_transcription_finished=None)`

`text()` is the main final transcription call.

Current behavior to preserve:

- Clears interrupt state before waiting.
- Calls `wait_audio()` to wait for an utterance when recording was not manually
  started and stopped.
- Returns `""` when shutdown or interruption prevents transcription.
- Without a callback, returns the final transcription string.
- With a callback, starts a thread that calls the callback with the final string.
- Calls `transcribe()` after audio has been selected.
- May block while waiting for VAD start/stop, queued audio, or transcription.

Common external usage:

```python
text = recorder.text()
recorder.text(process_text)
```

### `feed_audio(chunk, original_sample_rate=16000)`

`feed_audio()` is the public external-audio input path when
`use_microphone=False`.

Current behavior to preserve:

- Accepts raw 16-bit PCM bytes and NumPy arrays.
- Converts stereo NumPy arrays to mono by averaging channels.
- Resamples NumPy input to 16000 Hz when `original_sample_rate` differs.
- Converts NumPy input to `int16` bytes.
- Buffers partial input internally.
- Emits exact queue chunks of `2 * buffer_size` bytes.
- Preserves input order and leaves incomplete tail bytes in `recorder.buffer`.
- Does not return a value.

Callers should feed small ordered chunks and enough trailing silence for VAD to
finalize finite streams.

### `start(frames=None)`

`start()` manually starts recording without waiting for voice activity.

Current behavior to preserve:

- Returns `self`, enabling `recorder.stop().text()` style usage.
- Respects `min_gap_between_recordings`; if called too soon, it logs and returns
  `self` without starting.
- Sets state to `"recording"`.
- Clears previous text/realtime state for the new recording.
- Increments the realtime recording generation and resets the realtime
  stabilizer.
- Resets Silero VAD state.
- Sets `is_recording=True`, clears the stop event, and sets the start event.
- Runs `on_recording_start` when provided.
- Accepts optional initial `frames`, which tests use directly.

### `stop(backdate_stop_seconds=0.0, backdate_resume_seconds=0.0)`

`stop()` manually stops active recording.

Current behavior to preserve:

- Returns `self`.
- Respects `min_length_of_recording`; if called too soon, it logs and returns
  `self` without stopping.
- Copies current frames into `last_frames`.
- Queues the stopped recording for `wait_audio()`/`text()`.
- Stores backdate values so final audio can trim or retain samples correctly.
- Clears active frames and sets `is_recording=False`.
- Finalizes realtime stabilization when active.
- Resets Silero and WebRTC activity state.
- Clears the start event and sets the stop event.
- Runs `on_recording_stop` when provided.

### `wait_audio()`

`wait_audio()` is lower level than `text()`, but it is used by examples and
tests, so treat it as public enough to preserve.

Current behavior to preserve:

- Arms voice-activity recording when needed.
- Waits for recording start and stop unless interrupted.
- Consumes queued recordings before falling back to current or last frames.
- Converts selected int16 frames to float32 audio in `recorder.audio`.
- Applies stop/resume backdating.
- Clears consumed frames after recording is complete.
- Restores state to `"inactive"` when appropriate.
- Supports continuous listening after non-wake-word VAD completion.

### `transcribe()` And `perform_final_transcription(...)`

These methods are not the preferred user entry point, but tests and server
paths exercise them.

Current behavior to preserve:

- `transcribe()` copies `recorder.audio`, sets state to `"transcribing"`, runs
  `on_transcription_start` if present, and then calls final transcription unless
  the callback aborts it.
- `perform_final_transcription()` returns `""` for missing audio, submits work
  to either the internal worker or injected executor, updates detected language
  metadata, stores last transcription audio/metadata, applies text formatting,
  and raises on transcription errors.
- Injected executors must receive `(audio, language=None, use_prompt=True)` or
  equivalent method calls and return the same result contract used by engine
  adapters.

### `abort()`

`abort()` interrupts the current waiting/transcription flow.

Current behavior to preserve:

- Disables automatic start/stop gates.
- Sets the interrupt event.
- Waits for interruption acknowledgement when active.
- Moves state through `"transcribing"` when interruption happens outside the
  inactive state.
- Stops active recording if needed.
- Causes `text()`/transcription paths to return `""` when interrupted.

### `shutdown()`

`shutdown()` is the explicit cleanup method and is called by context manager
exit.

Current behavior to preserve:

- Is idempotent.
- Forces `wait_audio()` and `text()` to exit.
- Sets shutdown/running/recording state flags.
- Signals worker shutdown events.
- Joins recording, reader, transcription, and realtime workers where present.
- Terminates reader/transcription processes when they do not exit in time.
- Closes parent transcription pipes.
- Releases realtime model references and triggers garbage collection.

The context manager protocol must continue to call `shutdown()`:

```python
with AudioToTextRecorder() as recorder:
    print(recorder.text())
```

## Secondary Recorder Methods And State

These are smaller surfaces but already used by tests, examples, servers, or
legacy clients.

| API | Current expectation |
| --- | --- |
| `set_microphone(microphone_on=True)` | Toggles microphone capture state and logs the new value. |
| `wakeup()` | Simulates wake-word activation by setting the listen start time. |
| `listen()` | Enters listening state and arms voice activity start. |
| `clear_audio_queue()` | Clears pre-roll and drains pending audio queue items. |
| `has_pending_recordings()` | Returns whether queued stopped recordings are waiting to be consumed. |
| `flush_buffered_audio(min_abs_level=50)` | Queues non-silent buffered audio, stops first if currently recording, and returns whether audio was queued/stopped. |
| `is_recording` | Boolean state read by servers/tests to decide whether a session is active. |
| `state` | String state used for lifecycle visibility; current values include `"inactive"`, `"listening"`, `"wakeword"`, `"recording"`, and `"transcribing"`. |

Avoid renaming these without adding compatibility accessors.

## Callback Contract

Callbacks are a public integration surface. Preserve their names, timing, and
argument shapes.

| Callback | Current argument shape |
| --- | --- |
| `on_recording_start` | No arguments. |
| `on_recording_stop` | No arguments. |
| `on_transcription_start` | Receives a copy of the final audio. Truthy return aborts the built-in final transcription path. |
| `on_realtime_transcription_update` | Receives text. |
| `on_realtime_transcription_stabilized` | Receives text. |
| `on_realtime_text_stabilization_update` | Receives the structured stabilization event. |
| `on_vad_start`, `on_vad_stop` | No arguments. |
| `on_vad_detect_start`, `on_vad_detect_stop` | No arguments. |
| `on_turn_detection_start`, `on_turn_detection_stop` | No arguments. |
| `on_wakeword_detected`, `on_wakeword_timeout` | No arguments. |
| `on_wakeword_detection_start`, `on_wakeword_detection_end` | No arguments. |
| `on_recorded_chunk` | Receives raw recorded chunk bytes. |

Preserve callback ordering relative to state transitions and logs. If callback
execution moves between threads, keep `start_callback_in_new_thread` behavior
and document the migration.

## Engine And Result Contracts

The recorder's final and realtime transcription paths depend on the shared
engine contract in `RealtimeSTT/transcription_engines/base.py`.

Compatibility-sensitive types:

- `TranscriptionEngineConfig`
- `TranscriptionResult`
- `TranscriptionInfo`
- `BaseTranscriptionEngine`
- `StreamingTranscriptionSession`
- `TranscriptionEngineError`
- `UnsupportedTranscriptionEngineError`

Compatibility-sensitive behavior:

- `BaseTranscriptionEngine.transcribe()` returns `TranscriptionResult`.
- `StreamingTranscriptionSession.accept_audio()`, `decode()`, `get_result()`,
  `finish()`, `reset()`, and `close()` keep their meanings.
- Engine adapters keep optional dependencies lazy and raise
  `TranscriptionEngineError` or a subclass with useful install guidance.
- `factory.py` keeps existing aliases and normalizes names by lowercasing and
  replacing `-` with `_`.
- Unsupported-engine error messages should continue listing available engines.

## Client And Server Compatibility

`AudioToTextRecorderClient` mirrors part of the local recorder API over the
legacy websocket server. It remains a compatibility surface while
`RealtimeSTT_server` exists.

Preserve these client methods unless explicitly deprecating the legacy path:

- `text(on_transcription_finished=None)`
- `feed_audio(chunk, audio_meta_data, original_sample_rate=16000)`
- `set_microphone(microphone_on=True)`
- `abort()`
- `wakeup()`
- `clear_audio_queue()`
- `stop()`
- `shutdown()`
- context manager `__enter__`/`__exit__`
- `set_parameter()`, `get_parameter()`, and `call_method()`

The maintained FastAPI browser server does not replace the local recorder API.
It constructs recorder instances, feeds external audio, reads `is_recording`,
calls `text()`, `abort()`, and `shutdown()`, and injects scheduler-backed
transcription executors.

## Known Usage Patterns

Tests, examples, and docs currently exercise these patterns:

```python
with AudioToTextRecorder() as recorder:
    print(recorder.text())
```

```python
recorder = AudioToTextRecorder()
while True:
    recorder.text(process_text)
```

```python
recorder = AudioToTextRecorder(use_microphone=False)
recorder.feed_audio(pcm_bytes, original_sample_rate=16000)
print(recorder.text())
recorder.shutdown()
```

```python
recorder.start()
user_text = recorder.stop().text()
```

```python
recorder.set_microphone(False)
recorder.set_microphone(True)
```

```python
if not recorder.is_recording and not recorder.has_pending_recordings():
    ...
```

## Refactor Safety Checklist

Before changing any API surface above, answer these questions in the PR or
working notes:

- Does `from RealtimeSTT import AudioToTextRecorder` still lazy-load?
- Does `from RealtimeSTT.audio_recorder import AudioToTextRecorder` still work?
- Did any constructor parameter name, default, type meaning, or order change?
- Did any public method signature or return value change?
- Did `text()` blocking, callback, or interruption behavior change?
- Did `feed_audio()` chunking, resampling, type conversion, or ordering change?
- Did `start()`/`stop()` still return `self` and respect timing guards?
- Did callback timing, threading, or argument shape change?
- Did text formatting defaults change?
- Did logs, warnings, exceptions, or error messages change?
- Did optional dependencies remain lazy?
- Did old client/server protocols remain compatible?

## Validation Commands

For documentation-only edits:

```powershell
Get-Content docs\api-compatibility.md
git -c safe.directory=D:/Projekte/STT/RealtimeSTT/RealtimeSTT status --short docs/api-compatibility.md
```

For code refactors touching the recorder API, start with the smallest relevant
tests:

```powershell
python -m pytest tests\unit\test_audio_fixtures.py
python -m pytest tests\unit\test_slow_final_transcription_audio_gap.py
python -m pytest tests\unit\test_silero_vad_backend.py
python -m pytest tests\unit\test_realtime_text_stabilizer.py
python -m pytest tests\unit\test_realtime_streaming_transcription.py
python -m pytest tests\unit\test_fastapi_server_protocol.py
python -m pytest tests\unit\test_fastapi_server_multi_user.py
```

For engine contract changes, add the engine-specific test file and factory
coverage:

```powershell
python -m pytest tests\unit\test_additional_transcription_engines.py
python -m pytest tests\unit\test_faster_whisper_engine.py
python -m pytest tests\unit\test_kroko_onnx_engine.py
```

Run opt-in golden or live scripts only when the change touches real model,
device, websocket, or latency behavior.
