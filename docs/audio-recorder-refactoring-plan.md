# Audio Recorder Refactoring Plan

## Goal

Shrink `RealtimeSTT/audio_recorder.py` by moving complete recorder
responsibilities into internal modules while preserving the public recorder API.

Target structure:

```text
RealtimeSTT/
  audio_recorder.py
  core/
    __init__.py
    transcription.py
    realtime.py
    recording.py
    voice_activity.py
    wakeword.py
    initialization.py
    safepipe.py
    silero_vad.py
```

`RealtimeSTT/audio_recorder.py` remains the public facade. Callers must still be
able to import `AudioToTextRecorder` from `RealtimeSTT` and from
`RealtimeSTT.audio_recorder`.

## Governing Rule

A good refactor boundary is based on cohesion and coupling. Line count is
secondary.

Cohesion means the extracted code changes for the same reason. Coupling means
how much the extracted code depends on the rest of the recorder. For this file,
that means extracting recorder subsystems such as transcription, wake words,
voice activity, recording, and realtime processing. It does not mean extracting
fixed 50-line chunks.

## Compatibility Rules

- Do not change `AudioToTextRecorder.__init__` arguments, defaults, ordering, or
  accepted values.
- Do not change public imports from `RealtimeSTT` or `RealtimeSTT.audio_recorder`.
- Keep existing private method names as delegating wrappers during the split
  when tests, examples, or server code currently call them.
- Preserve log statements, log levels, message text, exception types, exception
  messages, callback order, queue behavior, thread lifecycle, and timing
  defaults.
- Wake-word internals live under `RealtimeSTT/core/wakeword.py`; the old
  `RealtimeSTT/wakeword_dependencies.py` facade has been removed.

## Baseline Validation

Run the smallest relevant set for each pass. Useful focused gates include:

```powershell
python -m pytest tests\unit\test_wakeword.py
python -m pytest tests\unit\test_silero_vad_backend.py tests\unit\test_slow_final_transcription_audio_gap.py
python -m pytest tests\unit\test_preroll.py tests\unit\test_audio_recorder_preroll_integration.py
python -m pytest tests\unit\test_realtime_text_stabilizer.py tests\unit\test_realtime_streaming_transcription.py
python -m pytest tests\unit\test_fastapi_server_protocol.py
```

For syntax-only confidence after mechanical moves:

```powershell
python -m py_compile RealtimeSTT\audio_recorder.py
```

## Milestone 0: Map and Test Gaps

Purpose: make the current behavior visible before moving more code.

Milestone 0 findings live in `docs/audio-recorder-refactoring-map.md`.

Actions:

- Produce a method-to-responsibility map for `audio_recorder.py`.
- List tests covering each responsibility.
- Add characterization tests only where a planned move has weak coverage.
- Record any private methods that tests or server code call directly.

Exit criteria:

- The next extraction has a named responsibility, target module, compatibility
  plan, and validation command.

## Milestone 1: Create Internal Package Shell

Purpose: introduce the destination without changing behavior.

Actions:

- Add `RealtimeSTT/core/__init__.py`.
- Do not move runtime behavior yet.
- Optionally add a short internal-package note that this is not a public API.

Validation:

```powershell
python -m py_compile RealtimeSTT\audio_recorder.py
```

## Milestone 2: Extract Transcription Runtime

Purpose: isolate final transcription process and executor plumbing.

Move to `RealtimeSTT/core/transcription.py`:

- `TranscriptionWorker`
- process entry helpers for transcription workers
- request submission and result receiving helpers, where practical

Compatibility plan:

- Keep `AudioToTextRecorder` methods with the same names as thin delegating
  wrappers if any internal tests or callers use them.
- Preserve pipe messages, queue behavior, model creation options, logging, and
  exception handling.

Validation:

```powershell
python -m py_compile RealtimeSTT\audio_recorder.py RealtimeSTT\core\transcription.py
python -m pytest tests\unit\test_slow_final_transcription_audio_gap.py
```

## Milestone 3: Extract Wake-Word Runtime

Purpose: put wake-word backend selection, optional dependency loading, setup,
processing, and cleanup behind one boundary.

Move or centralize in `RealtimeSTT/core/wakeword.py`:

- wake-word backend constants
- backend normalization
- optional dependency loading
- runtime processing currently handled by `_process_wakeword`
- setup/cleanup helpers currently embedded in constructor or shutdown paths

Compatibility plan:

- Import wake-word helpers directly from `RealtimeSTT.core.wakeword`.
- Keep `AudioToTextRecorder._process_wakeword()` as a wrapper at first.
- Preserve Porcupine/OpenWakeWord error messages and extra names.

Validation:

```powershell
python -m pytest tests\unit\test_wakeword.py
python -m pytest tests\unit\test_fastapi_server_multi_user.py
```

## Milestone 4: Extract Voice Activity and Pre-Roll Glue

Purpose: separate recorder-level VAD decisions from the main recorder facade.

Move to `RealtimeSTT/core/voice_activity.py`:

- Silero probability and state reset helpers
- WebRTC speech checks
- combined voice activity checks
- pre-recording buffer helpers
- pre-roll metadata and diagnostics glue

Compatibility plan:

- Keep existing recorder method names as wrappers during the migration.
- Do not change Silero backend behavior or `RealtimeSTT/core/preroll.py` public
  behavior; this milestone only moves recorder glue.
- Preserve async Silero generation behavior and stale-result protection.

Validation:

```powershell
python -m pytest tests\unit\test_silero_vad_backend.py tests\unit\test_slow_final_transcription_audio_gap.py
python -m pytest tests\unit\test_preroll.py tests\unit\test_audio_recorder_preroll_integration.py
```

## Milestone 5: Extract Recording Worker

Purpose: move the main audio queue consumption and recording state loop after
its wake-word and VAD dependencies have stable internal boundaries.

Move to `RealtimeSTT/core/recording.py`:

- `_recording_worker` loop body as `run_recording_worker(recorder)` or an
  equivalent internal class/function.
- small local helpers that only serve the recording loop.

Compatibility plan:

- Keep `AudioToTextRecorder._recording_worker()` as a delegating wrapper.
- Preserve state transitions, callback order, buffer overflow handling,
  `on_recorded_chunk`, early transcription behavior, and logging text.

Validation:

```powershell
python -m pytest tests\unit\test_slow_final_transcription_audio_gap.py
python -m pytest tests\unit\test_audio_recorder_preroll_integration.py
python -m pytest tests\unit\test_fastapi_server_protocol.py
```

## Milestone 6: Extract Realtime Worker

Purpose: isolate realtime transcription scheduling, streaming-session behavior,
partial text handling, and stabilization callbacks.

Move to `RealtimeSTT/core/realtime.py`:

- `_realtime_worker` loop body
- realtime frame snapshot and conversion helpers
- streaming-session lifecycle helpers
- realtime stabilization callback plumbing where practical

Compatibility plan:

- Keep `AudioToTextRecorder._realtime_worker()` as a delegating wrapper.
- Preserve the non-fatal realtime error policy: realtime failures are logged and
  skipped without crashing recording or final transcription.
- Preserve full-buffer behavior for non-streaming engines and incremental
  behavior for streaming engines.

Validation:

```powershell
python -m pytest tests\unit\test_realtime_text_stabilizer.py tests\unit\test_realtime_streaming_transcription.py
python -m pytest tests\unit\test_realtime_boundary_detector.py
```

## Milestone 7: Split Initialization Last

Purpose: make the constructor readable only after the runtime subsystems have
clear module boundaries.

Move to `RealtimeSTT/core/initialization.py`:

- grouped setup helpers for logging, callbacks, VAD, wake words, transcription
  engines, realtime state, queues, and worker startup.

Compatibility plan:

- Do not change the constructor signature.
- Do not introduce required config objects in the public API.
- Keep all stored public and semi-public attributes available with the same
  names.

Validation:

```powershell
python -m pytest tests\unit\test_silero_vad_backend.py
python -m pytest tests\unit\test_fastapi_server_protocol.py
python -m py_compile RealtimeSTT\audio_recorder.py RealtimeSTT\core\initialization.py
```

## Milestone 8: Cleanup and Boundary Enforcement

Purpose: remove only temporary internal duplication after behavior has stayed
stable across the previous milestones.

Actions:

- Remove internal wrappers only when no tests, examples, or server code depend
  on them.
- Keep public compatibility facades.
- Add a lightweight import-boundary check if needed to prevent new large
  responsibilities from accumulating in `audio_recorder.py`.

Validation:

```powershell
python -m pytest tests\unit
```

## Pass Template

Before each code-moving pass, state:

1. Current behavior.
2. Proposed structural change.
3. Public API compatibility plan.
4. Validation commands.

After each code-moving pass:

1. Run the focused validation.
2. Review the diff for behavior drift, removed logs, changed errors, changed
   callback order, and unrelated formatting churn.
3. Fix only regressions caused by that pass.
4. Commit or checkpoint before starting the next responsibility.
