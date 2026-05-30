# Audio Recorder Refactoring Map

## Purpose

This is the Milestone 0 map for splitting `RealtimeSTT/audio_recorder.py`.
It records the current responsibilities, known tests, direct private-method
callers, and the next extraction gate before more code is moved.

## Source Snapshot

`RealtimeSTT/audio_recorder.py` currently has 4,292 lines.

Large spans:

| Span | Code | Responsibility |
|---|---|---|
| 120-248 | `TranscriptionWorker` | Internal final transcription process, model creation, warmup, pipe polling, final ASR execution. |
| 264-1172 | `AudioToTextRecorder.__init__` | Constructor API, option normalization, state setup, model/process setup, worker startup. |
| 1282-1623 | `_audio_data_worker` | Microphone input, resampling, queueing normalized chunks. |
| 2275-2722 | `_recording_worker` | Main turn state machine, wake-word gating, VAD start/stop, early final transcription, recording lifecycle. |
| 2724-3557 | `_realtime_worker` | Realtime transcription scheduling, streaming-session path, partial text stabilization, realtime callbacks. |
| 3732-4079 | VAD and pre-roll helpers | Silero/WebRTC decisions, recurrent VAD reset, pre-recording buffer metadata and selection. |
| 4081-4292 | State, formatting, callbacks, context manager | State transitions, spinner text, transcript formatting, realtime callback publication, cleanup protocol. |

## Responsibility Map

| Responsibility | Current code | Side effects and coupling | Existing coverage |
|---|---|---|---|
| Public recorder API and constructor compatibility | `AudioToTextRecorder.__init__`, `text`, `start`, `stop`, `listen`, `feed_audio`, `shutdown`, context manager | Public constructor parameters, callback names, state attributes, worker startup, optional dependency behavior. Highly coupled. | `docs/api-compatibility.md`, `tests/unit/test_silero_vad_backend.py`, `tests/unit/test_fastapi_server_protocol.py`, server tests. |
| Final transcription runtime | `TranscriptionWorker`, `_transcription_worker`, `_call_transcription_executor`, `_submit_transcription_request`, `_receive_transcription_result`, `perform_final_transcription`, `transcribe` | Multiprocessing/threading, `SafePipe`, model factory, warmup WAV, external executor path, result queue, language metadata, interruption behavior. | External executor path in `tests/unit/test_fastapi_server_protocol.py`; gap for full internal worker run because it loads real ASR dependencies. |
| Audio ingestion | `_audio_data_worker`, `feed_audio`, `set_microphone` | PyAudio device reads, resampling, buffer-size chunking, interrupt events, queue writes. | `tests/unit/test_audio_fixtures.py` and integration/manual tests. |
| Recording lifecycle and queueing | `wakeup`, `abort`, `wait_audio`, `_set_audio_from_frames`, `_queue_recorded_audio`, `_get_next_recorded_audio`, `has_pending_recordings`, `flush_buffered_audio`, `start`, `stop`, `listen`, `shutdown` | State changes, locks, callbacks, finalization, queue consumption, thread joins, recurrent state reset. | `tests/unit/test_slow_final_transcription_audio_gap.py`, `tests/unit/test_realtime_text_stabilizer.py`, `tests/unit/test_fastapi_server_protocol.py`. |
| Wake-word runtime | `_normalize_wakeword_backend`, `_load_porcupine_module`, `_load_openwakeword_modules`, `_process_wakeword`, constructor setup and shutdown cleanup | Optional imports, Porcupine/OpenWakeWord model state, callback states, follow-up recording flow. | Dependency loading in `tests/unit/test_wakeword.py`; server wake-word config in `tests/unit/test_fastapi_server_multi_user.py`. Runtime detection is weakly characterized. |
| Voice activity and pre-roll glue | `_silero_vad_probability`, `_reset_silero_vad_state`, `_warmup_voice_activity_detectors`, `_is_silero_speech`, `_is_webrtc_speech`, `_check_voice_activity`, pre-recording buffer helpers | WebRTC/Silero timing, async stale-generation protection, VAD locks, pre-roll diagnostics, recording-boundary reset. | `tests/unit/test_silero_vad_backend.py`, `tests/unit/test_slow_final_transcription_audio_gap.py`, `tests/unit/test_preroll.py`, `tests/unit/test_audio_recorder_preroll_integration.py`. |
| Recording worker loop | `_recording_worker` | Consumes `audio_queue`, manages overflow, wake-word timeout, VAD start/stop, early final submission, stop decisions, callbacks. | Indirect tests via VAD/pre-roll/final path; no small direct worker-loop characterization. |
| Realtime worker loop | `_realtime_worker`, `_on_realtime_transcription_stabilized`, `_on_realtime_transcription_update` | Thread loop, frame snapshots, main/realtime model calls, streaming sessions, callback publication, non-fatal error policy. | `tests/unit/test_realtime_streaming_transcription.py`, `tests/unit/test_realtime_text_stabilizer.py`, `tests/unit/test_realtime_boundary_detector.py`. |
| State and callback dispatch | `_set_state`, `_run_callback`, `_set_state_after_transcription` | Callback ordering, spinner text, state strings, optional callback thread behavior. | `tests/unit/test_fastapi_server_protocol.py`, `tests/unit/test_realtime_text_stabilizer.py`, `tests/unit/test_slow_final_transcription_audio_gap.py`; architecture docs mark `_set_state` central. |
| Text post-processing and display | `format_number`, `_set_spinner`, `_preprocess_output`, `_find_tail_match_in_text`, `bcolors` | Console output, punctuation/casing behavior, spinner side effects. | Weak direct coverage. Keep in facade until a later low-risk formatting pass. |

## Direct Private-Method Callers

These names should stay as recorder methods or wrappers during the split.

| Private method | Direct callers outside implementation |
|---|---|
| `_selected_pre_recording_buffer_frames` | `tests/unit/test_audio_recorder_preroll_integration.py` |
| `_get_next_recorded_audio` | `tests/unit/test_slow_final_transcription_audio_gap.py` |
| `_is_voice_active` | `tests/unit/test_slow_final_transcription_audio_gap.py` |
| `_check_voice_activity` | `tests/unit/test_slow_final_transcription_audio_gap.py` |
| `_is_silero_speech` | `tests/unit/test_slow_final_transcription_audio_gap.py` |
| `_set_state` | monkeypatched in `tests/unit/test_fastapi_server_protocol.py`, `tests/unit/test_realtime_text_stabilizer.py`, and `tests/unit/test_slow_final_transcription_audio_gap.py`; documented as central in `docs/ARCHITECTURE.md`. |
| `_realtime_worker` | `tests/unit/test_realtime_streaming_transcription.py` |

No direct test or server caller currently invokes `_submit_transcription_request`,
`_receive_transcription_result`, `_call_transcription_executor`,
`_process_wakeword`, or `_recording_worker`. They are still used internally and
should remain as wrappers when moved because they are part of the current
recorder implementation surface.

## Test Coverage By Planned Milestone

| Planned milestone | Existing focused tests | Coverage notes |
|---|---|---|
| Milestone 2: transcription runtime | `tests/unit/test_fastapi_server_protocol.py::FastAPIServerProtocolTests::test_recorder_external_executor_preserves_final_transcription_path`; `tests/unit/test_slow_final_transcription_audio_gap.py` | External executor path is covered. Internal worker `run()` is not fully unit-covered because it creates and warms a real engine. Move the worker code first without changing its body. |
| Milestone 3: wake-word runtime | `tests/unit/test_wakeword.py`, `tests/unit/test_fastapi_server_multi_user.py` | Dependency loading and server flow are covered. Runtime Porcupine/OpenWakeWord scoring remains weak. |
| Milestone 4: voice activity and pre-roll | `tests/unit/test_silero_vad_backend.py`, `tests/unit/test_slow_final_transcription_audio_gap.py`, `tests/unit/test_preroll.py`, `tests/unit/test_audio_recorder_preroll_integration.py` | Good coverage for Silero state reset, stale generations, delayed confirmation, and pre-roll selection. |
| Milestone 5: recording worker | `tests/unit/test_slow_final_transcription_audio_gap.py`, `tests/unit/test_audio_recorder_preroll_integration.py`, `tests/unit/test_fastapi_server_protocol.py` | Indirect coverage only. Add characterization tests before changing loop behavior, but a pure delegation move can rely on existing focused gates. |
| Milestone 6: realtime worker | `tests/unit/test_realtime_streaming_transcription.py`, `tests/unit/test_realtime_text_stabilizer.py`, `tests/unit/test_realtime_boundary_detector.py` | Good coverage for streaming vs non-streaming behavior and stabilization reset. |
| Milestone 7: initialization | `tests/unit/test_silero_vad_backend.py`, `tests/unit/test_fastapi_server_protocol.py`, docs compatibility checklist | Constructor signature has focused coverage; full constructor side effects are broad and should be split last. |

## Next Extraction Gate

Recommended next extraction:

- Responsibility: final transcription runtime.
- Target: `RealtimeSTT/core/transcription.py`.
- First safe slice: move `TranscriptionWorker` and a module-level
  `run_transcription_worker(*args, **kwargs)` helper.
- Keep `AudioToTextRecorder._transcription_worker()` as a wrapper that calls the
  new helper.
- Leave `_submit_transcription_request`, `_receive_transcription_result`, and
  `_call_transcription_executor` on `AudioToTextRecorder` for the first
  Milestone 2 pass unless the move remains purely delegating. They are coupled to
  recorder instance state, early-final behavior, realtime fallback, queues, and
  locks.

Validation for the next pass:

```powershell
python -c "from pathlib import Path; [compile(Path(p).read_text(encoding='utf-8'), p, 'exec') for p in ['RealtimeSTT/audio_recorder.py', 'RealtimeSTT/core/transcription.py']]; print('compile ok')"
python -m pytest tests\unit\test_fastapi_server_protocol.py -k recorder_external_executor
python -m pytest tests\unit\test_slow_final_transcription_audio_gap.py
```

No extra characterization test is required before moving `TranscriptionWorker`
unchanged. Add one before changing worker behavior, pipe messages, result
formats, logging, warmup behavior, or executor semantics.
