# Release Notes

## 1.1.0rc1 - 2026-08-27

This is a release candidate. No CI, real-model, TestPyPI, or production-PyPI
publication is implied until the release-readiness checklist is complete.
See [the release-readiness checklist](docs/release-1.1.0rc1-readiness.md) for
the required evidence gates.

### Added

- Added correlated Preview `resume` handling for the versioned production
  WebSocket API, including an ordered acknowledgement barrier and explicit
  candidate/audio provenance fields.
- Added an opt-in authenticated late full-turn correction operation for
  Preview-only deployments, bounded by a configurable maximum audio duration.
- Added recorder Preview/tail transcription support with package-level result
  types and the documented dual-realtime merge contracts.

### Changed

- Preview-only production turns retain the complete logical-turn audio for model
  input; live text remains diagnostic and is never substituted for Preview or
  Final output.
- Empty Preview results use one bounded silence-padding retry, while Final
  remains authoritative whenever Final ASR is enabled.
- Preview admission is now bounded and coalesced: one native dispatch lane
  retains only the latest pending snapshot while an older inference is active.
- Unexpected live-lane exits now produce structured, observable degraded events
  while preserving the authoritative-final fallback path.
- Raw-PCM responses propagate provider-reported language and finite language
  probability metadata when the selected provider supplies it.
- Realtime punctuation splitting commits the frame boundary atomically, and the
  local Preview worker rejects new speculative requests once its bounded queue
  is full.
- Release packaging checks now validate the requested version, inspect archive
  names/content for private material, and install both wheel and sdist from
  outside the source checkout.

### Release boundaries

- The documented Nemotron-live/Parakeet-final profile targets Linux x86-64 with
  the pinned sherpa-onnx runtime. Native Windows remains a development target
  for this pair; use another authoritative final engine there unless the
  cumulative-recovery behavior is independently validated.
- Model weights remain external artifacts. Users must review and accept the
  exact model/runtime licenses before downloading or redistributing them.
- Real-model acceptance, CI results, and exact TestPyPI installation must be
  recorded for the final commit before any production release decision.

## 1.0.4 - 2026-08-21

### Fixed

- Replaced the production WebSocket recorder/VAD-derived final path with a
  turn-owned state machine and authoritative canonical PCM buffer. Every
  finalized turn now produces exactly one Parakeet final result followed by
  exactly one completion.
- Made accepted 16 kHz PCM invariant to 10/20/40/64/100 ms packet boundaries;
  the WebSocket final uses the same final scheduler lane, audio normalization,
  language, and prompt policy as raw-PCM HTTP transcription.
- Fenced late live/final/transport callbacks across cancel, reset, disconnect,
  and reconnect, including reconnects that reuse a session or turn id.
- Fixed live Nemotron input scaling, removed growing full-buffer snapshots,
  fed only new frames to one stream per logical turn, and deduplicated and
  rate-limited changed partials.
- Closed long-session lifecycle leaks by retiring completed final threads,
  draining cancelled live queues, cancelling timed-out live streams, pruning
  connection epochs, and resolving queued scheduler jobs during shutdown.
- Moved native live-stream creation and cancellation off the ASGI event loop,
  fenced streams returned after a turn was retired, and bounded concurrent
  native cancellation work process-wide.
- Made final/completion admission atomic across reset and cancel, scoped final
  timeout cancellation to the exact scheduler request, and bounded live
  backpressure by queued audio duration instead of client packet count.
- Bounded terminal-event reserve for slow WebSocket readers and close the
  connection with code 1013 instead of allowing finalized turns to grow an
  unbounded outbound queue.
- Removed bearer-token command-line flags; production tokens now come only
  from `REALTIMESTT_SERVER_BEARER_TOKEN` so they cannot appear in process
  listings or shell history.

### Changed

- Production WebSocket audio is explicitly canonical mono 16 kHz
  `pcm_s16le`. HTTP final transcription retains its documented supported input
  sample rates and performs one whole-request resample.
- Streaming benchmark pacing now uses the absolute audio clock and validates
  exact terminal counts, terminal/completion ordering, quiet completion tails,
  long-run repetitions, and bounded parallel concurrency.
- HTTP A/B and WebSocket benchmark reports redact endpoints, corpus paths,
  clip identifiers, references, partials, and finals by default. Sensitive
  per-record output requires an explicit protected-local opt-in.
- The production CPU profile keeps Nemotron live hypotheses display-only;
  Parakeet remains the sole authoritative turn final.

### Notes

- This release supersedes the old TestPyPI `1.0.3` artifact. PyPI
  files are immutable, so the corrected production release uses version `1.0.4`.
- The supported Nemotron/Parakeet production profile remains Linux x86-64.
  Keep the server on loopback behind an authenticated TLS/WSS endpoint and
  retain a local-live/final-server rollback path during client rollout.
- No production-PyPI publication is implied by this release candidate.

## 1.0.3 - 2026-08-20

### Added

- Added the experimental optional FunASR/SenseVoice transcription backend and
  install extra.
- Added `AudioToTextRecorder.feed_audio_file()` for feeding audio files through
  the manual audio input path.
- Added `TranscriptionResult.metadata` support and word timestamp plumbing for
  backends that can return word-level timing.
- Added public custom transcription engine interfaces through
  `RealtimeSTT.engines`, including `BaseEngine`, `TranscriptionEngineConfig`,
  `TranscriptionResult`, and `StreamingTranscriptionSession`.
- Added custom engine documentation showing `transcription_executor`,
  `realtime_transcription_executor`, and streaming-session integration.
- Added opt-in realtime punctuation splitting through
  `realtime_punctuation_split_marks`. The default remains `"off"`; `"sentence"`
  is the supported production mode for `.`, `?`, and `!`.
- Added the multilingual sherpa-onnx Nemotron 3.5 0.6B INT8 engine as a true
  streaming engine with per-stream language selection, incremental frame
  ingestion, final draining, reset, cancellation, and deterministic release.
- Added pinned manifests for the exact Nemotron streaming and Parakeet v3 final
  model archives, including source, license, archive integrity, and extracted
  file integrity metadata.
- Added `stt-install-sherpa-models` for resumable, verified, atomic installation
  and offline reuse of those pinned model archives in persistent storage.
- Added a packaged production FastAPI server with versioned liveness,
  readiness, capabilities, raw PCM16 final transcription, and ordered remote
  streaming WebSocket APIs. Non-loopback binds require bearer authentication
  and TLS. The server includes bounded queues and sessions, backpressure,
  limits, structured errors, warmup, and graceful resource cleanup.
- Added a reproducible same-host A/B benchmark harness for the raw PCM16 client
  contract and release CI for unit, distribution, and isolated wheel
  installation checks.

### Changed

- Hardened realtime punctuation splitting with repeated-observation checks,
  timestamp validation, guarded dash handling, and synchronized split-state
  bookkeeping.
- Expanded install-extra validation to include the FunASR extra and the `all`
  extra.
- Exposed the same engine authoring interfaces through lazy top-level package
  imports while keeping existing `BaseTranscriptionEngine` imports compatible.
- Pinned the sherpa-onnx optional dependency to the release-tested version and
  hardened empty-output/language handling for Parakeet final transcription.
- Made the production `server` extra install a local Silero ONNX VAD runtime,
  avoiding interactive Torch Hub model retrieval when WebSocket sessions start.
- Promoted the remote production server to the `stt-server-production` packaged
  entry point while preserving the legacy `stt-server` command.

### Notes

- Comma, dash, ellipsis, and `all` punctuation split modes remain available for
  experiments but are not promoted as production-supported in this release.
- Realtime punctuation splitting requires word timestamps. Built-in support is
  currently wired through `faster_whisper`; other built-in engines skip the
  split path.
- The production CPU pair uses Nemotron only for replaceable live hypotheses;
  Parakeet transcribes the authoritative final audio once and replaces the live
  hypothesis at turn completion.
- The exact model weights remain external artifacts governed by their own
  licenses. The package ships manifests and installation tooling, not weights.
- The pinned two-model production profile targets Linux x86-64. Native Windows
  remains suitable for development, but Parakeet can return empty output for
  some voiced cumulative-turn audio; use the Linux server or another final
  engine until that runtime combination is proven reliable.
- On the same Linux CPU host and final model, the 1.0.3 server matched the
  reference WER/CER/exact result, was at single-client latency parity, and at
  four concurrent clients improved throughput by 7.5% and p95 latency by 33.2%.

## 1.0.2 - 2026-05-31

### Changed

- Split the `AudioToTextRecorder` implementation behind the existing public
  facade into focused core modules for lifecycle, recording, realtime
  transcription, voice activity, wake-word handling, initialization, shutdown,
  and formatting.
- Refreshed recorder architecture and compatibility documentation so the public
  facade, threading boundaries, and regression checks are easier to audit.
- Normalized package docstrings and comments to use block-style summaries and
  focused runtime explanations.

### Removed

- Removed an unused internal console color helper from `audio_recorder.py`.

## 1.0.1 - 2026-05-20

### Added

- Added a generic streaming transcription session interface so engines can
  opt in to incremental realtime decoding while existing engines keep the
  full-buffer fallback behavior.
- Added `kroko_onnx` transcription engine for Kroko/Banafo `.data` streaming
  models.
- Added Kroko realtime preview support that feeds streaming engines only newly
  recorded audio frames through a persistent session.
- Added `stt-install-kroko`, exposed through the `kroko-builder` extra, to help
  build and install Kroko-ONNX for the active Python environment.
- Added focused Kroko and realtime streaming unit coverage plus a public manual
  `tests/realtimestt_kroko_test.py` smoke script.
- Added `omnilingual_asr` transcription engine for Meta
  Omnilingual ASR on Linux/WSL2 with Python 3.11.x, with support for the
  published CTC and LLM model cards. Native Windows is not supported because
  `fairseq2n` has no Windows wheel; Python 3.12.x is blocked by upstream
  `omnilingual-asr` package metadata.
- Added `docs/licenses.md` with engine and model-family license notes.

### Changed

- Kroko Community models with known public filenames can be auto-downloaded
  into the RealtimeSTT cache when `auto_download_model` is enabled.
- Kroko final transcription remains one-shot. The new streaming path is used
  for realtime previews only when the realtime engine advertises streaming
  support.
- Kroko model cadence is used to choose automatic finalization tail padding.
- Omnilingual ASR uses `omniASR_CTC_1B_v2` as the default when the
  recorder is still configured with a Whisper default model name.
- Omnilingual in-memory audio is passed to the backend as predecoded waveform
  dictionaries to avoid the upstream package treating raw float arrays as
  encoded audio bytes.

### Notes

- Install/build Kroko-ONNX separately with
  `pip install "RealtimeSTT[kroko-builder,silero-onnx-cpu]"` followed by
  `stt-install-kroko --build`, or install a compatible Kroko-ONNX wheel in the
  same Python environment. The `silero-onnx-cpu` extra provides the local VAD
  backend used by recorder-based Kroko smoke tests and live microphone use.
- Licensed Pro models require a Pro-capable Kroko wheel and a key supplied at
  runtime through configuration, CLI, or environment variables. Do not commit
  keys, Pro models, generated logs, local wheels, or local cache contents.
- `Pro-16-L` is the recommended realtime model for the fastest partials. Local
  private validation observed the expected low-latency partial behavior, but
  exact cadence depends on runtime, provider, hardware, and scheduling.
- `suppress_native_output=True` redirects Kroko native stdout/stderr during
  recognizer calls and sets `KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT=1`. Reliable
  suppression of asynchronous Pro license refresh messages requires a Kroko
  wheel rebuilt with RealtimeSTT's native quiet-output patch; older Kroko wheels
  may still print background license status text.
- Omnilingual ASR support is optional and Linux/WSL2 Python 3.11.x-oriented.
  Native Windows installs are not supported by the upstream dependency stack at
  this time, and Python 3.12.x cannot resolve the current upstream package.
- `omniASR_CTC_1B_v2` is the recommended Omnilingual starting point for local
  realtime tests in this release. Smaller/larger CTC and LLM models are exposed
  through model-card plumbing but require their own quality and memory checks.
