# Release Notes

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
