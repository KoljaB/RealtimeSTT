# Release Notes

## 1.0.1 - 2026-05-20

### Added

- Added a `kroko_onnx` transcription engine for Kroko/Banafo `.data` streaming
  models.
- Added a generic streaming transcription session interface so engines can
  opt in to incremental realtime decoding while existing engines keep the
  full-buffer fallback behavior.
- Added Kroko realtime preview support that feeds streaming engines only newly
  recorded audio frames through a persistent session.
- Added `stt-install-kroko`, exposed through the `kroko-builder` extra, to help
  build and install Kroko-ONNX for the active Python environment.
- Added focused Kroko and realtime streaming unit coverage plus a public manual
  `tests/realtimestt_kroko_test.py` smoke script.
- Added an experimental `omnilingual_asr` transcription engine for Meta
  Omnilingual ASR on Linux/WSL2, with support for the published CTC and LLM
  model cards.
- Added the `omnilingual`, `omnilingual-asr`, and `meta-omnilingual-asr`
  install extras. The dependency is guarded for non-Windows Python 3.10-3.12
  because `fairseq2n` has no native Windows wheel.
- Added focused Omnilingual ASR unit coverage using mocked runtime objects so
  normal test runs do not need the heavy optional backend installed.
- Added `docs/licenses.md` with engine and model-family license notes.

### Changed

- Kroko Community models with known public filenames can be auto-downloaded
  into the RealtimeSTT cache when `auto_download_model` is enabled.
- Kroko final transcription remains one-shot. The new streaming path is used
  for realtime previews only when the realtime engine advertises streaming
  support.
- Kroko model cadence is used to choose automatic finalization tail padding.
- Public docs now describe Community versus Pro model handling, ignored local
  test model caches, credential policy, and Kroko realtime recommendations.
- Omnilingual ASR uses `omniASR_CTC_300M_v2` as the practical default when the
  recorder is still configured with a Whisper default model name.
- Omnilingual CTC models omit language hints, while LLM model cards can receive
  Omnilingual language IDs such as `eng_Latn`.
- Omnilingual in-memory audio is passed to the backend as predecoded waveform
  dictionaries to avoid the upstream package treating raw float arrays as
  encoded audio bytes.

### Notes

- Kroko support is new and optional. Install/build Kroko-ONNX separately with
  `pip install "RealtimeSTT[kroko-builder]"` followed by
  `stt-install-kroko --build`, or install a compatible Kroko-ONNX wheel in the
  same Python environment.
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
- Omnilingual ASR support is optional and Linux/WSL2-oriented. Native Windows
  installs are not supported by the upstream dependency stack at this time.
- `omniASR_CTC_300M_v2` is the recommended Omnilingual starting point for local
  realtime tests. Larger CTC and LLM models are exposed through model-card
  plumbing but may exceed typical desktop GPU memory.
