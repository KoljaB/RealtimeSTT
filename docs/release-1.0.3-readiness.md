# RealtimeSTT 1.0.3 release readiness

Status date: 2026-08-20
Scope: current `codex/realtimestt-1.0.3-production-ready` checkout

This is an evidence ledger, not a claim that 1.0.3 has shipped. The source
implementation and substantial local/remote validation are present, but the
release is **not ready to announce** until the remaining CI, fresh-package,
TestPyPI, and publication gates below have an attached log or artifact.
“Implemented” means that the code/docs are present here; “validated” is used
only where a concrete test or benchmark result is recorded.

## Status key

- **Implemented** — present in the current checkout and described by source or
  documentation.
- **Validated** — a deterministic local, model, protocol, or remote check
  produced the stated result.
- **Pending** — requires a clean supported environment, CI run, real models,
  remote service, package index, or repository operation not evidenced here.
- **Limitation/fallback** — a known constraint that must remain visible in the
  release and deployment instructions.

## Release checklist

### Requested 1.0.3 product surface

| Area | Evidence in this checkout | Status / release interpretation |
| --- | --- | --- |
| Version and release notes | `setup.py` declares `1.0.3`; `RELEASE_NOTES.md` has the 1.0.3 feature, behavior, model, and license notes. | **Implemented.** A fresh package build still has to prove that the generated metadata is 1.0.3. |
| Public engine interfaces | `RealtimeSTT.engines` and `RealtimeSTT.transcription_engines` expose the custom engine/config/result/streaming-session interfaces; compatibility imports remain in place. | **Implemented.** API import checks are part of the package smoke gate below. |
| Audio and transcription additions | Manual audio-file feeding, result metadata/word timestamps, and recorder plumbing are present; the supported-venv unit discovery completed with 428 OK and 14 intentional opt-in skips. | **Validated.** The skips are the explicitly opt-in real-model/platform checks, not test discovery failures. |
| Realtime punctuation splitting | The default remains off; `sentence` (`.`, `?`, `!`) is the documented production mode. The path requires word timestamps and is wired for `faster_whisper`; other built-in engines skip it. Experimental comma/dash/ellipsis modes are not promoted. | **Implemented with an explicit scope limit.** Do not advertise punctuation splitting as a generic all-engine feature. |
| Packaged production server | `RealtimeSTT_server.production_server`, `stt-server-production`, versioned health/readiness/capabilities endpoints, raw PCM16 final transcription, ordered streaming WebSocket events, bounded sessions/queues, structured errors, bearer auth, TLS checks, and cleanup are present. | **Validated locally.** The focused production/manual-drain suite is 48/48 OK, and the Linux WebSocket check below is exact with no sequence failures. Clean wheel smoke and CI evidence remain pending. |
| CPU two-model path | The documented profile uses Nemotron for replaceable live hypotheses and Parakeet once for the authoritative final. The two engines, per-stream language handling, final drain/reset/cancel behavior, and pinned `sherpa-onnx==1.13.4` extra are present. | **Validated with boundaries.** A real Windows Nemotron+Parakeet golden run was exact, with Parakeet RTF 0.175; the Linux remote A/B is also recorded below. The known Windows cumulative-recovery limitation remains. |
| Model installation and integrity | `stt-install-sherpa-models`, immutable manifests, archive size/SHA-256 checks, extracted-file checks, safe extraction, resumable downloads, offline reuse, and atomic commit are present. Model weights remain external artifacts. | **Implemented and unit-checked.** Installer failure paths pass locally; real archive/offline acceptance in the release workflow remains pending. |
| Documentation and licensing | Installation/testing/server/engine/license docs, `RealtimeSTT_server/PRODUCTION_SERVER.md`, the remote benchmark contract, and release notes are present. | **Implemented.** Final link review and release-artifact review remain required. |

### Validation and release evidence

| Gate | Current evidence | Status |
| --- | --- | --- |
| Full supported-venv unit discovery | The supported virtual environment discovered 442 tests: **428 OK** and **14 intentional opt-in skips**. No dependency-import failure is part of this result. | **Validated locally.** Keep the 14 skips explicit: they are opt-in model/platform checks, not a green result for those real-model paths. |
| Focused production suite | The focused production and manual-drain suite completed **48/48 OK**. | **Validated locally.** |
| Full unit matrix | `.github/workflows/release-checks.yml` defines Ubuntu Python 3.11/3.12 and Windows Python 3.11 unit jobs. | **Pending CI evidence.** The workflow definition is not a successful CI run. |
| Real Nemotron + Parakeet acceptance | The Windows real-model golden run was exact for both Nemotron and Parakeet; the Parakeet final measured **RTF 0.175**. The workflow also defines a manually callable Linux job that installs/verifies both pinned archives, reuses them offline, and fails if either real golden test is skipped. | **Windows validated for the recorded fixture.** Linux workflow/archive acceptance is still pending; the Windows result does not remove the documented cumulative-recovery limitation. |
| Local multilingual streaming | The final local seven-language WebSocket runs at **100 ms** and **37 ms** chunks were exact **7/7**, with zero event-sequence and audio-sequence failures. Evidence is retained in `test-results/release-1.0.3/final-local-stream-100ms.*` and `final-local-stream-37ms.*`. | **Validated locally.** |
| Linux WebSocket contract | Seven English clips were exact, with **61 partials**, zero sequence failures, and a completion-after-finalize median of **0.1039 s**. | **Validated remotely.** Keep the protocol report with the release evidence. |
| Distribution build and wheel isolation | The workflow defines `python -m build`, `twine check`, an isolated wheel install, package imports, version assertion, and both console-script `--help` checks. Existing ignored `dist/` files are dated 2026-07-03 and are not evidence for the current tree. | **Pending.** Build fresh artifacts from the final tree; inspect sdist/wheel contents and run the isolated install. |
| Install-extra matrix and server dependency fix | The initial server-extra probe exposed a missing packaged Silero VAD dependency. `setup.py` now includes `silero-vad[onnx-cpu]` through `production_server_requirements`, fixing the source packaging declaration. | **Fixed in source; remote proof pending.** Re-run the server-extra install after the manual dependency install and capture the clean-package result. The broad optional-extra matrix remains pending. |
| Linux exact-source HTTP A/B | The exact-source **b593** candidate was compared with the Linux reference on **36 clips × 3**. Quality was identical: WER **0.0436**, CER **0.0242**, exact **0.8889**; there were **0 failures**. Sequential latency was candidate median **0.1319 s** vs reference **0.1311 s**, and candidate p95 **0.3179 s** vs reference **0.3488 s**. At concurrency 4, throughput was candidate **9.903** vs reference **9.214** requests/s, with p95 **0.5419 s** vs **0.8107 s**. | **Validated remotely.** These numbers are candidate-vs-reference evidence, not a substitute for CI or a published-package test. |
| Memory observation during HTTP A/B | Candidate observed idle-after-stream RSS was **2,652,468 KiB**; reference RSS was **1,096,164 KiB**. | **Validated with a capacity caveat.** The candidate retains materially more memory than the reference; deployment sizing must account for it. |
| TestPyPI package check | No current checkout evidence shows a fresh 1.0.3 upload or a clean environment installed from TestPyPI. | **Pending and release-blocking if TestPyPI is part of the release checklist.** Upload only after build/checks pass, then install by exact version in a clean environment and run the package smoke. |
| GitHub publication | The current worktree has many modified and untracked release files. There is no evidence here of a final commit, remote push, release tag, or accepted PR. | **Pending.** Root must review the final diff, commit the intended files, push the agreed branch/repository, and record the resulting commit/tag/CI status. |

## Platform boundaries and fallback

These are product boundaries, not reasons to silently claim broader support:

- The documented Nemotron-live/Parakeet-final production profile targets Linux
  x86-64 with `sherpa-onnx==1.13.4`. The exact-source Linux HTTP A/B above is
  evidence for the candidate/reference contract; the reproducible Linux
  real-model acceptance workflow is still pending.
- Native Windows now has an exact Nemotron+Parakeet golden result, with
  Parakeet RTF **0.175** on that fixture. Separately, 1.0.3 validation recorded
  intermittent empty Parakeet finals on voiced cumulative-recovery audio. An
  empty decode must not be treated as a valid transcription. Until that
  cumulative-recovery behavior is resolved or bounded by more evidence, use a
  different authoritative final engine such as `faster_whisper`, or run the
  pinned Nemotron/Parakeet pair on Linux; do not turn the single exact golden
  into a blanket Windows production guarantee.
- The core package targets Python 3.11 or newer, but optional stacks narrow the
  matrix: Omnilingual is Linux/WSL2 and Python 3.11.x-oriented; the Kroko
  Windows builder currently requires CPython 3.12 x64. These constraints must
  stay in installation documentation and release communication.
- macOS, CUDA variants, microphone devices, and optional engine/model families
  outside the release acceptance job have no blanket 1.0.3 production claim
  from the evidence recorded here. Use the corresponding engine's documented
  install and smoke-test path.
- Model archives are not shipped in the package. Users must accept the model
  licenses and install the exact external bundles into persistent storage; the
  manifests and installer provide integrity checks but do not grant model
  rights.
- The candidate's observed idle-after-stream RSS (2,652,468 KiB) is well above
  the reference (1,096,164 KiB). This is a deployment capacity constraint even
  though the recorded A/B run had zero request failures.

## Minimum close-out sequence

The following evidence is already recorded: supported-venv unit discovery
(428 OK, 14 intentional opt-in skips), focused production/manual-drain suite
(48/48 OK), Windows Nemotron+Parakeet exact golden with Parakeet
RTF 0.175, multilingual 100/37 ms streaming (7/7 exact with zero sequence
failures), Linux WebSocket contract, and the b593 Linux HTTP A/B report.

Before calling this release ready, still attach or record all of the following:

1. A successful release workflow run for the unit matrix and package/wheel
   smoke; run the explicit Linux real-model acceptance job and retain both
   real golden-test results.
2. Fresh distribution files reviewed with `twine check` and installed in an
   isolated environment, including imports, `stt-server-production --help`,
   `stt-install-sherpa-models --help`, and an exact `1.0.3` version check.
3. A clean server-extra install/retest after the `setup.py` Silero VAD fix;
   the source declaration is fixed, but remote proof after the manual install
   is still outstanding.
4. If required by the release checklist, a TestPyPI upload followed by a clean
   TestPyPI install/smoke using the exact 1.0.3 artifact.
5. A final diff review confirming that generated caches, model weights,
   credentials, private test material, and local benchmark output are not
   being published; then the approved commit/push/tag and resulting CI status.

Until those items are recorded, the honest release state is **implemented and
substantially validated, but not publication-ready**. The Windows
Parakeet-final cumulative-recovery limitation and the candidate memory
headroom requirement remain even after the pending gates pass.
