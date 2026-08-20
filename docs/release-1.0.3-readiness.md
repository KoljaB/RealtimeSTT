# RealtimeSTT 1.0.3 release readiness

Status date: 2026-08-20
Scope: current `codex/realtimestt-1.0.3-production-ready` checkout

**Superseded on 2026-08-20:** later real AgentTalk WebSocket testing disproved
this ledger's former claim that no technical blocker remained. The published
TestPyPI `1.0.3` files finalize recorder/VAD fragments rather than one explicit
client turn, can restart resampling at packet boundaries, and do not satisfy
the exact-one-final/completion or live-performance gates. They must not be
promoted to production PyPI. The corrected candidate is tracked separately as
`1.0.4rc1`; the evidence below is retained only as historical evidence for the
old commit and is not release approval.
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
| Version and release notes | `setup.py` declares `1.0.3`; `RELEASE_NOTES.md` has the 1.0.3 feature, behavior, model, and license notes. The clean wheel and sdist both expose version 1.0.3. | **Validated.** Fresh build metadata, TestPyPI metadata, and the isolated installed distribution all report 1.0.3. |
| Public engine interfaces | `RealtimeSTT.engines` and `RealtimeSTT.transcription_engines` expose the custom engine/config/result/streaming-session interfaces; compatibility imports remain in place. | **Implemented.** API import checks are part of the package smoke gate below. |
| Audio and transcription additions | Manual audio-file feeding, result metadata/word timestamps, and recorder plumbing are present; the supported-venv unit discovery completed with 428 OK and 14 intentional opt-in skips. | **Validated.** The skips are the explicitly opt-in real-model/platform checks, not test discovery failures. |
| Realtime punctuation splitting | The default remains off; `sentence` (`.`, `?`, `!`) is the documented production mode. The path requires word timestamps and is wired for `faster_whisper`; other built-in engines skip it. Experimental comma/dash/ellipsis modes are not promoted. | **Implemented with an explicit scope limit.** Do not advertise punctuation splitting as a generic all-engine feature. |
| Packaged production server | `RealtimeSTT_server.production_server`, `stt-server-production`, versioned health/readiness/capabilities endpoints, raw PCM16 final transcription, ordered streaming WebSocket events, bounded sessions/queues, structured errors, bearer auth, TLS checks, and cleanup are present. | **Validated.** The focused production/manual-drain suite is 48/48 OK, the Linux WebSocket check below is exact with no sequence failures, CI's isolated wheel smoke passed, and the TestPyPI console script/import smoke passed. |
| CPU two-model path | The documented profile uses Nemotron for replaceable live hypotheses and Parakeet once for the authoritative final. The two engines, per-stream language handling, final drain/reset/cancel behavior, and pinned `sherpa-onnx==1.13.4` extra are present. | **Validated with boundaries.** A real Windows Nemotron+Parakeet golden run was exact, with Parakeet RTF 0.175; the Linux remote A/B is also recorded below. The known Windows cumulative-recovery limitation remains. |
| Model installation and integrity | `stt-install-sherpa-models`, immutable manifests, archive size/SHA-256 checks, extracted-file checks, safe extraction, resumable downloads, offline reuse, and atomic commit are present. Model weights remain external artifacts. | **Validated.** Installer unit/failure paths pass locally; release CI installed and verified both pinned real bundles, reused the verified model root, and ran both real golden tests. |
| Documentation and licensing | Installation/testing/server/engine/license docs, `RealtimeSTT_server/PRODUCTION_SERVER.md`, the sanitized remote benchmark contract, and release notes are present. | **Reviewed for the candidate.** Wheel/sdist inspection found no model weights, credentials, private test material, local benchmark output, or development logs. |

### Validation and release evidence

| Gate | Current evidence | Status |
| --- | --- | --- |
| Full supported-venv unit discovery | The supported virtual environment discovered 442 tests: **428 OK** and **14 intentional opt-in skips**. No dependency-import failure is part of this result. | **Validated locally.** Keep the 14 skips explicit: they are opt-in model/platform checks, not a green result for those real-model paths. |
| Focused production suite | The focused production and manual-drain suite completed **48/48 OK**. | **Validated locally.** |
| Full unit matrix | Release run [32390235078](https://github.com/KoljaB/RealtimeSTT-development/actions/runs/32390235078) ran Ubuntu Python 3.11/3.12 and Windows Python 3.11 unit jobs against commit `7713522`; all completed successfully. | **Validated in CI.** The same run also passed the clean distribution/wheel job and real-model job. |
| Real Nemotron + Parakeet acceptance | The Windows real-model golden run was exact for both Nemotron and Parakeet; the Parakeet final measured **RTF 0.175**. In release run [32390235078](https://github.com/KoljaB/RealtimeSTT-development/actions/runs/32390235078), Linux installed and verified both pinned archives, then passed the real Nemotron and real Parakeet golden tests without skip fallback. | **Validated on Linux CI and the recorded Windows fixture.** The Windows result does not remove the documented cumulative-recovery limitation. |
| Local multilingual streaming | The final local seven-language WebSocket runs at **100 ms** and **37 ms** chunks were exact **7/7**, with zero event-sequence and audio-sequence failures. Evidence is retained in `test-results/release-1.0.3/final-local-stream-100ms.*` and `final-local-stream-37ms.*`. | **Validated locally.** |
| Linux WebSocket contract | Seven English clips were exact, with **61 partials**, zero sequence failures, and a completion-after-finalize median of **0.1039 s**. | **Validated remotely.** Keep the protocol report with the release evidence. |
| Distribution build and wheel isolation | A clean staged export built a 263,740-byte wheel (`33583740547632bc9f02e3f4a212871a9d3eac5a0abc57b283ae87e0f2e7e10e`) and 247,298-byte sdist (`a980b4dd96f9bcd9da2ad9efe6e3922e0a58ada666d13f30c59f007a2defe988`). `twine check`, content inspection, a full dependency install, `pip check`, imports/resources, and both console-script `--help` checks passed; CI independently rebuilt and smoked the wheel. | **Validated locally and in CI.** The inspected artifacts contain the new engine, installer, server package, guide, metadata, and entry points, with no private/reproducibility-excluded material. |
| Install-extra matrix and server dependency fix | The initial server-extra probe exposed a missing packaged Silero VAD dependency. `setup.py` now includes `silero-vad[onnx-cpu]`; the clean local wheel smoke and the exact TestPyPI install resolved `silero-vad==6.2.1` and `onnxruntime`, then passed `pip check` and server imports. | **Validated for the release server and sherpa extras.** Optional engine families outside this release scope retain their documented platform-specific install paths. |
| Linux exact-source HTTP A/B | The exact-source **b593** candidate was compared with the Linux reference on **36 clips × 3**. Quality was identical: WER **0.0436**, CER **0.0242**, exact **0.8889**; there were **0 failures**. Sequential latency was candidate median **0.1319 s** vs reference **0.1311 s**, and candidate p95 **0.3179 s** vs reference **0.3488 s**. At concurrency 4, throughput was candidate **9.903** vs reference **9.214** requests/s, with p95 **0.5419 s** vs **0.8107 s**. | **Validated remotely.** These numbers are candidate-vs-reference evidence, not a substitute for CI or a published-package test. |
| Memory observation during HTTP A/B | Candidate observed idle-after-stream RSS was **2,652,468 KiB**; reference RSS was **1,096,164 KiB**. | **Validated with a capacity caveat.** The candidate retains materially more memory than the reference; deployment sizing must account for it. |
| TestPyPI package check | [RealtimeSTT 1.0.3 on TestPyPI](https://test.pypi.org/project/realtimestt/1.0.3/) exposes the exact wheel and sdist hashes recorded above. A new Python 3.12 environment installed `RealtimeSTT[server,sherpa-onnx]==1.0.3` from TestPyPI with public PyPI only as the dependency index; `pip check`, imports, packaged resource lookup, version/entry-point assertions, and both release CLI help checks passed. | **Validated from the published index.** The install log fetched RealtimeSTT itself from `test-files.pythonhosted.org`, not the source checkout or local wheel. |
| GitHub publication | Commit [`7713522`](https://github.com/KoljaB/RealtimeSTT-development/commit/7713522ad33d2fc06f358ea210e7d41797e4c1b7) is pushed to `codex/realtimestt-1.0.3-production-ready` and `release/1.0.3-candidate` in `KoljaB/RealtimeSTT-development`; release run [32390235078](https://github.com/KoljaB/RealtimeSTT-development/actions/runs/32390235078) is green. | **Release candidate published and CI-validated.** Final merge/tag and production-PyPI upload have intentionally not been inferred from the request. |

## Platform boundaries and fallback

These are product boundaries, not reasons to silently claim broader support:

- The documented Nemotron-live/Parakeet-final production profile targets Linux
  x86-64 with `sherpa-onnx==1.13.4`. The exact-source Linux HTTP A/B above is
  evidence for the candidate/reference contract, and release CI independently
  installed both pinned bundles and passed both Linux real-model golden tests.
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

## Completed close-out and remaining maintainer actions

The close-out evidence now includes supported-venv unit discovery (428 OK,
14 intentional opt-in skips), the 48/48 focused production/manual-drain suite,
Windows and Linux real-model golden tests, multilingual 100/37 ms streaming,
the Linux WebSocket contract, the b593 Linux HTTP A/B, the full GitHub unit
matrix, clean distribution and wheel isolation, exact TestPyPI publication,
and a fresh TestPyPI install with the server and sherpa extras.

The old `1.0.3` candidate has technical release blockers and is retired. Its
remaining historical actions must not be executed:

1. Do not merge/tag `7713522` as the corrected production release.
2. Do not upload the existing `1.0.3` artifacts to production PyPI.
3. Use the `1.0.4rc1` readiness ledger and repeat every hard gate against its
   final commit and exact package-index artifact.

The Windows Parakeet-final cumulative-recovery limitation and the candidate
memory-headroom requirement remain release notes and deployment constraints;
they are not silently broadened into unsupported production claims.
