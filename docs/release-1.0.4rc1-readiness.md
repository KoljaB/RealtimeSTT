# RealtimeSTT 1.0.4rc1 release readiness

Status date: 2026-08-21

This ledger supersedes `release-1.0.3-readiness.md`. It records only sanitized
aggregate evidence. Private AgentTalk audio, paths, clip identifiers,
transcripts, and detailed reports remain ignored and local; none are release
artifacts.

## Corrected production contract

- The versioned production WebSocket path owns an explicit logical turn. It
  does not instantiate `AudioToTextRecorder` and does not treat recorder/VAD
  fragments as protocol finals.
- Accepted WebSocket audio is canonical mono 16 kHz PCM16. The server appends
  each accepted packet exactly once to the authoritative turn buffer and feeds
  the same new frames once to one Nemotron stream.
- Finalize seals live input, drains the stream, and submits the complete turn
  once to the same Parakeet scheduler lane and prompt policy as HTTP. Exactly
  one final outcome is emitted, followed by exactly one completion.
- Cancel, reset, disconnect, reconnect, queue rejection/drop, drain timeout,
  and late callback paths are fenced by turn generation and transport epoch.
- Changed partials are normalized, deduplicated, rate-limited to at most 15 Hz,
  and remain replaceable display output. They are never authoritative finals.

## Gate evidence

| Gate | Sanitized evidence | Status |
| --- | --- | --- |
| Deterministic unit/race coverage | Supported Windows environment: 463 tests passed, 15 explicit opt-in model/platform tests skipped. The suite includes synchronous final callback, duplicate finalize, atomic final/completion admission across reset/cancel, exact-request final-timeout cancellation, queue rejection/drop, sample-duration backpressure, cancellation-safe bounded native stream creation/reaping, bounded native cancellation, full-queue cancel, live-drain timeout, cancel/reset/new-turn late callbacks, disconnect/reconnect transport fencing, completed-thread pruning, scheduler shutdown drain, empty/silence/short turns, WebSocket=HTTP, canonical PCM invariance, and default report redaction. | **Validated locally.** |
| AgentTalk adapter and regressions | Full AgentTalk suite: 443 passed, one optional real DualTurn audio check skipped, with 69 parameterized subtests. Relevant ASR/remote adapter scope passed Ruff. The adapter has authenticated verified WSS, protected token lookup, CA support, bounded capture/send/receive queues, stateful anti-aliased 48-to-16 kHz conversion, contiguous 40 ms packets, FIFO media/control ordering, cancellation/reset/reconnect fencing, deterministic shutdown, and parallel local fallback. | **Validated locally.** |
| Chunk-boundary invariance | Real Linux candidate runs at 10/20/40/64/100 ms reconstructed identical canonical PCM and returned the same authoritative final, with one final, one completion, and zero audio/event sequence gaps for every packetization. | **Validated on the LAN candidate.** |
| Real final-ASR parity | 143/143 HTTP requests succeeded on both the unchanged reference and frozen candidate. All 143 normalized transcript hashes matched pairwise; both targets had six empty results and there were zero empty disagreements. Candidate/reference client medians were 0.2138/0.2318 s; p95 values were 0.6776/0.6323 s. Across all paired requests the median delta was -0.0061 s, mean delta +0.0012 s, and the candidate was faster on 84/143 requests. | **Validated on private data over the authorized LAN.** Central latency is at parity; the approximately 45 ms p95 difference is within the run-to-run variation seen in the independent pre-freeze repeat. |
| Representative live correctness | A 15-turn private live run produced 15/15 WebSocket successes, no multiple finals, no sequence gaps, and 15/15 authoritative WebSocket finals exactly equal to candidate HTTP finals. | **Validated on private data over the authorized LAN.** |
| Live latency versus local AgentTalk | Local baseline: first-partial median 1.9121 s, p95 3.0347 s, 106 changed partials. Three frozen-candidate canary repetitions each covered 15 clips and 114.912 seconds of audio; first-partial median was about 1.6074 s, p95 about 2.4458 s, with 98 changed remote partials, zero fallbacks, zero worker errors, and no fatal error per run. | **Validated at two live engine threads.** Remote live is materially faster than the local path. |
| Failure injection and recovery | During a real 15-clip canary the authenticated TLS tunnel was terminated. The run completed without a fatal error, switched from 23 remote to 67 local-fallback partials, reported the expected single remote worker error, and did not hang. Deterministic tests separately cover same-instance reconnect, cancellation, reset, stale media, and shutdown races. | **Validated locally and over the authorized LAN.** |
| Parallel reliability and thread selection | A frozen-candidate paced run at concurrency four and three repetitions completed 12/12 turns covering 144.796 seconds of audio, with zero protocol/sequence failures and no missing final or completion. Fixed-affinity sweeps over 1/2/4/6 final and live threads all passed; four final and two live threads were selected to balance final latency and live CPU capacity. | **Validated on the LAN candidate.** |
| 30-minute reliability | The exact commit candidate completed 152/152 paced turns covering 1,834.089 seconds of public-fixture audio. It produced 2,888 changed partials with zero failed records, zero event/audio sequence failures, zero missing finals, and zero missing completions. First-partial median/p95 were 0.7440/0.7547 s; completion-after-finalize median/p95 were 0.4910/0.5099 s. After idle, health reported zero active sessions, speakers, queued jobs, or scheduler sessions; the service had zero restarts and no warning journal entries. The resident two-model process used 2,105,272 kB RSS and 36 threads (2,118,062,080 cgroup bytes and 36 tasks). | **Validated on the exact LAN commit candidate.** |
| Packaging | The documentation-synchronized `1.0.4rc1` wheel/sdist passed `twine check` and the archive privacy inspection. The wheel is 269,947 bytes (`F3F3E7D644D81077E9D8B39D26F145447AB88257D5CDCE23616EF7AFE8E64CB1`); the sdist is 255,198 bytes (`2A71DB79BA6C9A4CC0CF68456B169998F9B45F4F005A8893F8E32CC513B84D0F`). A fresh Python 3.12 environment installed the exact wheel with production dependencies; `pip check`, imports outside the source tree, packaged warmup/document resources, all five entry-point help checks, and version `1.0.4rc1` passed. | **Validated locally.** |
| GitHub/CI/TestPyPI | No corrected commit or index artifact exists yet. | **Pending.** |

## Platform, security, and capacity boundaries

- The pinned Nemotron-live/Parakeet-final profile is supported for Linux
  x86-64 with the release-pinned sherpa-onnx runtime. Native Windows remains a
  development target for this pair because voiced cumulative-recovery audio
  has produced intermittent empty Parakeet finals there.
- Keep the ASR process on loopback and expose it through authenticated TLS/WSS,
  or configure direct TLS plus bearer authentication. Tokens come from a
  protected environment/secret source and are absent from configuration,
  process arguments, logs, capabilities, reports, and artifacts.
- WebSocket clients send canonical 16 kHz audio and keep one stateful resampler
  per logical turn. HTTP retains its documented whole-request resampling
  support for other advertised rates.
- Both models remain resident. Capacity planning must include measured process
  RSS, native runtime headroom, the authoritative PCM retained until final,
  and sample-duration-bounded per-session queues. Authoritative PCM is released
  as soon as an immutable final-worker copy exists. Model archive sizes are not
  a memory budget.
- Benchmark output is publish-safe by default. Raw paths, endpoint addresses,
  references, partials, and final transcripts require the explicit
  `--include-sensitive-details` protected-local opt-in and remain ignored.
- AgentTalk rollback remains one configuration change to local
  `SherpaNemotronLiveAsr` plus the unchanged reference final-ASR endpoint.
- No production-PyPI publication is authorized by this candidate. Only the
  unique `1.0.4rc1` TestPyPI artifact may be published after all hard gates and
  CI are green.

## Remaining close-out

1. Commit and push only the intended changes to the existing repository
   `master` branch.
2. Wait for release CI, publish the exact CI-equivalent `1.0.4rc1` wheel/sdist
   to TestPyPI, and install the exact files retrieved from that index in a new
   environment.
