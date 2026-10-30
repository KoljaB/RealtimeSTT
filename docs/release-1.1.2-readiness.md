# RealtimeSTT 1.1.2 release readiness

Status date: 2026-08-30

This checklist is evidence-conservative. An unchecked gate is not a release
claim, and this file does not authorize publication.

## Required gates

- [ ] Base the candidate on current public master and review the exact commit,
  linked worktrees, branch, tag, and remote divergence.
- [ ] Run the focused early-RMS, authoritative Preview, production-session,
  capabilities, package-version, and release-guard tests for the exact commit.
- [ ] Build the wheel and sdist once from a fresh clean worktree and validate
  their metadata, contents, filenames, and privacy checks.
- [ ] Install that exact wheel into the declared Linux runtime, verify all three
  import roots, restart the service, and run authenticated health,
  capabilities, representative Preview, and 25 ms early-RMS acceptance.
- [ ] Attest `RealtimeSTT`, `RealtimeSTT_server`, and
  `example_fastapi_server` against the exact deployed wheel and sdist no more
  than 30 minutes before publication.
- [ ] Publish only through `tools/release_guard.py publish --repository pypi`,
  proving the remote release branch and `v1.1.2` tag resolve to the attested
  commit and confirming both uploaded hashes through PyPI.
- [ ] Download the exact published wheel and sdist into fresh environments and
  repeat package provenance and smoke checks.

## Known boundaries

The exact advisory `previewMode = "early_rms"` uses one decode with 25 ms of
zero-PCM decoder-flush silence and no empty-result retry. Authoritative,
missing, unknown, and malformed modes retain the quality-preserving conditional
500 ms retry. Miep owns browser RMS timing and the decision to use the advisory
mode. LLM admission-proxy cancellation is outside this distribution.

The pinned Nemotron-live/Parakeet-final profile targets Linux x86-64. Native
Windows remains a development target for this pair. Model weights are external
artifacts and are not shipped in the Python distributions.

## Evidence recording

Exact-commit workflow URLs, deployed import paths, artifact hashes, runtime
smoke results, and PyPI confirmation belong in the matching GitHub release.