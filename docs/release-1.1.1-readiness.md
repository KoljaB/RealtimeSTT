# RealtimeSTT 1.1.1 release readiness

Status date: 2026-08-28

This checklist is evidence-conservative. An unchecked gate is not a release
claim, and this file does not authorize publication.

## Required gates

- [ ] Base the candidate on current public master and review the exact commit,
  linked worktrees, branch, tag, and remote divergence.
- [ ] Run the focused cancellation, production-session, server-protocol, and
  package-version tests for the exact candidate commit.
- [ ] Build the wheel and sdist once from a fresh clean worktree and validate
  their metadata and contents.
- [ ] Install that exact wheel into the declared Linux runtime, verify import
  provenance, restart the service, and run authenticated health, capabilities,
  and representative Preview cancellation acceptance.
- [ ] Attest `RealtimeSTT`, `RealtimeSTT_server`, and
  `example_fastapi_server` against the exact deployed wheel and sdist no more
  than 30 minutes before publication.
- [ ] Publish only through `tools/release_guard.py publish --repository pypi`,
  proving the remote release branch and `v1.1.1` tag resolve to the attested
  commit and confirming both uploaded hashes through PyPI.
- [ ] Download the exact published wheel and sdist into fresh environments and
  repeat package provenance and smoke checks.

## Known boundaries

The pinned Nemotron-live/Parakeet-final profile targets Linux x86-64. Native
Windows remains a development target for this pair. Model weights are external
artifacts and are not shipped in the Python distributions.

The 64 ms continuation hysteresis that decides when Miep supersedes a Preview
snapshot is a Miep integration change. This package release supplies the
request-scoped cancellation needed once that decision has been made.

## Evidence recording

Exact-commit workflow URLs, deployed import paths, artifact hashes, runtime
smoke results, and PyPI confirmation belong in the matching GitHub release.
