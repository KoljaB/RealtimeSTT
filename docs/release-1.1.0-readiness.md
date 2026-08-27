# RealtimeSTT 1.1.0 release readiness

Status date: 2026-08-27

This checklist is evidence-conservative. An unchecked gate is not a release
claim, and this file does not authorize publication.

## Required gates

- [ ] Base the final candidate on the current public master, then review the
  final commit, intended working-tree changes, branch, and remote divergence.
- [ ] Run `python tools/release_guard.py check-worktrees --repo .` before
  building; every linked worktree must be clean.
- [ ] Build the exact candidate with `python -m build` and validate metadata with
  `python -m twine check dist/*`.
- [ ] Resolve every declared base and extra dependency from the intended public
  index for the candidate Python matrix; reconcile any unavailable pin before
  publication.
- [ ] Inspect wheel and sdist names and contents for the target version, private
  paths or network addresses, credentials, reports, model weights, and
  development-only artifacts.
- [ ] Install the wheel in a fresh environment outside the source checkout;
  verify import provenance, package version, packaged resources, console help,
  and `pip check`.
- [ ] Repeat the isolated install, import-provenance, console-help, and
  `pip check` gates from the sdist.
- [ ] Record the supported unit/contract CI result for the exact final commit.
- [ ] Exercise the release-supported Python 3.11 and 3.12 matrix; Python 3.13
  and newer remain outside this release's `Requires-Python` range.
- [ ] Run Linux x86-64 Nemotron-live/Parakeet-final real-model acceptance with
  the release-pinned sherpa-onnx runtime and externally verified model bundles.
- [ ] Attest the exact deployed wheel/runtime with
  `python tools/release_guard.py attest ...` no more than 30 minutes before
  publication. Include `RealtimeSTT`, `RealtimeSTT_server`, and
  `example_fastapi_server`, the exact runtime Python/import paths, the required
  wheel and sdist, and the private attestation key.
- [ ] Verify and publish only through
  `python tools/release_guard.py publish ... --repository testpypi` (or
  `--repository pypi`); direct `twine upload` is prohibited. Publication must
  also prove that the remote release branch and version tag resolve to the
  attested commit, then confirm both artifact hashes through the package-index
  JSON API.
- [ ] Download the exact published wheel and sdist into fresh environments and
  repeat the package smoke checks.
- [ ] Review the final public Git tree and release notes for privacy, licenses,
  platform/model boundaries, rollback instructions, and user-facing accuracy.

## Known boundaries

The pinned Nemotron-live/Parakeet-final profile targets Linux x86-64. Native
Windows remains a development target for this pair because cumulative-recovery
authoritative finals require independent validation. Model weights are external
artifacts and are not shipped in the Python distributions.

## Evidence recording

This checklist defines gates. Exact-commit workflow URLs, published artifact
hashes, and real-model results are recorded in the matching GitHub release,
where the post-push and post-tag evidence can remain authoritative.

The release guard is the publication boundary. A clean source checkout or a
successful CI build does not prove that the running Linux service is the same
code. The guard compares all three shipped package trees across source, wheel,
sdist, and runtime, probes their import paths with the declared runtime Python,
and signs the result. Local runtimes are re-read during verification; remote
Linux attestations are accepted only with a valid private-key signature and a
maximum age of 30 minutes. `release_guard.py publish` additionally checks the
remote branch/tag and confirms the uploaded hashes from the package index.

## Local preflight evidence

The following preparation checks passed on the candidate tree on 2026-08-27.
They reduce local release risk but do not replace exact-commit CI, the Linux
real-model gate, or TestPyPI installation:

- Python 3.11.5: 574 unit tests passed with 14 environment/model skips; all 34
  top-level Preview and tail-transcription contract tests passed.
- Python 3.12.4: both the wheel and sdist installed outside the checkout from
  public dependencies with the `server,sherpa-onnx` extras. Package version,
  import provenance, packaged resources, both console entry points, and
  `pip check` passed for each artifact.
- PyPA build, Twine metadata validation, and the archive/source privacy scan
  passed for `realtimestt-1.1.0-py3-none-any.whl` and
  `realtimestt-1.1.0.tar.gz`.
- The pinned `halo==0.0.31` dependency resolved from public PyPI and built in
  the fresh Python 3.12 environment.
