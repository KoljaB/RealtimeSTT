#!/usr/bin/env python3
"""Fast, deterministic guard between a deployed Python runtime and publication."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
IGNORED_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


class GuardError(RuntimeError):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={repo.as_posix()}",
        "-C",
        str(repo),
        *args,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuardError(f"git command failed: {' '.join(args)}: {detail}")
    return result


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(data: bytes, relative: PurePosixPath, *, canonical_text: bool) -> str:
    if canonical_text and relative.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return _sha256_bytes(data)


def _included(relative: PurePosixPath) -> bool:
    return (
        "__pycache__" not in relative.parts
        and relative.suffix.lower() not in IGNORED_SUFFIXES
    )


def _hash_tree(root: Path, *, canonical_text: bool = False) -> dict[str, str]:
    if not root.is_dir():
        raise GuardError(f"package directory does not exist: {root}")
    files: dict[str, str] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _included(relative):
            files[str(relative)] = _content_hash(
                path.read_bytes(), relative, canonical_text=canonical_text
            )
    if not files:
        raise GuardError(f"package directory contains no releasable files: {root}")
    return files


def _wheel_package_hashes(
    wheel: Path, package_dir: str, *, canonical_text: bool = False
) -> dict[str, str]:
    prefix = package_dir.strip("/") + "/"
    files: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            relative = PurePosixPath(info.filename[len(prefix) :])
            if _included(relative):
                files[str(relative)] = _content_hash(
                    archive.read(info), relative, canonical_text=canonical_text
                )
    if not files:
        raise GuardError(f"wheel {wheel.name} does not contain package {package_dir!r}")
    return files


def _wheel_metadata(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise GuardError(f"wheel must contain exactly one METADATA file: {wheel}")
        metadata = archive.read(names[0]).decode("utf-8", errors="strict")
    result: dict[str, str] = {}
    for line in metadata.splitlines():
        if line.startswith("Name: ") and "name" not in result:
            result["name"] = line[6:].strip()
        elif line.startswith("Version: ") and "version" not in result:
            result["version"] = line[9:].strip()
    if not result.get("name") or not result.get("version"):
        raise GuardError(f"wheel metadata lacks Name or Version: {wheel}")
    return result


def _worktree_paths(repo: Path) -> list[Path]:
    output = _git(repo, "worktree", "list", "--porcelain").stdout
    paths: list[Path] = []
    for block in output.strip().split("\n\n"):
        lines = block.splitlines()
        if any(line.startswith("prunable") for line in lines):
            continue
        for line in lines:
            if line.startswith("worktree "):
                paths.append(Path(line[len("worktree ") :]).resolve())
                break
    return paths


def _assert_clean_worktrees(repo: Path) -> list[str]:
    dirty: list[str] = []
    checked: list[str] = []
    for worktree in _worktree_paths(repo):
        safe = worktree.as_posix()
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={safe}",
                "-C",
                str(worktree),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise GuardError(f"cannot inspect worktree {worktree}: {detail}")
        checked.append(str(worktree))
        if result.stdout.strip():
            dirty.append(f"{worktree}:\n{result.stdout.rstrip()}")
    if dirty:
        raise GuardError(
            "publication blocked: uncommitted files exist in linked worktrees:\n"
            + "\n".join(dirty)
        )
    return checked


def _assert_same_files(expected: dict[str, str], actual: dict[str, str], label: str) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])
    if missing or extra or changed:
        details: list[str] = [f"publication blocked: {label} differs"]
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("extra: " + ", ".join(extra))
        if changed:
            details.append("changed: " + ", ".join(changed))
        raise GuardError("\n".join(details))


def _artifact_records(paths: list[Path]) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise GuardError(f"release artifact does not exist: {resolved}")
        if resolved.name in seen:
            raise GuardError(f"duplicate artifact name: {resolved.name}")
        seen.add(resolved.name)
        records.append(
            {
                "filename": resolved.name,
                "sha256": _sha256_file(resolved),
                "size": resolved.stat().st_size,
            }
        )
    return records


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"cannot read deployment manifest {path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise GuardError(f"unsupported deployment manifest schema: {payload.get('schema_version')!r}")
    return payload


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def command_check(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo.resolve()
    checked = _assert_clean_worktrees(repo)
    return {"status": "ok", "head": _head(repo), "clean_worktrees": checked}


def command_attest(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo.resolve()
    wheel = args.wheel.resolve()
    artifacts = [wheel, *(path.resolve() for path in args.artifact)]
    checked = _assert_clean_worktrees(repo)
    source_files = _hash_tree(repo / args.package_dir, canonical_text=True)
    wheel_source_files = _wheel_package_hashes(
        wheel, args.package_dir, canonical_text=True
    )
    wheel_files = _wheel_package_hashes(wheel, args.package_dir)
    runtime_files = _hash_tree(args.runtime_package_dir.resolve())
    _assert_same_files(source_files, wheel_source_files, "source package and wheel")
    _assert_same_files(wheel_files, runtime_files, "deployed runtime and wheel")

    metadata = _wheel_metadata(wheel)
    versions: dict[str, str] = {}
    for distribution in [args.distribution, *args.dependency]:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise GuardError(f"runtime distribution is not installed: {distribution}") from exc
    if versions[args.distribution] != metadata["version"]:
        raise GuardError(
            f"runtime version {versions[args.distribution]!r} does not match wheel "
            f"version {metadata['version']!r}"
        )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "component": args.component,
        "runtime_label": args.runtime_label,
        "source_commit": _head(repo),
        "package_dir": args.package_dir,
        "wheel_metadata": metadata,
        "artifacts": _artifact_records(artifacts),
        "runtime_files": runtime_files,
        "distribution_versions": versions,
        "clean_worktrees": checked,
    }
    _atomic_json(args.output.resolve(), manifest)
    return {"status": "ok", "manifest": str(args.output.resolve()), **manifest}


def command_verify(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo.resolve()
    wheel = args.wheel.resolve()
    artifacts = [wheel, *(path.resolve() for path in args.artifact)]
    manifest = _load_manifest(args.manifest.resolve())
    checked = _assert_clean_worktrees(repo)
    head = _head(repo)
    if manifest.get("source_commit") != head:
        raise GuardError(
            "publication blocked: deployed source commit does not equal release HEAD "
            f"({manifest.get('source_commit')} != {head})"
        )
    if manifest.get("package_dir") != args.package_dir:
        raise GuardError("publication blocked: package directory differs from deployment manifest")

    source_files = _hash_tree(repo / args.package_dir, canonical_text=True)
    wheel_source_files = _wheel_package_hashes(
        wheel, args.package_dir, canonical_text=True
    )
    wheel_files = _wheel_package_hashes(wheel, args.package_dir)
    _assert_same_files(source_files, wheel_source_files, "source package and wheel")
    runtime_files = manifest.get("runtime_files")
    if not isinstance(runtime_files, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in runtime_files.items()
    ):
        raise GuardError("deployment manifest runtime_files is invalid")
    _assert_same_files(runtime_files, wheel_files, "attested runtime and release wheel")

    actual_artifacts = _artifact_records(artifacts)
    if manifest.get("artifacts") != actual_artifacts:
        raise GuardError("publication blocked: release artifact hashes differ from deployed artifacts")
    if manifest.get("wheel_metadata") != _wheel_metadata(wheel):
        raise GuardError("publication blocked: wheel metadata differs from deployment manifest")

    return {
        "status": "ok",
        "component": manifest.get("component"),
        "head": head,
        "artifacts": actual_artifacts,
        "clean_worktrees": checked,
        "parity": "exact",
    }


def command_publish(args: argparse.Namespace) -> dict[str, object]:
    result = command_verify(args)
    artifacts = [args.wheel.resolve(), *(path.resolve() for path in args.artifact)]
    command = [sys.executable, "-m", "twine", "upload", "--non-interactive"]
    if args.repository:
        command.extend(["--repository", args.repository])
    command.extend(str(path) for path in artifacts)
    upload = subprocess.run(command, check=False)
    if upload.returncode != 0:
        raise GuardError(f"twine upload failed with exit code {upload.returncode}")
    result["published"] = [path.name for path in artifacts]
    result["repository"] = args.repository or "pypi"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Block publication unless clean source, built artifacts and deployed runtime match."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-worktrees", help="Fail if any linked worktree is dirty")
    check.add_argument("--repo", type=Path, required=True)
    check.set_defaults(handler=command_check)

    attest = subparsers.add_parser(
        "attest", help="Record a clean, exact wheel-to-runtime deployment"
    )
    attest.add_argument("--repo", type=Path, required=True)
    attest.add_argument("--component", required=True)
    attest.add_argument("--distribution", required=True)
    attest.add_argument("--package-dir", required=True)
    attest.add_argument("--wheel", type=Path, required=True)
    attest.add_argument("--artifact", type=Path, action="append", default=[])
    attest.add_argument("--runtime-package-dir", type=Path, required=True)
    attest.add_argument("--dependency", action="append", default=[])
    attest.add_argument("--runtime-label", default="")
    attest.add_argument("--output", type=Path, required=True)
    attest.set_defaults(handler=command_attest)

    verify = subparsers.add_parser(
        "verify", help="Fail unless the release artifacts exactly match an attested runtime"
    )
    verify.add_argument("--repo", type=Path, required=True)
    verify.add_argument("--package-dir", required=True)
    verify.add_argument("--wheel", type=Path, required=True)
    verify.add_argument("--artifact", type=Path, action="append", default=[])
    verify.add_argument("--manifest", type=Path, required=True)
    verify.set_defaults(handler=command_verify)

    publish = subparsers.add_parser(
        "publish", help="Verify exact runtime parity and then invoke Twine"
    )
    publish.add_argument("--repo", type=Path, required=True)
    publish.add_argument("--package-dir", required=True)
    publish.add_argument("--wheel", type=Path, required=True)
    publish.add_argument("--artifact", type=Path, action="append", default=[])
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument("--repository", default="pypi")
    publish.set_defaults(handler=command_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except GuardError as exc:
        print(f"RELEASE_GUARD_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
