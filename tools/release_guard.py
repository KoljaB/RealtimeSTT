#!/usr/bin/env python3
"""Fast, deterministic guard between deployed Python code and publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 2
MAX_ATTESTATION_AGE_SECONDS = 30 * 60
SIGNATURE_NAMESPACE = "codex-release-guard"
IGNORED_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class GuardError(RuntimeError):
    pass


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
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


def _content_hash(
    data: bytes, relative: PurePosixPath, *, canonical_text: bool
) -> str:
    if canonical_text and relative.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return _sha256_bytes(data)


def _included(relative: PurePosixPath) -> bool:
    return (
        bool(relative.parts)
        and "__pycache__" not in relative.parts
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


def _sdist_package_hashes(
    sdist: Path, package_dir: str, *, canonical_text: bool = False
) -> dict[str, str]:
    package_parts = PurePosixPath(package_dir.strip("/")).parts
    files: dict[str, str] = {}
    with tarfile.open(sdist, "r:*") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) <= len(package_parts):
                continue
            after_root = parts[1:]
            if tuple(after_root[: len(package_parts)]) != package_parts:
                continue
            relative = PurePosixPath(*after_root[len(package_parts) :])
            if not _included(relative):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise GuardError(f"cannot read sdist member: {member.name}")
            files[str(relative)] = _content_hash(
                stream.read(), relative, canonical_text=canonical_text
            )
    if not files:
        raise GuardError(f"sdist {sdist.name} does not contain package {package_dir!r}")
    return files


def _wheel_metadata(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
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


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


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


def _assert_same_files(
    expected: dict[str, str], actual: dict[str, str], label: str
) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        path
        for path in set(expected) & set(actual)
        if expected[path] != actual[path]
    )
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
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def _sign_manifest_file(manifest: Path, signing_key: Path) -> Path:
    key = signing_key.resolve()
    if not key.is_file():
        raise GuardError(f"private signing key does not exist: {key}")
    signature = Path(str(manifest.resolve()) + ".sig")
    with tempfile.TemporaryDirectory(prefix="release-guard-sign-") as temporary:
        temporary_manifest = Path(temporary) / manifest.name
        temporary_manifest.write_bytes(manifest.read_bytes())
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                SIGNATURE_NAMESPACE,
                str(temporary_manifest),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        temporary_signature = Path(str(temporary_manifest) + ".sig")
        if result.returncode != 0 or not temporary_signature.is_file():
            detail = result.stderr.strip() or result.stdout.strip()
            raise GuardError(f"cannot sign deployment manifest: {detail}")
        os.replace(temporary_signature, signature)
    return signature


def _verify_manifest_signature(
    manifest: Path, signature: Path, allowed_signers: Path, signer: str
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", signer):
        raise GuardError("manifest signer identity is invalid")
    if not signature.resolve().is_file():
        raise GuardError(f"deployment signature does not exist: {signature}")
    if not allowed_signers.resolve().is_file():
        raise GuardError(f"allowed-signers file does not exist: {allowed_signers}")
    result = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(allowed_signers.resolve()),
            "-I",
            signer,
            "-n",
            SIGNATURE_NAMESPACE,
            "-s",
            str(signature.resolve()),
        ],
        input=manifest.resolve().read_bytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise GuardError(f"deployment manifest signature is invalid: {detail}")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(f"cannot read deployment manifest {path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise GuardError(
            f"unsupported deployment manifest schema: {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("signer"), str):
        raise GuardError("deployment manifest signer identity is invalid")
    return payload


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise GuardError("deployment manifest created_at is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GuardError("deployment manifest created_at is invalid") from exc
    if parsed.tzinfo is None:
        raise GuardError("deployment manifest created_at has no timezone")
    return parsed.astimezone(timezone.utc)


def _assert_fresh_manifest(payload: dict[str, object], max_age_seconds: int) -> None:
    if max_age_seconds <= 0 or max_age_seconds > MAX_ATTESTATION_AGE_SECONDS:
        raise GuardError(
            f"max attestation age must be between 1 and {MAX_ATTESTATION_AGE_SECONDS} seconds"
        )
    age = (
        datetime.now(timezone.utc) - _parse_timestamp(payload.get("created_at"))
    ).total_seconds()
    if age < -60:
        raise GuardError("deployment manifest timestamp is in the future")
    if age > max_age_seconds:
        raise GuardError(
            f"deployment manifest is stale ({age:.0f}s > {max_age_seconds}s); attest again"
        )


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _package_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    packages = list(args.package_dir)
    source_packages = list(args.source_package_dir) or packages
    runtime_dirs = [Path(path).resolve() for path in args.runtime_package_dir]
    modules = list(args.runtime_module)
    if (
        not packages
        or len(packages) != len(source_packages)
        or len(packages) != len(runtime_dirs)
        or len(packages) != len(modules)
    ):
        raise GuardError(
            "package, source-package, runtime-directory and runtime-module counts differ"
        )
    if len(set(packages)) != len(packages):
        raise GuardError("duplicate --package-dir value")
    if len(set(source_packages)) != len(source_packages):
        raise GuardError("duplicate --source-package-dir value")
    if len(set(modules)) != len(modules):
        raise GuardError("duplicate --runtime-module value")
    return [
        {
            "package_dir": package,
            "source_package_dir": source_package,
            "runtime_package_dir": str(runtime_dir),
            "runtime_module": module,
        }
        for package, source_package, runtime_dir, module in zip(
            packages, source_packages, runtime_dirs, modules, strict=True
        )
    ]


RUNTIME_PROBE = r"""
import importlib
import importlib.metadata
import json
import pathlib
import sys

request = json.loads(sys.argv[1])
imports = {}
for name in request["modules"]:
    module = importlib.import_module(name)
    paths = list(getattr(module, "__path__", ()))
    if paths:
        location = pathlib.Path(paths[0])
    else:
        location = pathlib.Path(module.__file__).parent
    imports[name] = str(location.resolve())
versions = {
    name: importlib.metadata.version(name)
    for name in request["distributions"]
}
print(json.dumps({
    "python": str(pathlib.Path(sys.executable).absolute()),
    "prefix": str(pathlib.Path(sys.prefix).resolve()),
    "imports": imports,
    "versions": versions,
}, sort_keys=True))
"""


def _runtime_probe(
    runtime_python: Path, modules: list[str], distributions: list[str]
) -> dict[str, object]:
    executable = _absolute_path(runtime_python)
    if not executable.is_file():
        raise GuardError(f"runtime Python does not exist: {executable}")
    request = json.dumps(
        {"modules": modules, "distributions": distributions}, sort_keys=True
    )
    result = subprocess.run(
        [str(executable), "-I", "-c", RUNTIME_PROBE, request],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuardError(f"runtime import probe failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise GuardError("runtime import probe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GuardError("runtime import probe returned an invalid object")
    return payload


def _same_path(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def _absolute_path(path: Path) -> Path:
    """Return an absolute path without resolving a venv interpreter symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def _same_executable_path(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _runtime_state(
    specs: list[dict[str, str]],
    runtime_python: Path,
    distribution: str,
    dependencies: list[str],
) -> dict[str, object]:
    distributions = [distribution, *dependencies]
    if len(set(map(_canonical_distribution, distributions))) != len(distributions):
        raise GuardError("duplicate runtime distribution")
    probe = _runtime_probe(
        runtime_python,
        [spec["runtime_module"] for spec in specs],
        distributions,
    )
    if not _same_executable_path(
        probe.get("python"), str(_absolute_path(runtime_python))
    ):
        raise GuardError("runtime import probe used a different Python executable")
    imports = probe.get("imports")
    versions = probe.get("versions")
    if not isinstance(imports, dict) or not isinstance(versions, dict):
        raise GuardError("runtime import probe omitted imports or versions")
    package_files: dict[str, dict[str, str]] = {}
    for spec in specs:
        module = spec["runtime_module"]
        runtime_dir = spec["runtime_package_dir"]
        if not _same_path(imports.get(module), runtime_dir):
            raise GuardError(
                f"runtime module {module!r} imports from {imports.get(module)!r}, "
                f"not declared directory {runtime_dir!r}"
            )
        package_files[spec["package_dir"]] = _hash_tree(Path(runtime_dir))
    if not all(isinstance(versions.get(name), str) for name in distributions):
        raise GuardError("runtime import probe omitted a distribution version")
    return {
        "python": str(_absolute_path(runtime_python)),
        "prefix": probe.get("prefix"),
        "imports": imports,
        "versions": {name: versions[name] for name in distributions},
        "package_files": package_files,
    }


def _validate_package_artifacts(
    repo: Path,
    wheel: Path,
    sdist: Path,
    specs: list[dict[str, str]],
    runtime_state: dict[str, object] | None,
) -> None:
    runtime_files = runtime_state.get("package_files") if runtime_state else None
    for spec in specs:
        package_dir = spec["package_dir"]
        source_package_dir = spec["source_package_dir"]
        source_files = _hash_tree(repo / source_package_dir, canonical_text=True)
        wheel_source_files = _wheel_package_hashes(
            wheel, package_dir, canonical_text=True
        )
        sdist_source_files = _sdist_package_hashes(
            sdist, package_dir, canonical_text=True
        )
        wheel_files = _wheel_package_hashes(wheel, package_dir)
        _assert_same_files(
            source_files, wheel_source_files, f"source package {package_dir} and wheel"
        )
        _assert_same_files(
            source_files, sdist_source_files, f"source package {package_dir} and sdist"
        )
        if runtime_state is not None:
            if not isinstance(runtime_files, dict):
                raise GuardError("runtime package hash evidence is invalid")
            current = runtime_files.get(package_dir)
            if not isinstance(current, dict):
                raise GuardError(f"runtime package evidence missing: {package_dir}")
            _assert_same_files(
                wheel_files,
                current,
                f"deployed runtime package {package_dir} and wheel",
            )


def _manifest_specs(payload: dict[str, object]) -> list[dict[str, str]]:
    specs = payload.get("packages")
    if not isinstance(specs, list) or not specs:
        raise GuardError("deployment manifest package list is invalid")
    result: list[dict[str, str]] = []
    for item in specs:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str)
            for key in (
                "package_dir",
                "source_package_dir",
                "runtime_package_dir",
                "runtime_module",
            )
        ):
            raise GuardError("deployment manifest package entry is invalid")
        result.append(
            {
                "package_dir": item["package_dir"],
                "source_package_dir": item["source_package_dir"],
                "runtime_package_dir": item["runtime_package_dir"],
                "runtime_module": item["runtime_module"],
            }
        )
    return result


def _verify_manifest_inputs(
    args: argparse.Namespace,
    payload: dict[str, object],
    repo: Path,
    wheel: Path,
    sdist: Path,
) -> tuple[list[dict[str, str]], dict[str, object] | None]:
    head = _head(repo)
    if payload.get("source_commit") != head:
        raise GuardError(
            "publication blocked: deployed source commit does not equal release HEAD "
            f"({payload.get('source_commit')} != {head})"
        )
    specs = _manifest_specs(payload)
    requested_packages = list(args.package_dir)
    if requested_packages != [spec["package_dir"] for spec in specs]:
        raise GuardError("publication blocked: package list differs from deployment manifest")
    requested_sources = list(args.source_package_dir) or requested_packages
    if requested_sources != [spec["source_package_dir"] for spec in specs]:
        raise GuardError(
            "publication blocked: source package list differs from deployment manifest"
        )
    metadata = _wheel_metadata(wheel)
    distribution = payload.get("distribution")
    dependencies = payload.get("dependencies")
    if not isinstance(distribution, str) or not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise GuardError("deployment manifest distribution metadata is invalid")
    if _canonical_distribution(metadata["name"]) != _canonical_distribution(distribution):
        raise GuardError(
            f"wheel distribution {metadata['name']!r} does not match {distribution!r}"
        )
    if payload.get("wheel_metadata") != metadata:
        raise GuardError("publication blocked: wheel metadata differs from deployment manifest")
    artifacts = [wheel, sdist, *(path.resolve() for path in args.artifact)]
    if payload.get("artifacts") != _artifact_records(artifacts):
        raise GuardError(
            "publication blocked: release artifact hashes differ from deployment artifacts"
        )

    runtime_state: dict[str, object] | None = None
    if not args.allow_remote_attestation:
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("python"), str):
            raise GuardError("deployment manifest runtime evidence is invalid")
        runtime_state = _runtime_state(
            specs,
            Path(runtime["python"]),
            distribution,
            dependencies,
        )
        if runtime_state != runtime:
            raise GuardError("publication blocked: current runtime differs from attested runtime")
    _validate_package_artifacts(repo, wheel, sdist, specs, runtime_state)
    return specs, runtime_state


def _assert_remote_release_refs(
    repo: Path, remote: str, branch: str, tag: str, expected_head: str, version: str
) -> dict[str, str]:
    if tag not in {version, f"v{version}"}:
        raise GuardError(f"release tag {tag!r} does not match wheel version {version!r}")
    remote_url = _git(repo, "remote", "get-url", remote).stdout.strip()
    branch_ref = f"refs/heads/{branch}"
    tag_ref = f"refs/tags/{tag}"
    peeled_ref = tag_ref + "^{}"
    result = subprocess.run(
        ["git", "ls-remote", remote_url, branch_ref, tag_ref, peeled_ref],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuardError(f"cannot verify remote release refs: {detail}")
    refs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            refs[fields[1]] = fields[0]
    tag_head = refs.get(peeled_ref, refs.get(tag_ref))
    if refs.get(branch_ref) != expected_head:
        raise GuardError(
            f"publication blocked: remote {branch_ref} is not release HEAD {expected_head}"
        )
    if tag_head != expected_head:
        raise GuardError(
            f"publication blocked: remote {tag_ref} is not release HEAD {expected_head}"
        )
    return {"remote": remote_url, "branch": branch_ref, "tag": tag_ref}


def _index_api(repository: str, name: str, version: str, index_url: str) -> str:
    if index_url:
        return index_url.format(
            name=urllib.parse.quote(name), version=urllib.parse.quote(version)
        )
    if repository == "pypi":
        base = "https://pypi.org"
    elif repository == "testpypi":
        base = "https://test.pypi.org"
    else:
        raise GuardError("custom Twine repositories require --index-url")
    return (
        f"{base}/pypi/{urllib.parse.quote(name)}/"
        f"{urllib.parse.quote(version)}/json"
    )


def _confirm_published(
    repository: str,
    metadata: dict[str, str],
    expected: list[dict[str, str | int]],
    index_url: str,
    timeout_seconds: int,
) -> list[dict[str, str | int]]:
    if timeout_seconds <= 0 or timeout_seconds > 180:
        raise GuardError("publish confirmation timeout must be between 1 and 180 seconds")
    url = _index_api(repository, metadata["name"], metadata["version"], index_url)
    deadline = time.monotonic() + timeout_seconds
    last_error = "release not visible"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                payload = json.load(response)
            urls = payload.get("urls") if isinstance(payload, dict) else None
            if not isinstance(urls, list):
                raise GuardError("package index returned invalid release JSON")
            published = {
                item.get("filename"): {
                    "filename": item.get("filename"),
                    "sha256": (item.get("digests") or {}).get("sha256"),
                    "size": item.get("size"),
                }
                for item in urls
                if isinstance(item, dict)
            }
            mismatches = [
                record
                for record in expected
                if published.get(record["filename"]) != record
            ]
            if not mismatches:
                return expected
            last_error = "published artifact hashes do not yet match"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise GuardError(
                f"upload completed but package-index confirmation failed: {last_error}"
            )
        time.sleep(2)


def command_check(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo.resolve()
    checked = _assert_clean_worktrees(repo)
    return {"status": "ok", "head": _head(repo), "clean_worktrees": checked}


def command_attest(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo.resolve()
    wheel = args.wheel.resolve()
    sdist = args.sdist.resolve()
    specs = _package_specs(args)
    checked = _assert_clean_worktrees(repo)
    metadata = _wheel_metadata(wheel)
    if _canonical_distribution(metadata["name"]) != _canonical_distribution(
        args.distribution
    ):
        raise GuardError(
            f"wheel distribution {metadata['name']!r} does not match {args.distribution!r}"
        )
    runtime_state = _runtime_state(
        specs,
        args.runtime_python,
        args.distribution,
        list(args.dependency),
    )
    versions = runtime_state["versions"]
    if not isinstance(versions, dict) or versions.get(args.distribution) != metadata["version"]:
        raise GuardError(
            "runtime distribution version does not match wheel version "
            f"{metadata['version']!r}"
        )
    _validate_package_artifacts(repo, wheel, sdist, specs, runtime_state)
    artifacts = [wheel, sdist, *(path.resolve() for path in args.artifact)]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signer": args.signer,
        "component": args.component,
        "runtime_label": args.runtime_label,
        "source_commit": _head(repo),
        "packages": specs,
        "distribution": args.distribution,
        "dependencies": list(args.dependency),
        "wheel_metadata": metadata,
        "artifacts": _artifact_records(artifacts),
        "runtime": runtime_state,
        "clean_worktrees": checked,
    }
    _atomic_json(args.output.resolve(), manifest)
    signature = _sign_manifest_file(
        args.output.resolve(), args.signing_key_file.resolve()
    )
    return {
        "status": "ok",
        "manifest": str(args.output.resolve()),
        "signature": str(signature),
        "manifest_sha256": _sha256_file(args.output.resolve()),
        **manifest,
    }


def command_verify(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo.resolve()
    wheel = args.wheel.resolve()
    sdist = args.sdist.resolve()
    _verify_manifest_signature(
        args.manifest.resolve(),
        args.signature_file.resolve(),
        args.allowed_signers_file.resolve(),
        args.signer,
    )
    payload = _load_manifest(args.manifest.resolve())
    if payload.get("signer") != args.signer:
        raise GuardError("deployment manifest signer differs from requested signer")
    _assert_fresh_manifest(payload, args.max_attestation_age_seconds)
    checked = _assert_clean_worktrees(repo)
    specs, runtime_state = _verify_manifest_inputs(args, payload, repo, wheel, sdist)
    return {
        "status": "ok",
        "component": payload.get("component"),
        "head": _head(repo),
        "packages": [spec["package_dir"] for spec in specs],
        "artifacts": _artifact_records(
            [wheel, sdist, *(path.resolve() for path in args.artifact)]
        ),
        "clean_worktrees": checked,
        "live_runtime_rechecked": runtime_state is not None,
        "parity": "exact",
    }


def command_publish(args: argparse.Namespace) -> dict[str, object]:
    result = command_verify(args)
    repo = args.repo.resolve()
    wheel = args.wheel.resolve()
    sdist = args.sdist.resolve()
    artifacts = [wheel, sdist, *(path.resolve() for path in args.artifact)]
    metadata = _wheel_metadata(wheel)
    result["remote_refs"] = _assert_remote_release_refs(
        repo,
        args.remote,
        args.branch,
        args.tag,
        _head(repo),
        metadata["version"],
    )
    command = [sys.executable, "-m", "twine", "upload", "--non-interactive"]
    if args.repository:
        command.extend(["--repository", args.repository])
    command.extend(str(path) for path in artifacts)
    upload = subprocess.run(command, check=False)
    if upload.returncode != 0:
        raise GuardError(f"twine upload failed with exit code {upload.returncode}")
    records = _artifact_records(artifacts)
    _confirm_published(
        args.repository or "pypi",
        metadata,
        records,
        args.index_url,
        args.publish_confirm_timeout,
    )
    result["published"] = records
    result["repository"] = args.repository or "pypi"
    result["index_roundtrip"] = "exact"
    return result


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package-dir", action="append", required=True)
    parser.add_argument(
        "--source-package-dir",
        action="append",
        default=[],
        help="Source path when it differs from the package path inside artifacts",
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])


def _add_verification_arguments(parser: argparse.ArgumentParser) -> None:
    _add_artifact_arguments(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signature-file", type=Path, required=True)
    parser.add_argument("--allowed-signers-file", type=Path, required=True)
    parser.add_argument("--signer", required=True)
    parser.add_argument(
        "--max-attestation-age-seconds",
        type=int,
        default=MAX_ATTESTATION_AGE_SECONDS,
    )
    parser.add_argument(
        "--allow-remote-attestation",
        action="store_true",
        help="Use a fresh signed remote attestation when runtime paths are not local",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Block publication unless clean source, wheel, sdist and deployed runtime match."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-worktrees", help="Fail if any linked worktree is dirty"
    )
    check.add_argument("--repo", type=Path, required=True)
    check.set_defaults(handler=command_check)

    attest = subparsers.add_parser(
        "attest", help="Record a fresh signed artifact-to-runtime deployment"
    )
    _add_artifact_arguments(attest)
    attest.add_argument("--component", required=True)
    attest.add_argument("--distribution", required=True)
    attest.add_argument(
        "--runtime-package-dir", type=Path, action="append", required=True
    )
    attest.add_argument("--runtime-module", action="append", required=True)
    attest.add_argument("--runtime-python", type=Path, required=True)
    attest.add_argument("--dependency", action="append", default=[])
    attest.add_argument("--runtime-label", default="")
    attest.add_argument("--signing-key-file", type=Path, required=True)
    attest.add_argument("--signer", required=True)
    attest.add_argument("--output", type=Path, required=True)
    attest.set_defaults(handler=command_attest)

    verify = subparsers.add_parser(
        "verify", help="Fail unless release artifacts match a fresh signed deployment"
    )
    _add_verification_arguments(verify)
    verify.set_defaults(handler=command_verify)

    publish = subparsers.add_parser(
        "publish", help="Verify exact parity, upload, then confirm index hashes"
    )
    _add_verification_arguments(publish)
    publish.add_argument("--repository", default="pypi")
    publish.add_argument("--remote", default="origin")
    publish.add_argument("--branch", required=True)
    publish.add_argument("--tag", required=True)
    publish.add_argument(
        "--index-url",
        default="",
        help="JSON URL template for custom indexes; supports {name} and {version}",
    )
    publish.add_argument("--publish-confirm-timeout", type=int, default=90)
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
