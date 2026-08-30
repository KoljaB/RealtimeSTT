#!/usr/bin/env python3
"""Fast, deterministic guard between deployed Python code and publication."""

from __future__ import annotations

import argparse
import base64
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
from pathlib import Path, PurePosixPath, PureWindowsPath

SCHEMA_VERSION = 3
MAX_ATTESTATION_AGE_SECONDS = 30 * 60
SIGNATURE_NAMESPACE = "codex-release-guard"
IGNORED_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {
    ".cfg",
    ".cmd",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# These profiles are deliberately code-owned. Release callers may supply runtime
# locations, but they cannot redefine what a component is, who may attest it, or
# which public repository/branch is released.
COMPONENT_PROFILES: dict[str, dict[str, object]] = {
    "RealtimeTTS": {
        "distribution": "realtimetts",
        "packages": (("RealtimeTTS", "RealtimeTTS", "RealtimeTTS"),),
        "signer": "linux-services",
        "signer_fingerprint": "SHA256:ODuksd5J17paccWV+N0zWfczcc1iV30V5mQytjiar2w",
        "remote_repository": "github.com/koljab/realtimetts",
        "remote_branch": "master",
        "service_required": True,
        "service_name": "wwz-qwen3-tts.service",
        "publishable": True,
    },
    "RealtimeSTT": {
        "distribution": "realtimestt",
        "packages": (
            ("RealtimeSTT", "RealtimeSTT", "RealtimeSTT"),
            ("RealtimeSTT_server", "RealtimeSTT_server", "RealtimeSTT_server"),
            (
                "example_fastapi_server",
                "example_fastapi_server",
                "example_fastapi_server",
            ),
        ),
        "signer": "linux-services",
        "signer_fingerprint": "SHA256:ODuksd5J17paccWV+N0zWfczcc1iV30V5mQytjiar2w",
        "remote_repository": "github.com/koljab/realtimestt",
        "remote_branch": "master",
        "service_required": True,
        "service_name": "realtimestt-104rc1-final-20260820.service",
        "publishable": True,
    },
    "echoff": {
        "distribution": "echoff",
        "packages": (("echoff", "src/echoff", "echoff"),),
        "signer": "echoff-windows",
        "signer_fingerprint": "SHA256:1FRbwkXHlurnoZOYAfK4yApbE9lbwmq8aoRSfJdEhQo",
        "remote_repository": "github.com/koljab/echoff",
        "remote_branch": "main",
        "service_required": False,
        "service_name": None,
        "publishable": True,
    },
    # Non-publishable profiles retain real CLI/unit coverage without weakening
    # any production component profile.
    "fixture": {
        "distribution": "guard-dist",
        "packages": (("GuardPkg", "GuardPkg", "GuardPkg"),),
        "signer": "guard-cli-test",
        "signer_fingerprint": None,
        "remote_repository": None,
        "remote_branch": None,
        "service_required": False,
        "service_name": None,
        "publishable": False,
    },
    "demo": {
        "distribution": "demo_distribution",
        "packages": (
            ("demo_pkg", "demo_pkg", "demo_pkg"),
            ("demo_server", "src/demo_server", "demo_server"),
        ),
        "signer": "test-runtime",
        "signer_fingerprint": None,
        "remote_repository": None,
        "remote_branch": None,
        "service_required": False,
        "service_name": None,
        "publishable": False,
    },
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


def _component_profile(component: object, *, publishing: bool = False) -> dict[str, object]:
    if not isinstance(component, str) or component not in COMPONENT_PROFILES:
        raise GuardError(f"unknown release component: {component!r}")
    profile = COMPONENT_PROFILES[component]
    if publishing and profile.get("publishable") is not True:
        raise GuardError(f"component {component!r} is not publishable")
    return profile


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    if "\\" in value:
        raise GuardError(f"{label} must use a repository-relative POSIX path")
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise GuardError(f"{label} must stay inside its declared root: {value!r}")
    return relative


def _repo_member(repo: Path, relative: str, label: str) -> Path:
    member = _safe_relative_path(relative, label)
    root = repo.resolve()
    resolved = (root / Path(*member.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GuardError(f"{label} escapes repository root: {relative!r}") from exc
    return resolved


def _lexically_within(child: object, root: object) -> bool:
    if not isinstance(child, str) or not isinstance(root, str) or not child or not root:
        return False
    path_type = PureWindowsPath if re.match(r"^[A-Za-z]:[\\/]", root) else PurePosixPath
    child_path = path_type(child)
    root_path = path_type(root)
    if not child_path.is_absolute() or not root_path.is_absolute():
        return False
    if ".." in child_path.parts or ".." in root_path.parts:
        return False
    try:
        child_path.relative_to(root_path)
    except ValueError:
        return False
    return child_path != root_path


def _validate_hash_tree(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise GuardError(f"{label} hash evidence is invalid")
    result: dict[str, str] = {}
    for relative, digest in value.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise GuardError(f"{label} hash evidence is invalid")
        _safe_relative_path(relative, f"{label} file")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GuardError(f"{label} hash evidence is invalid")
        result[relative] = digest
    return result


def _ssh_key_fingerprint(key_type: str, encoded_key: str) -> str:
    if not re.fullmatch(r"(?:ssh|ecdsa|sk)-[A-Za-z0-9@._+-]+", key_type):
        raise GuardError("allowed signer key type is invalid")
    try:
        key_blob = base64.b64decode(encoded_key, validate=True)
    except (ValueError, TypeError) as exc:
        raise GuardError("allowed signer public key is invalid") from exc
    encoded = base64.b64encode(hashlib.sha256(key_blob).digest()).decode("ascii")
    return "SHA256:" + encoded.rstrip("=")


def _public_key_parts(line: str) -> tuple[str, str]:
    fields = line.split()
    for index, field in enumerate(fields[:-1]):
        if re.fullmatch(r"(?:ssh|ecdsa|sk)-[A-Za-z0-9@._+-]+", field):
            return field, fields[index + 1]
    raise GuardError("allowed signer entry has no supported public key")


def _allowed_signer_fingerprint(allowed_signers: Path, signer: str) -> str:
    matches: list[str] = []
    for raw_line in allowed_signers.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        principals = line.split(maxsplit=1)[0].split(",")
        if signer in principals:
            key_type, encoded_key = _public_key_parts(line)
            matches.append(_ssh_key_fingerprint(key_type, encoded_key))
    if len(matches) != 1:
        raise GuardError(f"allowed-signers must contain exactly one key for {signer!r}")
    return matches[0]


def _signing_key_fingerprint(signing_key: Path) -> str:
    result = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(signing_key.resolve())],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuardError(f"cannot read signing public key: {detail}")
    key_type, encoded_key = _public_key_parts(result.stdout.strip())
    return _ssh_key_fingerprint(key_type, encoded_key)


def _canonical_remote_url(value: str) -> str:
    candidate = value.strip()
    scp = re.fullmatch(r"(?:[^@/:]+@)?([^/:]+):(.+)", candidate)
    if scp and "://" not in candidate:
        host, path = scp.groups()
    else:
        parsed = urllib.parse.urlsplit(candidate)
        host = parsed.hostname or ""
        path = parsed.path
    if not host or not path:
        raise GuardError("release remote must be a network repository URL")
    normalized_path = path.strip("/")
    if normalized_path.lower().endswith(".git"):
        normalized_path = normalized_path[:-4]
    if not normalized_path:
        raise GuardError("release remote repository path is empty")
    return f"{host.lower()}/{normalized_path.lower()}"


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
    manifest: Path,
    signature: Path,
    allowed_signers: Path,
    signer: str,
    expected_fingerprint: object,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", signer):
        raise GuardError("manifest signer identity is invalid")
    if not signature.resolve().is_file():
        raise GuardError(f"deployment signature does not exist: {signature}")
    if not allowed_signers.resolve().is_file():
        raise GuardError(f"allowed-signers file does not exist: {allowed_signers}")
    if expected_fingerprint is not None:
        actual_fingerprint = _allowed_signer_fingerprint(
            allowed_signers.resolve(), signer
        )
        if actual_fingerprint != expected_fingerprint:
            raise GuardError(
                "allowed signer key does not match the component trust profile "
                f"({actual_fingerprint} != {expected_fingerprint})"
            )
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


def _package_specs(
    args: argparse.Namespace, repo: Path, profile: dict[str, object]
) -> list[dict[str, str]]:
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
    declared = tuple(zip(packages, source_packages, modules, strict=True))
    if declared != profile.get("packages"):
        raise GuardError(
            "component package contract differs from the pinned release profile: "
            f"{declared!r} != {profile.get('packages')!r}"
        )
    for package, source_package, module in declared:
        _safe_relative_path(package, "package directory")
        _repo_member(repo, source_package, "source package directory")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            raise GuardError(f"runtime module is invalid: {module!r}")
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
    absolute = Path(os.path.abspath(os.fspath(path)))
    return absolute.resolve() if os.name == "nt" else absolute


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
    prefix = probe.get("prefix")
    if (
        not isinstance(imports, dict)
        or not isinstance(versions, dict)
        or not isinstance(prefix, str)
    ):
        raise GuardError("runtime import probe omitted imports or versions")
    runtime_python_path = str(_absolute_path(runtime_python))
    if not _lexically_within(runtime_python_path, prefix):
        raise GuardError("runtime Python is outside the probed environment prefix")
    package_files: dict[str, dict[str, str]] = {}
    for spec in specs:
        module = spec["runtime_module"]
        runtime_dir = spec["runtime_package_dir"]
        if not _lexically_within(runtime_dir, prefix):
            raise GuardError(
                f"runtime package {runtime_dir!r} is outside environment prefix {prefix!r}"
            )
        if not _same_path(imports.get(module), runtime_dir):
            raise GuardError(
                f"runtime module {module!r} imports from {imports.get(module)!r}, "
                f"not declared directory {runtime_dir!r}"
            )
        package_files[spec["package_dir"]] = _hash_tree(Path(runtime_dir))
    if not all(isinstance(versions.get(name), str) for name in distributions):
        raise GuardError("runtime import probe omitted a distribution version")
    return {
        "python": runtime_python_path,
        "prefix": prefix,
        "imports": imports,
        "versions": {name: versions[name] for name in distributions},
        "package_files": package_files,
    }


def _validated_attested_runtime_state(
    runtime: object,
    specs: list[dict[str, str]],
    distribution: str,
    dependencies: list[str],
) -> dict[str, object]:
    if not isinstance(runtime, dict):
        raise GuardError("deployment manifest runtime evidence is invalid")
    python = runtime.get("python")
    prefix = runtime.get("prefix")
    imports = runtime.get("imports")
    versions = runtime.get("versions")
    package_files = runtime.get("package_files")
    if (
        not isinstance(python, str)
        or not isinstance(prefix, str)
        or not isinstance(imports, dict)
        or not isinstance(versions, dict)
        or not isinstance(package_files, dict)
    ):
        raise GuardError("deployment manifest runtime evidence is invalid")
    if not _lexically_within(python, prefix):
        raise GuardError("attested runtime Python is outside its environment prefix")

    expected_modules = {spec["runtime_module"] for spec in specs}
    expected_distributions = {distribution, *dependencies}
    expected_packages = {spec["package_dir"] for spec in specs}
    if set(imports) != expected_modules:
        raise GuardError("attested runtime import set differs from component contract")
    if set(versions) != expected_distributions or not all(
        isinstance(versions[name], str) for name in expected_distributions
    ):
        raise GuardError("attested runtime distribution set differs from manifest")
    if set(package_files) != expected_packages:
        raise GuardError("attested runtime package set differs from component contract")

    for spec in specs:
        module = spec["runtime_module"]
        runtime_dir = spec["runtime_package_dir"]
        imported = imports.get(module)
        if imported != runtime_dir or not _lexically_within(runtime_dir, prefix):
            raise GuardError(
                f"attested runtime module {module!r} is outside or differs from its prefix"
            )
        _validate_hash_tree(
            package_files.get(spec["package_dir"]),
            f"attested runtime package {spec['package_dir']}",
        )
    return runtime


def _service_name(profile: dict[str, object], requested: str) -> str | None:
    pinned = profile.get("service_name")
    required = profile.get("service_required") is True
    if pinned is not None:
        if requested and requested != pinned:
            raise GuardError(
                f"service {requested!r} differs from pinned component service {pinned!r}"
            )
        requested = str(pinned)
    if required and not requested:
        raise GuardError("this component requires a live --service-name attestation")
    if requested and not re.fullmatch(r"[A-Za-z0-9_.@:-]+\.service", requested):
        raise GuardError(f"systemd service name is invalid: {requested!r}")
    if not required and requested:
        raise GuardError("this component profile does not permit a service attestation")
    return requested or None


def _systemd_user_service_state(
    service: str, runtime_state: dict[str, object]
) -> dict[str, object]:
    if os.name != "posix":
        raise GuardError("systemd service attestation must run on the Linux service host")
    properties = (
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecStart",
        "WorkingDirectory",
        "FragmentPath",
    )
    command = ["systemctl", "--user", "show", service]
    command.extend(f"--property={name}" for name in properties)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuardError(f"cannot inspect live systemd service {service}: {detail}")
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in properties:
            values[key] = value
    if values.get("ActiveState") != "active" or values.get("SubState") != "running":
        raise GuardError(f"live service {service} is not active/running")
    try:
        main_pid = int(values.get("MainPID", "0"))
    except ValueError as exc:
        raise GuardError(f"live service {service} returned an invalid MainPID") from exc
    if main_pid <= 0:
        raise GuardError(f"live service {service} has no running MainPID")

    process = Path("/proc") / str(main_pid)
    try:
        raw_command = (process / "cmdline").read_bytes()
        raw_environment = (process / "environ").read_bytes()
        process_cwd = os.readlink(process / "cwd")
    except OSError as exc:
        raise GuardError(f"cannot inspect live service process {main_pid}: {exc}") from exc
    argv = [part.decode(errors="replace") for part in raw_command.split(b"\0") if part]
    environment = [
        part.decode(errors="replace")
        for part in raw_environment.split(b"\0")
        if part
    ]
    runtime_python = runtime_state.get("python")
    runtime_prefix = runtime_state.get("prefix")
    if not argv or not _same_executable_path(argv[0], runtime_python):
        raise GuardError(
            f"live service {service} does not run the attested runtime Python"
        )
    if any(item.startswith("PYTHONPATH=") and item != "PYTHONPATH=" for item in environment):
        raise GuardError(f"live service {service} has a shadowing PYTHONPATH")

    exec_start = values.get("ExecStart", "")
    match = re.search(r"(?:^|[ {;])path=([^ ;}]+)", exec_start)
    launcher = match.group(1) if match else ""
    if (
        not isinstance(runtime_prefix, str)
        or not launcher
        or not _lexically_within(launcher, runtime_prefix)
        or not Path(launcher).is_file()
    ):
        raise GuardError(f"live service {service} launcher is outside the runtime")
    if len(argv) < 2 or not _same_executable_path(argv[1], launcher):
        raise GuardError(f"live service {service} process does not use its systemd launcher")

    configured_cwd = values.get("WorkingDirectory", "")
    if configured_cwd and not _same_path(configured_cwd, process_cwd):
        raise GuardError(f"live service {service} process cwd differs from its unit")
    fragment = Path(values.get("FragmentPath", ""))
    if not fragment.is_file():
        raise GuardError(f"live service {service} unit file does not exist")
    runtime_python_path = Path(str(runtime_python))
    if not runtime_python_path.is_file():
        raise GuardError(f"live service {service} runtime Python does not exist")
    return {
        "manager": "systemd-user",
        "name": service,
        "active_state": "active",
        "sub_state": "running",
        "main_pid": main_pid,
        "runtime_python": str(runtime_python),
        "runtime_python_sha256": _sha256_file(runtime_python_path),
        "launcher": launcher,
        "launcher_sha256": _sha256_file(Path(launcher)),
        "command_sha256": _sha256_bytes(raw_command),
        "working_directory": process_cwd,
        "fragment_path": str(fragment.resolve()),
        "fragment_sha256": _sha256_file(fragment),
        "pythonpath_shadowing": False,
    }


def _validated_attested_service_state(
    value: object,
    profile: dict[str, object],
    runtime_state: dict[str, object],
) -> dict[str, object] | None:
    if profile.get("service_required") is not True:
        if value is not None:
            raise GuardError("manifest contains service evidence forbidden by its profile")
        return None
    if not isinstance(value, dict):
        raise GuardError("deployment manifest service evidence is invalid")
    required_strings = (
        "manager",
        "name",
        "active_state",
        "sub_state",
        "runtime_python",
        "runtime_python_sha256",
        "launcher",
        "launcher_sha256",
        "command_sha256",
        "working_directory",
        "fragment_path",
        "fragment_sha256",
    )
    if not all(isinstance(value.get(key), str) for key in required_strings):
        raise GuardError("deployment manifest service evidence is invalid")
    if (
        value.get("manager") != "systemd-user"
        or value.get("active_state") != "active"
        or value.get("sub_state") != "running"
        or value.get("pythonpath_shadowing") is not False
        or not isinstance(value.get("main_pid"), int)
        or value["main_pid"] <= 0
    ):
        raise GuardError("deployment manifest service is not an active clean process")
    pinned_name = profile.get("service_name")
    if pinned_name is not None and value.get("name") != pinned_name:
        raise GuardError("deployment manifest service differs from component profile")
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+\.service", str(value.get("name"))):
        raise GuardError("deployment manifest service name is invalid")
    prefix = runtime_state.get("prefix")
    if (
        value.get("runtime_python") != runtime_state.get("python")
        or not _lexically_within(value.get("launcher"), prefix)
        or not _lexically_within(value.get("runtime_python"), prefix)
    ):
        raise GuardError("deployment manifest service is outside the attested runtime")
    for key in (
        "runtime_python_sha256",
        "launcher_sha256",
        "command_sha256",
        "fragment_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key))):
            raise GuardError("deployment manifest service hash evidence is invalid")
    return value


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
        source_root = _repo_member(
            repo, source_package_dir, f"source package {package_dir}"
        )
        source_files = _hash_tree(source_root, canonical_text=True)
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
            current = _validate_hash_tree(
                current, f"runtime package {package_dir}"
            )
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
    declared = tuple(
        (
            item["package_dir"],
            item["source_package_dir"],
            item["runtime_module"],
        )
        for item in result
    )
    if len({item[0] for item in declared}) != len(declared):
        raise GuardError("deployment manifest contains duplicate packages")
    for package, source_package, module in declared:
        _safe_relative_path(package, "manifest package directory")
        _safe_relative_path(source_package, "manifest source package directory")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            raise GuardError("deployment manifest runtime module is invalid")
    return result


def _verify_manifest_inputs(
    args: argparse.Namespace,
    payload: dict[str, object],
    repo: Path,
    wheel: Path,
    sdist: Path,
) -> tuple[list[dict[str, str]], dict[str, object], bool, bool]:
    head = _head(repo)
    if payload.get("source_commit") != head:
        raise GuardError(
            "publication blocked: deployed source commit does not equal release HEAD "
            f"({payload.get('source_commit')} != {head})"
        )
    specs = _manifest_specs(payload)
    profile = _component_profile(payload.get("component"))
    declared = tuple(
        (
            spec["package_dir"],
            spec["source_package_dir"],
            spec["runtime_module"],
        )
        for spec in specs
    )
    if declared != profile.get("packages"):
        raise GuardError("deployment manifest packages differ from component profile")
    for spec in specs:
        _repo_member(
            repo,
            spec["source_package_dir"],
            f"manifest source package {spec['package_dir']}",
        )
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
    if distribution != profile.get("distribution"):
        raise GuardError("deployment manifest distribution differs from component profile")
    if len(set(map(_canonical_distribution, [distribution, *dependencies]))) != (
        1 + len(dependencies)
    ):
        raise GuardError("deployment manifest contains duplicate distributions")
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

    attested_runtime = _validated_attested_runtime_state(
        payload.get("runtime"), specs, distribution, dependencies
    )
    versions = attested_runtime.get("versions")
    if not isinstance(versions, dict) or versions.get(distribution) != metadata["version"]:
        raise GuardError("attested runtime distribution version differs from wheel")
    attested_service = _validated_attested_service_state(
        payload.get("service"), profile, attested_runtime
    )

    live_runtime_rechecked = not args.allow_remote_attestation
    live_service_rechecked = False
    runtime_state = attested_runtime
    if live_runtime_rechecked:
        current_runtime = _runtime_state(
            specs,
            Path(str(attested_runtime["python"])),
            distribution,
            dependencies,
        )
        if current_runtime != attested_runtime:
            raise GuardError("publication blocked: current runtime differs from attested runtime")
        runtime_state = current_runtime
        if attested_service is not None:
            current_service = _systemd_user_service_state(
                str(attested_service["name"]), current_runtime
            )
            if current_service != attested_service:
                raise GuardError(
                    "publication blocked: current live service differs from attested service"
                )
            live_service_rechecked = True
    _validate_package_artifacts(repo, wheel, sdist, specs, runtime_state)
    return specs, runtime_state, live_runtime_rechecked, live_service_rechecked


def _assert_remote_release_refs(
    repo: Path,
    remote: str,
    branch: str,
    tag: str,
    expected_head: str,
    version: str,
    profile: dict[str, object],
) -> dict[str, str]:
    expected_branch = profile.get("remote_branch")
    expected_repository = profile.get("remote_repository")
    if not isinstance(expected_branch, str) or not isinstance(expected_repository, str):
        raise GuardError("component has no publishable remote profile")
    if branch != expected_branch:
        raise GuardError(
            f"release branch {branch!r} differs from pinned branch {expected_branch!r}"
        )
    if tag not in {version, f"v{version}"}:
        raise GuardError(f"release tag {tag!r} does not match wheel version {version!r}")
    remote_url = _git(repo, "remote", "get-url", remote).stdout.strip()
    canonical_remote = _canonical_remote_url(remote_url)
    if canonical_remote != expected_repository:
        raise GuardError(
            "release remote differs from pinned component repository "
            f"({canonical_remote!r} != {expected_repository!r})"
        )
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
        detail = detail.replace(remote_url, canonical_remote)
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
    return {"remote": canonical_remote, "branch": branch_ref, "tag": tag_ref}


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
            expected_names = {str(record["filename"]) for record in expected}
            published_names = {name for name in published if isinstance(name, str)}
            if not mismatches and published_names == expected_names:
                return expected
            if published_names != expected_names:
                last_error = (
                    "published artifact set is not exact "
                    f"({sorted(published_names)!r} != {sorted(expected_names)!r})"
                )
            else:
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
    profile = _component_profile(args.component)
    expected_distribution = str(profile.get("distribution"))
    if args.distribution != expected_distribution:
        raise GuardError(
            f"distribution {args.distribution!r} differs from component profile "
            f"{expected_distribution!r}"
        )
    expected_signer = str(profile.get("signer"))
    if args.signer != expected_signer:
        raise GuardError(
            f"signer {args.signer!r} differs from pinned signer {expected_signer!r}"
        )
    expected_fingerprint = profile.get("signer_fingerprint")
    if expected_fingerprint is not None:
        actual_fingerprint = _signing_key_fingerprint(args.signing_key_file)
        if actual_fingerprint != expected_fingerprint:
            raise GuardError(
                "private signing key does not match the component trust profile "
                f"({actual_fingerprint} != {expected_fingerprint})"
            )
    specs = _package_specs(args, repo, profile)
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
    service_name = _service_name(profile, getattr(args, "service_name", ""))
    service_state = (
        _systemd_user_service_state(service_name, runtime_state)
        if service_name is not None
        else None
    )
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
        "service": service_state,
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
    payload = _load_manifest(args.manifest.resolve())
    profile = _component_profile(payload.get("component"))
    expected_signer = str(profile.get("signer"))
    if args.signer and args.signer != expected_signer:
        raise GuardError(
            f"requested signer {args.signer!r} differs from pinned signer {expected_signer!r}"
        )
    _verify_manifest_signature(
        args.manifest.resolve(),
        args.signature_file.resolve(),
        args.allowed_signers_file.resolve(),
        expected_signer,
        profile.get("signer_fingerprint"),
    )
    if payload.get("signer") != expected_signer:
        raise GuardError("deployment manifest signer differs from component profile")
    _assert_fresh_manifest(payload, args.max_attestation_age_seconds)
    checked = _assert_clean_worktrees(repo)
    specs, _runtime_state_evidence, runtime_rechecked, service_rechecked = (
        _verify_manifest_inputs(args, payload, repo, wheel, sdist)
    )
    return {
        "status": "ok",
        "component": payload.get("component"),
        "head": _head(repo),
        "packages": [spec["package_dir"] for spec in specs],
        "artifacts": _artifact_records(
            [wheel, sdist, *(path.resolve() for path in args.artifact)]
        ),
        "clean_worktrees": checked,
        "signed_runtime_validated": True,
        "live_runtime_rechecked": runtime_rechecked,
        "live_service_rechecked": service_rechecked,
        "parity": "exact",
    }


def command_publish(args: argparse.Namespace) -> dict[str, object]:
    result = command_verify(args)
    profile = _component_profile(result.get("component"), publishing=True)
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
        profile,
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
    parser.add_argument(
        "--signer",
        default="",
        help="Optional assertion; the component profile pins the actual signer",
    )
    parser.add_argument(
        "--max-attestation-age-seconds",
        type=int,
        default=MAX_ATTESTATION_AGE_SECONDS,
    )
    parser.add_argument(
        "--allow-remote-attestation",
        action="store_true",
        help=(
            "Validate fresh signed runtime/service evidence when their paths are remote; "
            "signed wheel/runtime hashes are still compared"
        ),
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
    attest.add_argument(
        "--service-name",
        default="",
        help="Running systemd user service; required or pinned by Linux profiles",
    )
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
