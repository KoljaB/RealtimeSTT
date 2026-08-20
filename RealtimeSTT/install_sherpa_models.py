"""Install the pinned sherpa-onnx ASR model bundles.

The model archives are deliberately kept outside the Python package.  This
module provides the small amount of lifecycle management needed for a
repeatable installation: archives are cached under a caller-selected
persistent directory, interrupted downloads resume from a stable ``.part``
file, and extraction is committed with one atomic directory rename.

The command is intentionally stdlib-only so it can be used immediately after
installing RealtimeSTT::

    python -m RealtimeSTT.install_sherpa_models --root D:/Models/RealtimeSTT \
        --model all

The resulting directories are ``<root>/<manifest.model_id>``.  They can be
passed directly as ``model`` or ``engine_options["model_dir"]`` to the
sherpa-onnx engines.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import re
import shutil
import sys
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model_manifests import (
    ModelManifest,
    SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST,
    SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST,
)


class ModelInstallError(RuntimeError):
    """Raised when a model cannot be verified or installed safely."""


MODEL_MANIFESTS: Mapping[str, ModelManifest] = {
    "nemotron": SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST,
    "parakeet": SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST,
}
MODEL_SELECTIONS = ("nemotron", "parakeet", "all")

# These names are part of the on-disk contract.  Do not use a temporary
# directory for either cache: a model install should survive process exits.
ARCHIVE_CACHE_DIRNAME = ".archives"
PARTIAL_CACHE_DIRNAME = ".partials"
STAGING_DIRNAME = ".staging"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)
_Urlopen = Callable[..., object]


def manifests_for_selection(selection: str) -> Tuple[Tuple[str, ModelManifest], ...]:
    """Return canonical manifest pairs for ``nemotron``, ``parakeet`` or ``all``."""

    value = str(selection).strip().lower()
    if value == "all":
        return tuple((name, MODEL_MANIFESTS[name]) for name in ("nemotron", "parakeet"))
    if value in MODEL_MANIFESTS:
        return ((value, MODEL_MANIFESTS[value]),)
    raise ModelInstallError(
        "Unknown sherpa model selection %r; choose nemotron, parakeet, or all."
        % selection
    )


def _manifest_for_selection(selection: Union[str, ModelManifest]) -> ModelManifest:
    if isinstance(selection, ModelManifest):
        return selection
    value = str(selection).strip().lower()
    if value in MODEL_MANIFESTS:
        return MODEL_MANIFESTS[value]
    raise ModelInstallError(
        "Unknown sherpa model selection %r; choose nemotron or parakeet."
        % selection
    )


def _ensure_root(root: Union[str, os.PathLike]) -> Path:
    path = Path(root).expanduser()
    if path.exists() and not path.is_dir():
        raise ModelInstallError("Model root exists but is not a directory: %s" % path)
    if path.is_symlink():
        raise ModelInstallError("Refusing to use a symlink as the model root: %s" % path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelInstallError("Cannot create model root %s: %s" % (path, exc)) from exc
    return path.resolve()


def _model_destination(root: Path, manifest: ModelManifest) -> Path:
    # Manifests are shipped metadata, but keep this check so a future manifest
    # cannot turn the caller's root into an extraction target outside itself.
    name = Path(manifest.model_id)
    if name.name != manifest.model_id or name in (Path("."), Path("..")):
        raise ModelInstallError("Invalid model id in manifest: %r" % manifest.model_id)
    return root / name


def _validate_model_directory(manifest: ModelManifest, path: Path) -> Tuple[bool, str]:
    """Return whether an extracted directory satisfies the manifest contract."""

    if path.is_symlink():
        return False, "destination is a symlink"
    if not path.exists():
        return False, "destination does not exist"
    if not path.is_dir():
        return False, "destination is not a directory"

    missing = []
    for relative_name in manifest.expected_files:
        candidate = path / relative_name
        if candidate.is_symlink():
            return False, "expected model file is a symlink: %s" % relative_name
        if not candidate.is_file():
            missing.append(relative_name)
    if missing:
        return False, "missing expected model file(s): %s" % ", ".join(missing)

    # ``expected_files`` catches an incomplete layout cheaply, while the
    # immutable file manifest proves that a cached ONNX/token file is the
    # exact pinned file.  Optional records (currently Nemotron's README) are
    # intentionally not required for cache validity.
    file_metadata = getattr(manifest, "file_metadata", ())
    required_metadata = tuple(record for record in file_metadata if record.required)
    for record in required_metadata:
        candidate = path / record.filename
        if candidate.is_symlink():
            return False, "required model file is a symlink: %s" % record.filename
    if required_metadata:
        try:
            invalid = manifest.invalid_files(path, include_optional=False)
        except OSError as exc:
            return False, "cannot verify extracted model files: %s" % exc
        if invalid:
            return False, "file verification failed: %s" % "; ".join(invalid)
    return True, "verified extracted model files"


def _destination_state(manifest: ModelManifest, destination: Path) -> str:
    valid, reason = _validate_model_directory(manifest, destination)
    if valid:
        return "valid"
    if not destination.exists() and not destination.is_symlink():
        return "missing"
    raise ModelInstallError(
        "Refusing to overwrite an existing invalid model destination %s (%s). "
        "Repair or move it, then retry." % (destination, reason)
    )


def _safe_archive_path(directory: Path, filename: str) -> Path:
    name = Path(filename)
    if name.name != filename or name in (Path("."), Path("..")):
        raise ModelInstallError("Invalid archive filename in manifest: %r" % filename)
    return directory / name


def _archive_digest(path: Path) -> Tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ModelInstallError("Archive cache is not a regular file: %s" % path)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ModelInstallError("Cannot read archive %s: %s" % (path, exc)) from exc
    return size, digest.hexdigest()


def verify_archive(path: Union[str, os.PathLike], manifest: ModelManifest) -> bool:
    """Return whether ``path`` exactly matches the manifest size and SHA-256."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        return False
    try:
        size, digest = _archive_digest(candidate)
    except ModelInstallError:
        return False
    return size == manifest.archive_size_bytes and digest.lower() == manifest.archive_sha256.lower()


def _integrity_error(path: Path, manifest: ModelManifest) -> ModelInstallError:
    try:
        size, digest = _archive_digest(path)
    except ModelInstallError as exc:
        return exc
    return ModelInstallError(
        "Archive verification failed for %s: expected %d bytes and SHA-256 %s; "
        "got %d bytes and SHA-256 %s. The partial file was retained for inspection "
        "or a later resumed download."
        % (
            path,
            manifest.archive_size_bytes,
            manifest.archive_sha256,
            size,
            digest,
        )
    )


def _response_status(response: object) -> Optional[int]:
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _response_header(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get(name)
        if value is not None:
            return str(value)
    getter = getattr(response, "getheader", None)
    if getter is not None:
        value = getter(name)
        if value is not None:
            return str(value)
    return ""


def _range_start(response: object) -> Optional[int]:
    value = _response_header(response, "Content-Range").strip()
    match = _CONTENT_RANGE_RE.match(value)
    if not match:
        return None
    return int(match.group(1))


def _call_urlopen(opener: Optional[_Urlopen], request: Request, timeout: int):
    function = opener or urlopen
    try:
        return function(request, timeout=timeout)
    except TypeError:
        # Small test doubles and a few urllib-compatible wrappers only accept
        # the request object.  The real urllib function accepts timeout.
        return function(request)


def _open_download(
    manifest: ModelManifest,
    partial: Path,
    *,
    timeout: int,
    opener: Optional[_Urlopen],
):
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "RealtimeSTT/%s sherpa-model-installer" % "1.0.3"}
    if offset:
        headers["Range"] = "bytes=%d-" % offset
    request = Request(manifest.archive_url, headers=headers)
    try:
        return _call_urlopen(opener, request, timeout), offset
    except HTTPError as exc:
        if exc.code != http.client.REQUESTED_RANGE_NOT_SATISFIABLE or not offset:
            raise
        # A stale/oversized partial cannot be resumed.  Ask for a fresh full
        # response; the partial is retained until the response verifies.
        request = Request(
            manifest.archive_url,
            headers={"User-Agent": headers["User-Agent"]},
        )
        return _call_urlopen(opener, request, timeout), 0


def _download_archive(
    manifest: ModelManifest,
    root: Path,
    *,
    offline: bool,
    timeout: int,
    opener: Optional[_Urlopen],
) -> Path:
    archive_dir = root / ARCHIVE_CACHE_DIRNAME
    partial_dir = root / PARTIAL_CACHE_DIRNAME
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        partial_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelInstallError("Cannot create model archive cache under %s: %s" % (root, exc)) from exc

    archive = _safe_archive_path(archive_dir, manifest.archive_filename)
    partial = _safe_archive_path(partial_dir, manifest.archive_filename + ".part")

    if archive.exists() or archive.is_symlink():
        if archive.is_symlink():
            raise ModelInstallError("Refusing to use a symlink as the archive cache: %s" % archive)
        if verify_archive(archive, manifest):
            return archive
        if offline:
            raise _integrity_error(archive, manifest)

    if partial.exists() or partial.is_symlink():
        if partial.is_symlink():
            raise ModelInstallError("Refusing to use a symlink as the partial cache: %s" % partial)
        if verify_archive(partial, manifest):
            try:
                os.replace(str(partial), str(archive))
            except OSError as exc:
                raise ModelInstallError("Cannot promote verified archive cache %s: %s" % (partial, exc)) from exc
            return archive
        if offline:
            raise _integrity_error(partial, manifest)

    if offline:
        raise ModelInstallError(
            "Offline model install requires a verified archive or extracted cache for %s. "
            "No verified cache was found under %s." % (manifest.model_id, root)
        )

    try:
        response, requested_offset = _open_download(
            manifest,
            partial,
            timeout=timeout,
            opener=opener,
        )
    except (HTTPError, URLError, OSError, http.client.HTTPException, ValueError) as exc:
        raise ModelInstallError(
            "Could not download %s from %s: %s. Any partial download was retained at %s."
            % (manifest.model_id, manifest.archive_url, exc, partial)
        ) from exc

    status = _response_status(response)
    append = requested_offset > 0 and status == 206 and _range_start(response) == requested_offset
    # A server that ignores Range must not be appended to the partial: doing
    # so would create a byte-valid-looking but unverifiable concatenation.
    mode = "ab" if append else "wb"
    try:
        with response:
            with partial.open(mode) as output:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except (OSError, HTTPError, URLError, http.client.HTTPException, ValueError) as exc:
        raise ModelInstallError(
            "Download of %s was interrupted: %s. The partial download was retained at %s."
            % (manifest.model_id, exc, partial)
        ) from exc

    if not verify_archive(partial, manifest):
        raise _integrity_error(partial, manifest)
    try:
        os.replace(str(partial), str(archive))
    except OSError as exc:
        raise ModelInstallError("Cannot promote verified archive cache %s: %s" % (partial, exc)) from exc
    return archive


def _member_target(extraction_root: Path, member_name: str) -> Optional[Path]:
    """Return a safe target for a tar member, rejecting traversal/links."""

    if not member_name or "\x00" in member_name:
        raise ModelInstallError("Archive contains an invalid empty/NUL member name")
    # Backslashes are not path separators in a tar format, but accepting them
    # would make the same archive behave differently on Windows and POSIX.
    if "\\" in member_name:
        raise ModelInstallError("Archive member uses a backslash path: %r" % member_name)
    path = PurePosixPath(member_name)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ModelInstallError("Archive member escapes the extraction directory: %r" % member_name)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        return None
    target = extraction_root.joinpath(*parts)
    try:
        target.relative_to(extraction_root)
    except ValueError as exc:
        raise ModelInstallError("Archive member escapes the extraction directory: %r" % member_name) from exc
    return target


def _extract_archive_safely(archive: Path, extraction_root: Path) -> None:
    """Extract a tar archive without following or creating filesystem links."""

    seen = set()
    try:
        with tarfile.open(str(archive), mode="r:*") as bundle:
            for member in bundle.getmembers():
                target = _member_target(extraction_root, member.name)
                if target is None:
                    continue
                relative = str(target.relative_to(extraction_root))
                if relative in seen:
                    raise ModelInstallError("Archive contains duplicate member: %r" % member.name)
                seen.add(relative)

                if member.issym() or member.islnk() or member.isdev() or not (member.isdir() or member.isreg()):
                    raise ModelInstallError(
                        "Archive contains an unsupported link/device/special member: %r" % member.name
                    )

                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                # The staging tree is newly created, so exclusive creation
                # catches a file/dir collision rather than replacing content.
                try:
                    with target.open("xb") as output:
                        source = bundle.extractfile(member)
                        if source is None:
                            raise ModelInstallError("Cannot read archive member: %r" % member.name)
                        with source:
                            copied = 0
                            while True:
                                chunk = source.read(DOWNLOAD_CHUNK_SIZE)
                                if not chunk:
                                    break
                                copied += len(chunk)
                                output.write(chunk)
                            if copied != member.size:
                                raise ModelInstallError(
                                    "Archive member size changed while extracting: %r" % member.name
                                )
                except FileExistsError as exc:
                    raise ModelInstallError("Archive member collides with an existing path: %r" % member.name) from exc
    except (tarfile.TarError, OSError, EOFError) as exc:
        if isinstance(exc, ModelInstallError):
            raise
        raise ModelInstallError("Cannot safely extract archive %s: %s" % (archive, exc)) from exc


def _find_extracted_model(manifest: ModelManifest, extraction_root: Path) -> Path:
    candidates = []
    for directory, directory_names, _file_names in os.walk(str(extraction_root), followlinks=False):
        base = Path(directory)
        for name in directory_names:
            if (base / name).is_symlink():
                raise ModelInstallError("Extracted archive contains a directory symlink: %s" % (base / name))
        valid, _reason = _validate_model_directory(manifest, base)
        if valid:
            candidates.append(base)
    if not candidates:
        raise ModelInstallError(
            "Archive %s extracted, but no directory contains all expected model files: %s"
            % (archive_name_for_error(manifest), ", ".join(manifest.expected_files))
        )
    if len(candidates) > 1:
        raise ModelInstallError(
            "Archive contains multiple possible model directories; refusing ambiguous extraction: %s"
            % ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def archive_name_for_error(manifest: ModelManifest) -> str:
    """Return a stable archive label for errors and callers."""

    return manifest.archive_filename


def _install_manifest(
    manifest: ModelManifest,
    root: Path,
    *,
    offline: bool,
    timeout: int,
    opener: Optional[_Urlopen],
) -> Path:
    destination = _model_destination(root, manifest)
    state = _destination_state(manifest, destination)
    if state == "valid":
        return destination

    archive = _download_archive(
        manifest,
        root,
        offline=offline,
        timeout=timeout,
        opener=opener,
    )
    staging_root = root / STAGING_DIRNAME
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = staging_root / (manifest.model_id + "." + uuid.uuid4().hex)
    extraction_root = stage / "payload"
    stage_created = False
    try:
        stage.mkdir(parents=False, exist_ok=False)
        stage_created = True
        extraction_root.mkdir(parents=False, exist_ok=False)
        _extract_archive_safely(archive, extraction_root)
        candidate = _find_extracted_model(manifest, extraction_root)

        # Re-check immediately before the commit.  An existing incomplete
        # destination is never silently replaced, including when another
        # installer raced us while the archive was downloading.
        state = _destination_state(manifest, destination)
        if state == "valid":
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(str(candidate), str(destination))
        except FileExistsError as exc:
            raise ModelInstallError(
                "Model destination appeared during installation; refusing to overwrite: %s" % destination
            ) from exc
        except OSError as exc:
            raise ModelInstallError("Atomic model extraction commit failed for %s: %s" % (destination, exc)) from exc
        return destination
    finally:
        # Only this invocation's staging directory is ever removed.  Archive,
        # partial, and an existing destination are intentionally untouched.
        if stage_created and stage.exists():
            shutil.rmtree(str(stage), ignore_errors=False)


def install_model(
    selection: Union[str, ModelManifest],
    root: Union[str, os.PathLike],
    *,
    offline: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    urlopen_fn: Optional[_Urlopen] = None,
) -> Path:
    """Install one selected model and return its persistent extracted path."""

    if timeout <= 0:
        raise ModelInstallError("Download timeout must be greater than zero seconds.")
    manifest = _manifest_for_selection(selection)
    model_root = _ensure_root(root)
    return _install_manifest(
        manifest,
        model_root,
        offline=bool(offline),
        timeout=int(timeout),
        opener=urlopen_fn,
    )


def install_models(
    root: Union[str, os.PathLike],
    model: str = "all",
    *,
    offline: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    urlopen_fn: Optional[_Urlopen] = None,
) -> Dict[str, Path]:
    """Install one or both pinned models under ``root``."""

    if timeout <= 0:
        raise ModelInstallError("Download timeout must be greater than zero seconds.")
    model_root = _ensure_root(root)
    installed = {}
    for name, manifest in manifests_for_selection(model):
        installed[name] = _install_manifest(
            manifest,
            model_root,
            offline=bool(offline),
            timeout=int(timeout),
            opener=urlopen_fn,
        )
    return installed


def install_selected_models(
    root: Union[str, os.PathLike],
    selection: str = "all",
    **kwargs,
) -> Dict[str, Path]:
    """Explicit alias for callers that prefer ``selection`` terminology."""

    return install_models(root, model=selection, **kwargs)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by :func:`main`."""

    parser = argparse.ArgumentParser(
        prog="stt-install-sherpa-models",
        description=(
            "Download, verify, and atomically install the pinned RealtimeSTT "
            "sherpa-onnx Nemotron and Parakeet model bundles."
        ),
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Persistent model root; archives and extracted models are stored below it.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_SELECTIONS,
        default="all",
        help="Model to install (default: all).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only verified extracted models or archive caches; never access the network.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP connection/read timeout in seconds (default: %(default)s).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the model installer CLI and return a process exit status."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        installed = install_models(
            args.root,
            model=args.model,
            offline=args.offline,
            timeout=args.timeout,
        )
    except ModelInstallError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: model installation interrupted", file=sys.stderr)
        return 130

    for name, path in installed.items():
        print("%s: %s" % (name, path))
    return 0


__all__ = [
    "ARCHIVE_CACHE_DIRNAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "MODEL_MANIFESTS",
    "MODEL_SELECTIONS",
    "ModelInstallError",
    "PARTIAL_CACHE_DIRNAME",
    "STAGING_DIRNAME",
    "archive_name_for_error",
    "build_argument_parser",
    "install_model",
    "install_models",
    "install_selected_models",
    "main",
    "manifests_for_selection",
    "verify_archive",
]


if __name__ == "__main__":
    raise SystemExit(main())
