"""Immutable metadata for model bundles used by built-in engines.

Model weights are intentionally not downloaded by RealtimeSTT.  The manifest
objects make the user-supplied, persistent model-directory contract explicit
and give applications enough information to verify an archive before
extracting it.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ModelFileManifest:
    """Immutable size and SHA-256 metadata for one extracted model file."""

    filename: str
    size_bytes: int
    sha256: str
    required: bool = True


@dataclass(frozen=True)
class ModelManifest:
    """Describes one pinned model archive and its extracted files.

    The archive size and SHA-256 are the integrity boundary for a release:
    callers verify those values before extraction.  ``expected_files`` is the
    post-extraction runtime contract and catches an incomplete or incorrectly
    rooted extraction.  ``file_metadata`` records the exact extracted files
    from the pinned release; callers can opt into checking these when opening
    a user-supplied model directory.
    """

    model_id: str
    archive_url: str
    archive_filename: str
    archive_size_bytes: int
    archive_sha256: str
    expected_files: Tuple[str, ...]
    license_name: str
    license_url: str
    runtime_license_name: str = "Apache-2.0"
    license_id: str = ""
    file_metadata: Tuple[ModelFileManifest, ...] = ()

    @property
    def url(self):
        """Backward-friendly short alias for :attr:`archive_url`."""

        return self.archive_url

    @property
    def archive_size(self):
        """Short alias for the archive size in bytes."""

        return self.archive_size_bytes

    @property
    def sha256(self):
        """Short alias for the archive SHA-256 digest."""

        return self.archive_sha256

    @property
    def expected_file_metadata(self) -> Tuple[ModelFileManifest, ...]:
        """Alias for the immutable extracted-file records."""

        return self.file_metadata

    def invalid_files(
        self,
        model_dir: Path,
        *,
        include_optional: bool = False,
    ) -> Tuple[str, ...]:
        """Return extracted files that do not match pinned metadata.

        ``expected_files`` remains the cheap existence-only check used by
        default engine startup.  This method is intentionally explicit because
        hashing a large ONNX encoder can take noticeable time on Windows.
        """

        root = Path(model_dir).expanduser()
        invalid = []
        for record in self.file_metadata:
            if not record.required and not include_optional:
                continue
            path = root / record.filename
            if not path.is_file():
                invalid.append("%s (missing)" % record.filename)
                continue
            actual_size = path.stat().st_size
            if actual_size != record.size_bytes:
                invalid.append(
                    "%s (size %d, expected %d)"
                    % (record.filename, actual_size, record.size_bytes)
                )
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256.lower() != record.sha256.lower():
                invalid.append(
                    "%s (SHA-256 %s, expected %s)"
                    % (record.filename, actual_sha256, record.sha256)
                )
        return tuple(invalid)

    def describe_invalid_files(
        self,
        model_dir: Path,
        *,
        include_optional: bool = False,
    ) -> str:
        """Build a stable setup error for mismatched extracted files."""

        invalid = self.invalid_files(model_dir, include_optional=include_optional)
        if not invalid:
            return ""
        return (
            "Extracted model file verification failed for %s: %s. Verify "
            "archive size %d bytes and SHA-256 %s before extraction, then "
            "pass a persistent directory containing the exact files."
            % (
                self.model_id,
                "; ".join(invalid),
                self.archive_size_bytes,
                self.archive_sha256,
            )
        )

    def missing_files(self, model_dir: Path) -> Tuple[str, ...]:
        """Return expected relative files absent from ``model_dir``."""

        root = Path(model_dir).expanduser()
        return tuple(name for name in self.expected_files if not (root / name).is_file())

    def describe_missing_files(self, model_dir: Path) -> str:
        """Build a stable setup error for an incomplete extracted bundle."""

        missing = self.missing_files(model_dir)
        if not missing:
            return ""
        return (
            "Missing model file(s) for %s: %s. Download the pinned archive "
            "from %s, verify size %d bytes and SHA-256 %s, extract it into a "
            "persistent directory, and pass that directory as model or "
            "engine_options['model_dir']."
            % (
                self.model_id,
                ", ".join(missing),
                self.archive_url,
                self.archive_size_bytes,
                self.archive_sha256,
            )
        )


SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST = ModelManifest(
    model_id="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    archive_url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
    ),
    archive_filename="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
    archive_size_bytes=487170055,
    archive_sha256="5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
    expected_files=(
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ),
    license_name="CC-BY-4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    license_id="CC-BY-4.0",
    file_metadata=(
        ModelFileManifest(
            "encoder.int8.onnx",
            652184281,
            "acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247",
        ),
        ModelFileManifest(
            "decoder.int8.onnx",
            11845275,
            "179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e",
        ),
        ModelFileManifest(
            "joiner.int8.onnx",
            6355277,
            "3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3",
        ),
        ModelFileManifest(
            "tokens.txt",
            93939,
            "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
        ),
    ),
)


SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST = ModelManifest(
    model_id=(
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
    ),
    archive_url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2"
    ),
    archive_filename=(
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2"
    ),
    archive_size_bytes=475271763,
    archive_sha256="c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a",
    expected_files=(
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ),
    license_name="NVIDIA Open Model Data Warehouse License Agreement v1.1",
    license_url="https://www.nvidia.com/en-us/open-model-license/",
    license_id="OpenMDW-1.1",
    file_metadata=(
        ModelFileManifest(
            "encoder.int8.onnx",
            657601403,
            "012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68",
        ),
        ModelFileManifest(
            "decoder.int8.onnx",
            14978075,
            "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0",
        ),
        ModelFileManifest(
            "joiner.int8.onnx",
            9504438,
            "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce",
        ),
        ModelFileManifest(
            "tokens.txt",
            131440,
            "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a",
        ),
        ModelFileManifest(
            "README.md",
            214,
            "4cec75ccd38f289f3bd39055bd7033bfcbaa145d38b85b31e3943b8f03ae86f1",
            required=False,
        ),
    ),
)


# Short aliases are useful to callers that do not need the sherpa-specific
# prefix.  They all point at the same frozen objects.
PARAKEET_V3_INT8_MANIFEST = SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST
NEMOTRON_560MS_INT8_MANIFEST = SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST


__all__ = [
    "ModelFileManifest",
    "ModelManifest",
    "SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST",
    "SHERPA_ONNX_NEMOTRON_560MS_INT8_MANIFEST",
    "PARAKEET_V3_INT8_MANIFEST",
    "NEMOTRON_560MS_INT8_MANIFEST",
]
