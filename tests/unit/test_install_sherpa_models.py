import hashlib
import io
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from RealtimeSTT import install_sherpa_models as installer
from RealtimeSTT.model_manifests import ModelFileManifest, ModelManifest


class _Response:
    def __init__(self, payload, status=200, headers=None, fail_after=None):
        self._payload = payload
        self._position = 0
        self.status = status
        self.headers = headers or {}
        self._fail_after = fail_after
        self._failed = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return False

    def read(self, size=-1):
        if self._fail_after is not None and self._position >= self._fail_after and not self._failed:
            self._failed = True
            raise OSError("simulated connection drop")
        if size is None or size < 0:
            size = len(self._payload) - self._position
        end = min(len(self._payload), self._position + size)
        if self._fail_after is not None:
            end = min(end, self._fail_after)
        chunk = self._payload[self._position:end]
        self._position = end
        return chunk


def _fixture_manifest(model_id="fixture-model"):
    return ModelManifest(
        model_id=model_id,
        archive_url="https://example.invalid/%s.tar.bz2" % model_id,
        archive_filename="%s.tar.bz2" % model_id,
        archive_size_bytes=0,
        archive_sha256="",
        expected_files=("encoder.int8.onnx", "tokens.txt"),
        license_name="Test",
        license_url="https://example.invalid/license",
    )


def _archive_bytes(manifest, *, malicious=None, link=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as bundle:
        root = manifest.model_id + "/"
        directory = tarfile.TarInfo(root)
        directory.type = tarfile.DIRTYPE
        bundle.addfile(directory)
        for name, content in (
            ("encoder.int8.onnx", b"encoder"),
            ("tokens.txt", b"tokens"),
        ):
            member = tarfile.TarInfo(root + name)
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
        if malicious is not None:
            member = tarfile.TarInfo(malicious)
            member.size = 4
            bundle.addfile(member, io.BytesIO(b"evil"))
        if link:
            member = tarfile.TarInfo(root + "linked.txt")
            member.type = tarfile.LNKTYPE
            member.linkname = root + "tokens.txt"
            bundle.addfile(member)
    return output.getvalue()


def _with_integrity(manifest, archive):
    file_contents = {
        "encoder.int8.onnx": b"encoder",
        "tokens.txt": b"tokens",
    }
    return ModelManifest(
        model_id=manifest.model_id,
        archive_url=manifest.archive_url,
        archive_filename=manifest.archive_filename,
        archive_size_bytes=len(archive),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        expected_files=manifest.expected_files,
        license_name=manifest.license_name,
        license_url=manifest.license_url,
        runtime_license_name=manifest.runtime_license_name,
        license_id=manifest.license_id,
        file_metadata=tuple(
            ModelFileManifest(
                filename,
                len(file_contents[filename]),
                hashlib.sha256(file_contents[filename]).hexdigest(),
            )
            for filename in manifest.expected_files
        ),
    )


class InstallSherpaModelsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "models"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resume_uses_range_and_commits_atomically(self):
        original = _fixture_manifest()
        archive = _archive_bytes(original)
        manifest = _with_integrity(original, archive)
        split = len(archive) // 3
        calls = []

        def opener(request, timeout=None):
            range_header = request.get_header("Range")
            calls.append((range_header, timeout))
            if range_header is None:
                return _Response(archive[:split], fail_after=split)
            self.assertEqual("bytes=%d-" % split, range_header)
            return _Response(
                archive[split:],
                status=206,
                headers={"Content-Range": "bytes %d-%d/%d" % (split, len(archive) - 1, len(archive))},
            )

        with self.assertRaises(installer.ModelInstallError):
            installer.install_model(manifest, self.root, urlopen_fn=opener)
        partial = self.root / installer.PARTIAL_CACHE_DIRNAME / (manifest.archive_filename + ".part")
        self.assertEqual(split, partial.stat().st_size)
        self.assertFalse((self.root / manifest.model_id).exists())

        destination = installer.install_model(manifest, self.root, urlopen_fn=opener)
        self.assertTrue((destination / "tokens.txt").is_file())
        self.assertEqual(2, len(calls))
        self.assertEqual("bytes=%d-" % split, calls[1][0])
        self.assertFalse(partial.exists())
        self.assertTrue(
            installer.verify_archive(
                self.root / installer.ARCHIVE_CACHE_DIRNAME / manifest.archive_filename,
                manifest,
            )
        )
        self.assertEqual([], list((self.root / installer.STAGING_DIRNAME).iterdir()))

    def test_checksum_mismatch_retains_partial_and_never_creates_destination(self):
        original = _fixture_manifest()
        archive = _archive_bytes(original)
        manifest = _with_integrity(original, archive)

        def opener(_request, timeout=None):
            return _Response(b"wrong archive")

        with self.assertRaisesRegex(installer.ModelInstallError, "Archive verification failed"):
            installer.install_model(manifest, self.root, urlopen_fn=opener)
        partial = self.root / installer.PARTIAL_CACHE_DIRNAME / (manifest.archive_filename + ".part")
        self.assertTrue(partial.is_file())
        self.assertFalse((self.root / manifest.model_id).exists())
        self.assertFalse((self.root / installer.STAGING_DIRNAME).exists())

    def test_verified_archive_cache_is_reused_offline(self):
        original = _fixture_manifest()
        archive = _archive_bytes(original)
        manifest = _with_integrity(original, archive)
        calls = []

        def opener(_request, timeout=None):
            calls.append(True)
            return _Response(archive)

        first = installer.install_model(manifest, self.root, urlopen_fn=opener)
        shutil.rmtree(str(first))

        def offline_opener(_request, timeout=None):
            raise AssertionError("offline cache reuse attempted a network request")

        second = installer.install_model(
            manifest,
            self.root,
            offline=True,
            urlopen_fn=offline_opener,
        )
        self.assertEqual(first, second)
        self.assertEqual([True], calls)
        self.assertTrue((second / "encoder.int8.onnx").is_file())

    def test_existing_invalid_destination_is_not_overwritten(self):
        original = _fixture_manifest()
        archive = _archive_bytes(original)
        manifest = _with_integrity(original, archive)
        destination = self.root / manifest.model_id
        destination.mkdir(parents=True)
        sentinel = destination / "user-owned.txt"
        sentinel.write_text("keep", encoding="utf-8")
        (destination / manifest.expected_files[0]).write_bytes(b"incomplete")

        with self.assertRaisesRegex(installer.ModelInstallError, "invalid model destination"):
            installer.install_model(manifest, self.root, urlopen_fn=lambda *_args, **_kwargs: None)
        self.assertEqual("keep", sentinel.read_text(encoding="utf-8"))
        self.assertFalse((self.root / installer.ARCHIVE_CACHE_DIRNAME).exists())

    def test_corrupt_complete_destination_is_rejected_online_and_offline(self):
        original = _fixture_manifest()
        archive = _archive_bytes(original)
        manifest = _with_integrity(original, archive)

        destination = installer.install_model(
            manifest,
            self.root,
            urlopen_fn=lambda _request, timeout=None: _Response(archive),
        )
        corrupt_file = destination / "encoder.int8.onnx"
        corrupt_file.write_bytes(b"corrupt-but-present")

        def network_must_not_be_used(_request, timeout=None):
            raise AssertionError("corrupt destination must fail before downloading")

        for offline in (False, True):
            with self.assertRaisesRegex(installer.ModelInstallError, "invalid model destination"):
                installer.install_model(
                    manifest,
                    self.root,
                    offline=offline,
                    urlopen_fn=network_must_not_be_used,
                )
            self.assertEqual(b"corrupt-but-present", corrupt_file.read_bytes())

    def test_safe_extraction_rejects_traversal_and_links_and_cleans_staging(self):
        original = _fixture_manifest()
        traversal_archive = _archive_bytes(original, malicious="../outside.txt")
        traversal_manifest = _with_integrity(original, traversal_archive)
        with self.assertRaisesRegex(installer.ModelInstallError, "escapes"):
            installer.install_model(
                traversal_manifest,
                self.root,
                urlopen_fn=lambda _request, timeout=None: _Response(traversal_archive),
            )
        self.assertFalse((self.root / "outside.txt").exists())
        self.assertFalse((self.root / traversal_manifest.model_id).exists())
        staging = self.root / installer.STAGING_DIRNAME
        self.assertEqual([], list(staging.iterdir()))

        link_archive = _archive_bytes(original, link=True)
        link_manifest = _with_integrity(original, link_archive)
        with self.assertRaisesRegex(installer.ModelInstallError, "link/device/special"):
            installer.install_model(
                link_manifest,
                self.root,
                urlopen_fn=lambda _request, timeout=None: _Response(link_archive),
            )
        self.assertEqual([], list(staging.iterdir()))

    def test_malformed_verified_archive_has_atomic_no_destination_and_clean_stage(self):
        original = _fixture_manifest()
        archive = b"this has the right digest but is not a tar archive"
        manifest = _with_integrity(original, archive)
        with self.assertRaisesRegex(installer.ModelInstallError, "safely extract"):
            installer.install_model(
                manifest,
                self.root,
                urlopen_fn=lambda _request, timeout=None: _Response(archive),
            )
        self.assertFalse((self.root / manifest.model_id).exists())
        self.assertEqual([], list((self.root / installer.STAGING_DIRNAME).iterdir()))

    def test_cli_passes_model_selection_and_root(self):
        root = self.root / "cli"
        expected = {"nemotron": root / "nemotron"}
        with patch.object(installer, "install_models", return_value=expected) as install:
            status = installer.main(["--root", str(root), "--model", "nemotron", "--offline"])
        self.assertEqual(0, status)
        install.assert_called_once_with(
            root,
            model="nemotron",
            offline=True,
            timeout=installer.DEFAULT_TIMEOUT_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
