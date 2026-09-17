"""Orukeet selection and the publisher-manifest integrity boundary."""

import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from RealtimeSTT import install_sherpa_models as installer
from RealtimeSTT.model_manifests import (
    ModelFileManifest,
    SHERPA_ONNX_ORUKEET_INT8_MANIFEST as ORUKEET,
    SHERPA_ONNX_PARAKEET_V3_INT8_MANIFEST as PARAKEET,
)
from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
)
from RealtimeSTT.transcription_engines.sherpa_onnx_engine import (
    SherpaOnnxParakeetBackend,
)
from tests.unit.test_install_sherpa_models import (
    _archive_bytes,
    _fixture_manifest,
    _with_integrity,
)
from tests.unit.test_sherpa_onnx_engine import FakeSherpaRecognizer, PARAKEET_FILES


class OrukeetModelTests(unittest.TestCase):
    def test_explicit_installer_selection_preserves_default_pair(self):
        self.assertEqual(
            installer.manifests_for_selection("orukeet"), (("orukeet", ORUKEET),)
        )
        self.assertEqual(
            [name for name, _ in installer.manifests_for_selection("all")],
            ["nemotron", "parakeet"],
        )

    def test_named_orukeet_uses_correct_verification_with_relocated_files(self):
        with tempfile.TemporaryDirectory() as root:
            for name in PARAKEET_FILES:
                (Path(root) / name).touch()
            config = TranscriptionEngineConfig(
                model="oruk/orukeet",
                engine_options={"model_dir": root, "verify_model_files": True},
            )
            with patch.object(
                type(ORUKEET), "describe_invalid_files", autospec=True, return_value=""
            ) as verify:
                backend = SherpaOnnxParakeetBackend(
                    config, recognizer_cls=FakeSherpaRecognizer
                )
            self.assertIs(backend.model_manifest, ORUKEET)
            self.assertIs(verify.call_args.args[0], ORUKEET)
            self.assertEqual(backend.recognizer.kwargs["feature_dim"], 128)
            self.assertEqual(backend.recognizer.kwargs["model_type"], "nemo_transducer")
            # Selection is per instance and must not change future Parakeet loads.
            default = SherpaOnnxParakeetBackend(
                TranscriptionEngineConfig(model=root),
                recognizer_cls=FakeSherpaRecognizer,
            )
            self.assertIs(default.model_manifest, PARAKEET)

    def test_directory_model_selects_orukeet_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / ORUKEET.model_id
            path.mkdir()
            for name in PARAKEET_FILES:
                (path / name).touch()
            backend = SherpaOnnxParakeetBackend(
                TranscriptionEngineConfig(model=str(path)),
                recognizer_cls=FakeSherpaRecognizer,
            )
            self.assertIs(backend.model_manifest, ORUKEET)

    def test_conflicting_known_model_and_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for manifest in (ORUKEET, PARAKEET):
                model_dir = Path(root) / manifest.model_id
                model_dir.mkdir()
                for name in PARAKEET_FILES:
                    (model_dir / name).touch()
            cases = (
                ("oruk/orukeet", PARAKEET),
                ("nvidia/parakeet-tdt-0.6b-v3", ORUKEET),
                (str(Path(root) / ORUKEET.model_id), PARAKEET),
                (str(Path(root) / PARAKEET.model_id), ORUKEET),
            )
            for model, directory_manifest in cases:
                with self.subTest(model=model):
                    config = TranscriptionEngineConfig(
                        model=model,
                        engine_options={
                            "model_dir": str(Path(root) / directory_manifest.model_id),
                        },
                    )
                    with patch.object(FakeSherpaRecognizer, "from_transducer") as load:
                        with self.assertRaisesRegex(
                            TranscriptionEngineError,
                            "Conflicting sherpa-onnx model identities",
                        ):
                            SherpaOnnxParakeetBackend(
                                config, recognizer_cls=FakeSherpaRecognizer
                            )
                        load.assert_not_called()

    def test_matching_alias_and_directory_preserve_features_and_overrides(self):
        cases = (
            ("oruk/orukeet", ORUKEET, 128),
            ("nvidia/parakeet-tdt-0.6b-v3", PARAKEET, 80),
        )
        with tempfile.TemporaryDirectory() as root:
            for model, manifest, default_features in cases:
                model_dir = Path(root) / manifest.model_id
                model_dir.mkdir()
                for name in PARAKEET_FILES:
                    (model_dir / name).touch()
                for override in (None, 64):
                    with self.subTest(model=model, feature_dim=override):
                        options = {"model_dir": str(model_dir)}
                        if override is not None:
                            options["feature_dim"] = override
                        backend = SherpaOnnxParakeetBackend(
                            TranscriptionEngineConfig(
                                model=model, engine_options=options
                            ),
                            recognizer_cls=FakeSherpaRecognizer,
                        )
                        self.assertIs(backend.model_manifest, manifest)
                        self.assertEqual(
                            backend.recognizer.kwargs["feature_dim"],
                            default_features if override is None else override,
                        )

    def test_manifest_hash_and_archive_identity_are_both_checked(self):
        payload = json.dumps(
            {
                "archive": ORUKEET.archive_filename,
                "archive_bytes": ORUKEET.archive_size_bytes,
                "archive_sha256": ORUKEET.archive_sha256,
            }
        ).encode()
        manifest = replace(
            ORUKEET,
            release_manifest=ModelFileManifest(
                "manifest.json", len(payload), hashlib.sha256(payload).hexdigest()
            ),
        )
        requested = []

        def opener(request, timeout):
            requested.append(request.full_url)
            return io.BytesIO(payload)

        installer._verify_release_manifest(manifest, timeout=5, opener=opener)
        self.assertTrue(requested[0].endswith("/onnx/manifest.json"))
        with self.assertRaisesRegex(installer.ModelInstallError, "verification failed"):
            installer._verify_release_manifest(
                manifest, timeout=5, opener=lambda *args, **kwargs: io.BytesIO(b"bad")
            )
        with self.assertRaisesRegex(installer.ModelInstallError, "disagrees"):
            installer._verify_release_manifest(
                replace(manifest, archive_size_bytes=1), timeout=5, opener=opener
            )


class OrukeetInstallerTests(unittest.TestCase):
    def make_release(self):
        manifest = _fixture_manifest("orukeet-install-fixture")
        archive = _archive_bytes(manifest)
        manifest = _with_integrity(manifest, archive)
        payload = json.dumps(
            {
                "archive": manifest.archive_filename,
                "archive_bytes": manifest.archive_size_bytes,
                "archive_sha256": manifest.archive_sha256,
            }
        ).encode()
        manifest = replace(
            manifest,
            release_manifest=ModelFileManifest(
                "manifest.json", len(payload), hashlib.sha256(payload).hexdigest()
            ),
        )
        return manifest, archive, payload

    def forbid_download(self, *args, **kwargs):
        self.fail("A verified cache must not request the release manifest or archive")

    def test_public_install_checks_manifest_and_reuses_extracted_cache(self):
        manifest, archive, payload = self.make_release()
        manifest_url = manifest.archive_url.rsplit("/", 1)[0] + "/manifest.json"
        requests = []

        def opener(request, timeout):
            requests.append(request.full_url)
            return io.BytesIO(payload if request.full_url == manifest_url else archive)

        with tempfile.TemporaryDirectory() as root:
            destination = installer.install_model(manifest, root, urlopen_fn=opener)
            self.assertEqual(requests, [manifest_url, manifest.archive_url])
            self.assertEqual(manifest.invalid_files(destination), ())
            for offline in (False, True):
                with self.subTest(offline=offline):
                    self.assertEqual(
                        installer.install_model(
                            manifest, root, offline=offline,
                            urlopen_fn=self.forbid_download,
                        ),
                        destination,
                    )

    def test_public_install_manifest_failure_stops_before_archive(self):
        manifest, _, _ = self.make_release()
        requests = []

        def opener(request, timeout):
            requests.append(request.full_url)
            return io.BytesIO(b"corrupt release manifest")

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(installer.ModelInstallError, "verification failed"):
                installer.install_model(manifest, root, urlopen_fn=opener)
            self.assertEqual(
                requests, [manifest.archive_url.rsplit("/", 1)[0] + "/manifest.json"]
            )
            self.assertFalse((Path(root) / manifest.model_id).exists())

    def test_verified_archive_cache_skips_manifest_and_network(self):
        manifest, archive, _ = self.make_release()
        for offline in (False, True):
            with self.subTest(offline=offline), tempfile.TemporaryDirectory() as root:
                cache = Path(root) / installer.ARCHIVE_CACHE_DIRNAME
                cache.mkdir()
                (cache / manifest.archive_filename).write_bytes(archive)
                destination = installer.install_model(
                    manifest, root, offline=offline, urlopen_fn=self.forbid_download
                )
                self.assertEqual(manifest.invalid_files(destination), ())


if __name__ == "__main__":
    unittest.main()
