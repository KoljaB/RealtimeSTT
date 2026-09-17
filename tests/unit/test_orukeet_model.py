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
from RealtimeSTT.transcription_engines import TranscriptionEngineConfig
from RealtimeSTT.transcription_engines.sherpa_onnx_engine import (
    SherpaOnnxParakeetBackend,
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


if __name__ == "__main__":
    unittest.main()
