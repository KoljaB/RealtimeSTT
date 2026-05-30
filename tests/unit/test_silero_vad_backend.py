import importlib.util
import inspect
import unittest
from unittest import mock

import numpy as np

from RealtimeSTT import silero_vad

try:
    from RealtimeSTT.audio_recorder import (
        AudioToTextRecorder,
        DEACTIVITY_SILENCE_CONFIRMATION_DURATION,
    )
except Exception as exc:  # pragma: no cover - optional runtime deps may be absent
    AudioToTextRecorder = None
    DEACTIVITY_SILENCE_CONFIRMATION_DURATION = None
    AUDIO_RECORDER_IMPORT_ERROR = exc
else:
    AUDIO_RECORDER_IMPORT_ERROR = None


class SileroVadBackendSelectionTests(unittest.TestCase):
    def test_default_omitted_legacy_flag_resolves_to_auto(self):
        self.assertEqual(
            silero_vad.resolve_silero_backend("auto", silero_use_onnx=None),
            silero_vad.SILERO_BACKEND_AUTO,
        )

    def test_legacy_bool_flag_preserves_old_paths(self):
        self.assertEqual(
            silero_vad.resolve_silero_backend("auto", silero_use_onnx=True),
            silero_vad.SILERO_BACKEND_LEGACY,
        )
        self.assertEqual(
            silero_vad.resolve_silero_backend("auto", silero_use_onnx=False),
            silero_vad.SILERO_BACKEND_LEGACY,
        )

    def test_explicit_backend_wins_over_legacy_flag(self):
        self.assertEqual(
            silero_vad.resolve_silero_backend(
                "pytorch_cuda",
                silero_use_onnx=True,
            ),
            silero_vad.SILERO_BACKEND_PYTORCH_CUDA,
        )

    def test_auto_tries_ifless_raw_onnx_first(self):
        marker = object()
        calls = []

        def fake_raw(model_name, backend_name, *_args):
            calls.append((model_name, backend_name))
            return marker

        with mock.patch(
            "RealtimeSTT.silero_vad._create_raw_onnx_vad",
            side_effect=fake_raw,
        ):
            model = silero_vad.create_silero_vad_model()

        self.assertIs(model, marker)
        self.assertEqual(
            calls,
            [
                (
                    silero_vad.SILERO_OP18_IFLESS_MODEL,
                    silero_vad.SILERO_BACKEND_RAW_ONNX_IFLESS,
                )
            ],
        )

    def test_auto_falls_back_to_regular_raw_onnx(self):
        marker = object()
        calls = []

        def fake_raw(model_name, backend_name, *_args):
            calls.append((model_name, backend_name))
            if backend_name == silero_vad.SILERO_BACKEND_RAW_ONNX_IFLESS:
                raise silero_vad.SileroVadError("ifless missing")
            return marker

        with mock.patch(
            "RealtimeSTT.silero_vad._create_raw_onnx_vad",
            side_effect=fake_raw,
        ):
            model = silero_vad.create_silero_vad_model()

        self.assertIs(model, marker)
        self.assertEqual(
            calls,
            [
                (
                    silero_vad.SILERO_OP18_IFLESS_MODEL,
                    silero_vad.SILERO_BACKEND_RAW_ONNX_IFLESS,
                ),
                (
                    silero_vad.SILERO_ONNX_MODEL,
                    silero_vad.SILERO_BACKEND_RAW_ONNX,
                ),
            ],
        )

    def test_auto_falls_back_to_pytorch_cpu(self):
        marker = object()

        with mock.patch(
            "RealtimeSTT.silero_vad._create_raw_onnx_vad",
            side_effect=silero_vad.SileroVadError("onnx missing"),
        ), mock.patch(
            "RealtimeSTT.silero_vad._create_pytorch_vad",
            return_value=marker,
        ) as pytorch_vad:
            model = silero_vad.create_silero_vad_model()

        self.assertIs(model, marker)
        pytorch_vad.assert_called_once_with("cpu")

    def test_explicit_legacy_false_uses_old_torch_hub_pytorch_path(self):
        marker = object()
        with mock.patch(
            "RealtimeSTT.silero_vad._create_legacy_vad",
            return_value=marker,
        ) as legacy_vad:
            model = silero_vad.create_silero_vad_model(silero_use_onnx=False)

        self.assertIs(model, marker)
        legacy_vad.assert_called_once_with(False)

    def test_explicit_legacy_true_uses_old_torch_hub_onnx_path(self):
        marker = object()
        with mock.patch(
            "RealtimeSTT.silero_vad._create_legacy_vad",
            return_value=marker,
        ) as legacy_vad:
            model = silero_vad.create_silero_vad_model(silero_use_onnx=True)

        self.assertIs(model, marker)
        legacy_vad.assert_called_once_with(True)


class SileroVadPublicApiTests(unittest.TestCase):
    def setUp(self):
        if AUDIO_RECORDER_IMPORT_ERROR is not None:
            self.skipTest(
                "AudioToTextRecorder import failed: %s"
                % AUDIO_RECORDER_IMPORT_ERROR
            )

    def test_constructor_keeps_legacy_and_new_silero_options(self):
        signature = inspect.signature(AudioToTextRecorder.__init__)
        self.assertIn("silero_use_onnx", signature.parameters)
        self.assertIn("silero_backend", signature.parameters)
        self.assertIn("silero_onnx_model_path", signature.parameters)
        self.assertIn("silero_onnx_threads", signature.parameters)
        self.assertIsNone(signature.parameters["silero_use_onnx"].default)
        self.assertEqual(signature.parameters["silero_backend"].default, "auto")
        names = list(signature.parameters)
        self.assertEqual(
            names.index("silero_deactivity_detection"),
            names.index("silero_use_onnx") + 1,
        )
        self.assertLess(
            names.index("silero_deactivity_detection"),
            names.index("silero_backend"),
        )

    def test_constructor_exposes_deactivity_silence_confirmation_duration(self):
        signature = inspect.signature(AudioToTextRecorder.__init__)
        parameter = signature.parameters[
            "deactivity_silence_confirmation_duration"
        ]

        self.assertEqual(
            parameter.default,
            DEACTIVITY_SILENCE_CONFIRMATION_DURATION,
        )


class SileroVadBackendIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("silero_vad")
        and importlib.util.find_spec("onnxruntime")
        and importlib.util.find_spec("torch"),
        "silero-vad, onnxruntime, and torch are required",
    )
    def test_raw_ifless_onnx_probabilities_match_pytorch_cpu(self):
        raw_model_path = silero_vad.find_silero_model_file(
            silero_vad.SILERO_OP18_IFLESS_MODEL
        )
        if not raw_model_path:
            self.skipTest("silero_vad_op18_ifless.onnx is not installed")

        raw = silero_vad.create_silero_vad_model(
            backend="raw_onnx_ifless",
            onnx_model_path=raw_model_path,
        )
        pytorch = silero_vad.create_silero_vad_model(backend="pytorch_cpu")

        rng = np.random.default_rng(1234)
        frames = [
            np.zeros(512, dtype=np.float32),
            (0.03 * np.sin(2 * np.pi * 220.0 * np.arange(512) / 16000.0)).astype(np.float32),
            rng.normal(0.0, 0.01, 512).astype(np.float32),
        ]

        raw.reset_states()
        pytorch.reset_states()
        for frame in frames:
            raw_prob = raw(frame, 16000)
            torch_prob = pytorch(frame, 16000)
            self.assertLessEqual(abs(raw_prob - torch_prob), 1e-4)
            self.assertEqual(raw_prob > 0.5, torch_prob > 0.5)


if __name__ == "__main__":
    unittest.main()
