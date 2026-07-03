import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from RealtimeSTT import (
    BaseEngine as PackageBaseEngine,
    BaseTranscriptionEngine as PackageBaseTranscriptionEngine,
    StreamingTranscriptionSession as PackageStreamingTranscriptionSession,
    TranscriptionEngineConfig as PackageTranscriptionEngineConfig,
    TranscriptionEngineError as PackageTranscriptionEngineError,
    TranscriptionInfo as PackageTranscriptionInfo,
    TranscriptionResult as PackageTranscriptionResult,
)
from RealtimeSTT.engines import (
    BaseEngine,
    BaseTranscriptionEngine,
    StreamingTranscriptionSession,
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)
from RealtimeSTT.engines.base_engine import BaseEngine as ModuleBaseEngine
from RealtimeSTT.transcription_engines import BaseEngine as TranscriptionPackageBase

try:
    from RealtimeSTT.core.transcription import call_transcription_executor
except ModuleNotFoundError as exc:
    if exc.name == "soundfile":
        call_transcription_executor = None
    else:
        raise


class EchoEngine(BaseEngine):
    engine_name = "echo"

    def __init__(self, config):
        super().__init__(config)
        self.calls = []

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs):
        audio = self._normalize_audio(audio)
        prompt = self._get_prompt(use_prompt)
        self.calls.append(
            {
                "language": language,
                "prompt": prompt,
                "kwargs": dict(kwargs),
                "peak": float(abs(audio).max()),
            }
        )
        return TranscriptionResult(
            text="echo",
            info=TranscriptionInfo(
                language=language,
                language_probability=1.0,
            ),
            metadata={"prompt": prompt},
        )


class CountingStreamingSession(StreamingTranscriptionSession):
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = 0
        self.decode_count = 0

    def accept_audio(self, audio, sample_rate=None):
        self.samples += int(getattr(audio, "size", 0))

    def decode(self):
        self.decode_count += 1

    def get_result(self):
        return TranscriptionResult(
            text="samples=%d decode=%d" % (self.samples, self.decode_count)
        )


class CustomTranscriptionEngineApiTests(unittest.TestCase):
    def test_public_base_engine_import_paths_share_the_same_class(self):
        self.assertIs(BaseEngine, BaseTranscriptionEngine)
        self.assertIs(ModuleBaseEngine, BaseEngine)
        self.assertIs(TranscriptionPackageBase, BaseEngine)
        self.assertIs(PackageBaseEngine, BaseEngine)
        self.assertIs(PackageBaseTranscriptionEngine, BaseTranscriptionEngine)
        self.assertIs(
            PackageStreamingTranscriptionSession,
            StreamingTranscriptionSession,
        )
        self.assertIs(PackageTranscriptionEngineConfig, TranscriptionEngineConfig)
        self.assertIs(PackageTranscriptionEngineError, TranscriptionEngineError)
        self.assertIs(PackageTranscriptionInfo, TranscriptionInfo)
        self.assertIs(PackageTranscriptionResult, TranscriptionResult)

    @unittest.skipIf(np is None, "NumPy is required for custom engine tests")
    def test_custom_engine_uses_shared_config_helpers(self):
        engine = EchoEngine(
            TranscriptionEngineConfig(
                model="custom",
                initial_prompt="context prompt",
                normalize_audio=True,
            )
        )

        result = engine.transcribe(
            np.array([0.0, 2.0], dtype=np.float32),
            language="en",
        )

        self.assertEqual(result.text, "echo")
        self.assertEqual(result.info.language, "en")
        self.assertEqual(result.metadata["prompt"], "context prompt")
        self.assertAlmostEqual(engine.calls[-1]["peak"], 0.95, places=6)

    @unittest.skipIf(np is None, "NumPy is required for custom engine tests")
    @unittest.skipIf(
        call_transcription_executor is None,
        "soundfile is required to import the transcription executor helper",
    )
    def test_custom_engine_works_as_transcription_executor(self):
        engine = EchoEngine(TranscriptionEngineConfig(model="custom"))

        result = call_transcription_executor(
            engine,
            np.ones(8, dtype=np.float32),
            language="de",
            use_prompt=False,
            word_timestamps=True,
        )

        self.assertEqual(result.text, "echo")
        self.assertEqual(result.info.language, "de")
        self.assertEqual(engine.calls[-1]["kwargs"], {"word_timestamps": True})

    def test_non_streaming_base_engine_rejects_streaming_session_creation(self):
        engine = EchoEngine(TranscriptionEngineConfig(model="custom"))

        with self.assertRaisesRegex(TranscriptionEngineError, "does not support"):
            engine.create_streaming_session()

    @unittest.skipIf(np is None, "NumPy is required for custom engine tests")
    def test_streaming_session_finish_decodes_and_returns_result(self):
        session = CountingStreamingSession()
        session.accept_audio(np.ones(4, dtype=np.float32), sample_rate=16000)

        result = session.finish()

        self.assertEqual(result.text, "samples=4 decode=1")


if __name__ == "__main__":
    unittest.main()
