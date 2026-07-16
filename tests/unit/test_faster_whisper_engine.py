import unittest
from unittest.mock import patch

from RealtimeSTT.transcription_engines.base import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
)
from RealtimeSTT.transcription_engines.faster_whisper_engine import FasterWhisperEngine


class FasterWhisperEngineDependencyTests(unittest.TestCase):
    def test_missing_dependency_mentions_extra(self):
        config = TranscriptionEngineConfig(model="tiny")
        with patch(
            "RealtimeSTT.transcription_engines.faster_whisper_engine.import_module",
            side_effect=ModuleNotFoundError("No module named 'faster_whisper'"),
        ):
            with self.assertRaisesRegex(
                TranscriptionEngineError,
                r"RealtimeSTT\[faster-whisper\]",
            ):
                FasterWhisperEngine(config)


if __name__ == "__main__":
    unittest.main()
