import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_RECORDER_DIR = PROJECT_ROOT / "RealtimeSTT" / "_audio_recorder"


class AudioRecorderBoundaryTests(unittest.TestCase):
    def test_public_recorder_imports_stay_compatible(self):
        from RealtimeSTT import AudioToTextRecorder as package_recorder
        from RealtimeSTT.audio_recorder import (
            AudioToTextRecorder as module_recorder,
            DEACTIVITY_SILENCE_CONFIRMATION_DURATION,
        )

        self.assertIs(package_recorder, module_recorder)
        self.assertEqual(DEACTIVITY_SILENCE_CONFIRMATION_DURATION, 0.16)

    def test_internal_recorder_modules_do_not_import_public_facade(self):
        offenders = []

        for path in sorted(INTERNAL_RECORDER_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "RealtimeSTT.audio_recorder":
                            offenders.append(path.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "RealtimeSTT.audio_recorder":
                        offenders.append(path.name)
                    elif node.module == "RealtimeSTT" and any(
                        alias.name == "audio_recorder" for alias in node.names
                    ):
                        offenders.append(path.name)
                    elif node.level > 0 and node.module == "audio_recorder":
                        offenders.append(path.name)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
