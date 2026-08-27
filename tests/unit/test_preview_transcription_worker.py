import unittest
from types import SimpleNamespace

from RealtimeSTT.core.preview_transcription import PreviewTranscriptionWorker


class PreviewTranscriptionWorkerTests(unittest.TestCase):
    def test_submit_rejects_instead_of_growing_queue_without_bound(self):
        recorder = SimpleNamespace(_preview_uses_shared_final_worker=False)
        worker = PreviewTranscriptionWorker(recorder)
        self.addCleanup(worker.stop)

        self.assertTrue(
            worker.submit(b"first", "first live text", recording_id=1)
        )
        self.assertTrue(
            worker.submit(b"second", "second live text", recording_id=2)
        )
        with self.assertLogs("realtimestt", level="WARNING"):
            accepted = worker.submit(
                b"third",
                "third live text",
                recording_id=3,
            )

        self.assertFalse(accepted)
        self.assertEqual(worker.queue.maxsize, 2)
        self.assertEqual(worker.queue.qsize(), 2)
