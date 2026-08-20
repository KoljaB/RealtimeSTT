import queue
import threading
import unittest
from types import SimpleNamespace

from RealtimeSTT.core.manual_audio_input import flush_audio_input
from RealtimeSTT.core.recording import drain_audio_input


class ManualAudioDrainTests(unittest.TestCase):
    def test_partial_manual_input_is_queued_before_drain(self):
        recorder = SimpleNamespace(
            buffer=bytearray(b"partial-pcm"),
            audio_queue=queue.Queue(),
        )

        self.assertTrue(flush_audio_input(recorder))
        self.assertEqual(recorder.buffer, bytearray())
        self.assertEqual(recorder.audio_queue.get_nowait(), b"partial-pcm")
        self.assertFalse(flush_audio_input(recorder))

    def test_drain_marker_acknowledges_all_earlier_audio(self):
        recorder = SimpleNamespace(audio_queue=queue.Queue())
        recorder.audio_queue.put(b"first")
        recorder.audio_queue.put(b"second")
        consumed = []

        def consume_until_marker():
            while True:
                item = recorder.audio_queue.get(timeout=1.0)
                marker_event = getattr(item, "complete", None)
                if marker_event is not None:
                    marker_event.set()
                    return
                consumed.append(item)

        worker = threading.Thread(target=consume_until_marker)
        worker.start()
        try:
            self.assertTrue(drain_audio_input(recorder, timeout=1.0))
        finally:
            worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(consumed, [b"first", b"second"])

    def test_drain_timeout_is_reported(self):
        recorder = SimpleNamespace(audio_queue=queue.Queue())

        self.assertFalse(drain_audio_input(recorder, timeout=0.0))


if __name__ == "__main__":
    unittest.main()
