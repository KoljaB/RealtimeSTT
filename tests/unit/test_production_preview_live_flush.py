import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from RealtimeSTT_server import production_server as production

SR = production.SERVER_SAMPLE_RATE


class _Manager:
    def __init__(self):
        self._loop = object()
        self.events = []
        self.changed = threading.Condition()

    def set_audio_sequence(self, session_id, sequence):
        return None

    def publish_session(self, session_id, event, authoritative=False):
        with self.changed:
            self.events.append(dict(event))
            self.changed.notify_all()
        delivered = production.concurrent.futures.Future()
        delivered.set_result(True)
        return delivered

    def wait_for(self, count, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self.changed:
            while len(self.events) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"expected {count} events, got {self.events}")
                self.changed.wait(remaining)


class _Service:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe_turn(self, audio, language, use_prompt):
        self.calls += 1
        return SimpleNamespace(text=self.text)


class _FlushSession:
    def __init__(self, text, refined=None):
        self.text = text
        self.refined = refined

    def flush_snapshot(self, refine_chunks=None, on_first=None):
        first = {"text": self.text, "live_text": "live", "steps": 1, "flush_ms": 50.0}
        if not refine_chunks:
            return first
        on_first(first)
        return {**first, "refined_text": self.refined, "refine_steps": 1, "refined_ms": 95.0}


class LiveFlushPreviewTests(unittest.TestCase):
    def _protocol(self, *, flush_text, record_text="record text", enabled=True):
        manager = _Manager()
        service = _Service(record_text)
        protocol = production.ProductionSessionProtocol(
            service,
            manager,
            "session",
            production.ProductionServerSettings(preview_live_flush=enabled),
        )
        protocol.turn = production.TurnState(
            turn_id="turn",
            language="en",
            generation=1,
            pcm_buffer=bytearray(b"\x01\x00" * (SR * 2)),
        )
        protocol.turn.live_queue = production._SampleBoundedQueue(SR * 10)
        protocol.turn.live_stream = _FlushSession(flush_text)
        return protocol, service, manager

    @staticmethod
    def _request(protocol, source="live_flush", refine=False):
        payload = {"type": "preview", "turnId": "turn", "previewRequestId": "r1"}
        if source:
            payload["pauseTextSource"] = source
        if refine:
            payload["pauseTextRefine"] = True
        return asyncio.run(protocol.preview(payload))

    def test_refined_flush_follows_the_preview(self):
        protocol, _service, manager = self._protocol(flush_text="that sounds awe")
        protocol.turn.live_stream = _FlushSession("that sounds awe", refined="that sounds awesome")
        self._request(protocol, refine=True)
        item = protocol.turn.live_queue.get_nowait()
        self.assertTrue(item.refine)
        protocol._run_live_flush("turn", 1, protocol.turn.live_stream, item, SR * 2)
        manager.wait_for(3)
        preview, refined, record = manager.events
        self.assertEqual(preview["type"], "preview")
        self.assertEqual(preview["text"], "that sounds awe")
        self.assertTrue(preview["refinePending"])
        self.assertEqual(refined["type"], "previewRefined")
        self.assertEqual(refined["previewRequestId"], "r1")
        self.assertEqual(refined["text"], "that sounds awesome")
        self.assertTrue(refined["changed"])
        self.assertEqual(record["type"], "previewRecord")
        # The refined event does not consume or replace the Preview result.
        self.assertEqual(protocol.turn.preview_status, "flush")

    def test_unrequested_refine_sends_one_preview_without_pending_flag(self):
        protocol, _service, manager = self._protocol(flush_text="hello there")
        self._request(protocol)
        item = protocol.turn.live_queue.get_nowait()
        self.assertFalse(item.refine)
        protocol._run_live_flush("turn", 1, protocol.turn.live_stream, item, SR * 2)
        manager.wait_for(2)
        self.assertFalse(manager.events[0]["refinePending"])
        self.assertEqual([e["type"] for e in manager.events], ["preview", "previewRecord"])

    def test_flush_answers_preview_and_record_follows(self):
        protocol, service, manager = self._protocol(flush_text="hello there")
        self._request(protocol)
        item = protocol.turn.live_queue.get_nowait()
        self.assertIsInstance(item, production._LiveFlushRequest)
        # The record decode waits for the flush.
        time.sleep(0.05)
        self.assertEqual(manager.events, [])
        protocol._run_live_flush("turn", 1, protocol.turn.live_stream, item, SR * 2)
        manager.wait_for(2)
        first, second = manager.events
        self.assertEqual(first["type"], "preview")
        self.assertEqual(first["status"], "flush")
        self.assertEqual(first["text"], "hello there")
        self.assertEqual(first["inputSampleRange"], {"start": 0, "end": SR * 2})
        self.assertEqual(first["previewTiming"]["flushSteps"], 1)
        self.assertEqual(second["type"], "previewRecord")
        self.assertEqual(second["text"], "record text")
        self.assertEqual(protocol.turn.preview_status, "flush")

    def test_empty_flush_lets_the_record_decode_answer(self):
        protocol, _service, manager = self._protocol(flush_text="")
        self._request(protocol)
        item = protocol.turn.live_queue.get_nowait()
        protocol._run_live_flush("turn", 1, protocol.turn.live_stream, item, SR * 2)
        manager.wait_for(1)
        time.sleep(0.05)
        self.assertEqual(len(manager.events), 1)
        self.assertEqual(manager.events[0]["type"], "preview")
        self.assertEqual(manager.events[0]["status"], "full_buffer")
        self.assertEqual(manager.events[0]["text"], "record text")

    def test_requests_without_live_flush_keep_the_record_preview(self):
        protocol, _service, manager = self._protocol(flush_text="hello")
        self._request(protocol, source=None)
        self.assertEqual(protocol.turn.live_queue.unfinished_tasks, 0)
        manager.wait_for(1)
        self.assertEqual(manager.events[0]["status"], "full_buffer")

    def test_disabled_server_ignores_the_request_field(self):
        protocol, _service, manager = self._protocol(flush_text="hello", enabled=False)
        self._request(protocol)
        self.assertEqual(protocol.turn.live_queue.unfinished_tasks, 0)
        manager.wait_for(1)
        self.assertEqual(manager.events[0]["status"], "full_buffer")

    def test_flush_marker_has_no_queue_weight(self):
        queue = production._SampleBoundedQueue(1)
        marker = production._LiveFlushRequest(
            request_id="r", preview_mode="early_rms", preview_epoch=1,
            cancelled=threading.Event(), requested_at=0.0, audio_revision=1,
            audio_frames=1, audio_packets=1, input_scope="full_buffer",
            input_sample_start=0, input_sample_end=1, candidate_start_sample=0,
            candidate_start_byte=0, candidate_id=None, candidate_base_text="",
            resume_request_id=None, resume_epoch=0,
        )
        self.assertTrue(queue.can_put(marker))


if __name__ == "__main__":
    unittest.main()
