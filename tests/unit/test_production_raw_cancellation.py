import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from RealtimeSTT_server import production_server as production


class RawTranscriptionCancellationTests(unittest.TestCase):
    def make_service(self, *, complete=False, accepted=True, cancel_error=None):
        service = SimpleNamespace(
            _pending_recorder_lock=threading.Lock(),
            _pending_recorder_results={},
            scheduler=mock.Mock(),
        )
        service._pop_pending_recorder_result = service._pending_recorder_results.pop
        service.result = SimpleNamespace(text="hello", error=None)

        def submit(job):
            if complete:
                holder = service._pending_recorder_results[job.request_id]
                holder["result"] = service.result
                holder["event"].set()
            return SimpleNamespace(accepted=accepted, reason="queue full")

        service.scheduler.submit.side_effect = submit
        service.scheduler.cancel_request.side_effect = cancel_error
        return service

    def test_timeout_cancels_exact_request_and_removes_waiter(self):
        service = self.make_service()
        with self.assertRaisesRegex(TimeoutError, "^final transcription timed out$"):
            production._service_raw_transcription(service, [0.0], "en", 0)
        job = service.scheduler.submit.call_args.args[0]
        service.scheduler.cancel_request.assert_called_once_with(job.request_id)
        self.assertEqual(service._pending_recorder_results, {})

    def test_cancellation_failure_preserves_timeout_and_cleans_waiter(self):
        service = self.make_service(cancel_error=RuntimeError("cancel failed"))
        with self.assertLogs(production.LOGGER, level="WARNING") as logs:
            with self.assertRaisesRegex(TimeoutError, "^final transcription timed out$"):
                production._service_raw_transcription(service, [0.0], "en", 0)
        self.assertIn("cancel failed", "\n".join(logs.output))
        self.assertEqual(service._pending_recorder_results, {})

    def test_success_returns_result_without_cancelling(self):
        service = self.make_service(complete=True)
        result = production._service_raw_transcription(service, [0.0], "en", 0)
        self.assertIs(result, service.result)
        service.scheduler.cancel_request.assert_not_called()
        self.assertEqual(service._pending_recorder_results, {})

    def test_rejection_cleans_waiter_without_cancelling(self):
        service = self.make_service(accepted=False)
        with self.assertRaisesRegex(RuntimeError, "queue full"):
            production._service_raw_transcription(service, [0.0], "en", 0)
        service.scheduler.cancel_request.assert_not_called()
        self.assertEqual(service._pending_recorder_results, {})

    def test_submission_error_cleans_waiter(self):
        service = self.make_service()
        service.scheduler.submit.side_effect = RuntimeError("submit failed")
        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            production._service_raw_transcription(service, [0.0], "en", 0)
        service.scheduler.cancel_request.assert_not_called()
        self.assertEqual(service._pending_recorder_results, {})


if __name__ == "__main__":
    unittest.main()
