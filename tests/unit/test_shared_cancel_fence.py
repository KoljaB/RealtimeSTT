import threading
from types import SimpleNamespace
from example_fastapi_server.server import SharedEngineWorker


def test_cancel_cannot_cross_into_successor_native_job():
    worker = SharedEngineWorker.__new__(SharedEngineWorker)
    worker.name = "test"
    worker._active_job_lock = threading.Lock()
    worker._active_request_id = "old"
    worker._native_request_id = "old"
    worker._active_cancel_event = threading.Event()
    worker._cancel_requested_ids = {}
    entered, release, successor = (threading.Event() for _ in range(3))
    observed = []
    def cancel():
        entered.set()
        assert release.wait(2)
        observed.append(worker._active_request_id)
    worker.engine = SimpleNamespace(cancel_active=cancel)
    cancel_thread = threading.Thread(target=worker.cancel_request, args=("old",))
    def advance():
        worker._end_native_job("old")
        worker._release_job("old")
        worker._claim_job("next")
        successor.set()
    next_thread = threading.Thread(target=advance)
    cancel_thread.start()
    try:
        assert entered.wait(1)
        next_thread.start()
        assert not successor.wait(.03)
    finally:
        release.set()
        cancel_thread.join(1)
        next_thread.join(1)
    assert observed == ["old"]
    assert successor.is_set()
    assert worker._active_request_id == "next"
    assert worker.cancel_request("old") is False
    assert not worker._active_cancel_event.is_set()
