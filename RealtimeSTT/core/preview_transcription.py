"""Low-latency speculative transcription from the active speech tail."""

from dataclasses import dataclass
import copy
import logging
import queue
import threading
from typing import Optional

from .state import run_callback
from .tail_transcription import (
    MIN_LIVE_WORDS_FOR_FUZZY_REPAIR,
    merge_live_and_tail_transcription,
)
from .text_formatting import preprocess_output
from .transcription import call_transcription_executor


logger = logging.getLogger("realtimestt")
_PREVIEW_QUEUE_MAX_SIZE = 2


@dataclass(frozen=True)
class PreviewTranscriptionResult:
    """Result of one speculative tail transcription."""

    text: str
    live_text: str
    tail_text: str
    status: str
    recording_id: int = 0
    matched: bool = False
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    distance: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class _PreviewRequest:
    """One immutable request consumed by the Preview worker."""

    tail_audio: object
    live_text: str
    recording_id: int
    use_prompt: bool
    dispatch_event: object


def _format_preview_text(recorder, text):
    """Applies the same text formatting rules as Final ASR."""

    return preprocess_output(
        str(text or "").strip(),
        ensure_sentence_starting_uppercase=(
            recorder.ensure_sentence_starting_uppercase
        ),
        ensure_sentence_ends_with_period=(
            recorder.ensure_sentence_ends_with_period
        ),
    )


def _run_preview_model(
    recorder,
    tail_audio,
    language,
    use_prompt,
    dispatch_event=None,
):
    """Runs the configured independent Preview model or executor."""

    if getattr(recorder, "_uses_external_preview_transcription_executor", False):
        if (
            dispatch_event is not None
            and hasattr(recorder.preview_transcription_executor, "transcribe_preview")
        ):
            return recorder.preview_transcription_executor.transcribe_preview(
                tail_audio,
                language=language if language else None,
                use_prompt=use_prompt,
                dispatch_event=dispatch_event,
            )
        return call_transcription_executor(
            recorder.preview_transcription_executor,
            tail_audio,
            language,
            use_prompt,
        )

    model = getattr(recorder, "preview_transcription_model", None)
    if model is None:
        raise RuntimeError("Preview transcription model is not initialized")

    return model.transcribe(
        tail_audio,
        language=language if language else None,
        use_prompt=use_prompt,
    )


def transcribe_preview(
    recorder,
    tail_audio,
    live_text,
    recording_id=0,
    use_prompt=True,
    dispatch_event=None,
):
    """Transcribes and safely merges one bounded tail without Final fallback."""

    live_text = str(live_text or "").strip()
    result = _run_preview_model(
        recorder,
        tail_audio,
        getattr(recorder, "language", ""),
        use_prompt,
        dispatch_event=dispatch_event,
    )
    tail_text = str(getattr(result, "text", "") or "").strip()

    if live_text:
        merge_result = merge_live_and_tail_transcription(
            live_text,
            tail_text,
            min_live_words_for_fuzzy_repair=getattr(
                recorder,
                "preview_transcription_min_live_words_for_fuzzy_repair",
                MIN_LIVE_WORDS_FOR_FUZZY_REPAIR,
            ),
        )
        if merge_result.matched:
            status = "fuzzy" if merge_result.used_fuzzy_match else "exact"
            preview_text = merge_result.text
        else:
            # Preview must never block on or invoke the full Final model. The
            # safe output on an untrusted tail is the Live boundary snapshot.
            status = "alignment_failed"
            preview_text = live_text

        return PreviewTranscriptionResult(
            text=_format_preview_text(recorder, preview_text),
            live_text=live_text,
            tail_text=tail_text,
            status=status,
            recording_id=recording_id,
            matched=merge_result.matched,
            used_fuzzy_match=merge_result.used_fuzzy_match,
            anchor_length=merge_result.anchor_length,
            distance=merge_result.distance,
        )

    status = "tail_only" if tail_text else "empty"
    return PreviewTranscriptionResult(
        text=_format_preview_text(recorder, tail_text),
        live_text="",
        tail_text=tail_text,
        status=status,
        recording_id=recording_id,
    )


def _preview_error_result(recorder, request, error):
    """Converts a Preview exception into a non-blocking result."""

    live_text = request.live_text
    return PreviewTranscriptionResult(
        text=_format_preview_text(recorder, live_text),
        live_text=live_text,
        tail_text="",
        status="error",
        recording_id=request.recording_id,
        error=str(error),
    )


class PreviewTranscriptionWorker:
    """Serializes independent Preview requests without blocking VAD or Final."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.queue = queue.Queue(maxsize=_PREVIEW_QUEUE_MAX_SIZE)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="RealtimeSTTPreviewTranscription",
            daemon=True,
        )

    def start(self):
        """Starts the worker once."""

        if not self.thread.is_alive():
            self.thread.start()

    def submit(self, tail_audio, live_text, recording_id=0, use_prompt=True):
        """Queues one copied tail and the Live snapshot captured at VAD."""

        if self.stop_event.is_set():
            return False
        if self.queue.full():
            logger.warning(
                "Preview transcription queue is full; dropping speculative work"
            )
            return False
        dispatch_event = threading.Event()
        try:
            self.queue.put_nowait(
                _PreviewRequest(
                    tail_audio=copy.deepcopy(tail_audio),
                    live_text=str(live_text or "").strip(),
                    recording_id=recording_id,
                    use_prompt=use_prompt,
                    dispatch_event=dispatch_event,
                )
            )
        except queue.Full:
            logger.warning(
                "Preview transcription queue is full; dropping speculative work"
            )
            return False
        if getattr(self.recorder, "_preview_uses_shared_final_worker", False):
            dispatch_event.wait(timeout=1.0)
        return True

    def stop(self):
        """Requests shutdown and waits briefly for the current decode."""

        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            # A full queue means the worker is already active or will wake on
            # queued work; stop_event prevents another request from starting.
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=10)

    def _publish(self, result):
        recorder = self.recorder
        recorder.last_preview_transcription_result = result
        recorder.last_preview_transcription = result.text
        callback = getattr(recorder, "on_preview_transcription_finished", None)
        if callback is None:
            return
        try:
            run_callback(recorder, callback, result)
        except Exception:
            logger.error("Preview transcription callback failed", exc_info=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                request = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if request is None:
                break

            try:
                result = transcribe_preview(
                    self.recorder,
                    request.tail_audio,
                    request.live_text,
                    recording_id=request.recording_id,
                    use_prompt=request.use_prompt,
                    dispatch_event=request.dispatch_event,
                )
            except Exception as error:
                logger.warning(
                    "Preview transcription failed without blocking Final ASR: %s",
                    error,
                    exc_info=True,
                )
                result = _preview_error_result(self.recorder, request, error)

            self._publish(result)


def start_preview_transcription_worker(recorder):
    """Creates and starts the optional Preview worker."""

    if not getattr(recorder, "enable_preview_transcription", False):
        return None

    worker = PreviewTranscriptionWorker(recorder)
    recorder.preview_transcription_worker = worker
    recorder.preview_transcription_queue = worker.queue
    recorder.preview_transcription_stop_event = worker.stop_event
    recorder.preview_transcription_thread = worker.thread
    worker.start()
    return worker


def submit_preview_transcription_request(
    recorder,
    tail_audio,
    live_text,
    recording_id=0,
    use_prompt=True,
):
    """Queues Preview work and returns whether it was accepted."""

    worker = getattr(recorder, "preview_transcription_worker", None)
    if worker is None:
        return False
    return worker.submit(
        tail_audio,
        live_text,
        recording_id=recording_id,
        use_prompt=use_prompt,
    )


def stop_preview_transcription_worker(recorder):
    """Stops the optional Preview worker."""

    worker = getattr(recorder, "preview_transcription_worker", None)
    if worker is None:
        return
    worker.stop()
