"""
Internal final-transcription worker runtime.
"""

import copy
import itertools
import logging
import os
import queue
import signal as system_signal
import threading
import time

import soundfile as sf

from ..transcription_engines import (
    TranscriptionEngineConfig,
    create_transcription_engine,
)


TIME_SLEEP = 0.02
_SHARED_PREVIEW_REQUEST = "__realtimestt_shared_preview__"


class TranscriptionWorker:
    """
    Runs the final-transcription model worker.
    """

    def __init__(self, conn, stdout_pipe, transcription_engine, transcription_engine_options, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter, normalize_audio,
                 shared_preview_result_queue=None):
        """
        Initializes worker state and communication channels.
        """
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.transcription_engine = transcription_engine
        self.transcription_engine_options = transcription_engine_options or {}
        self.model_path = model_path
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.batch_size = batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.normalize_audio = normalize_audio
        self.shared_preview_result_queue = shared_preview_result_queue
        self.queue = queue.Queue()
        self.engine = None
        self._engine_close_lock = threading.Lock()
        self._engine_closed = False


    def custom_print(self, *args, **kwargs):
        """
        Forwards worker print output through the stdout pipe.
        """
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        """
        Transfers pipe messages into the worker queue.
        """
        while not self.shutdown_event.is_set():
            try:
                if self.conn.poll(0.01):  # Short poll keeps shutdown responsive.
                    data = self.conn.recv()
                    self.queue.put(data)
                else:
                    time.sleep(TIME_SLEEP)
            except Exception as e:
                logging.error(f"Error receiving data from connection: {e}", exc_info=True)
                time.sleep(TIME_SLEEP)

    def _close_engine_once(self):
        """Closes the resident engine exactly once from any worker thread."""

        with self._engine_close_lock:
            if self._engine_closed:
                return
            self._engine_closed = True
            close = getattr(self.engine, "close", None)
            if not callable(close):
                return
            try:
                close()
            except Exception:
                logging.error(
                    "Could not close the %s transcription engine.",
                    self.transcription_engine,
                    exc_info=True,
                )

    def _close_engine_on_shutdown(self):
        """Cancels native inference as soon as recorder shutdown is requested."""

        self.shutdown_event.wait()
        self._close_engine_once()

    def run(self):
        """
        Initializes the engine and processes queued transcription requests.
        """
        if __name__ == "__main__":
             system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
             __builtins__['print'] = self.custom_print

        logging.info(
            f"Initializing {self.transcription_engine} main transcription model {self.model_path}"
        )

        try:
            self.engine = create_transcription_engine(
                self.transcription_engine,
                TranscriptionEngineConfig(
                    model=self.model_path,
                    download_root=self.download_root,
                    compute_type=self.compute_type,
                    gpu_device_index=self.gpu_device_index,
                    device=self.device,
                    beam_size=self.beam_size,
                    initial_prompt=self.initial_prompt,
                    suppress_tokens=self.suppress_tokens,
                    batch_size=self.batch_size,
                    vad_filter=self.faster_whisper_vad_filter,
                    normalize_audio=self.normalize_audio,
                    engine_options=self.transcription_engine_options,
                ),
            )

            # Warmup pays model startup cost before the first user request.
            current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            warmup_audio_path = os.path.join(
                current_dir, "assets", "warmup_audio.wav"
            )
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            self.engine.warmup(warmup_audio_data)
        except Exception as e:
            self._close_engine_once()
            logging.exception(
                f"Error initializing main {self.transcription_engine} transcription model: {e}"
            )
            raise

        engine_close_thread = threading.Thread(
            target=self._close_engine_on_shutdown,
            name="RealtimeSTTFinalEngineShutdown",
            daemon=True,
        )
        engine_close_thread.start()

        self.ready_event.set()
        logging.debug(
            f"{self.transcription_engine} main speech to text transcription model initialized successfully"
        )

        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    request = self.queue.get(timeout=0.1)
                    is_shared_preview = (
                        isinstance(request, tuple)
                        and bool(request)
                        and isinstance(request[0], str)
                        and request[0] == _SHARED_PREVIEW_REQUEST
                    )
                    if is_shared_preview:
                        (
                            _,
                            request_id,
                            audio,
                            language,
                            use_prompt,
                            options,
                        ) = request
                    else:
                        audio, language, use_prompt = request[:3]
                        options = request[3] if len(request) > 3 else {}

                    try:
                        logging.debug(
                            "%s transcribing audio with language %s",
                            "Preview" if is_shared_preview else "Final",
                            language,
                        )
                        start_t = time.time()
                        transcription_result = self.engine.transcribe(
                            audio,
                            language=language,
                            use_prompt=use_prompt,
                            **options,
                        )
                        elapsed = time.time() - start_t
                        logging.debug(
                            "%s text detected with main model: %s in %.4fs",
                            "Preview" if is_shared_preview else "Final",
                            transcription_result.text,
                            elapsed,
                        )
                        if is_shared_preview:
                            self.shared_preview_result_queue.put(
                                (request_id, "success", transcription_result)
                            )
                        else:
                            self.conn.send(('success', transcription_result))
                    except Exception as e:
                        if self.shutdown_event.is_set():
                            logging.debug(
                                "%s transcription cancelled during shutdown.",
                                "Preview" if is_shared_preview else "Final",
                            )
                            break
                        logging.error(f"General error in transcription: {e}", exc_info=True)
                        if is_shared_preview:
                            self.shared_preview_result_queue.put(
                                (request_id, "error", str(e))
                            )
                        else:
                            self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}", exc_info=True)
        finally:
            self.shutdown_event.set()
            self._close_engine_once()
            engine_close_thread.join()
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            polling_thread.join()  # Wait for the polling thread to finish


def run_transcription_worker(*args, **kwargs):
    """
    Runs the final-transcription worker process.
    """
    worker = TranscriptionWorker(*args, **kwargs)
    worker.run()


class SharedFinalModelExecutor:
    """Uses the already-loaded Final model for Preview requests."""

    def __init__(self, recorder):
        self.recorder = recorder
        self._request_ids = itertools.count(1)
        self._lock = threading.Lock()

    def transcribe(self, audio, language=None, use_prompt=True, **options):
        """Submits one Preview request and waits for its isolated response."""

        return self._transcribe(
            audio,
            language=language,
            use_prompt=use_prompt,
            dispatch_event=None,
            options=options,
        )

    def transcribe_preview(
        self,
        audio,
        language=None,
        use_prompt=True,
        dispatch_event=None,
        **options,
    ):
        """Dispatches Preview before returning control to the VAD thread."""

        return self._transcribe(
            audio,
            language=language,
            use_prompt=use_prompt,
            dispatch_event=dispatch_event,
            options=options,
        )

    def _transcribe(
        self,
        audio,
        language,
        use_prompt,
        dispatch_event,
        options,
    ):
        """Submits one request and waits for the isolated response."""

        parent_pipe = getattr(
            self.recorder,
            "parent_transcription_pipe",
            None,
        )
        result_queue = getattr(
            self.recorder,
            "shared_preview_transcription_result_queue",
            None,
        )
        if parent_pipe is None or result_queue is None:
            raise RuntimeError("Shared Final model Preview channel is unavailable")

        with self._lock:
            request_id = next(self._request_ids)
            try:
                parent_pipe.send(
                    (
                        _SHARED_PREVIEW_REQUEST,
                        request_id,
                        copy.deepcopy(audio),
                        language,
                        use_prompt,
                        options,
                    )
                )
            finally:
                if dispatch_event is not None:
                    dispatch_event.set()

            while True:
                shutdown_event = getattr(self.recorder, "shutdown_event", None)
                if getattr(self.recorder, "is_shut_down", False) or (
                    shutdown_event is not None and shutdown_event.is_set()
                ):
                    raise RuntimeError("Shared Final model Preview was interrupted")

                try:
                    response_id, status, result = result_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if response_id != request_id:
                    logging.warning(
                        "Discarding unexpected shared Preview response %s; expected %s",
                        response_id,
                        request_id,
                    )
                    continue
                if status == "success":
                    return result
                raise RuntimeError(result)


def call_transcription_executor(executor, audio, language, use_prompt, **options):
    """
    Calls object-style or function-style transcription executors.
    """
    if hasattr(executor, "transcribe"):
        return executor.transcribe(
            audio,
            language=language if language else None,
            use_prompt=use_prompt,
            **options,
        )
    return executor(
        audio,
        language=language if language else None,
        use_prompt=use_prompt,
        **options,
    )


def submit_transcription_request(recorder, audio, language, use_prompt, options=None):
    """
    Submits audio for final transcription.
    """
    if recorder._uses_external_transcription_executor:
        audio_copy = copy.deepcopy(audio)

        def _run_external_transcription():
            """
            Runs final transcription through an external executor.
            """

            try:
                result = call_transcription_executor(
                    recorder.transcription_executor,
                    audio_copy,
                    language,
                    use_prompt,
                    **(options or {}),
                )
                recorder._external_transcription_results.put(("success", result))
            except Exception as exc:
                recorder._external_transcription_results.put(("error", str(exc)))

        recorder.transcribe_count += 1
        thread = threading.Thread(
            target=_run_external_transcription,
            name="RealtimeSTTExternalFinalTranscription",
            daemon=True,
        )
        recorder._external_transcription_threads.append(thread)
        thread.start()
        return

    request = (audio, language, use_prompt)
    if options:
        request = (audio, language, use_prompt, options)
    recorder.parent_transcription_pipe.send(request)
    recorder.transcribe_count += 1


def receive_transcription_result(recorder, timeout=0.1):
    """
    Receives a final-transcription result when one is ready.
    """
    if recorder._uses_external_transcription_executor:
        try:
            return recorder._external_transcription_results.get(timeout=timeout)
        except queue.Empty:
            return None

    if not recorder.parent_transcription_pipe.poll(timeout):
        return None
    return recorder.parent_transcription_pipe.recv()
