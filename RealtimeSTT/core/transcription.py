"""Internal final-transcription worker runtime."""

import copy
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


class TranscriptionWorker:
    """
    Runs the final-transcription model worker.
    """

    def __init__(self, conn, stdout_pipe, transcription_engine, transcription_engine_options, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter, normalize_audio):
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
        self.queue = queue.Queue()

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
            engine = create_transcription_engine(
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
            engine.warmup(warmup_audio_data)
        except Exception as e:
            logging.exception(
                f"Error initializing main {self.transcription_engine} transcription model: {e}"
            )
            raise

        self.ready_event.set()
        logging.debug(
            f"{self.transcription_engine} main speech to text transcription model initialized successfully"
        )

        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language, use_prompt = self.queue.get(timeout=0.1)
                    try:
                        logging.debug(f"Transcribing audio with language {language}")
                        start_t = time.time()
                        transcription_result = engine.transcribe(
                            audio,
                            language=language,
                            use_prompt=use_prompt,
                        )
                        elapsed = time.time() - start_t
                        logging.debug(
                            f"Final text detected with main model: {transcription_result.text} in {elapsed:.4f}s"
                        )
                        self.conn.send(('success', transcription_result))
                    except Exception as e:
                        logging.error(f"General error in transcription: {e}", exc_info=True)
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
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish


def run_transcription_worker(*args, **kwargs):
    """
    Runs the final-transcription worker process.
    """
    worker = TranscriptionWorker(*args, **kwargs)
    worker.run()


def call_transcription_executor(executor, audio, language, use_prompt):
    """
    Calls object-style or function-style transcription executors.
    """
    if hasattr(executor, "transcribe"):
        return executor.transcribe(
            audio,
            language=language if language else None,
            use_prompt=use_prompt,
        )
    return executor(
        audio,
        language=language if language else None,
        use_prompt=use_prompt,
    )


def submit_transcription_request(recorder, audio, language, use_prompt):
    """
    Submits audio for final transcription.
    """
    if recorder._uses_external_transcription_executor:
        audio_copy = copy.deepcopy(audio)

        def _run_external_transcription():
            try:
                result = call_transcription_executor(
                    recorder.transcription_executor,
                    audio_copy,
                    language,
                    use_prompt,
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

    recorder.parent_transcription_pipe.send((audio, language, use_prompt))
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
