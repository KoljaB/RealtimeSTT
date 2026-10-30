import types
import sys
import unittest
from unittest import mock


try:
    import numpy as np
    import scipy.signal  # noqa: F401

    from RealtimeSTT.core.audio_input_worker import run_audio_data_worker
except ModuleNotFoundError as exc:
    np = None
    run_audio_data_worker = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FakeQueue:
    def __init__(self, shutdown_event):
        self.items = []
        self.shutdown_event = shutdown_event

    def put(self, item):
        self.items.append(item)
        self.shutdown_event.set()


class FakeEvent:
    def __init__(self):
        self._is_set = False

    def is_set(self):
        return self._is_set

    def set(self):
        self._is_set = True


class FakeUseMicrophone:
    value = True


class FakeStream:
    def __init__(self):
        self.closed = False

    def start_stream(self):
        pass

    def read(self, chunk_size, exception_on_overflow=False):
        return np.arange(chunk_size, dtype=np.int16).tobytes()

    def stop_stream(self):
        pass

    def close(self):
        self.closed = True


class FakeAudioInterface:
    def __init__(self):
        self.open_rates = []
        self.terminated = False

    def get_device_count(self):
        return 1

    def get_device_info_by_index(self, device_index):
        return {
            "index": device_index,
            "maxInputChannels": 1,
            "defaultSampleRate": 48000,
        }

    def get_default_input_device_info(self):
        return self.get_device_info_by_index(0)

    def open(
            self,
            *,
            format,
            channels,
            rate,
            input,
            frames_per_buffer,
            input_device_index,
            start=True,
    ):
        self.open_rates.append(rate)
        if rate == 16000:
            raise OSError("16 kHz unsupported by this fake device")
        return FakeStream()

    def terminate(self):
        self.terminated = True


class AudioInputWorkerTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"Audio input worker dependency missing: {IMPORT_ERROR}")

    def test_worker_falls_back_to_device_rate_when_target_rate_fails(self):
        shutdown_event = FakeEvent()
        queue = FakeQueue(shutdown_event)
        interrupt_stop_event = FakeEvent()
        audio_interface = FakeAudioInterface()
        fake_pyaudio = types.SimpleNamespace(
            paInt16=8,
            paInputOverflowed=-9981,
            PyAudio=lambda: audio_interface,
        )

        def stop_retries(*_args, **_kwargs):
            shutdown_event.set()

        with mock.patch.dict(sys.modules, {"pyaudio": fake_pyaudio}):
            with mock.patch(
                    "RealtimeSTT.core.audio_input_worker.time.sleep",
                    side_effect=stop_retries,
            ):
                run_audio_data_worker(
                    queue,
                    target_sample_rate=16000,
                    buffer_size=64,
                    input_device_index=None,
                    shutdown_event=shutdown_event,
                    interrupt_stop_event=interrupt_stop_event,
                    use_microphone=FakeUseMicrophone(),
                )

        self.assertIn(48000, audio_interface.open_rates)
        self.assertTrue(queue.items)
        self.assertTrue(audio_interface.terminated)


if __name__ == "__main__":
    unittest.main()
