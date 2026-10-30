import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from RealtimeSTT.core import initialization


class FakeStreamingModel:
    supports_streaming = True

    def __init__(self):
        self.warmup_calls = 0

    def create_streaming_session(self, language=None, use_prompt=True):
        return object()

    def warmup(self, audio):
        self.warmup_calls += 1


def make_recorder(*, slow_model="slow-1120ms", fast_model="fast-80ms"):
    return SimpleNamespace(
        enable_realtime_transcription=True,
        use_main_model_for_realtime=False,
        _uses_external_realtime_transcription_executor=False,
        realtime_transcription_engine="sherpa_onnx",
        realtime_transcription_engine_options={"provider": "cpu"},
        realtime_model_type=slow_model,
        realtime_transcription_model=None,
        ultrafast_realtime_transcription_engine="sherpa_onnx",
        ultrafast_realtime_transcription_engine_options={"provider": "cpu"},
        ultrafast_realtime_model_type=fast_model,
        ultrafast_realtime_transcription_model=None,
        _ultrafast_uses_realtime_model=False,
        device="cpu",
        compute_type="int8",
        gpu_device_index=0,
        download_root=None,
        beam_size_realtime=3,
        initial_prompt_realtime=None,
        suppress_tokens=[-1],
        realtime_batch_size=1,
        faster_whisper_vad_filter=False,
        normalize_audio=False,
    )


class DualRealtimeInitializationTests(unittest.TestCase):
    def test_different_models_load_and_warm_independently(self):
        recorder = make_recorder()
        slow = FakeStreamingModel()
        fast = FakeStreamingModel()

        with mock.patch.object(
            initialization,
            "create_transcription_engine",
            side_effect=[slow, fast],
        ) as create_engine:
            with mock.patch.object(
                initialization.sf,
                "read",
                return_value=(np.zeros(160, dtype=np.float32), 16000),
            ):
                initialization._initialize_realtime_transcription_model(
                    recorder
                )

        self.assertEqual(create_engine.call_count, 2)
        self.assertEqual(
            create_engine.call_args_list[0].args[1].model,
            "slow-1120ms",
        )
        self.assertEqual(
            create_engine.call_args_list[1].args[1].model,
            "fast-80ms",
        )
        self.assertIs(recorder.realtime_transcription_model, slow)
        self.assertIs(recorder.ultrafast_realtime_transcription_model, fast)
        self.assertEqual(slow.warmup_calls, 1)
        self.assertEqual(fast.warmup_calls, 1)
        self.assertFalse(recorder._ultrafast_uses_realtime_model)

    def test_identical_configs_share_loaded_model_and_single_warmup(self):
        recorder = make_recorder(
            slow_model="shared-streaming",
            fast_model="shared-streaming",
        )
        shared = FakeStreamingModel()

        with mock.patch.object(
            initialization,
            "create_transcription_engine",
            return_value=shared,
        ) as create_engine:
            with mock.patch.object(
                initialization.sf,
                "read",
                return_value=(np.zeros(160, dtype=np.float32), 16000),
            ):
                initialization._initialize_realtime_transcription_model(
                    recorder
                )

        create_engine.assert_called_once()
        self.assertIs(recorder.realtime_transcription_model, shared)
        self.assertIs(recorder.ultrafast_realtime_transcription_model, shared)
        self.assertEqual(shared.warmup_calls, 1)
        self.assertTrue(recorder._ultrafast_uses_realtime_model)


if __name__ == "__main__":
    unittest.main()
