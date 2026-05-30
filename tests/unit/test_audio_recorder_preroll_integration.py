import collections
import unittest

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
    from RealtimeSTT.core.preroll import PrerollFrameMetadata
except Exception as exc:  # pragma: no cover - optional runtime deps may be absent
    AudioToTextRecorder = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


SAMPLE_RATE = 16000
FRAME_SAMPLES = 320


def chunk(value, sample_count=FRAME_SAMPLES):
    return int(value).to_bytes(2, byteorder="little", signed=True) * sample_count


def metadata(is_speech, rms):
    return PrerollFrameMetadata(
        sample_count=FRAME_SAMPLES,
        is_speech=is_speech,
        rms=rms,
    )


class AudioRecorderPrerollIntegrationTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"RealtimeSTT import failed: {IMPORT_ERROR}")

    def test_selected_prebuffer_frames_are_the_frames_taken_into_recording(self):
        audio_frames = [chunk(0) for _ in range(30)] + [chunk(2000) for _ in range(8)]
        frame_metadata = [metadata(False, 5.0) for _ in range(30)] + [
            metadata(True, 300.0) for _ in range(8)
        ]
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.sample_rate = SAMPLE_RATE
        recorder.audio_buffer = collections.deque(audio_frames, maxlen=len(audio_frames))
        recorder.audio_buffer_metadata = collections.deque(
            frame_metadata,
            maxlen=len(frame_metadata),
        )
        recorder.pre_recording_buffer_trim_config = {
            "enabled": True,
            "min_silence_ms": 200.0,
            "guard_ms": 160.0,
            "max_gap_ms": 80.0,
            "min_included_ms": 600.0,
        }
        recorder._pending_preroll_selection = None

        selected_frames = recorder._selected_pre_recording_buffer_frames()

        selection = recorder._pending_preroll_selection
        self.assertIsNotNone(selection)
        self.assertEqual(selection.selected_frame_count, len(selected_frames))
        self.assertEqual(audio_frames[selection.start_index:], selected_frames)
        self.assertGreaterEqual(selection.included_seconds, 0.60)
        self.assertLess(len(selected_frames), len(audio_frames))


if __name__ == "__main__":
    unittest.main()
