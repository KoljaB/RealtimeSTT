from RealtimeSTT import AudioToTextRecorder
import os


def test_offline_vad():
    local_dir = os.path.abspath("./silero-vad")
    source = "local"
    recorder = AudioToTextRecorder(
        silero_repo_or_dir=local_dir, silero_source=source, silero_deactivity_detection=True)
    assert recorder.silero_vad_model is not None, "Failed to load Silero VAD model offline"
    print("Offline VAD test passed!")


if __name__ == '__main__':
    test_offline_vad()
