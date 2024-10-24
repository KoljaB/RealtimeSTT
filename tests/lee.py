if __name__ == '__main__':

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from RealtimeSTT.audio_recorder import AudioToTextRecorder

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.2,
        webrtc_sensitivity=3,
        model="small",
        language="ko",
        )

    print("Say something...")
    
    try:
        while (True):
            print("Detected text: " + recorder.text())
    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")