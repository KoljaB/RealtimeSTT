from RealtimeSTT import AudioToTextRecorder
if __name__ == '__main__':
    recorder = AudioToTextRecorder(spinner=False, model="tiny.en", language="en")

    print("Say something...")
    while (True): print(recorder.text(), end=" ", flush=True)