from RealtimeSTT import AudioToTextRecorder
if __name__ == '__main__':
    recorder = AudioToTextRecorder(spinner=False)

    print("Say something...")
    while (True): print(recorder.text(), end=" ", flush=True)