from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(spinner=False)

print("Say something...")

while (True): print(recorder.text(), end=" ", flush=True)