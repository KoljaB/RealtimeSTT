from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder()

print("Say something...")

while (True): print(recorder.text(), end=" ", flush=True)