import RealtimeSTT 
import logging

recorder = RealtimeSTT.AudioToTextRecorder(level=logging.DEBUG)

print("Say something...")
print(recorder.text())